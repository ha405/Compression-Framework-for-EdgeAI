import math
import os
import sys
import threading
import time
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn.modules.conv import _ConvNd

from ..looper.named_module import NamedModule
from ..quantization import QuantizeConfig
from ..utils.logger import setup_logger
from ..utils.torch import auto_select_torch_device, torch_compile, torch_sync
from .quantizer import HF_OPTIMUM, Quantizer

log = setup_logger()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CPU = torch.device("cpu")
DEVICE_0 = auto_select_torch_device(index=0)
DEVICE_1 = auto_select_torch_device(index=1)

lock = threading.Lock()

def get_number_of_rows_and_cols(layer: nn.Module):
    if isinstance(layer, NamedModule):
        layer = layer.module

    if isinstance(layer, transformers.Conv1D):
        return layer.weight.shape[1], layer.weight.shape[0]
    else:
        return layer.weight.shape[0], np.prod(layer.weight.shape[1:])

def _layer_forward(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(layer, (nn.Linear, transformers.Conv1D)):
        W_layer = layer.weight.t() if isinstance(layer, transformers.Conv1D) else layer.weight
        original_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        if isinstance(layer, transformers.Conv1D):
             pass

        out = F.linear(x, W_layer, layer.bias)

        if out.dim() > 1 and x.dim() > 2:
             out = out.reshape(*original_shape[:-1], out.shape[-1])
        return out
    else:
        log.warning(f"KL Divergence calculation not implemented for layer type {type(layer)}. Skipping KL term.")
        return torch.zeros(x.shape[0], layer.weight.shape[0], device=x.device)


class GPTQ:
    def __init__(self, module: nn.Module, qcfg: Optional[QuantizeConfig]=None):
        self.W_orig = module.weight.data.clone()
        self.bias_orig = module.bias.data.clone() if module.bias is not None else None

        self.rows, self.columns = get_number_of_rows_and_cols(module)
        if isinstance(module, NamedModule):
            self.module = module.module
            self.name = module.name
        else:
            self.name = HF_OPTIMUM
            self.module = module

        self._validate_module(self.module)

        self.qcfg = qcfg if qcfg else QuantizeConfig()
        self.device = self.module.weight.device

        self.module_copy = None

        self.H = None
        self.A = None
        self.G = None
        self.nsamples = 0

        self.quantizer = self.create_quantizer(name=self.name)

        self.fwd_inputs_buffered = True
        self.fwd_inputs_buffered_data: List[torch.Tensor] = []
        self.buffered_inputs_device = DEVICE_1 if DEVICE_0.index != DEVICE_1.index else CPU

        self.fwd_counter = 0

    @staticmethod
    def _validate_module(module):
        assert isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, transformers.Conv1D)), f"We supports only linear and convolutional layers. actual = `{module}`"

    def create_quantizer(self, name: str) -> Quantizer:
        return Quantizer(qcfg=self.qcfg, name=name)

    def shape(self):
        return (self.rows, self.columns)

    def _clone_module_weights(self, copy=True, device: torch.device = None):
        if not device:
            device = self.W_orig.device

        clone = self.W_orig.to(copy=copy, device=device)

        if isinstance(self.module, _ConvNd):
            clone = clone.flatten(1)
        if isinstance(self.module, transformers.Conv1D):
            clone = clone.t()

        return clone.float()

    @torch.inference_mode()
    def block_cholesky_inverse(self, L: torch.Tensor, upper=False, block_size=512):
        assert L.dim() == 2 and L.size(0) == L.size(1), "Input must be square"
        n = L.size(0)
        device = L.device
        dtype = L.dtype

        if upper:
            L = L.t()

        invA = torch.zeros_like(L)
        num_blocks = math.ceil(n / block_size)
        invL_cache = {}
        precomputed_terms = {} # Cache for L_ik @ invL_kk

        for k in reversed(range(num_blocks)):
            k_start = k * block_size
            k_end = min((k + 1) * block_size, n)
            k_size = k_end - k_start

            L_block = L[k_start:k_end, k_start:k_end]
            invL_block = torch.linalg.solve_triangular(
                L_block, torch.eye(k_size, device=device, dtype=dtype), upper=False
            )
            invL_cache[k] = invL_block
            invA[k_start:k_end, k_start:k_end] = invL_block.t() @ invL_block

            for i in range(k + 1, num_blocks):
                 i_start = i * block_size
                 i_end = min((i + 1) * block_size, n)
                 L_ik = L[i_start:i_end, k_start:k_end]
                 precomputed_terms[(i, k)] = L_ik @ invL_block
                 del L_ik

            for j in range(k):
                j_start = j * block_size
                j_end = min((j + 1) * block_size, n)
                j_size = j_end - j_start
                L_jk = L[j_start:j_end, k_start:k_end]

                invL_ik_blocks = []
                for i in range(k, num_blocks):
                    i_start = i * block_size
                    i_end = min((i + 1) * block_size, n)
                    i_size = i_end - i_start
                    if i == k: invL_ik = invL_block
                    else:
                        invL_ii = invL_cache[i]
                        # Use precomputed L_ik @ invL_kk if available
                        if (i, k) in precomputed_terms:
                             L_ik_invL_kk = precomputed_terms[(i, k)]
                        else:
                             # Fallback if precomputation wasn't done (should not happen with current structure)
                             L_ik_block = L[i_start:i_end, k_start:k_end]
                             L_ik_invL_kk = L_ik_block @ invL_block
                             del L_ik_block
                        invL_ik = -invL_ii @ L_ik_invL_kk
                        # del invL_ii # Keep ii block in cache
                        # del L_ik_invL_kk # No need to delete if from dict
                    invL_ik_blocks.append(invL_ik)

                temp_col = torch.cat([
                    (invL_ik.t() @ L_jk.t()) for invL_ik in invL_ik_blocks
                ], dim=0)

                del invL_ik_blocks

                invA[j_start:j_end, k_start:k_end] = temp_col[:j_size].t()
                invA[k_start:k_end, j_start:j_end] = temp_col[:j_size]
                del temp_col, L_jk

            # Clear precomputed terms related to column k after use
            keys_to_del = [key for key in precomputed_terms if key[1] == k]
            for key in keys_to_del:
                del precomputed_terms[key]

        del invL_cache
        del precomputed_terms

        return invA

    def add_batch(self, inp: torch.Tensor, out: Optional[torch.Tensor]=None):
        self.fwd_counter += 1
        self.fwd_inputs_buffered_data.append(inp.to(device=self.buffered_inputs_device, non_blocking=True))
        self.process_batch(inp.to(device=DEVICE_1, non_blocking=True))

    def process_batch(self, inp: torch.Tensor):
        reshaped_inp = inp.to(dtype=torch.float32)

        if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
            if reshaped_inp.shape[-1] != self.columns:
                 reshaped_inp = reshaped_inp.reshape(-1, reshaped_inp.shape[-1])
                 if reshaped_inp.shape[-1] != self.columns:
                      raise ValueError(f"Input shape {inp.shape} incompatible with layer columns {self.columns}")
            else:
                 reshaped_inp = reshaped_inp.reshape(-1, self.columns)

        elif isinstance(self.module, nn.Conv1d) or isinstance(self.module, nn.Conv2d):
             log.warning(f"Input reshaping for KL/Hessian might need adjustments for {type(self.module)}")
             reshaped_inp = reshaped_inp.reshape(-1, self.columns)

        batch_token_size = reshaped_inp.shape[0]
        if batch_token_size == 0:
             log.warning("Received batch with 0 tokens.")
             return 0, None, 0.0, 1.0

        if self.H is None:
            self.H = torch.zeros((self.columns, self.columns),
                                 dtype=torch.float32,
                                 device=DEVICE_1)

        if self.nsamples == 0:
             beta = 0.0
             alpha = 2.0 / batch_token_size if batch_token_size > 0 else 0.0
        else:
             beta = self.nsamples / (self.nsamples + batch_token_size)
             alpha = 2.0 / (self.nsamples + batch_token_size)

        self.H.addmm_(reshaped_inp.T, reshaped_inp, beta=beta, alpha=alpha)
        self.nsamples += batch_token_size
        del reshaped_inp
        return batch_token_size, None, alpha, beta

    @torch.inference_mode()
    def _get_teacher_outputs(self, calib_data: List[torch.Tensor]) -> torch.Tensor:
        log.info("Computing teacher outputs for KL divergence...")
        teacher_outputs = []
        # Determine original layer shape correctly
        if isinstance(self.module, transformers.Conv1D):
             in_features, out_features = self.W_orig.shape
        else: # Linear, Conv2d (flattened)
             out_features, in_features = self.W_orig.shape

        # Recreate layer based on type
        if isinstance(self.module, nn.Linear):
             orig_layer = nn.Linear(in_features, out_features, bias=self.bias_orig is not None, device=self.W_orig.device, dtype=self.W_orig.dtype)
        elif isinstance(self.module, transformers.Conv1D):
             orig_layer = transformers.Conv1D(out_features, in_features) # nx, nf format for Conv1D
             orig_layer = orig_layer.to(device=self.W_orig.device, dtype=self.W_orig.dtype)
        # Add other layer types if needed
        else:
             raise TypeError(f"Layer type {type(self.module)} not supported for teacher output generation.")

        orig_layer.weight.data = self.W_orig
        if self.bias_orig is not None:
             orig_layer.bias.data = self.bias_orig
        else:
             orig_layer.bias = None

        for inp in calib_data:
            inp_device = inp.to(DEVICE_0)
            logits = _layer_forward(orig_layer, inp_device)
            p_t = torch.softmax(logits / self.qcfg.kl_tau, dim=-1)
            teacher_outputs.append(p_t.to(self.buffered_inputs_device))
            del inp_device, logits, p_t

        del orig_layer
        return torch.cat(teacher_outputs, dim=0)

    @torch.inference_mode()
    def _compute_distillation_hessian(self, p_t: torch.Tensor, calib_data: List[torch.Tensor]) -> torch.Tensor:
        log.info("Computing KL distillation Hessian A...")
        A = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=DEVICE_1)
        nsamples_processed = 0
        p_t_idx = 0

        for inp in calib_data:
            inp_device = inp.to(DEVICE_1)
            if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
                 reshaped_inp = inp_device.reshape(-1, self.columns).to(torch.float32)
            else:
                 reshaped_inp = inp_device.reshape(-1, self.columns).to(torch.float32)

            batch_size = reshaped_inp.shape[0]
            if batch_size == 0: continue
            current_p_t = p_t[p_t_idx : p_t_idx + batch_size].to(DEVICE_1)

            for i in range(batch_size):
                xt = reshaped_inp[i:i+1, :]
                pt_i = current_p_t[i:i+1, :]
                pt_scalar_weight = torch.sum(pt_i * (1.0 - pt_i))
                xt_outer = xt.T @ xt
                A.add_(xt_outer, alpha=pt_scalar_weight.item())

            p_t_idx += batch_size
            nsamples_processed += batch_size
            del inp_device, reshaped_inp, current_p_t

        if nsamples_processed > 0:
            A /= nsamples_processed

        log.info(f"Computed KL Hessian A using {nsamples_processed} samples.")
        return A

    def _compute_kl_gradients(self, W_current: torch.Tensor, calib_data: List[torch.Tensor], p_t_teacher: torch.Tensor) -> torch.Tensor:
        log.info("Computing KL gradients G...")
        G_device = DEVICE_1
        G = torch.zeros_like(W_current, device=G_device)
        W_current_device = W_current.device

        q_t_student_list = []
        # Determine original layer shape correctly
        if isinstance(self.module, transformers.Conv1D):
             in_features, out_features = self.W_orig.shape
             temp_layer = transformers.Conv1D(out_features, in_features).to(W_current_device)
             temp_layer.weight.data = W_current.t() # Conv1D expects (in, out)
        elif isinstance(self.module, nn.Linear):
             out_features, in_features = W_current.shape
             temp_layer = nn.Linear(in_features, out_features, bias=self.bias_orig is not None, device=W_current_device)
             temp_layer.weight.data = W_current
        else:
              raise TypeError(f"Layer type {type(self.module)} not supported for KL gradient generation.")

        if self.bias_orig is not None:
             temp_layer.bias.data = self.bias_orig.to(W_current_device)
        else:
             temp_layer.bias = None

        with torch.inference_mode():
             for inp in calib_data:
                 inp_device = inp.to(W_current_device)
                 logits_student = _layer_forward(temp_layer, inp_device)
                 q_t = torch.softmax(logits_student / self.qcfg.kl_tau, dim=-1)
                 q_t_student_list.append(q_t.to(self.buffered_inputs_device))
                 del inp_device, logits_student, q_t
        del temp_layer
        q_t_student = torch.cat(q_t_student_list, dim=0)
        del q_t_student_list

        nsamples_processed = 0
        p_t_idx = 0
        beta_tau_factor = -self.qcfg.kl_beta / self.qcfg.kl_tau

        for inp in calib_data:
            inp_device = inp.to(G_device)
            if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
                 reshaped_inp = inp_device.reshape(-1, self.columns).to(torch.float32)
            else:
                 reshaped_inp = inp_device.reshape(-1, self.columns).to(torch.float32)

            batch_size = reshaped_inp.shape[0]
            if batch_size == 0: continue
            p_t_batch = p_t_teacher[p_t_idx : p_t_idx + batch_size].to(G_device)
            q_t_batch = q_t_student[p_t_idx : p_t_idx + batch_size].to(G_device)

            diff = p_t_batch - q_t_batch
            G.addmm_(diff.T, reshaped_inp, alpha=beta_tau_factor)

            p_t_idx += batch_size
            nsamples_processed += batch_size
            del inp_device, reshaped_inp, p_t_batch, q_t_batch, diff

        if nsamples_processed > 0:
            G /= nsamples_processed

        log.info(f"Computed KL gradients G using {nsamples_processed} samples.")
        return G.to(W_current_device)

    def hf_quantize(
        self,
        blocksize=128,
        percdamp=0.01,
        damp_auto_increment=0.0015,
        group_size=-1,
        actorder=False,
        static_groups=False,
        kl_divergence=False,
        kl_strategy='global',
        kl_beta=0.1,
        kl_tau=1.0,
    ):
        self.qcfg.group_size = group_size
        self.qcfg.damp_percent = percdamp
        self.qcfg.damp_auto_increment = damp_auto_increment
        self.qcfg.desc_act = actorder
        self.qcfg.static_groups = static_groups
        self.qcfg.kl_divergence = kl_divergence
        self.qcfg.kl_strategy = kl_strategy
        self.qcfg.kl_beta = kl_beta
        self.qcfg.kl_tau = kl_tau

        (Q, scale, zero, g_idx, duration, avg_loss, damp_percent, nsamples) = self.quantize(blocksize=blocksize)

        if isinstance(self.module, transformers.Conv1D):
             Q_final = Q.t()
        elif isinstance(self.module, _ConvNd):
             Q_final = Q.reshape(self.module.weight.shape)
        else:
             Q_final = Q

        self.module.weight.data = Q_final.to(self.module.weight.device, dtype=self.module.weight.dtype)

        return scale, zero, g_idx, duration, avg_loss, damp_percent

    @torch.inference_mode()
    def hessian_inverse(self, H_eff: torch.Tensor):
        damp = self.qcfg.damp_percent
        H_damped = H_eff.clone()
        H_damped_inv = None # Initialize

        while 1 > damp > 0:
            try:
                diag_indices = torch.arange(self.columns, device=H_damped.device)
                damping_value = damp * torch.mean(H_damped[diag_indices, diag_indices])

                current_H_damped = H_damped.clone()
                current_H_damped[diag_indices, diag_indices] += damping_value

                with lock:
                    L = torch.linalg.cholesky(current_H_damped)
                    try:
                        H_damped_inv = torch.cholesky_inverse(L)
                    except torch.OutOfMemoryError:
                        log.warning("OOM in torch.cholesky_inverse, falling back to block_cholesky_inverse.")
                        H_damped_inv = self.block_cholesky_inverse(L, block_size=self.columns // 2)
                        log.warning("Quantization: OOM bypassed via low memory math at a cost of lower accuracy: `cholesky_inverse`")

                del current_H_damped, L
                break

            except torch._C._LinAlgError as e:
                 if 'current_H_damped' in locals(): del current_H_damped
                 if self.qcfg.damp_auto_increment > 0:
                    old_damp = damp
                    damp += self.qcfg.damp_auto_increment
                    log.warning(
                        f"Quantization: Cholesky failed with damp={old_damp:.5f}. Increasing to {damp:.5f}.")
                 else:
                    log.error(
                        f"Quantization: Cholesky failed with damp={damp:.5f}. Auto-increment is off. Increase damp or nsamples.")
                    raise e

        if not (0 < damp < 1):
            raise ValueError(f"Quantization: `damp_percent` ended outside (0, 1): {damp}")
        if H_damped_inv is None:
            raise RuntimeError("Hessian inversion failed to produce a result.")

        log.info(f"Hessian inversion successful with final damp={damp:.5f}")
        return H_damped_inv, damp

    @torch.inference_mode()
    def quantize(
        self,
        blocksize=128,
    ):
        start_time = time.time()
        log.info(f"Starting quantization for layer {self.name}...")

        if self.qcfg.kl_divergence and not self.fwd_inputs_buffered_data:
            log.warning("KL divergence enabled but no calibration data was buffered. Skipping KL term.")
            self.qcfg.kl_divergence = False

        all_calib_inputs = self.fwd_inputs_buffered_data
        self.fwd_inputs_buffered_data = []

        p_t_teacher = None
        if self.qcfg.kl_divergence:
            p_t_teacher = self._get_teacher_outputs(all_calib_inputs)

        if self.H is None or self.nsamples == 0:
             raise RuntimeError("Hessian H not computed. Ensure calibration data was added.")

        H_eff = self.H.clone()

        if self.qcfg.kl_divergence and self.qcfg.kl_strategy == 'global':
            self.A = self._compute_distillation_hessian(p_t_teacher, all_calib_inputs)
            log.info(f"Applying Global KL strategy with beta={self.qcfg.kl_beta:.4f}")
            H_eff += self.qcfg.kl_beta * self.A
            if hasattr(self, 'A'): del self.A

        diag_H_eff = torch.diag(H_eff)
        dead = diag_H_eff == 0
        W = self._clone_module_weights(device=DEVICE_1)
        if torch.any(dead):
             log.warning(f"Found {torch.sum(dead)} dead columns based on Hessian diagonal. Setting H_eff[dead, dead]=1 and W[:, dead]=0.")
             H_eff[dead, dead] = 1.0
             W[:, dead] = 0.0

        perm, invperm = None, None
        if self.qcfg.desc_act:
            log.info("Applying activation ordering...")
            perm = torch.argsort(diag_H_eff, descending=True)
            W = W[:, perm]
            H_eff = H_eff[perm][:, perm]
            invperm = torch.argsort(perm)
        del diag_H_eff # Free memory

        Hinv, final_damp = self.hessian_inverse(H_eff)
        del H_eff

        self.G = None
        if self.qcfg.kl_divergence and self.qcfg.kl_strategy == 'local':
            log.info(f"Applying Local KL strategy with beta={self.qcfg.kl_beta:.4f}")
            self.G = self._compute_kl_gradients(W, all_calib_inputs, p_t_teacher)

        del all_calib_inputs
        if p_t_teacher is not None: del p_t_teacher

        Q = torch.zeros_like(W)
        Losses = torch.zeros((self.columns,), device=DEVICE_1) # Store per-column loss sum

        scale = []
        zero = []
        group_quantizer_index = 0

        static_groups_quantizers = []
        if self.qcfg.group_size != -1 and self.qcfg.static_groups:
             import copy
             log.info(f"Using static groups (size={self.qcfg.group_size})")
             num_groups = math.ceil(self.columns / self.qcfg.group_size)
             current_quantizer = copy.deepcopy(self.quantizer)

             for group_idx in range(num_groups):
                 g_start = group_idx * self.qcfg.group_size
                 g_end = min(g_start + self.qcfg.group_size, self.columns)
                 current_quantizer.find_params(W[:, g_start:g_end], weight=True)
                 scale.append(current_quantizer.scale.clone())
                 zero.append(current_quantizer.zero.clone())
                 static_groups_quantizers.append(copy.deepcopy(current_quantizer))

        log.info(f"Starting column-wise quantization loop (blocksize={blocksize})...")
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W_block = W[:, i1:i2].clone()
            Q_block = torch.zeros_like(W_block)
            Err_block = torch.zeros_like(W_block)
            Hinv_block = Hinv[i1:i2, i1:i2].clone()

            for i in range(count):
                col_idx_global = i1 + i
                w = W_block[:, i]

                h_inv_ii = Hinv_block[i, i]
                if h_inv_ii <= 0:
                     h_inv_ii = torch.tensor(1e-8, device=h_inv_ii.device, dtype=h_inv_ii.dtype)

                d_sqrt = torch.sqrt(h_inv_ii)

                current_col_quantizer = self.quantizer
                if self.qcfg.group_size != -1:
                    current_global_col = col_idx_global
                    if self.qcfg.desc_act:
                        current_global_col = perm[current_global_col].item()
                    group_idx = current_global_col // self.qcfg.group_size
                    if self.qcfg.static_groups:
                         current_col_quantizer = static_groups_quantizers[group_idx]
                    else:
                         is_group_start = (current_global_col % self.qcfg.group_size) == 0
                         if is_group_start:
                             g_end = min(current_global_col + self.qcfg.group_size, self.columns)
                             self.quantizer.find_params(W[:, current_global_col : g_end], weight=True)
                             if group_idx == group_quantizer_index:
                                  scale.append(self.quantizer.scale.clone())
                                  zero.append(self.quantizer.zero.clone())
                                  group_quantizer_index += 1

                w_target = w
                if self.qcfg.kl_divergence and self.qcfg.kl_strategy == 'local':
                    gi = self.G[:, col_idx_global]
                    w_target = w + (self.qcfg.kl_beta / 2.0) * h_inv_ii * gi

                q = current_col_quantizer.quantize(w_target.unsqueeze(1)).flatten()
                Q_block[:, i] = q

                err = w_target - q
                loss_col = torch.sum((w - q)**2) / (2.0 * h_inv_ii)
                Losses[col_idx_global] = loss_col # Store sum of squared errors for the column

                err1 = err / d_sqrt
                Err_block[:, i] = err1

                if i < count - 1:
                     update_term = err1.unsqueeze(1).matmul(Hinv_block[i, i+1:].unsqueeze(0))
                     W_block[:, i+1:] -= update_term
                     del update_term

            Q[:, i1:i2] = Q_block

            if i2 < self.columns:
                inter_block_update = Err_block.matmul(Hinv[i1:i2, i2:])
                W[:, i2:] -= inter_block_update
                del inter_block_update

            del W_block, Q_block, Err_block, Hinv_block
            if i1 % (blocksize*4) == 0 : torch_sync() # Sync periodically

        del Hinv
        if hasattr(self, 'G'): del self.G

        total_loss = torch.sum(Losses).item()
        avg_loss = total_loss / self.nsamples if self.nsamples > 0 else 0.0

        if math.isnan(avg_loss):
            log.error(f"Quantization resulted in NaN loss for layer {self.name}. Original loss sum: {total_loss}")
            raise ValueError(f"Quantization failed due to NaN loss for `{self.name}`")
        log.info(f"Quantization avg loss = {avg_loss:.6f}")

        del Losses

        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns
        g_idx = None
        if self.qcfg.group_size != -1:
            if self.qcfg.desc_act:
                 original_col_indices = invperm
                 g_idx = [original_col_indices[i].item() // group_size for i in range(self.columns)]
            else:
                 g_idx = [i // group_size for i in range(self.columns)]
            g_idx = torch.tensor(g_idx, dtype=torch.int32, device=DEVICE_1)

        if self.qcfg.desc_act:
            log.info("Applying inverse permutation to Q.")
            Q = Q[:, invperm]

        if not scale:
             log.warning("Scale/zero list is empty. Using last quantizer params.")
             if self.qcfg.group_size != -1 and self.qcfg.static_groups:
                  pass
             else:
                  scale.append(self.quantizer.scale)
                  zero.append(self.quantizer.zero)

        final_scale = torch.cat(scale, dim=1).to(DEVICE_1)
        final_zero = torch.cat(zero, dim=1).to(DEVICE_1)
        del scale, zero

        duration = time.time() - start_time
        log.info(f"Quantization finished in {duration:.2f} seconds.")

        return Q, final_scale, final_zero, g_idx, duration, avg_loss, final_damp, self.nsamples

    def free(self):
        log.debug(f"Freeing resources for GPTQ layer {self.name}")
        if hasattr(self, "H"): del self.H
        if hasattr(self, "A"): del self.A
        if hasattr(self, "G"): del self.G
        if hasattr(self, "module_copy"): del self.module_copy
        if hasattr(self, "W_orig"): del self.W_orig
        if hasattr(self, "bias_orig"): del self.bias_orig
        self.fwd_inputs_buffered_data.clear()

__all__ = ["GPTQ"]