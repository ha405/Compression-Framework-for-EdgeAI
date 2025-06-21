import math
import os
import sys
import threading
import time
from typing import Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    class Conv1D(nn.Module): pass

from torch.nn.modules.conv import _ConvNd

from ..looper.named_module import NamedModule
from ..quantization import QuantizeConfig
from ..utils.logger import setup_logger
from ..utils.torch import auto_select_torch_device, torch_compile, torch_sync

try:
    from .quantizer import HF_OPTIMUM, Quantizer
except ImportError:
    try: from quantizer import HF_OPTIMUM, Quantizer
    except ImportError: HF_OPTIMUM="hf_optimum"; Quantizer=None

log = setup_logger()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CPU = torch.device("cpu")
try:
    if torch.cuda.is_available():
        DEVICE_0 = auto_select_torch_device(index=0)
        DEVICE_1 = auto_select_torch_device(index=1) if torch.cuda.device_count() > 1 else DEVICE_0
    else:
        DEVICE_0 = CPU
        DEVICE_1 = CPU
except Exception as e:
    log.warning(f"Could not automatically select torch devices: {e}. Defaulting to CPU.")
    DEVICE_0 = CPU
    DEVICE_1 = CPU

lock = threading.Lock()

def get_number_of_rows_and_cols(layer: nn.Module):
    if isinstance(layer, NamedModule): layer = layer.module
    if isinstance(layer, transformers.Conv1D): return layer.weight.shape[1], layer.weight.shape[0]
    elif isinstance(layer, (nn.Linear, nn.Conv2d)): return layer.weight.shape[0], int(np.prod(layer.weight.shape[1:]))
    else:
        if hasattr(layer, 'weight') and isinstance(layer.weight, torch.Tensor) and layer.weight.ndim >= 2:
             log.warning(f"Attempting fallback shape calculation for layer type: {type(layer)}")
             return layer.weight.shape[0], int(np.prod(layer.weight.shape[1:]))
        raise TypeError(f"Unsupported layer type for get_number_of_rows_and_cols: {type(layer)}")

class GPTQ:
    def __init__(self, module: nn.Module, qcfg: Optional[QuantizeConfig]=None):
        self.W_ref = module.weight
        self.rows, self.columns = get_number_of_rows_and_cols(module)
        if isinstance(module, NamedModule):
            self.module = module.module
            self.name = module.name
        else:
            self.name = HF_OPTIMUM
            self.module = module

        self._validate_module(self.module)

        self.qcfg = qcfg if qcfg else QuantizeConfig()
        
        # --- START OF THE DEFINITIVE FIX ---
        # This prevents an invalid group_size from ever being used for a layer.
        if self.qcfg.group_size != -1 and self.columns < self.qcfg.group_size:
            log.warning(
                f"For layer {self.name}, group_size ({self.qcfg.group_size}) is larger than "
                f"the number of input features ({self.columns}). "
                f"Setting group_size = -1 (per-channel quantization) for this layer to prevent errors."
            )
            self.qcfg = copy.copy(self.qcfg)
            self.qcfg.group_size = -1
        # --- END OF THE DEFINITIVE FIX ---

        if not hasattr(self.qcfg, 'beta'): self.qcfg.beta = 0.0
        if not hasattr(self.qcfg, 'tau'): self.qcfg.tau = 1.0
        if not hasattr(self.qcfg, 'damp_percent'): self.qcfg.damp_percent = 0.01
        if not hasattr(self.qcfg, 'damp_auto_increment'): self.qcfg.damp_auto_increment = 0.0015
        if not hasattr(self.qcfg, 'gamma'): self.qcfg.gamma = 0.0

        self.device = self.module.weight.device
        self.compute_device = DEVICE_1 if DEVICE_1.type != 'cpu' else CPU
        log.info(f"GPTQ layer {self.name} on {self.device}, compute on {self.compute_device}")

        self.W_orig = self._clone_module_weight(self.compute_device)
        expected_shape = (self.rows, self.columns)
        if self.W_orig.shape != expected_shape:
            raise RuntimeError(f"W_orig shape mismatch for {self.name}. Expected {expected_shape}, Got {self.W_orig.shape}")

        self.H: Optional[torch.Tensor] = None
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.nsamples: int = 0

        if Quantizer is None: raise ImportError("Quantizer class not found.")
        self.quantizer: Quantizer = self.create_quantizer(name=self.name)

        self.fwd_inputs_buffered = False
        self.fwd_inputs_buffered_data = []
        self.fwd_counter = 0

    @staticmethod
    def _validate_module(module):
        supported_types = (nn.Linear, nn.Conv2d, transformers.Conv1D)
        if not isinstance(module, supported_types):
             if not (hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor)):
                  raise TypeError(f"GPTQ supports Linear, Conv2d, transformers.Conv1D. Found: {type(module)}")
             else: log.warning(f"Module {type(module)} not explicitly supported but has .weight.")

    def create_quantizer(self, name: str) -> Quantizer:
        return Quantizer(qcfg=self.qcfg, name=name)

    def _clone_module_weight(self, target_device: torch.device) -> torch.Tensor:
        source_module = self.module
        source_weight = source_module.weight.data
        if source_weight.device == target_device: clone = source_weight.clone()
        else: clone = source_weight.to(device=target_device, copy=True)
        if isinstance(source_module, nn.Conv2d): clone = clone.flatten(1)
        elif isinstance(source_module, transformers.Conv1D): clone = clone.t()
        return clone.float()

    @torch.inference_mode()
    def block_cholesky_inverse(self, L: torch.Tensor, upper=False, block_size=512):
        n = L.size(0); device = L.device; dtype = L.dtype
        if upper: L = L.t()
        invA = torch.zeros_like(L); num_blocks = math.ceil(n / block_size); invL_cache = {}
        for k in reversed(range(num_blocks)):
            k_start = k * block_size; k_end = min((k + 1) * block_size, n); k_size = k_end - k_start
            L_block = L[k_start:k_end, k_start:k_end]
            invL_block = torch.linalg.solve_triangular(L_block, torch.eye(k_size, device=device, dtype=dtype), upper=False)
            invL_cache[k] = invL_block; invA[k_start:k_end, k_start:k_end] = invL_block.t() @ invL_block
            for j in range(k):
                j_start = j * block_size; j_end = min((j + 1) * block_size, n); j_size = j_end - j_start
                invL_ik_blocks = []
                start_row_idx = k_start
                for i in range(k, num_blocks):
                    i_start = i * block_size; i_end = min((i + 1) * block_size, n); i_size = i_end - i_start
                    if i == k: invL_ik = invL_block
                    else:
                        invL_ii = invL_cache.get(i)
                        if invL_ii is None:
                            invL_ii = torch.linalg.solve_triangular(L[i_start:i_end, i_start:i_end], torch.eye(i_size, device=device, dtype=dtype), upper=False)
                            invL_cache[i] = invL_ii
                        L_ik = L[i_start:i_end, k_start:k_end]; invL_ik = -invL_ii @ (L_ik @ invL_block)
                    invL_ik_blocks.append(invL_ik)
                    start_row_idx = i_end
                temp_col_parts_matmul = []
                start_col_idx_Ljk = k_start
                for invL_ik_part in invL_ik_blocks:
                     end_col_idx_Ljk = min(start_col_idx_Ljk + invL_ik_part.shape[1], k_end)
                     L_jk_part = L[j_start:j_end, start_col_idx_Ljk:end_col_idx_Ljk]
                     if invL_ik_part.shape[0] != L_jk_part.shape[1]:
                          raise RuntimeError(f"Shape mismatch in block Cholesky matmul: invL_ik_part.t() ({invL_ik_part.shape[1]}x{invL_ik_part.shape[0]}) @ L_jk_part.t() ({L_jk_part.shape[1]}x{L_jk_part.shape[0]})")
                     temp_col_parts_matmul.append(invL_ik_part.t() @ L_jk_part.t())
                     start_col_idx_Ljk = end_col_idx_Ljk
                temp_col = torch.cat(temp_col_parts_matmul, dim=0)
                invA[j_start:j_end, k_start:k_end] = temp_col[:j_size].t()
                invA[k_start:k_end, j_start:j_end] = temp_col[:j_size]
        del invL_cache; return invA

    def add_batch(self, inp: torch.Tensor, out: Optional[torch.Tensor]=None):
        self.fwd_counter += 1
        buffer_device = CPU if DEVICE_0.index == DEVICE_1.index else DEVICE_1
        if self.fwd_inputs_buffered:
            self.fwd_inputs_buffered_data.append(inp.to(device=buffer_device, non_blocking=True))
        else:
            self.process_batch(inp)

    def process_batch(self, inp: torch.Tensor):
        inp_compute = inp.to(device=self.compute_device, dtype=torch.float32)
        original_shape = inp_compute.shape; reshaped_inp = None; source_module = self.module
        if isinstance(source_module, (nn.Linear, transformers.Conv1D)):
            if inp_compute.ndim > 2: reshaped_inp = inp_compute.reshape(-1, original_shape[-1])
            else: reshaped_inp = inp_compute
        elif isinstance(source_module, nn.Conv2d):
            unfolded_inp = F.unfold(inp_compute, kernel_size=source_module.kernel_size, dilation=source_module.dilation, padding=source_module.padding, stride=source_module.stride)
            reshaped_inp = unfolded_inp.permute(0, 2, 1).reshape(-1, unfolded_inp.shape[1])
        else:
            if hasattr(source_module, 'weight') and inp_compute.ndim > 2:
                  try: reshaped_inp = inp_compute.reshape(-1, self.columns)
                  except Exception as reshape_err: raise TypeError(f"Unsupported layer {type(source_module)}, fallback failed: {reshape_err}")
            else: raise TypeError(f"Unsupported layer type for process_batch: {type(source_module)}")
        if reshaped_inp is None or reshaped_inp.shape[1] != self.columns:
            raise ValueError(f"Shape mismatch {self.name}: Input {original_shape} -> reshaped {reshaped_inp.shape if reshaped_inp is not None else 'None'}. Expected cols ({self.columns})")
        batch_token_size = reshaped_inp.shape[0]
        if batch_token_size == 0: return
        total_samples = self.nsamples + batch_token_size
        if total_samples == 0: return
        beta_scale = float(self.nsamples) / total_samples
        alpha_scale = 2.0 / total_samples
        if self.H is None: self.H = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=self.compute_device)
        self.H.addmm_(reshaped_inp.T, reshaped_inp, beta=beta_scale, alpha=alpha_scale)
        if self.qcfg.beta > 0:
            if self.A is None: self.A = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=self.compute_device)
            output = reshaped_inp @ self.W_orig.T
            if torch.isnan(output).any() or torch.isinf(output).any(): log.warning(f"NaN/Inf in pre-softmax output {self.name}. Clamping."); output = torch.nan_to_num(output)
            tau = max(self.qcfg.tau, 1e-6)
            try:
                pt = F.softmax(output / tau, dim=-1)
                if torch.isnan(pt).any(): raise ValueError("NaN detected in softmax output")
                kl_weights = torch.sum(pt * (1.0 - pt), dim=-1); kl_weights = torch.clamp(kl_weights, min=0.0)
            except ValueError:
                log.warning(f"Could not compute valid softmax/KL weights {self.name}. Using zero weights.")
                kl_weights = torch.zeros(batch_token_size, device=self.compute_device, dtype=torch.float32)
            weighted_inp = reshaped_inp * torch.sqrt(kl_weights).unsqueeze(1)
            self.A.addmm_(weighted_inp.T, weighted_inp, beta=beta_scale, alpha=alpha_scale)
            if self.qcfg.gamma > 0:
                if self.B is None: self.B = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=self.compute_device)
                qt = F.softmax(output / tau, dim=-1); ce_weights = torch.sum(qt * (1.0 - qt), dim=-1).clamp(min=0.0)
                weighted_inp_ce = reshaped_inp * torch.sqrt(ce_weights).unsqueeze(1)
                self.B.addmm_(weighted_inp_ce.T, weighted_inp_ce, beta=beta_scale, alpha=alpha_scale)
        self.nsamples += batch_token_size

    def hf_quantize(self, blocksize=128, percdamp=0.01, damp_auto_increment=0.0015, group_size=-1, actorder=False, static_groups=False):
        self.qcfg.group_size = group_size; self.qcfg.damp_percent = percdamp; self.qcfg.damp_auto_increment = damp_auto_increment;
        self.qcfg.desc_act = actorder; self.qcfg.static_groups = static_groups;
        (Q, scale, zero, g_idx, duration, avg_loss, damp_percent, nsamples) = self.quantize(blocksize=blocksize)
        self.module.weight.data = Q.to(self.device, dtype=self.W_ref.dtype)
        return scale, zero, g_idx, duration, avg_loss, damp_percent

    @torch.inference_mode()
    def hessian_inverse(self, H: torch.Tensor) -> Tuple[torch.Tensor, float]:
        damp = self.qcfg.damp_percent; max_damp = 1.0 - 1e-6
        if not (0 < damp < max_damp):
             if damp <= 0: raise ValueError(f"Initial damp_percent ({self.qcfg.damp_percent}) must be > 0.")
        while 0 < damp < max_damp:
            H_damped = H.clone()
            try:
                diag_H = torch.diag(H); diag_H_pos = diag_H[diag_H > 0]
                diag_mean = torch.mean(diag_H_pos).item() if len(diag_H_pos) > 0 else 1.0
                if not np.isfinite(diag_mean) or diag_mean <= 0: diag_mean = 1.0
                diag_indices = torch.arange(self.columns, device=H_damped.device)
                H_damped[diag_indices, diag_indices] += damp * diag_mean
                if not torch.isfinite(H_damped).all(): raise ValueError(f"NaN/Inf in damped Hessian (damp={damp:.5f})")
                with lock: L = torch.linalg.cholesky(H_damped)
                try: H_inv = torch.cholesky_inverse(L, upper=False)
                except torch.cuda.OutOfMemoryError:
                    log.warning(f"OOM using cholesky_inverse {self.name}. Falling back to block inverse."); H_inv = self.block_cholesky_inverse(L, block_size=self.columns // 2)
                if not torch.isfinite(H_inv).all(): raise ValueError(f"NaN/Inf in inverse Hessian (damp={damp:.5f})")
                return H_inv, damp
            except (torch._C._LinAlgError, ValueError, RuntimeError) as e:
                log.warning(f"Hessian inverse failed for damp={damp:.5f}. Reason: {e}")
                if self.qcfg.damp_auto_increment > 0:
                    damp = min(damp + self.qcfg.damp_auto_increment, max_damp); log.info(f"Increasing damp to {damp:.5f}")
                else: raise ValueError(f"Hessian inverse failed. Last damp: {damp:.5f}") from e
        raise ValueError(f"Could not compute Hessian inverse {self.name}. Max damp ({max_damp}) reached.")

    @torch.inference_mode()
    def quantize(self, blocksize=128):
        start = time.time()
        if self.fwd_inputs_buffered and len(self.fwd_inputs_buffered_data) > 0:
            log.info(f"Processing {len(self.fwd_inputs_buffered_data)} buffered batches for {self.name}...")
            if self.compute_device.type == 'cuda': torch.cuda.synchronize()
            for inp_batch in self.fwd_inputs_buffered_data:
                self.process_batch(inp_batch.to(device=self.compute_device))
            self.fwd_inputs_buffered_data.clear()
            if self.compute_device.type == 'cuda': torch.cuda.synchronize()
            log.info(f"Finished processing buffered inputs for {self.name}.")
        if self.nsamples == 0: raise ValueError(f"No samples collected {self.name}")
        if self.H is None: raise RuntimeError(f"Hessian is None {self.name}")
        H_tot = self.H
        if self.qcfg.beta > 0 and self.A is not None: log.debug(f"Adding KL Hessian with beta={self.qcfg.beta}"); H_tot = H_tot + self.qcfg.beta * self.A
        if self.qcfg.gamma > 0 and hasattr(self, 'B') and self.B is not None: log.debug(f"Adding CE Hessian with gamma={self.qcfg.gamma}"); H_tot = H_tot + self.qcfg.gamma * self.B
        del self.H; self.H = None
        if hasattr(self, 'A'): del self.A; self.A = None
        if hasattr(self, 'B'): del self.B; self.B = None
        W = self.W_orig.clone(); del self.W_orig; self.W_orig = None
        self.quantizer.find_params(W, weight=True)
        H = H_tot.clone(); del H_tot
        dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
        scale = []; zero = []; now_idx = 1; groups = []
        if self.qcfg.static_groups and self.qcfg.group_size != -1:
            log.debug(f"Using static groups (size={self.qcfg.group_size}) for {self.name}")
            group_size = self.qcfg.group_size
            for i in range(0, self.columns, group_size):
                quantizer_group = copy.deepcopy(self.quantizer)
                end_col = min(i + group_size, self.columns)
                quantizer_group.find_params(W[:, i:end_col], weight=True)
                scale.append(quantizer_group.scale); zero.append(quantizer_group.zero); groups.append(quantizer_group)
        perm, invperm = None, None
        if self.qcfg.desc_act:
            log.debug(f"Applying activation order (desc_act=True) for {self.name}")
            diag_H = torch.diag(H).clone(); perm = torch.argsort(diag_H, descending=True); del diag_H
            W = W[:, perm]; H = H[perm][:, perm]; invperm = torch.argsort(perm)
        Hinv, damp = self.hessian_inverse(H); del H
        Losses = torch.zeros_like(W); Q = torch.zeros_like(W)
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns); count = i2 - i1
            W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); Err1 = torch.zeros_like(W1); Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]; d = Hinv1[i, i]
                if self.qcfg.group_size != -1:
                    idx_global = i1 + i
                    group_idx = idx_global // self.qcfg.group_size
                    if self.qcfg.static_groups:
                        if group_idx < len(groups): current_quantizer = groups[group_idx]
                        else: log.error(f"Static group index {group_idx} OOB! Using global."); current_quantizer = self.quantizer
                    else:
                        if idx_global % self.qcfg.group_size == 0:
                            start_g = idx_global; end_g = min(start_g + self.qcfg.group_size, self.columns)
                            self.quantizer.find_params(W[:, start_g:end_g], weight=True)
                            if ((idx_global) // self.qcfg.group_size) - now_idx == -1:
                                scale.append(self.quantizer.scale); zero.append(self.quantizer.zero); now_idx += 1
                        current_quantizer = self.quantizer
                else: current_quantizer = self.quantizer
                q = current_quantizer.quantize(w.unsqueeze(1)).flatten(); Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2; err1 = (w - q) / d
                if i + 1 < count: W1[:, i+1:] -= err1.unsqueeze(1).matmul(Hinv1[i, i+1:].unsqueeze(0))
                Err1[:, i] = err1
            Q[:, i1:i2] = Q1; Losses[:, i1:i2] = Losses1 / 2
            if i2 < self.columns: W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        del Hinv, W1, Q1, Err1, Losses1; torch_sync()
        if self.nsamples != 0: avg_loss = torch.sum(Losses).item() / self.nsamples;
        else: avg_loss = -1.0; log.warning(f"Quantization {self.name} nsamples=0.")
        if math.isnan(avg_loss): avg_loss = float('inf'); log.error(f"Quantization NaN loss for {self.name}.")
        del Losses
        group_size_eff = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns
        if group_size_eff <=0 : group_size_eff = self.columns
        if self.qcfg.static_groups and self.qcfg.desc_act:
             if perm is None: raise RuntimeError("perm is None for g_idx calculation")
             g_idx_list = [perm[i].item() // group_size_eff for i in range(self.columns)]
        else: g_idx_list = [i // group_size_eff for i in range(self.columns)]
        g_idx = torch.tensor(g_idx_list, dtype=torch.int32, device=CPU)
        if self.qcfg.desc_act:
            if invperm is None: raise RuntimeError("invperm is None for unpermute")
            Q = Q[:, invperm]; g_idx = g_idx[invperm.to(CPU)]
        if isinstance(self.module, transformers.Conv1D): Q = Q.t()
        if Q.shape != self.W_ref.shape: Q = Q.reshape(self.W_ref.shape).to(dtype=self.W_ref.dtype)
        else: Q = Q.to(dtype=self.W_ref.dtype)
        Q = Q.to(device=self.compute_device)
        if not scale: scale.append(self.quantizer.scale); zero.append(self.quantizer.zero)
        try:
            scale = torch.cat(scale, dim=1).to(device=self.compute_device)
            zero = torch.cat(zero, dim=1).to(device=self.compute_device)
        except Exception as e:
             log.error(f"Error concatenating scale/zero {self.name}: {e}")
             scale = self.quantizer.scale.to(device=self.compute_device)
             zero = self.quantizer.zero.to(device=self.compute_device)
        duration = time.time() - start
        log.info(f"Finished quantization for {self.name} in {duration:.2f} seconds. Final damp: {damp:.5f}")
        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    def free(self):
        log.debug(f"Freeing resources for GPTQ layer {self.name}")
        for attr in ["H", "A", "W_orig", "quantizer", "module", "W_ref", "fwd_inputs_buffered_data", "module_copy"]:
            if hasattr(self, attr): delattr(self, attr)
        if 'fwd_inputs_buffered_data' not in locals() and hasattr(self, 'fwd_inputs_buffered_data'): self.fwd_inputs_buffered_data.clear()

__all__ = ["GPTQ"]