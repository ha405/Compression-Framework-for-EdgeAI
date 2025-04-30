# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

import math
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
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
# device_1 may be same as device_0 if there is only 1 visible/active device
DEVICE_1 = auto_select_torch_device(index=1)

lock = threading.Lock()

def get_number_of_rows_and_cols(layer: nn.Module):
    # return layer.weight.shape[0], np.prod(layer.weight.shape[1:])
    if isinstance(layer, NamedModule):
        layer = layer.module

    if isinstance(layer, transformers.Conv1D):
        # transformers.Conv1D: weight shape is (n_in, n_out)
        return layer.weight.shape[1], layer.weight.shape[0]
    else:
        # weight shape is (n_out, n_in)
        return layer.weight.shape[0], np.prod(layer.weight.shape[1:])




#NEW SHITT 2 FUNCTIONS
# Helper function for KL Hessian update (Eq. in Sec 3.2)
# Note: Assumes inputs are per-batch. Accumulation handled in process_batch.
# This calculates the matrix to be added/averaged into H_kl.
@torch.inference_mode()
def calculate_A_update_matrix(
    xt: torch.Tensor, # Shape: (batch_tokens, N=columns)
    qt: torch.Tensor, # Shape: (batch_tokens, vocab_size) - Student logits/probs
    tau: float
    ) -> torch.Tensor:
    """Calculates the batch contribution to the KL Hessian A."""
    if xt.numel() == 0 or qt.numel() == 0:
        return torch.zeros((xt.shape[1], xt.shape[1]), device=xt.device, dtype=xt.dtype)

    batch_tokens, N = xt.shape
    vocab_size = qt.shape[1]

    # Ensure qt represents probabilities (apply softmax if needed, depends on what qt is passed)
    # Assuming qt passed are already probabilities after softmax(logits/tau)
    # qt_probs = torch.softmax(qt / tau, dim=-1) # Apply if qt are logits
    qt_probs = qt # Assuming qt are already probabilities

    # Calculate diagonal term: sum_k [ qt,k(1 - qt,k) ]
    # qt * (1 - qt) -> shape (batch_tokens, vocab_size)
    # sum over k (vocab_size dim) -> shape (batch_tokens)
    diag_term = torch.sum(qt_probs * (1.0 - qt_probs), dim=1) # Shape: (batch_tokens)

    # Weight input outer product xt*xt' by the diag_term and 1/tau
    # Need efficient batch matmul: diag_term acts as a per-sample weight
    # (diag_term / tau).unsqueeze(1) -> (batch_tokens, 1)
    # xt -> (batch_tokens, N)
    # Weighted xt: diag_term_weighted_xt = (diag_term / tau).sqrt().unsqueeze(1) * xt # Use sqrt for correct outer product sum
    # A_update = weighted_xt.T @ weighted_xt does NOT average correctly here.

    # We need sum_t [ weight_t * xt_t * xt_t.T ]
    # Use einsum or loop (einsum is better)
    # weight = (diag_term / tau) # Shape: (batch_tokens)
    # A_update = torch.einsum('t,ti,tj->ij', weight, xt, xt) # 't'=batch, 'i','j'=features

    # Alternative using broadcasting and matmul (potentially faster)
    weight = (diag_term / tau).unsqueeze(1) # Shape: (batch_tokens, 1)
    weighted_xt = xt * weight # Multiply weight per sample: (batch_tokens, N)
    A_update = xt.T @ weighted_xt # Correct way: (N, batch_tokens) @ (batch_tokens, N) -> (N, N)

    # This A_update is the sum over the batch. Average is handled by alpha_scale later.
    return A_update

# Helper function for CE Hessian update (Eq. in Sec 3.2)
@torch.inference_mode()
def calculate_B_update_matrix(
    xt: torch.Tensor, # Shape: (batch_tokens, N=columns)
    qt: torch.Tensor  # Shape: (batch_tokens, vocab_size) - Student probs
    ) -> torch.Tensor:
    """Calculates the batch contribution to the CE Hessian B."""
    if xt.numel() == 0 or qt.numel() == 0:
        return torch.zeros((xt.shape[1], xt.shape[1]), device=xt.device, dtype=xt.dtype)

    batch_tokens, N = xt.shape
    vocab_size = qt.shape[1]

    # Ensure qt represents probabilities
    # qt_probs = torch.softmax(qt / tau, dim=-1) # Apply if qt are logits
    qt_probs = qt # Assuming qt are already probabilities

    # Calculate [diag(qt) - qt*qt'] term - this happens per sample
    # diag(qt) is tricky in batch. Equivalent to multiplying elementwise then summing.
    # Let M_t = diag(qt_t) - qt_t * qt_t.T  (vocab_size, vocab_size)
    # We need Sum_t [ xt_t.T @ M_t @ xt_t ] ?? No, formula is simpler in document.
    # Formula: [diag(qt) - qt*qt'] * xt*xt' - Seems like elementwise application per sample?

    # Let's re-read the formula structure. It looks like the outer product xt*xt'
    # is weighted by a scalar derived from [diag(qt) - qt*qt']. This isn't standard.
    # The standard CE Hessian involves P(1-P) similar to KL.
    # Assuming the doc means a scalar weight per sample 't' derived from qt_t somehow?
    # OR does it mean the Hessian *of the loss wrt logits*, which gives P-P^2 structure?

    # *** Reinterpreting Sec 3.2 Formula for B ***
    # The standard Hessian of CE loss w.r.t logits involves diag(p) - p*p'.
    # Let's assume the document *intended* to use a structure similar to A,
    # perhaps weighted differently based on CE loss properties.
    # If we use the standard CE Hessian structure w.r.t logits:
    # weight_ce = torch.sum(qt_probs * (1.0 - qt_probs), dim=1) # SAME AS KL DIAGONAL!
    # This seems unlikely unless beta and gamma are meant to combine on the same structure.

    # Let's strictly interpret `[diag(qt) - qt*qt'] xt*xt'`
    # This doesn't seem directly computable into a single (N,N) matrix easily.
    # It might imply an expectation over the vocabulary probabilities.

    # **** SAFER ASSUMPTION / ALTERNATIVE ****
    # Often in practice, the Fisher Information Matrix (FIM) is used as an approximation
    # to the Hessian. For CE loss with softmax, the FIM diagonal approximation
    # involves terms like E[dL/dlogit_k^2] ~ p_k(1-p_k).
    # Let's tentatively USE THE SAME STRUCTURE AS A, assuming B captures a different
    # aspect perhaps related to label smoothing or target distribution variance, OR
    # that the trace normalization handles the difference. This requires clarification
    # on the exact meaning of the formula for B.

    # USING A's STRUCTURE FOR B (tentative, needs validation based on formula intent)
    log.warning("Interpreting CE Hessian B with structure similar to KL Hessian A. Verify formula intent.")
    diag_term_ce = torch.sum(qt_probs * (1.0 - qt_probs), dim=1) # Shape: (batch_tokens)
    # weight = diag_term_ce # Shape: (batch_tokens)
    # B_update = torch.einsum('t,ti,tj->ij', weight, xt, xt)

    weight = diag_term_ce.unsqueeze(1) # Shape: (batch_tokens, 1)
    weighted_xt = xt * weight # Multiply weight per sample: (batch_tokens, N)
    B_update = xt.T @ weighted_xt # (N, batch_tokens) @ (batch_tokens, N) -> (N, N)

    # If the true labels 'yt' were available, B could be calculated differently,
    # potentially using the gradient of the CE loss. But the doc uses only qt.

    return B_update








# ... (GPTQ Class Definition starts here) ...

class GPTQ:
    def __init__(self, module: nn.Module, qcfg: Optional[QuantizeConfig]=None):
        # self.num_tied_handles = 0
        # if qcfg.tied_gptq_handle is not None:
        #     qcfg.tied_gptq_handle.num_tied_handles += 1

        # Flags indicating issues
        # self.issue_zero_samples = False
        # self.issue_nan_hessian = False
        # self.issue_non_invertible = False

        self.W = module.weight
        self.rows, self.columns = get_number_of_rows_and_cols(module)
        if isinstance(module, NamedModule):
            self.module = module.module
            self.name = module.name
        else:
            self.name = HF_OPTIMUM
            self.module = module

        self._validate_module(self.module)

        self.qcfg = qcfg if qcfg else QuantizeConfig() # HF compat will not pass qcfg
        self.device = self.module.weight.device

        self.module_copy = None

        self.H = None
        #NEW SHITTTTTTT
        self.H_mse = None # Original Hessian for MSE reconstruction
        self.H_kl = None  # Hessian for KL divergence
        self.H_ce = None  # Hessian for Cross-Entropy (optional)
        # Store hyperparameters from qcfg
        self.beta = getattr(qcfg, 'beta', 0.0)
        self.gamma = getattr(qcfg, 'gamma', 0.0)
        self.tau = getattr(qcfg, 'tau', 1.0)



        self.nsamples = 0

        self.quantizer = self.create_quantizer(name=self.name)

        # fwd input buffer
        self.fwd_inputs_buffered = False
        self.fwd_inputs_buffered_data = []

        # fwd counter
        self.fwd_counter = 0
        



    @staticmethod
    def _validate_module(module):
        assert isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, transformers.Conv1D)), f"We supports only linear and convolutional layers. actual = `{module}`"

    # def has_hessian_issues(self) -> bool:
    #     return any([self.issue_zero_samples, self.issue_nan_hessian, self.issue_non_invertible])

    def create_quantizer(self, name: str) -> Quantizer:
        return Quantizer(qcfg=self.qcfg, name=name)

    def shape(self):
        if hasattr(self, "module"):
            return self.module.weight.shape
        else:
            return (0, 0)

    def _clone_module(self, copy=True, device: torch.device = None):
        if not device:
            device = self.module.weight.data.device

        clone = self.module.weight.data.to(copy=copy, device=device)

        if isinstance(self.module, _ConvNd):
            clone = clone.flatten(1)

        if isinstance(self.module, transformers.pytorch_utils.Conv1D):
            clone = clone.t()

        return clone.float()

    @torch.inference_mode()
    def block_cholesky_inverse(self, L: torch.Tensor, upper=False, block_size=512):
        """
        Optimized Cholesky inverse with O(block_size^2) memory usage.
        Args:
            L (torch.Tensor): Cholesky factor (lower triangular)
            upper (bool): If True, L is upper triangular
            block_size (int): Processing block size (tunes memory/performance)
        Returns:
            torch.Tensor: The inverse matrix
        """
        assert L.dim() == 2 and L.size(0) == L.size(1), "Input must be square"
        n = L.size(0)
        device = L.device
        dtype = L.dtype

        if upper:
            L = L.t()

        invA = torch.zeros_like(L)
        num_blocks = math.ceil(n / block_size)

        # Cache for invL blocks to avoid recomputation
        invL_cache = {}

        for k in reversed(range(num_blocks)):
            k_start = k * block_size
            k_end = min((k + 1) * block_size, n)
            k_size = k_end - k_start

            # Diagonal block inversion
            L_block = L[k_start:k_end, k_start:k_end]
            invL_block = torch.linalg.solve_triangular(
                L_block,
                torch.eye(k_size, device=device, dtype=dtype),
                upper=False
            )
            invL_cache[k] = invL_block

            # Diagonal block contribution
            invA[k_start:k_end, k_start:k_end] = invL_block.t() @ invL_block

            # Process off-diagonal blocks in parallel where possible
            for j in range(k):
                j_start = j * block_size
                j_end = min((j + 1) * block_size, n)
                j_size = j_end - j_start

                # Compute all required invL_ik blocks first
                invL_ik_blocks = []
                for i in range(k, num_blocks):
                    i_start = i * block_size
                    i_end = min((i + 1) * block_size, n)

                    if i == k:
                        invL_ik = invL_block
                    else:
                        if i in invL_cache:
                            invL_ii = invL_cache[i]
                        else:
                            L_ii = L[i_start:i_end, i_start:i_end]
                            invL_ii = torch.linalg.solve_triangular(
                                L_ii,
                                torch.eye(i_end - i_start, device=device, dtype=dtype),
                                upper=False
                            )
                            invL_cache[i] = invL_ii

                        L_ik = L[i_start:i_end, k_start:k_end]
                        invL_ik = -invL_ii @ (L_ik @ invL_block)
                        del invL_ii

                    invL_ik_blocks.append(invL_ik)
                    del invL_ik

                # Stack blocks for batched operations
                L_jk = L[j_start:j_end, k_start:k_end]

                # Compute contributions in a more vectorized way
                temp_col = torch.cat([
                    (invL_ik.t() @ L_jk.t()) for invL_ik in invL_ik_blocks
                ], dim=0)

                del invL_ik_blocks

                # Accumulate to output
                invA[j_start:j_end, k_start:k_end] = temp_col[:j_size].t()
                invA[k_start:k_end, j_start:j_end] = temp_col[:j_size]

                del temp_col

        del invL_cache
        return invA




    # def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
    #     self.fwd_counter += 1

    #     if self.fwd_inputs_buffered:
    #         if DEVICE_0.index != DEVICE_1.index:
    #             self.fwd_inputs_buffered_data.append(inp.to(device=DEVICE_1, non_blocking=True))
    #         else:
    #             self.fwd_inputs_buffered_data.append(inp.to(device=CPU))
    #     else:
    #         self.process_batch(inp)



    def add_batch(self, inp: torch.Tensor, out: Optional[torch.Tensor]=None, # Original signature often has out=None
                  pt: Optional[torch.Tensor]=None, qt: Optional[torch.Tensor]=None): # Add pt, qt
        # --- START MODIFICATION ---
        if self.beta > 0 and pt is None:
            raise ValueError("Teacher probabilities 'pt' required when beta > 0")
        if (self.beta > 0 or self.gamma > 0) and qt is None:
            raise ValueError("Student probabilities 'qt' required when beta > 0 or gamma > 0")
        # --- END MODIFICATION ---

        self.fwd_counter += 1

        if self.fwd_inputs_buffered:
            # --- START MODIFICATION: Buffer pt/qt too ---
            current_device = inp.device # Assume all inputs are on same device for buffering
            buffer_target_device = DEVICE_1 if DEVICE_0.index != DEVICE_1.index else CPU
            buffered_inp = inp.to(device=buffer_target_device, non_blocking=True)
            buffered_pt = pt.to(device=buffer_target_device, non_blocking=True) if pt is not None else None
            buffered_qt = qt.to(device=buffer_target_device, non_blocking=True) if qt is not None else None
            self.fwd_inputs_buffered_data.append( (buffered_inp, buffered_pt, buffered_qt) )
            # --- END MODIFICATION ---
        else:
            self.process_batch(inp, pt, qt) # Pass pt and qt

    # def process_batch(self, inp: torch.Tensor):
    #     reshaped_inp = inp.to(device=DEVICE_1, dtype=torch.float32)
    #     del inp

    #     # input reshaping
    #     if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
    #         reshaped_inp = reshaped_inp.reshape(-1, reshaped_inp.shape[-1])
    #     else:
    #         unfold = nn.Unfold(
    #             self.module.kernel_size,
    #             dilation=self.module.dilation,
    #             padding=self.module.padding,
    #             stride=self.module.stride,
    #         )
    #         # output size (batch_size, channels * \prod kernel_size, num_patches)
    #         reshaped_inp = unfold(reshaped_inp)
    #         reshaped_inp = reshaped_inp.transpose(1, 2).flatten(0, 1)

    #     batch_token_size = reshaped_inp.shape[0]

    #     if self.H is None:
    #         self.H = torch.zeros((self.columns, self.columns),
    #                              dtype=torch.float32,
    #                              device=DEVICE_1)

    #     beta = self.nsamples / (self.nsamples + batch_token_size)
    #     alpha = 2.0 / (self.nsamples + batch_token_size)
    #     self.H.addmm_(reshaped_inp.T, reshaped_inp, beta=beta, alpha=alpha)

    #     # update number of collected samples
    #     self.nsamples += batch_token_size

    #     # inp returned here is flattened/reshaped original inp
    #     return batch_token_size, reshaped_inp, alpha, beta


# Modify process_batch signature and logic
    def process_batch(self, inp: torch.Tensor, pt: Optional[torch.Tensor], qt: Optional[torch.Tensor]):
        # --- START MODIFICATION ---
        # Move inputs to computation device DEVICE_1
        reshaped_inp = inp.to(device=DEVICE_1, dtype=torch.float32)
        pt = pt.to(device=DEVICE_1, dtype=torch.float32) if pt is not None else None
        qt = qt.to(device=DEVICE_1, dtype=torch.float32) if qt is not None else None
        # --- END MODIFICATION ---
        del inp # Free original tensor memory if possible

        # input reshaping (same as before)
        if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
            reshaped_inp = reshaped_inp.reshape(-1, reshaped_inp.shape[-1])
        # ... (add Conv1d, Conv2d reshaping if necessary) ...
        else:
             log.warning(f"Unsupported layer type for input reshaping: {type(self.module)}. Assuming linear reshape.")
             reshaped_inp = reshaped_inp.reshape(-1, reshaped_inp.shape[-1])


        batch_token_size = reshaped_inp.shape[0]
        if batch_token_size == 0:
             log.warning(f"Empty batch encountered for layer {self.name}. Skipping Hessian update.")
             return # Skip update if batch is empty

        # Initialize Hessians if first batch
        if self.H_mse is None:
            self.H_mse = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=DEVICE_1)
        if self.beta > 0 and self.H_kl is None:
            self.H_kl = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=DEVICE_1)
        if self.gamma > 0 and self.H_ce is None:
            self.H_ce = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=DEVICE_1)

        # Calculate updates using running average factors
        # Note: Using 1/(N+M) style average, not simple 1/T average from doc.
        # This gives more weight to later batches. Use simple average if preferred.
        beta_accum = self.nsamples / (self.nsamples + batch_token_size)
        alpha_accum = 1.0 / (self.nsamples + batch_token_size) # Scale factor for the new batch contribution

        # 1. MSE Hessian Update (Original logic, but using alpha/beta_accum)
        # Calculate batch contribution first: inp.T @ inp
        H_mse_batch = reshaped_inp.T @ reshaped_inp
        self.H_mse.mul_(beta_accum).add_(H_mse_batch, alpha=alpha_accum) # Weighted average update

        # 2. KL Hessian Update
        if self.beta > 0:
            # Calculate the sum over the batch: Sum_t [ weight_t * xt_t * xt_t.T ]
            A_batch = calculate_A_update_matrix(reshaped_inp, qt, self.tau)
            self.H_kl.mul_(beta_accum).add_(A_batch, alpha=alpha_accum) # Weighted average update

        # 3. CE Hessian Update
        if self.gamma > 0:
            # Calculate the sum over the batch
            B_batch = calculate_B_update_matrix(reshaped_inp, qt) # Uses qt, assuming probabilities
            self.H_ce.mul_(beta_accum).add_(B_batch, alpha=alpha_accum) # Weighted average update


        # update number of collected samples
        self.nsamples += batch_token_size

        # Clean up potentially large intermediate tensors
        del reshaped_inp, pt, qt, H_mse_batch
        if self.beta > 0: del A_batch
        if self.gamma > 0: del B_batch
        # torch.cuda.empty_cache() # Optional: Aggressive cleanup

        return # No return needed















    # FIXME, optimum needs fasterquant, we need to remove it
    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        damp_auto_increment=0.0015,
        group_size=-1,
        actorder=False,
        static_groups=False,
    ):
        return self.hf_quantize(blocksize, percdamp, damp_auto_increment, group_size, actorder, static_groups)

    # public api exposed to hf
    def hf_quantize(
        self,
        blocksize=128,
        percdamp=0.01,
        damp_auto_increment=0.0015,
        group_size=-1,
        actorder=False,
        static_groups=False,
    ):
        self.qcfg.group_size = group_size
        self.qcfg.damp_percent = percdamp
        self.qcfg.damp_auto_increment = damp_auto_increment
        self.qcfg.desc_act = actorder
        self.qcfg.static_groups = static_groups
        (Q, scale, zero, g_idx, duration, avg_loss, damp_percent, nsamples) = self.quantize(blocksize=blocksize)
        self.module.weight.data = Q
        return scale, zero, g_idx, duration, avg_loss, damp_percent

    
    # @torch.inference_mode()

    # def hessian_inverse(self, H: torch.Tensor):
    #     damp = self.qcfg.damp_percent
    #     while 1 > damp > 0:
    #         try:
    #             diag = torch.arange(self.columns, device=DEVICE_1)
    #             H[diag, diag] += damp * torch.mean(torch.diag(H))

    #             with lock:
    #                 # print(f"H SHAPE: {H.shape}")
    #                 H = torch.linalg.cholesky(H)

    #                 try:
    #                     # H = self.block_cholesky_inverse(H, block_size=H.shape[0])
    #                     H = torch.cholesky_inverse(H)
    #                 except torch.OutOfMemoryError:
    #                     # half the block size will use ~18% less memory but at higher accuracy loss: 1^-2 vs 1^-8
    #                     # worth the tradeoff since it's either oom or slightly higher accuracy loss
    #                     H = self.block_cholesky_inverse(H, block_size=self.columns // 2)
    #                     log.warn(
    #                         "Quantization: OOM bypassed via low memory math at a cost of lower accuracy: `cholesky_inverse`")

    #                 Hinv = torch.linalg.cholesky(H, upper=True)
    #             break
    #         except torch._C._LinAlgError as e:
    #             if self.qcfg.damp_auto_increment != 0:
    #                 log.warn(
    #                     f"Quantization: Current `damp_percent = {damp:.5f}` is too low, auto-incrementing by `{self.qcfg.damp_auto_increment:.5f}`")
    #                 damp += self.qcfg.damp_auto_increment
    #             else:
    #                 log.warn(
    #                     "Quantization: Please increase damp or nsamples for calibration data to avoid the following quant error: current damp_percent=`{damp_percent:.5f}`")
    #                 raise e

    #     if not (0 < damp < 1):
    #         raise ValueError(f"Quantization: `damp_percent` must between 0 and 1. current is {damp}")

    #     return Hinv, damp






    @torch.inference_mode()
    # --- START MODIFICATION for Step 5 ---
    # Function now operates on the combined Hessian H_combined
    def hessian_inverse(self, H_combined: torch.Tensor):
        """
        Computes the inverse of the (potentially combined) Hessian matrix H_combined,
        applying damping for numerical stability.

        Args:
            H_combined (torch.Tensor): The symmetric Hessian matrix (H_mse or H_tot)
                                        to invert, expected on DEVICE_1.

        Returns:
            Tuple[torch.Tensor, float]:
                - Hinv (torch.Tensor): The inverse of the damped H_combined (Htot^-1).
                # Original returned Cholesky(Hinv, upper=True), changing to return Hinv itself
                # as the quantize loop seems to use the matrix directly.
                - damp (float): The final damping percentage used.
        """
        # Start with the initial damping percentage from config
        damp = self.qcfg.damp_percent
        if not (0 < damp < 1):
             # Initial validation of damp_percent from config
             log.warning(f"Initial damp_percent ({damp}) is not between 0 and 1. Using default 0.01.")
             damp = 0.01 # Use a default safe value

        # Ensure H_combined is on the correct device (should be DEVICE_1)
        if H_combined.device != DEVICE_1:
             log.warning(f"H_combined received by hessian_inverse is on {H_combined.device}, expected {DEVICE_1}. Moving...")
             H_combined = H_combined.to(DEVICE_1)

        # Calculate mean diagonal for damping term scaling
        diag_H = torch.diag(H_combined)
        diag_mean = torch.mean(diag_H)

        # Handle edge cases for diag_mean
        if torch.isnan(diag_mean) or torch.isinf(diag_mean):
            log.error(f"Mean of H_combined diagonal is NaN or Inf for layer {self.name}. Cannot compute stable damping.")
            # Option 1: Raise error
            raise ValueError(f"NaN/Inf in H_combined diagonal mean for layer {self.name}")
            # Option 2: Use a tiny fallback, but this might hide issues
            # log.warning("Using fallback damping value 1e-8 due to invalid diag_mean.")
            # diag_mean = torch.tensor(1e-8, device=DEVICE_1, dtype=torch.float32)
        elif diag_mean == 0:
            log.warning(f"Mean of H_combined diagonal is zero for layer {self.name}. Using fallback damping value 1e-8.")
            # Using a small positive value if mean is exactly zero
            diag_mean = torch.tensor(1e-8, device=DEVICE_1, dtype=torch.float32)
        elif diag_mean < 0:
            # This shouldn't happen for valid Hessians (outer products), but check anyway
            log.warning(f"Mean of H_combined diagonal is negative ({diag_mean:.4e}) for layer {self.name}. Using its absolute value for damping.")
            diag_mean = torch.abs(diag_mean)


        while True: # Loop to increase damping if inversion fails
            try:
                # Clone H_combined for damping to avoid modifying the original input tensor
                H_damped = H_combined.clone()
                diag_indices = torch.arange(self.columns, device=DEVICE_1)

                # Apply damping: H' = H + λ * mean(diag(H)) * I
                # The document used lambda * tr(H)/N * I. mean(diag(H)) is tr(H)/N.
                damping_value = damp * diag_mean
                H_damped[diag_indices, diag_indices] += damping_value

                # Check for NaNs/Infs introduced by damping (unlikely unless diag_mean was huge/invalid)
                if torch.isnan(H_damped).any() or torch.isinf(H_damped).any():
                   log.error(f"NaN or Inf detected in H_damped after adding damping={damp:.5f} (damping_value={damping_value:.4e}).")
                   raise ValueError(f"NaN/Inf in damped Hessian for layer {self.name}")

                # --- Perform Inversion using Cholesky Decomposition ---
                # 1. Compute Cholesky decomposition: H_damped = L @ L.T
                # torch.linalg.cholesky expects positive definite matrix
                L = torch.linalg.cholesky(H_damped, upper=False) # Get lower triangular factor L

                # 2. Compute inverse using Cholesky factor: H_damped^-1 = (L.T)^-1 @ L^-1
                # torch.cholesky_inverse calculates this efficiently from L
                Hinv = torch.cholesky_inverse(L, upper=False) # Calculate inverse from lower L

                # --- Sanity Check the Inverse ---
                # Check for NaNs/Infs in the resulting inverse
                if torch.isnan(Hinv).any() or torch.isinf(Hinv).any():
                   log.error(f"NaN or Inf detected in the computed inverse Hinv with damp={damp:.5f}.")
                   raise ValueError(f"NaN/Inf in inverted Hessian for layer {self.name}")

                # Optional: Check symmetry (numerical errors might make it slightly asymmetric)
                # if not torch.allclose(Hinv, Hinv.T, atol=1e-5):
                #    log.warning(f"Computed inverse Hinv is slightly asymmetric for layer {self.name}.")

                # If we reached here, inversion was successful
                break # Exit the while loop

            except torch._C._LinAlgError as e:
                # This specific error usually means the matrix wasn't positive definite
                # even after damping, or other numerical issues occurred during Cholesky.
                log.warning(f"Cholesky decomposition failed for layer {self.name} with damp={damp:.5f}: {e}")
                if self.qcfg.damp_auto_increment > 0:
                    # Increase damping factor
                    damp += self.qcfg.damp_auto_increment
                    if damp >= 1.0: # Set a reasonable upper limit for damping
                       log.error(f"Damping factor reached >= 1.0 ({damp:.5f}). Cannot invert Hessian for layer {self.name}. Check calibration data or model stability.")
                       # Raise a more specific error perhaps
                       raise RuntimeError(f"Failed to invert Hessian for layer {self.name} after increasing damping to {damp:.5f}. Check calibration/model stability.") from e
                    log.warning(f"Retrying Cholesky with increased damp: {damp:.5f}")
                    # Continue to the next iteration of the while loop
                else:
                    # Damping failed and auto-increment is off
                    log.error(f"Cholesky failed for layer {self.name} with damp={damp:.5f}, and damp_auto_increment is 0. Increase damp_percent in config or check calibration data.")
                    raise e # Re-raise the LinAlgError

            except ValueError as e:
                # Catch NaNs/Infs we raised or other value errors
                log.error(f"ValueError during Hessian inversion for layer {self.name} with damp={damp:.5f}: {e}")
                # Decide whether to retry with more damping or just fail
                # Retrying might not help if the issue is NaNs in H_combined itself.
                # For now, re-raise the ValueError.
                raise e
            except Exception as e:
                # Catch any other unexpected errors
                log.error(f"Unexpected error during Hessian inversion for layer {self.name} with damp={damp:.5f}: {e}")
                raise e # Re-raise

        # Return the computed inverse matrix Hinv (Htot^-1) and the final damping value used
        return Hinv, damp
    # --- END MODIFICATION for Step 5 ---






    # @torch.inference_mode()
    # def quantize(
    #     self,
    #     blocksize=128,
    # ):

    #     #self.H = self.H.to(device=CUDA_0)
    #     # log.info(f"Quantization `{self.name}` using samples: `{self.nsamples}`")
    #     start = time.time()

    #     self.hessian_inverse = torch_compile(self.hessian_inverse)

    #     # process buffered inputs
    #     if len(self.fwd_inputs_buffered_data) > 0:
    #         torch.cuda.synchronize()

    #         for inp in self.fwd_inputs_buffered_data:
    #             self.process_batch(inp)

    #         # release buffer
    #         del self.fwd_inputs_buffered_data

    #     # if self.device.type not in ["mps", "cpu"]:
    #     #     self.module.weight.data = self.module.weight.data.cpu()

    #     # TODO: waiting for pytorch implementation of ops for MPS
    #     if sys.platform == "darwin" and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
    #         raise RuntimeError("For MacOS you must set env `PYTORCH_ENABLE_MPS_FALLBACK=1` before running quantization.")

    #     if self.module_copy is None:
    #         # log.info("copy W to cuda_1")
    #         W = self._clone_module(device=DEVICE_1)
    #     else:
    #         W = self.module_copy
    #         self.module_copy = None

    #     self.quantizer.find_params(W, weight=True)

    #     H = self.H
    #     del self.H

    #     dead = torch.diag(H) == 0
    #     H[dead, dead] = 1
    #     W[:, dead] = 0

    #     # g_idx = []
    #     scale = []
    #     zero = []
    #     now_idx = 1

    #     if self.qcfg.static_groups:
    #         import copy

    #         groups = []
    #         for i in range(0, self.columns, self.qcfg.group_size):
    #             quantizer = copy.deepcopy(self.quantizer)
    #             quantizer.find_params(W[:, i : (i + self.qcfg.group_size)], weight=True)

    #             scale.append(quantizer.scale)
    #             zero.append(quantizer.zero)
    #             groups.append(quantizer)

    #     if self.qcfg.desc_act:
    #         perm = torch.argsort(torch.diag(H), descending=True)
    #         W = W[:, perm]
    #         H = H[perm][:, perm]
    #         invperm = torch.argsort(perm)

    #     Losses = torch.zeros_like(W)
    #     Q = torch.zeros_like(W)

    #     Hinv, damp = self.hessian_inverse(H)

    #     for i1 in range(0, self.columns, blocksize):
    #         i2 = min(i1 + blocksize, self.columns)
    #         count = i2 - i1

    #         W1 = W[:, i1:i2].clone()
    #         Q1 = torch.zeros_like(W1)
    #         Err1 = torch.zeros_like(W1)
    #         Losses1 = torch.zeros_like(W1)
    #         Hinv1 = Hinv[i1:i2, i1:i2]

    #         for i in range(count):
    #             w = W1[:, i]
    #             d = Hinv1[i, i]

    #             if self.qcfg.group_size != -1:
    #                 if not self.qcfg.static_groups:
    #                     if (i1 + i) % self.qcfg.group_size == 0:
    #                         self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + self.qcfg.group_size)], weight=True)

    #                     if ((i1 + i) // self.qcfg.group_size) - now_idx == -1:
    #                         scale.append(self.quantizer.scale)
    #                         zero.append(self.quantizer.zero)
    #                         now_idx += 1
    #                 else:
    #                     idx = i1 + i
    #                     if self.qcfg.desc_act:
    #                         idx = perm[idx]

    #                     self.quantizer = groups[idx // self.qcfg.group_size]

    #             q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
    #             Q1[:, i] = q
    #             Losses1[:, i] = (w - q) ** 2 / d**2

    #             err1 = (w - q) / d
    #             W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
    #             Err1[:, i] = err1

    #         Q[:, i1:i2] = Q1
    #         Losses[:, i1:i2] = Losses1 / 2

    #         W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    #     del Hinv
    #     torch_sync()

    #     avg_loss = torch.sum(Losses).item() / self.nsamples

    #     if math.isnan(avg_loss):
    #         print("Losses sum item:", torch.sum(Losses).item())
    #         raise ValueError(f"Quantization: Failed due to `NaN` loss for `{self.name}`")

    #     del Losses

    #     group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns

    #     if self.qcfg.static_groups and self.qcfg.desc_act:
    #         g_idx = [perm[i] // group_size for i in range(self.columns)]
    #     else:
    #         g_idx = [i // group_size for i in range(self.columns)]

    #     g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

    #     if self.qcfg.desc_act:
    #         Q = Q[:, invperm]
    #         g_idx = g_idx[invperm]

    #     if isinstance(self.module, transformers.Conv1D):
    #         Q = Q.t()

    #     if Q.shape != self.module.weight.shape:
    #         Q = Q.reshape(self.module.weight.shape).type_as(self.module.weight.data)
    #     else:
    #         Q = Q.type_as(self.module.weight.data)

    #     Q = Q.to(device=DEVICE_1)

    #     if scale == []:
    #         scale.append(self.quantizer.scale)
    #         zero.append(self.quantizer.zero)

    #     scale = torch.cat(scale, dim=1)
    #     zero = torch.cat(zero, dim=1)

    #     duration = time.time() - start

    #     return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    # Inside /kaggle/input/gptmain/GPTQModel/gptqmodel/quantization/gptq.py
# Inside class GPTQ:

    @torch.inference_mode()
    def quantize(
        self,
        blocksize=128,
    ):
        start_time = time.time() # Use different variable name

        # --- Process Buffered Inputs ---
        if self.fwd_inputs_buffered and len(self.fwd_inputs_buffered_data) > 0:
            log.info(f"Processing {len(self.fwd_inputs_buffered_data)} buffered batches for layer {self.name}...")
            # Clear buffer first to prevent recursive buffering if process_batch calls add_batch somehow
            buffered_data_copy = self.fwd_inputs_buffered_data
            self.fwd_inputs_buffered_data = []
            self.fwd_inputs_buffered = False # Mark as not buffering anymore

            for buffered_inp, buffered_pt, buffered_qt in buffered_data_copy:
                self.process_batch(buffered_inp, buffered_pt, buffered_qt)
            del buffered_data_copy # Free memory
            log.info(f"Finished processing buffered data for layer {self.name}.")

        # --- Sanity Checks ---
        if self.nsamples == 0:
             raise RuntimeError(f"Layer {self.name}: No samples processed. Cannot quantize.")
        if self.H_mse is None:
             raise RuntimeError(f"Layer {self.name}: H_mse not computed.")
        if self.beta > 0 and self.H_kl is None:
             raise RuntimeError(f"Layer {self.name}: H_kl not computed, but beta > 0.")
        if self.gamma > 0 and self.H_ce is None:
             raise RuntimeError(f"Layer {self.name}: H_ce not computed, but gamma > 0.")

        log.info(f"Quantizing layer {self.name} with {self.nsamples} samples. beta={self.beta}, gamma={self.gamma}")

        # --- Get Weights ---
        W = self._clone_module(device=DEVICE_1) # Clone W to computation device

        # --- Combine Hessians ---
        H_mse = self.H_mse
        H_kl = self.H_kl if self.beta > 0 else None
        H_ce = self.H_ce if self.gamma > 0 else None

        # Move Hessians off self to allow freeing memory
        self.H_mse, self.H_kl, self.H_ce = None, None, None

        # Trace Normalization
        with torch.no_grad(): # Ensure no gradients calculated here
             sH = torch.trace(H_mse)
             if sH == 0 or torch.isnan(sH) or torch.isinf(sH):
                 log.warning(f"Trace of H_mse is {sH} for layer {self.name}. Using 1.0 for normalization.")
                 sH = 1.0 # Avoid division by zero/nan, effectively disabling norm if trace is invalid

             A_prime = torch.zeros_like(H_mse)
             if H_kl is not None:
                 sA = torch.trace(H_kl)
                 if sA != 0 and not torch.isnan(sA) and not torch.isinf(sA):
                     A_prime = (sH / sA) * H_kl
                 else:
                     log.warning(f"Trace of H_kl is {sA} for layer {self.name}. KL term A' set to zero.")
                 del H_kl # Free memory

             B_prime = torch.zeros_like(H_mse)
             if H_ce is not None:
                 sB = torch.trace(H_ce)
                 if sB != 0 and not torch.isnan(sB) and not torch.isinf(sB):
                     B_prime = (sH / sB) * H_ce
                 else:
                     log.warning(f"Trace of H_ce is {sB} for layer {self.name}. CE term B' set to zero.")
                 del H_ce # Free memory

             # Combine
             H_tot = H_mse + self.beta * A_prime + self.gamma * B_prime
             del H_mse, A_prime, B_prime # Free memory

        log.info(f"Combined Hessian H_tot created for layer {self.name}.")

        # --- Handle Dead Weights (based on H_tot diagonal) ---
        diag_H_tot = torch.diag(H_tot)
        # Check for NaNs/Infs BEFORE checking for zeros
        if torch.isnan(diag_H_tot).any() or torch.isinf(diag_H_tot).any():
             nan_inf_indices = torch.where(torch.isnan(diag_H_tot) | torch.isinf(diag_H_tot))[0]
             log.error(f"NaN or Inf detected in H_tot diagonal for layer {self.name} at indices: {nan_inf_indices.tolist()}. Cannot proceed.")
             raise ValueError(f"NaN/Inf in H_tot diagonal for layer {self.name}")
        dead = diag_H_tot == 0
        num_dead = torch.sum(dead).item()
        if num_dead > 0:
             log.warning(f"Layer {self.name}: Found {num_dead} dead weights based on H_tot diagonal.")
        H_tot[dead, dead] = 1.0 # Use 1.0 for stability, was 1 before
        W[:, dead] = 0.0

        # --- Activation Ordering (Optional, based on H_tot) ---
        perm = None
        invperm = None
        if self.qcfg.desc_act:
            log.info(f"Applying activation order for layer {self.name} based on H_tot.")
            perm = torch.argsort(diag_H_tot, descending=True)
            W = W[:, perm]
            H_tot = H_tot[perm][:, perm]
            invperm = torch.argsort(perm)
        del diag_H_tot # Free memory

        # --- Invert the Combined Hessian ---
        log.info(f"Inverting H_tot (shape: {H_tot.shape}) for layer {self.name}...")
        # Assuming hessian_inverse returns Htot^-1 (the actual inverse matrix)
        Hinv_tot_matrix, damp_used = self.hessian_inverse(H_tot)
        log.info(f"H_tot inverted using damp={damp_used:.5f} for layer {self.name}.")
        del H_tot # Free memory

        # --- Initial Quantizer Setup (scale/zero based on W) ---
        # self.quantizer.find_params(W, weight=True) # Already done before H_tot combination?
        # Re-running find_params might be needed if W changed significantly, but usually done once.
        # Let's assume it was done correctly before.

        # --- Column-wise Quantization Loop ---
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        # g_idx_list = [] # Using list might be slow, preallocate tensor if size known
        scale_list = [] # Using lists for scale/zero might be easier due to grouping
        zero_list = []
        group_quantizer_map = {} # For static groups cache

        # --- Static Group Precomputation (if enabled) ---
        if self.qcfg.static_groups:
             log.info(f"Precomputing quantizers for static groups in layer {self.name}")
             import copy
             # Need original W if desc_act was used? Or use permuted W? Use permuted W.
             for i_group in range(0, self.columns, self.qcfg.group_size):
                 group_W = W[:, i_group : min(i_group + self.qcfg.group_size, self.columns)]
                 quantizer_copy = copy.deepcopy(self.quantizer)
                 # Configure quantizer (bits, sym) - might be needed if Quantizer state changes
                 quantizer_copy.configure(bits=self.qcfg.bits, sym=self.qcfg.sym)
                 quantizer_copy.find_params(group_W, weight=True)
                 group_quantizer_map[i_group // self.qcfg.group_size] = quantizer_copy
                 scale_list.append(quantizer_copy.scale)
                 zero_list.append(quantizer_copy.zero)
             log.info(f"Finished precomputing {len(group_quantizer_map)} static group quantizers.")


        log.info(f"Starting column loop for layer {self.name}...")
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone() # Weights for the current block
            Q1 = torch.zeros_like(W1) # Quantized weights for the block
            Err1 = torch.zeros_like(W1) # Scaled errors for the block
            Losses1 = torch.zeros_like(W1) # <<< DECLARATION HERE

            # Get the corresponding block from the inverted combined Hessian
            Hinv1_block = Hinv_tot_matrix[i1:i2, i1:i2].clone() # Get block H^-1_ii

            # --- Column Loop Inside Block ---
            for i in range(count): # i is the index within the block (0 to count-1)
                global_col_idx = i1 + i # Actual column index in the full matrix
                w_col = W1[:, i] # Current column weights

                # Get the diagonal element for scaling error
                d = Hinv1_block[i, i].item() # Use .item() for scalar
                if d <= 1e-8: # Check against small threshold
                    if d <= 0:
                       log.warning(f"Non-positive diagonal Hinv_tot element at ({global_col_idx},{global_col_idx}): {d}. Using 1e-8.")
                    d = 1e-8 # Use small positive value

                # --- Determine Quantizer (Group Logic) ---
                current_quantizer = self.quantizer # Default
                if self.qcfg.group_size != -1:
                    group_idx = global_col_idx // self.qcfg.group_size
                    if self.qcfg.static_groups:
                        current_quantizer = group_quantizer_map[group_idx]
                    else: # Dynamic groups: find params if at group boundary
                        if global_col_idx % self.qcfg.group_size == 0:
                             group_W = W[:, global_col_idx : min(global_col_idx + self.qcfg.group_size, self.columns)]
                             # Make sure quantizer state (bits, sym) is correct before find_params
                             self.quantizer.configure(bits=self.qcfg.bits, sym=self.qcfg.sym)
                             self.quantizer.find_params(group_W, weight=True)
                             # Store scale/zero for dynamic groups
                             scale_list.append(self.quantizer.scale)
                             zero_list.append(self.quantizer.zero)
                        # Need logic here if scale/zero list index doesn't match group_idx directly

                # Quantize the current column using the determined quantizer
                q_col = current_quantizer.quantize(w_col.unsqueeze(1)).flatten()
                Q1[:, i] = q_col

                # Calculate loss contribution for this column (optional tracking)
                Losses1[:, i] = (w_col - q_col)**2 / (2 * d) # Formula using d

                # Calculate the scaled error for propagation: err_i = (w_i - q_i) / d_i
                err_col = (w_col - q_col) / d # Shape: (rows)

                # Propagate error to remaining columns *in this block*
                # Formula: w_j <- w_j - Hinv_ij / Hinv_ii * err_i = w_j - Hinv_ij / d * err_i
                if i < count - 1: # Check if there are remaining columns j = i+1 ... count-1
                    # Hinv_ij elements for this row 'i' and remaining columns 'j'
                    Hinv_ij_row = Hinv1_block[i, i+1:] # Shape: (count - 1 - i)
                    # Calculate update amount: err_col * (Hinv_ij_row / d)
                    # Need to broadcast err_col (rows, 1) with update factors (1, count-1-i)
                    update_values = err_col.unsqueeze(1) * (Hinv_ij_row.unsqueeze(0) / d) # Shape: (rows, count-1-i)
                    W1[:, i+1:] -= update_values # Update remaining columns

                Err1[:, i] = err_col # Store scaled error for block-to-block propagation

            # --- End Column Loop Inside Block ---

            # Store results for the processed block
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 # Store block loss if needed

            # Propagate error to subsequent blocks
            if i2 < self.columns:
                # Use the off-diagonal block Hinv_tot_matrix[i1:i2, i2:]
                Hinv_offdiag_block = Hinv_tot_matrix[i1:i2, i2:] # Shape: (blocksize, remaining_cols)
                # Calculate update: Err1 @ Hinv_offdiag_block
                # Err1 shape: (rows, blocksize)
                update_for_next_blocks = Err1.matmul(Hinv_offdiag_block) # Shape: (rows, remaining_cols)
                W[:, i2:] -= update_for_next_blocks # Update weights of subsequent blocks

            # Log progress periodically
            if (i1 // blocksize) % 10 == 0 or i2 == self.columns:
                 log.info(f"Layer {self.name}: Processed columns {i1}-{i2-1}/{self.columns}")


        log.info(f"Finished column loop for layer {self.name}.")
        # --- Final Steps ---
        del Hinv_tot_matrix, W1, Q1, Err1, Hinv1_block # Free memory

        # Calculate average loss if Losses were tracked
        total_loss_tensor = torch.sum(Losses) # Keep as tensor initially
        avg_loss_float = total_loss_tensor.item() / self.nsamples # C
        # avg_loss = -1 # Placeholder if loss not calculated
        if torch.isnan(total_loss_tensor):
            log.error(f"Quantization failed due to NaN loss for layer {self.name}")
            raise ValueError(f"NaN loss encountered for layer {self.name}")
        
        if self.nsamples > 0:
            avg_loss_float = total_loss_tensor.item() / self.nsamples
        else:
            log.warning(f"Layer {self.name}: nsamples is zero, setting avg_loss to 0.0")
            avg_loss_float = 0.0 # Assign a default value
        del Losses, total_loss_tensor

        # --- Group Index Generation ---
        # Needs careful handling depending on static/dynamic groups and desc_act
        if self.qcfg.group_size != -1:
             group_size = self.qcfg.group_size
             # If static groups, g_idx corresponds to precomputed groups.
             # If dynamic, it corresponds to groups processed during the loop.
             # If desc_act, need to use 'invperm'.
             g_idx_tensor = torch.arange(self.columns) // group_size
             if self.qcfg.desc_act and invperm is not None:
                 g_idx_permuted = torch.zeros_like(g_idx_tensor)
                 g_idx_permuted[invperm] = g_idx_tensor # Map original group index to permuted position
                 g_idx_tensor = g_idx_permuted
             g_idx = g_idx_tensor.to(dtype=torch.int32, device=Q.device)
        else:
             g_idx = torch.zeros(self.columns, dtype=torch.int32, device=Q.device) # No groups


        # Activation ordering cleanup
        if self.qcfg.desc_act and invperm is not None:
            log.info(f"Reverting activation order for layer {self.name}.")
            Q = Q[:, invperm]
            # g_idx already handled above if desc_act was used
            del invperm, perm

        # Finalize Q dtype and shape
        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()
        Q = Q.reshape(self.module.weight.shape).type_as(self.module.weight.data)

        # Finalize scale/zero tensors
        if not scale_list: # Handle case where no groups/-1 or only one group processed
             scale_list.append(self.quantizer.scale)
             zero_list.append(self.quantizer.zero)
        final_scale = torch.cat(scale_list, dim=1) # Cat along channel/group dim
        final_zero = torch.cat(zero_list, dim=1) # Cat along channel/group dim

        duration = time.time() - start_time
        log.info(f"Layer {self.name}: Quantization finished. Avg Loss={avg_loss_float:.6f}, Time={duration:.2f}s, Damp Used={damp_used:.5f}")

        # Return results (ensure signature matches what caller expects)
        return Q, final_scale, final_zero, g_idx, duration, avg_loss_float, damp_used, self.nsamples





    def free(self):
        if hasattr(self, "H"):
            del self.H
        del self.quantizer
        del self.module_copy
        del self.module

        # torch_empty_cache(self.device)

__all__ = ["GPTQ"]
