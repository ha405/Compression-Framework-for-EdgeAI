import copy
import math
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch as t
import torch.nn as nn
import transformers
from torch.nn.modules.conv import _ConvNd
from ...models._const import DEVICE, PLATFORM
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger

log = setup_logger()

class BaseQuantLinear(nn.Module):
    SUPPORTS_BITS: List[int] = [2, 3, 4, 8]
    SUPPORTS_GROUP_SIZE: List[int] = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT: List[bool] = [True, False]
    SUPPORTS_SYM: List[bool] = [True, False]
    SUPPORTS_SHARDS: bool = True
    SUPPORTS_TRAINING: bool = True
    SUPPORTS_TRAINING_USE_TORCH_KERNEL: bool = False
    SUPPORTS_AUTO_PADDING: bool = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY: List[int] = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY: List[int] = [1]
    SUPPORTS_PACK_DTYPES: List[t.dtype] = [t.int8, t.int16, t.int32, t.int64]
    SUPPORTS_DEVICES: List[DEVICE] = [DEVICE.ALL]
    SUPPORTS_PLATFORM: List[PLATFORM] = [PLATFORM.ALL]
    SUPPORTS_DTYPES: List[t.dtype] = [t.float16, t.bfloat16]

    def __init__(self,
                 bits: int,
                 group_size: int,
                 desc_act: bool,
                 sym: bool,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 pack_dtype: t.dtype,
                 backend: BACKEND,
                 name: str = None,
                 register_buffers: bool = False,
                 register_buffers_in_features: int = None,
                 register_buffers_out_features: int = None,
                 **kwargs):
        super().__init__()
        if name is None:
            name = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size if group_size != -1 else in_features
        self.bits = bits
        self.desc_act = desc_act
        self.pack_dtype = pack_dtype
        self.backend = backend
        self.maxq = 2 ** self.bits - 1
        self.pack_dtype = pack_dtype
        self.optimized = False
        if self.pack_dtype == t.int8:
            self.pack_dtype_bits, self.pack_np_dtype, self.pack_np_math_dtype = 8, np.int8, np.uint8
        elif self.pack_dtype == t.int16:
            self.pack_dtype_bits, self.pack_np_dtype, self.pack_np_math_dtype = 16, np.int16, np.uint16
        elif self.pack_dtype == t.int32:
            self.pack_dtype_bits, self.pack_np_dtype, self.pack_np_math_dtype = 32, np.int32, np.uint32
        elif self.pack_dtype == t.int64:
            self.pack_dtype_bits, self.pack_np_dtype, self.pack_np_math_dtype = 64, np.int64, np.uint64
        else:
            raise ValueError("Unsupported weight_dtype. Only int16 and int32 are supported.")
        self.pack_factor = self.pack_dtype_bits // self.bits
        _, err = self._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, in_features=in_features, out_features=out_features, pack_dtype=pack_dtype)
        if err: raise err
        self._qzeros_format = 1
        if register_buffers:
            in_features_reg = self.in_features if not register_buffers_in_features else register_buffers_in_features
            out_features_reg = self.out_features if not register_buffers_out_features else register_buffers_out_features
            padded_in_f = in_features_reg
            if padded_in_f % self.pack_factor != 0:
                padded_in_f += self.pack_factor - (padded_in_f % self.pack_factor)
            self.register_buffer("qweight", t.zeros((padded_in_f // self.pack_factor, out_features_reg), dtype=self.pack_dtype))
            self.register_buffer("qzeros", t.zeros((math.ceil(in_features_reg / self.group_size), out_features_reg // self.pack_factor), dtype=self.pack_dtype))
            self.register_buffer("scales", t.zeros((math.ceil(in_features_reg / self.group_size), out_features_reg), dtype=t.float16))
            self.register_buffer("g_idx", t.tensor([i // self.group_size for i in range(in_features_reg)], dtype=t.int32))
            if bias: self.register_buffer("bias", t.zeros(out_features_reg, dtype=t.float16))
            else: self.bias = None

    def list_buffers(self):
        buf = []
        for name in ["qweight", "qzeros", "scales", "g_idx", "bias"]:
            if hasattr(self, name) and getattr(self, name) is not None: buf.append(getattr(self, name))
        return buf

    def qzero_format(self, format: int = None):
        if format is None: return self._qzeros_format
        if format not in [1, 2]: raise ValueError("Unsupported qzero format.")
        self._qzeros_format = format
        return self._qzeros_format

    def post_init(self): pass
    @classmethod
    def validate(cls, **kwargs): return cls._validate(**kwargs)
    @classmethod
    def verify_supports_params(cls): pass
    @classmethod
    def _validate(cls, **kwargs): return True, None
    @classmethod
    def validate_device(cls, device: DEVICE): pass
    def optimize(self, **kwargs): self.optimized = True
    def train(self, mode=True): return super().train(mode)

class PackableQuantLinear(BaseQuantLinear):
    def post_init(self, **kwargs):
        if self.bits in [2, 4, 8]: wf = t.tensor(list(range(0, self.pack_dtype_bits, self.bits)), dtype=t.int32).unsqueeze(0).to(self.g_idx.device)
        elif self.bits == 3: wf = t.tensor([[0,3,6,9,12,15,18,21,24,27,30,0],[0,1,4,7,10,13,16,19,22,25,28,31],[0,2,5,8,11,14,17,20,23,26,29,0]], dtype=t.int32).reshape(1,3,12).to(self.g_idx.device)
        self.wf_unsqueeze_zero = wf.unsqueeze(0).to(self.g_idx.device)
        self.wf_unsqueeze_neg_one = wf.unsqueeze(-1).to(self.g_idx.device)
        super().post_init(**kwargs)

    def list_buffers(self):
        return super().list_buffers() + [self.wf_unsqueeze_zero, self.wf_unsqueeze_neg_one]

    def dequantize_weight(self, num_itr: int = 1):
        if self.bits in [2, 4, 8]:
            zeros = t.bitwise_right_shift(t.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor), self.wf_unsqueeze_zero)
            zeros = t.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)
            weight = t.bitwise_right_shift(t.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1), self.wf_unsqueeze_neg_one)
            weight = t.bitwise_and(weight, self.maxq).reshape(-1, self.out_features)
        else: # bits == 3 logic is preserved
            zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1,-1,-1,12) >> self.wf_unsqueeze_zero
            zeros[:,:,0,10] = (zeros[:,:,0,10]&3)|((zeros[:,:,1,0]<<2)&4); zeros[:,:,1,11] = (zeros[:,:,1,11]&1)|((zeros[:,:,2,0]<<1)&6)
            zeros = t.cat([zeros[:,:,0,:11], zeros[:,:,1,1:12], zeros[:,:,2,1:11]], dim=2).reshape(self.scales.shape)
            weight = self.qweight.reshape(self.qweight.shape[0]//3,3,1,self.qweight.shape[1]).expand(-1,-1,12,-1) >> self.wf_unsqueeze_neg_one
            weight[:,0,10] = (weight[:,0,10]&3)|((weight[:,1,0]<<2)&4); weight[:,1,11] = (weight[:,1,11]&1)|((weight[:,2,0]<<1)&6)
            weight = t.cat([weight[:,0,:11], weight[:,1,1:12], weight[:,2,1:11]], dim=1).reshape(-1, self.out_features)
        
        weight = weight[:self.in_features]
        weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])
        return weights

    def pack(self, linear: nn.Module, scales: t.Tensor, zeros: t.Tensor, g_idx: t.Tensor=None):
        W = linear.weight.data.clone()
        if isinstance(linear, _ConvNd):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.T
        
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
        
        # --- START OF THE DEFINITIVE FIX ---
        # The scales and zeros are passed with shape (num_groups, out_features)
        # `W` has shape (out_features, in_features)
        
        # We must transpose W to align its `in_features` dimension with `g_idx` for the math.
        W_t = W.T.contiguous() # Shape: [in_features, out_features]
        
        # Expand scales and zeros to match the dimensions of the transposed weight tensor
        # `g_idx` has shape [in_features], mapping each input feature to a group index.
        # `scales` and `zeros` have shape [num_groups, out_features].
        # Indexing them with g_idx creates tensors of shape [in_features, out_features].
        
        expanded_scales = scales[self.g_idx.long()]
        expanded_zeros = zeros[self.g_idx.long()]
        
        # Now the formula Q = round(W_t/S + Z) is applied element-wise with correct shapes.
        int_weight = t.round(W_t / expanded_scales + expanded_zeros).to(t.int32)
        # --- END OF THE DEFINITIVE FIX ---

        self.scales = scales.clone().to(dtype=t.float16)
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=t.float16)
        
        int_weight = int_weight.contiguous()
        
        # Padding for non-divisible in_features before bit-packing
        padded_in_features = self.in_features
        if self.in_features % self.pack_factor != 0:
            padded_in_features += self.pack_factor - (self.in_features % self.pack_factor)
        
        if int_weight.shape[0] < padded_in_features:
            pad_len = padded_in_features - int_weight.shape[0]
            int_weight = t.cat([int_weight, int_weight.new_zeros(pad_len, int_weight.shape[1])], dim=0)

        int_weight = int_weight.numpy().astype(self.pack_np_math_dtype)
        
        qweight = np.zeros((int_weight.shape[0] // self.pack_factor, int_weight.shape[1]), dtype=self.pack_np_math_dtype)
        if self.bits in [2, 4, 8]:
            for row in range(qweight.shape[0]):
                for j in range(self.pack_factor):
                    qweight[row] |= int_weight[row * self.pack_factor + j] << (self.bits * j)
        elif self.bits == 3:
            i, row = 0, 0
            while row < qweight.shape[0]:
                for j in range(i, i + 10): qweight[row] |= int_weight[j] << (3 * (j - i));
                i += 10; qweight[row] |= int_weight[i] << 30; row += 1
                qweight[row] |= (int_weight[i] >> 2) & 1; i += 1
                for j in range(i, i + 10): qweight[row] |= int_weight[j] << (3 * (j - i) + 1);
                i += 10; qweight[row] |= int_weight[i] << 31; row += 1
                qweight[row] |= (int_weight[i] >> 1) & 0x3; i += 1
                for j in range(i, i + 10): qweight[row] |= int_weight[j] << (3 * (j - i) + 2);
                i += 10; row += 1
        
        self.qweight = t.from_numpy(qweight.astype(self.pack_np_dtype))
        
        unpacked_zeros = zeros.numpy().astype(self.pack_np_math_dtype)
        qzeros = np.zeros((unpacked_zeros.shape[0], unpacked_zeros.shape[1] // self.pack_factor), dtype=self.pack_np_math_dtype)
        if self.bits in [2, 4, 8]:
            for row in range(qzeros.shape[0]): # Iterate through groups
                for col in range(qzeros.shape[1]): # Iterate through packed columns
                     for j in range(self.pack_factor):
                        qzeros[row, col] |= unpacked_zeros[row, col * self.pack_factor + j] << (self.bits * (j % self.pack_factor))
        elif self.bits == 3:
            i, col = 0, 0
            while col < qzeros.shape[1]:
                for j in range(i, i + 10): qzeros[:, col] |= unpacked_zeros[:, j] << (3 * (j - i));
                i += 10; qzeros[:, col] |= unpacked_zeros[:, i] << 30; col += 1
                qzeros[:, col] |= (unpacked_zeros[:, i] >> 2) & 1; i += 1
                for j in range(i, i + 10): qzeros[:, col] |= unpacked_zeros[:, j] << (3 * (j - i) + 1);
                i += 10; qzeros[:, col] |= unpacked_zeros[:, i] << 31; col += 1
                qzeros[:, col] |= (unpacked_zeros[:, i] >> 1) & 0x3; i += 1
                for j in range(i, i + 10): qzeros[:, col] |= unpacked_zeros[:, j] << (3 * (j - i) + 2);
                i += 10; col += 1
        
        self.qzeros = t.from_numpy(qzeros.astype(self.pack_np_dtype))