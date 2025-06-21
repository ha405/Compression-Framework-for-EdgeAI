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
    SUPPORTS_BITS: List[int] = None
    SUPPORTS_GROUP_SIZE: List[int] = None
    SUPPORTS_DESC_ACT: List[bool] = None
    SUPPORTS_SYM: List[bool] = None
    SUPPORTS_SHARDS: bool = None
    SUPPORTS_TRAINING: bool = None
    SUPPORTS_TRAINING_USE_TORCH_KERNEL: bool = False

    SUPPORTS_AUTO_PADDING: bool = None
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY: List[int] = None
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY: List[int] = None

    SUPPORTS_PACK_DTYPES: List[t.dtype] = None
    SUPPORTS_DEVICES: List[DEVICE] = None
    SUPPORTS_PLATFORM: List[PLATFORM] = None

    SUPPORTS_DTYPES: List[t.dtype] = None

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
            self.pack_dtype_bits = 8
            self.pack_np_dtype = np.int8
            self.pack_np_math_dtype = np.uint8
        elif self.pack_dtype == t.int16:
            self.pack_dtype_bits = 16
            self.pack_np_dtype = np.int16
            self.pack_np_math_dtype = np.uint16
        elif self.pack_dtype == t.int32:
            self.pack_dtype_bits = 32
            self.pack_np_dtype = np.int32
            self.pack_np_math_dtype = np.uint32
        elif self.pack_dtype == t.int64:
            self.pack_dtype_bits = 64
            self.pack_np_dtype = np.int64
            self.pack_np_math_dtype = np.uint64
        else:
            raise ValueError("Unsupported weight_dtype. Only int16 and int32 are supported.")

        self.pack_factor = self.pack_dtype_bits // self.bits
        _, err = self._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, in_features=in_features, out_features=out_features, pack_dtype=pack_dtype)
        if err:
            raise err

        self._qzeros_format = 1

        if register_buffers:
            in_features_reg = self.in_features if not register_buffers_in_features else register_buffers_in_features
            out_features_reg = self.out_features if not register_buffers_out_features else register_buffers_out_features

            self.register_buffer(
                "qweight",
                t.zeros((math.ceil(in_features_reg / self.pack_factor), out_features_reg), dtype=self.pack_dtype),
            )
            self.register_buffer(
                "qzeros",
                t.zeros(
                    (
                        math.ceil(in_features_reg / self.group_size),
                        out_features_reg // self.pack_factor,
                    ),
                    dtype=self.pack_dtype,
                ),
            )
            self.register_buffer(
                "scales",
                t.zeros(
                    (math.ceil(in_features_reg / self.group_size), out_features_reg),
                    dtype=t.float16,
                ),
            )
            self.register_buffer(
                "g_idx",
                t.tensor([i // self.group_size for i in range(in_features_reg)], dtype=t.int32),
            )
            if bias:
                self.register_buffer("bias", t.zeros(out_features_reg, dtype=t.float16))
            else:
                self.bias = None


    def list_buffers(self) -> List:
        buf = []
        if hasattr(self, "qweight") and self.qweight is not None:
            buf.append(self.qweight)
        if hasattr(self, "qzeros") and self.qzeros is not None:
            buf.append(self.qzeros)
        if hasattr(self, "scales") and self.scales is not None:
            buf.append(self.scales)
        if hasattr(self, "g_idx") and self.g_idx is not None:
            buf.append(self.g_idx)
        if hasattr(self, "bias") and self.bias is not None:
            buf.append(self.bias)

        return buf

    def qzero_format(self, format: int = None) -> int:
        if format is None:
            return self._qzeros_format

        if format not in [1, 2]:
            raise ValueError("Unsupported qzero format. Only 1 and 2 are supported.")

        self._qzeros_format = format
        return self._qzeros_format

    def post_init(self):
        pass

    @classmethod
    def validate(
            cls,
            bits: int,
            group_size: int,
            desc_act: bool,
            sym: bool,
            in_features:int=None,
            out_features:int=None,
            pack_dtype:t.dtype=None,
            dynamic:Optional[dict]=None,
            device:Optional[DEVICE]=None,
            trainable:Optional[bool]=None,
    ) -> Tuple[
        bool, Optional[Exception]]:
        return cls._validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym,
                             in_features=in_features, out_features=out_features, pack_dtype=pack_dtype,
                             dynamic=dynamic, device=device, trainable=trainable)

    @classmethod
    def verify_supports_params(cls):
        """
        Validate that SUPPORTS parameters are not None or empty lists, raising an exception if the validation fails.
        """
        base_supports_variables = [
            (name, value) for name, value in BaseQuantLinear.__dict__.items()
            if name.startswith("SUPPORTS") and not callable(value) and value is None
        ]
        child_supports_variables = [
            (name, value) for name, value in cls.__dict__.items()
            if name.startswith("SUPPORTS") and not callable(value)
        ]

        base_supports_variables.sort(key=lambda x: x[0])
        child_supports_variables.sort(key=lambda x: x[0])

        base_variable_names = {name for name, value in base_supports_variables}
        child_variable_names = {name for name, value in child_supports_variables}

        missing_variables = base_variable_names - child_variable_names

        if missing_variables:
            raise ValueError(
                f"{cls.__name__} these SUPPORTS variables are not overridden: {', '.join(sorted(missing_variables))}")

        for name, value in child_supports_variables:
            if not name.startswith("SUPPORTS") or callable(value):
                continue
            if value is None:
                raise ValueError(f"{cls.__name__}.{name} cannot be None.")


    @classmethod
    def _validate(cls, bits: int=4, group_size: int=128, desc_act: bool=False, sym: bool=False, pack_dtype:t.dtype=None, dynamic:Optional[dict]=None, in_features:int=None,
                  out_features:int=None, device:Optional[DEVICE]=None, trainable:Optional[bool]=None) -> Tuple[bool, Optional[Exception]]:
        cls.verify_supports_params()

        if pack_dtype not in cls.SUPPORTS_PACK_DTYPES:
            err = f"{cls} does not support `pack_dtype`: {pack_dtype}"
            return False, NotImplementedError(err)

        if PLATFORM.ALL not in cls.SUPPORTS_PLATFORM and sys.platform not in cls.SUPPORTS_PLATFORM:
            err = f"{cls} does not support platform: {sys.platform}"
            return False, NotImplementedError(err)

        if DEVICE.ALL not in cls.SUPPORTS_DEVICES and device is not None:
            try:
                cls.validate_device(device)
            except NotImplementedError:
                e = f"{cls} does not support device: {device}"
                return False, NotImplementedError(e)

        if trainable and not cls.SUPPORTS_TRAINING:
            err = f"{cls} does not support training."
            return False, NotImplementedError(err)

        if bits not in cls.SUPPORTS_BITS:
            err = f"{cls} only supports `{cls.SUPPORTS_BITS}` bits: actual bits = `{bits}`"
            return False, NotImplementedError(err)
        if group_size not in cls.SUPPORTS_GROUP_SIZE and group_size != in_features:
            err = f"{cls} only supports `{cls.SUPPORTS_GROUP_SIZE}` group_size: actual group_size = `{group_size}`"
            return False, NotImplementedError(err)
        if sym not in cls.SUPPORTS_SYM:
            err = f"{cls} only supports `{cls.SUPPORTS_SYM}` bits: actual sym = `{sym}`"
            return False, NotImplementedError(err)
        if desc_act not in cls.SUPPORTS_DESC_ACT:
            err = f"{cls} only supports `{cls.SUPPORTS_DESC_ACT}` bits: actual desc_act = `{desc_act}`"
            return False, NotImplementedError(err)
        if dynamic is not None:
            dynamic_bits = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_bits[pattern] = pattern_dict.get("bits", bits)
            if len(cls.SUPPORTS_BITS) == 1:
                err = f"{cls} not supported dynamic_bits, only support `{cls.SUPPORTS_BITS}` bits"
                return False, NotImplementedError(err)
            else:
                for layer, bits in dynamic_bits.items():
                    if bits not in cls.SUPPORTS_BITS:
                        err = f"{cls} only supports `{cls.SUPPORTS_BITS}` bits: actual dynamic_bits = `{bits}` for layer `{layer}`"
                        return False, NotImplementedError(err)

            dynamic_group_size = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_group_size[pattern] = pattern_dict.get("group_size", group_size)
            for layer, group_size in dynamic_group_size.items():
                if group_size not in cls.SUPPORTS_GROUP_SIZE:
                    err = f"{cls} only supports `{cls.SUPPORTS_GROUP_SIZE}` group_size: actual group_size = `{group_size}` for layer `{layer}`"
                    return False, NotImplementedError(err)

            dynamic_sym = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_sym[pattern] = pattern_dict.get("sym", sym)
            for layer, sym in dynamic_sym.items():
                if sym not in cls.SUPPORTS_SYM:
                    err = f"{cls} only supports `{cls.SUPPORTS_SYM}` bits: actual sym = `{sym}` for layer `{layer}`"
                    return False, NotImplementedError(err)

            dynamic_desc_act = {}
            for pattern, pattern_dict in dynamic.items():
                dynamic_desc_act[pattern] = pattern_dict.get("desc_act", desc_act)
            for layer, desc_act in dynamic_desc_act.items():
                if desc_act not in cls.SUPPORTS_DESC_ACT:
                    err = f"{cls} only supports `{cls.SUPPORTS_DESC_ACT}` bits: actual desc_act = `{desc_act}` for layer `{layer}`"
                    return False, NotImplementedError(err)

        if in_features is not None:
            validate = all(in_features % in_fea == 0 for in_fea in cls.SUPPORTS_IN_FEATURES_DIVISIBLE_BY)
            if not validate:
                err = f"{cls}: `in_features`: {in_features} must be divisible by {cls.SUPPORTS_IN_FEATURES_DIVISIBLE_BY}."
                return False, NotImplementedError(err)

            validate = in_features % group_size == 0 or cls.SUPPORTS_AUTO_PADDING
            if not validate:
                err = f"{cls}: `in_features`: {in_features} must be divisible by `group_size: {group_size}`."
                return False, NotImplementedError(err)
        if out_features is not None:
            validate = all(out_features % out_fea == 0 for out_fea in cls.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY)
            if not validate:
                err = f"{cls}: `out_features`: {out_features} must be divisible by {cls.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY}."
                return False, NotImplementedError(err)
        return True, None

    @classmethod
    def validate_device(cls, device: DEVICE):
        assert isinstance(device, DEVICE)

        if device not in cls.SUPPORTS_DEVICES:
            raise NotImplementedError(f"{cls} only supports `{cls.SUPPORTS_DEVICES}`: actual device = `{device}`")

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        self.optimized = True
        log.info.once(f"Optimize: `{self.__class__.__name__}` compilation triggered.")
        pass

    def train(self, mode=True):
        old_mode = self.training

        if old_mode == mode:
            return self

        if mode:
            if not self.SUPPORTS_TRAINING:
                err = f"{self.__class__.__name__}: `{self.name}` switching to training mode."
                log.error(err)
                raise NotImplementedError(err)
            else:
                pass
        else:
            pass

        return super().train(mode)

class PackableQuantLinear(BaseQuantLinear):
    def post_init(self, **kwargs):
        if self.bits in [2, 4, 8]:
            wf = t.tensor(list(range(0, self.pack_dtype_bits, self.bits)), dtype=t.int32).unsqueeze(0).to(
                device=self.g_idx.device)
        elif self.bits == 3:
            wf = t.tensor(
                [
                    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                    [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                    [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],
                ],
                dtype=t.int32,
            ).reshape(1, 3, 12).to(device=self.g_idx.device)

        self.wf_unsqueeze_zero = wf.unsqueeze(0).to(device=self.g_idx.device)
        self.wf_unsqueeze_neg_one = wf.unsqueeze(-1).to(device=self.g_idx.device)

        super().post_init(**kwargs)

    def list_buffers(self):
        return super().list_buffers() + [
            self.wf_unsqueeze_zero,
            self.wf_unsqueeze_neg_one,
        ]

    def dequantize_weight(self, num_itr: int = 1):
        # --- START OF THE DEFINITIVE FIX ---
        # 1. Unpack qzeros to get a tensor of shape (num_groups, out_features)
        if self.bits in [2, 4, 8]:
            zeros = t.bitwise_right_shift(t.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor), self.wf_unsqueeze_zero)
            zeros = t.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)
        else: # bits == 3
            zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1).expand(-1, -1, -1, 12)
            zeros = zeros >> self.wf_unsqueeze_zero
            zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
            zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
            zeros = zeros & 0x7
            zeros = t.cat([zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]], dim=2).reshape(self.scales.shape)

        # 2. Unpack qweight to get a tensor of shape (padded_in_features, out_features)
        if self.bits in [2, 4, 8]:
            weight = t.bitwise_right_shift(t.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1), self.wf_unsqueeze_neg_one)
            weight = t.bitwise_and(weight, self.maxq)
            weight = weight.reshape(-1, self.out_features)
        else: # bits == 3
            weight = self.qweight.reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
            weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = t.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
            weight = weight.reshape(-1, self.out_features)

        # 3. Slice the unpacked weight to match the true `in_features` dimension, removing any padding.
        weight = weight[:self.in_features, :]
        
        # 4. Use g_idx to expand scales and zeros to match the weight tensor's shape.
        # This is the only robust way to handle grouping for all cases.
        weights = self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])
        # --- END OF THE DEFINITIVE FIX ---

        return weights

    def pack(self,
             linear: nn.Module,
             scales: t.Tensor,
             zeros: t.Tensor,
             g_idx: t.Tensor = None):
        """
        Quantize and bit-pack the weight from `linear` into `self.qweight`/`self.qzeros`.
        Handles nn.Linear, HF Conv1D, and torch._ConvNd by flattening appropriately.

        Args:
          linear: nn.Linear, Conv1D, or ConvNd module.
          scales: Tensor [num_groups, out_features] of scale factors.
          zeros:  Tensor [num_groups, out_features] of zero points.
          g_idx:  Optional Tensor [in_features] mapping inputs to groups.
        """
        # 1) Extract weight as [out_features, in_total]
        W = linear.weight.data.clone()
        if isinstance(linear, _ConvNd):
            # e.g. Conv2d: [out, in, kH, kW]
            W = W.flatten(1)  # [out, in * kH * kW]
        elif isinstance(linear, transformers.pytorch_utils.Conv1D):
            # HF Conv1D stores [in, out]
            W = W.T            # -> [out, in]
        # else nn.Linear is already [out, in]

        out_features, in_total = W.shape
        assert out_features == self.out_features, f"out_features mismatch: {out_features} vs {self.out_features}"

        # 2) Build full-length group index for every input column
        base_gidx = (g_idx.clone() if g_idx is not None else self.g_idx).long()
        # if flattened conv, repeat per kernel element
        kernel_elements = in_total // base_gidx.shape[0]
        if kernel_elements > 1:
            g_idx_full = base_gidx.repeat_interleave(kernel_elements)
        else:
            g_idx_full = base_gidx
        assert g_idx_full.shape[0] == in_total

        # 3) Prepare [out, num_groups]
        scales_og = scales.T.contiguous()  # [out, groups]
        zeros_og  = zeros.T.contiguous()   # [out, groups]

        # 4) Expand to [out, in_total]
        exp_s = scales_og[:, g_idx_full]  # [out, in_total]
        exp_z = zeros_og[:,  g_idx_full]  # [out, in_total]

        # 5) Quantize: round(W / scale + zero)
        intW = t.round((W / exp_s) + exp_z).to(t.int32)

        # 6) Store float16 metadata
        self.scales = scales.clone().to(dtype=t.float16)
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(dtype=t.float16)

        # 7) Pad in_total to multiple of pack_factor
        pad = (-in_total) % self.pack_factor
        if pad:
            pad_tensor = intW.new_zeros(out_features, pad)
            intW = t.cat([intW, pad_tensor], dim=1)
            in_total += pad

        # 8) Move to numpy [in_padded, out]
        int_np = intW.T.contiguous().cpu().numpy().astype(self.pack_np_math_dtype)

        # 9) Bit-pack into qweight
        num_rows = in_total // self.pack_factor
        qw = np.zeros((num_rows, out_features), dtype=self.pack_np_math_dtype)
        if self.bits in [2, 4, 8]:
            for r in range(num_rows):
                for j in range(self.pack_factor):
                    qw[r] |= int_np[r * self.pack_factor + j] << (self.bits * j)
        else:  # 3-bit special
            i = 0
            row = 0
            while row < num_rows:
                # first 10 values
                for j in range(i, i + 10):
                    qw[row] |= int_np[j] << (3 * (j - i))
                i += 10
                qw[row] |= int_np[i] << 30
                row += 1

                qw[row] |= (int_np[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qw[row] |= int_np[j] << (3 * (j - i) + 1)
                i += 10
                qw[row] |= int_np[i] << 31
                row += 1

                qw[row] |= (int_np[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qw[row] |= int_np[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
        self.qweight = t.from_numpy(qw.astype(self.pack_np_dtype))

        # 10) Pack qzeros similarly
        zeros_np = zeros_og.cpu().numpy().astype(self.pack_np_math_dtype)  # [out, groups]
        num_cols = zeros_np.shape[1] // self.pack_factor
        qz = np.zeros((zeros_np.shape[0], num_cols), dtype=self.pack_np_math_dtype)
        if self.bits in [2, 4, 8]:
            for c in range(num_cols):
                for j in range(self.pack_factor):
                    qz[:, c] |= zeros_np[:, c * self.pack_factor + j] << (self.bits * j)
        else:  # 3-bit
            i = 0
            col = 0
            while col < num_cols:
                for j in range(i, i + 10):
                    qz[:, col] |= zeros_np[:, j] << (3 * (j - i))
                i += 10
                qz[:, col] |= zeros_np[:, i] << 30
                col += 1

                qz[:, col] |= (zeros_np[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qz[:, col] |= zeros_np[:, j] << (3 * (j - i) + 1)
                i += 10
                qz[:, col] |= zeros_np[:, i] << 31
                col += 1

                qz[:, col] |= (zeros_np[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qz[:, col] |= zeros_np[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1
        self.qzeros = t.from_numpy(qz.astype(self.pack_np_dtype))
