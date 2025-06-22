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
        # 1) Unpack qzeros → (num_groups, out_features)
        if self.bits in [2, 4, 8]:
            zeros = (
                t.bitwise_and(
                    t.bitwise_right_shift(
                        t.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
                        self.wf_unsqueeze_zero
                    ),
                    self.maxq
                )
                .reshape(self.scales.shape)
            )
        else:  # 3-bit special
            zeros = (
                self.qzeros
                .reshape(self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1)
                .expand(-1, -1, -1, 12)
            )
            zeros = zeros >> self.wf_unsqueeze_zero
            zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
            zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
            zeros = (zeros & 0x7)
            zeros = (
                t.cat([zeros[:, :, 0, :11],
                       zeros[:, :, 1, 1:12],
                       zeros[:, :, 2, 1:11]], dim=2)
                .reshape(self.scales.shape)
            )

        # 2) Unpack qweight → (padded_in_features, out_features)
        if self.bits in [2, 4, 8]:
            weight = (
                t.bitwise_and(
                    t.bitwise_right_shift(
                        t.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                        self.wf_unsqueeze_neg_one
                    ),
                    self.maxq
                )
                .reshape(-1, self.out_features)
            )
        else:  # 3-bit special
            weight = (
                self.qweight
                .reshape(self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1])
                .expand(-1, -1, 12, -1)
            )
            weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7
            weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
            weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
            weight = weight & 0x7
            weight = (
                t.cat([weight[:, 0, :11],
                       weight[:, 1, 1:12],
                       weight[:, 2, 1:11]], dim=1)
                .reshape(-1, self.out_features)
            )

        # 3) Trim off any pad so we only have exactly in_features rows
        weight = weight[: self.in_features, :]

        # 4) Finally dequantize per-group via g_idx
        #    scales & zeros are (num_groups, out_features)
        #    g_idx is length in_features → for each row choose its group
        return self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()])


    def pack(
        self,
        linear: nn.Module,
        scales: t.Tensor,
        zeros: t.Tensor,
        g_idx: t.Tensor = None,
    ):
        """
        Quantize & bit-pack `linear.weight` into self.qweight/self.qzeros.
        Supports:
          - torch._ConvNd (Conv1d/2d/3d): flattens weight to [out, in_total]
          - transformers.pytorch_utils.Conv1D: transposes weight to [out, in]
          - nn.Linear
        """
        # 1) Grab and flatten weight
        W = linear.weight.data.clone().float()
        if isinstance(linear, _ConvNd):
            out_c, in_c, *kernel = W.shape
            in_total = in_c * math.prod(kernel)
            W = W.flatten(1)                   # → [out_c, in_total]
        elif isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.T                            # HF Conv1D stored [in, out]
            out_c, in_total = W.shape
        else:
            out_c, in_total = W.shape         # nn.Linear

        # 2) Build full-length group index
        if g_idx is None:
            base = t.arange(self.in_features, device=W.device) // self.group_size
        else:
            base = g_idx.clone().long()
        repeats = in_total // base.numel()
        g_idx_full = base.repeat_interleave(repeats)

        # 3) Normalize scales/zeros to [groups, out_c]
        num_groups = math.ceil(self.in_features / self.group_size)
        if scales.shape == (num_groups, out_c):
            s_go, z_go = scales, zeros
        else:
            # maybe passed in as [out_c, groups]
            s_go, z_go = scales.T.contiguous(), zeros.T.contiguous()

        # 4) Expand to per-input
        exp_s = s_go[g_idx_full, :]    # [in_total, out_c]
        exp_z = z_go[g_idx_full, :]    # [in_total, out_c]

        # 5) Integerize
        # W is [out_c, in_total], so transpose metadata
        W_int = t.round((W + exp_z.T) / exp_s.T).to(t.int32)

        # 6) Store float16 metadata for forward
        self.scales = s_go.to(t.float16)
        if getattr(linear, "bias", None) is not None:
            self.bias = linear.bias.clone().to(t.float16)

        # 7) Pad to pack_factor
        pad = (-in_total) % self.pack_factor
        if pad > 0:
            extra = t.zeros(out_c, pad, dtype=W_int.dtype, device=W_int.device)
            W_int = t.cat([W_int, extra], dim=1)
            in_total += pad

        # 8) Bit-pack qweight → [in_total//pack_factor, out_c]
        int_np = W_int.T.cpu().numpy().astype(self.pack_np_math_dtype)  # [in_total, out_c]
        rows = in_total // self.pack_factor
        qw = np.zeros((rows, out_c), dtype=self.pack_np_math_dtype)
        for r in range(rows):
            block = int_np[r*self.pack_factor:(r+1)*self.pack_factor, :]
            for b in range(self.pack_factor):
                qw[r] |= (block[b] & self.maxq) << (self.bits * b)
        self.qweight = t.from_numpy(qw.astype(self.pack_np_dtype)).to(self.scales.device)

        # 9) Bit-pack qzeros → [num_groups, out_c//pack_factor]
        z_np = z_go.cpu().numpy().astype(self.pack_np_math_dtype)        # [groups, out_c]
        cols = out_c // self.pack_factor
        qz = np.zeros((num_groups, cols), dtype=self.pack_np_math_dtype)
        for g in range(num_groups):
            for c in range(cols):
                base_idx = c * self.pack_factor
                acc = 0
                for b in range(self.pack_factor):
                    acc |= int(z_np[g, base_idx + b] & self.maxq) << (self.bits * b)
                qz[g, c] = acc
        self.qzeros = t.from_numpy(qz.astype(self.pack_np_dtype)).to(self.scales.device)

