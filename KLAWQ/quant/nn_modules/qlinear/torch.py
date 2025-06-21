import torch
import torch.nn as nn
import torch.nn.functional as F # MODIFIED: Import functional for F.conv2d
from transformers import PreTrainedModel

from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.torch import torch_compile

log = setup_logger()

class TorchQuantLinear(PackableQuantLinear):
    SUPPORTS_BITS = [2, 3, 4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int8, torch.int16, torch.int32]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    QUANT_TYPE = "torch"

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        register_buffers: bool = True,
        **kwargs,
    ):
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.TORCH),
            register_buffers=register_buffers,
            **kwargs)
            
        # --- START OF MODIFICATIONS ---
        # Add flags and storage for convolutional layer parameters
        self.is_conv = False
        self.conv_kwargs = {}
        # --- END OF MODIFICATIONS ---

        self.dequant_dtype = torch.int16 if self.bits == 8 else torch.int8

    # --- ADDED NEW METHOD ---
    # This method is called during the model packing stage to let the layer know
    # if it's replacing a Conv2d layer.
    def set_conv_parameters_from_linear(self, linear_layer: nn.Module):
        if isinstance(linear_layer, nn.Conv2d):
            self.is_conv = True
            self.conv_kwargs = {
                "stride": linear_layer.stride,
                "padding": linear_layer.padding,
                "dilation": linear_layer.dilation,
                "groups": linear_layer.groups
            }
            # The original shape is crucial for reshaping the dequantized weight
            self.register_buffer('original_shape', torch.tensor(linear_layer.weight.shape))

    def post_init(self):
        super().post_init()
        self.optimize()

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        if self.optimized:
            return
        self.dequantize_weight = torch_compile(self.dequantize_weight, backend=backend, mode=mode, fullgraph=fullgraph)
        super().optimize()

    def train(self, mode: bool = True):
        old_train = self.training
        if mode == old_train:
            return self
        return super().train(mode=mode)

    def forward(self, x: torch.Tensor):
        # --- START OF THE DEFINITIVE FIX ---
        # This is the core logic change.
        if self.is_conv:
            # If it's a convolution, we dequantize the full weight and use F.conv2d
            # The weight must be reshaped back to its original 4D kernel shape.
            weight = self.dequantize_weight().reshape(self.original_shape.tolist())
            return F.conv2d(x, weight, self.bias, **self.conv_kwargs)
        else:
            # This is the original, unchanged logic for Linear layers
            out_shape = x.shape[:-1] + (self.out_features,)
            x = x.reshape(-1, x.shape[-1])
            out = self._forward(x, out_shape)
            return out
        # --- END OF THE DEFINITIVE FIX ---

    def _forward(self, x, out_shape):
        # This method is now ONLY called for Linear layers, so its logic is correct.
        num_itr = self.g_idx.shape[0] // x.shape[-1]
        weights = self.dequantize_weight(num_itr=num_itr).to(x.dtype)
        out = torch.matmul(x, weights).reshape(out_shape)
        if self.bias is not None:
            out.add_(self.bias)
        return out

    def _empty_gptq_only_weights(self):
        self.qzeros = None
        self.qweight = None
        self.g_idx = None
        self.scales = None

def dequantize_model(model: PreTrainedModel):
    for name, module in model.named_modules():
        if isinstance(module, BaseQuantLinear) and not isinstance(module, TorchQuantLinear):
            raise ValueError(
                "Only models loaded using TorchQuantLinear are supported for dequantization. "
                "Please load model using backend=BACKEND.TORCH."
            )

        if isinstance(module, TorchQuantLinear):
            # MODIFIED: Handle both Conv2d and Linear dequantization
            if module.is_conv:
                new_module = nn.Conv2d(
                    in_channels=module.original_shape[1] * module.conv_kwargs.get("groups", 1),
                    out_channels=module.original_shape[0],
                    kernel_size=module.original_shape[2:],
                    stride=module.conv_kwargs.get("stride", 1),
                    padding=module.conv_kwargs.get("padding", 0),
                    dilation=module.conv_kwargs.get("dilation", 1),
                    groups=module.conv_kwargs.get("groups", 1),
                    bias=module.bias is not None
                )
                dequantized_weight = module.dequantize_weight().reshape(module.original_shape)
                new_module.weight = nn.Parameter(dequantized_weight.detach().to("cpu", torch.float16))
            else:
                new_module = nn.Linear(module.in_features, module.out_features)
                new_module.weight = nn.Parameter(module.dequantize_weight().T.detach().to("cpu", torch.float16))
            
            if module.bias is not None:
                new_module.bias = torch.nn.Parameter(module.bias)

            parent = model
            if '.' in name:
                parent_name, module_name = name.rsplit('.', 1)
                parent = dict(model.named_modules())[parent_name]
            else:
                module_name = name
            setattr(parent, module_name, new_module)

    if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
        del model.config.quantization_config
    return model

# To make this work, the pack_module function in `utils/model.py` must be updated
# to call `set_conv_parameters_from_linear` after creating the TorchQuantLinear layer.

__all__ = ["TorchQuantLinear", "dequantize_model"]