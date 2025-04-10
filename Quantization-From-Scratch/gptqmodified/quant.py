import numpy as np
import torch
import torch.nn as nn

# quantize function remains the same, used by Quantizer internally
def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)
        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4: x = x.permute([1, 0, 2, 3]).flatten(1)
                elif len(shape) == 3: x = x.reshape((-1, shape[-1])).t()
                elif len(shape) == 2: x = x.t()
                else: x = x.flatten().unsqueeze(0)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp): xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0); xmin[tmp] = -1; xmax[tmp] = +1

        if self.maxq < 0: self.scale = xmax; self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym: self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else: self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid; xmin1 = p * xmin; xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x; q.abs_(); q.pow_(self.norm); err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp): best[tmp] = err[tmp]; self.scale[tmp] = scale1[tmp]; self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight: tmp = shape[0]
            else: tmp = shape[1] if len(shape) == 4 else shape[-1]
            self.scale = self.scale.repeat(tmp); self.zero = self.zero.repeat(tmp)

        if weight:
            shape_new = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape_new); self.zero = self.zero.reshape(shape_new)
            return
        if len(shape) == 4: self.scale = self.scale.reshape((1, -1, 1, 1)); self.zero = self.zero.reshape((1, -1, 1, 1))
        elif len(shape) == 3: self.scale = self.scale.reshape((1, 1, -1)); self.zero = self.zero.reshape((1, 1, -1))
        elif len(shape) == 2: self.scale = self.scale.unsqueeze(0); self.zero = self.zero.unsqueeze(0)

    def quantize(self, x): return quantize(x, self.scale, self.zero, self.maxq) if self.ready() else x
    def enabled(self): return self.maxq != 0
    def ready(self): return torch.all(self.scale != 0)

try: import quant_cuda
except ImportError: print('CUDA extension quant_cuda not found.'); quant_cuda = None
except Exception as e: print(f'Error importing quant_cuda: {e}'); quant_cuda = None

class Quant3Linear(nn.Module):
    def __init__(self, infeatures, outfeatures):
        super().__init__()
        self.infeatures = infeatures; self.outfeatures = outfeatures
        self.register_buffer('zeros', torch.zeros(outfeatures, dtype=torch.uint8))
        self.register_buffer('scales', torch.zeros(outfeatures))
        self.register_buffer('bias', torch.zeros(outfeatures))
        if infeatures % 4 != 0: raise ValueError("infeatures must be divisible by 4")
        self.register_buffer('qweight', torch.zeros((infeatures // 4, outfeatures), dtype=torch.uint32)) # Ensure uint32

    def pack(self, linear, scales, zeros_float):
        self.scales = scales.clone().reshape(self.outfeatures)
        self.zeros = zeros_float.round().clamp(0, 255).to(torch.uint8).reshape(self.outfeatures)

        if linear.bias is not None:
            self.bias = linear.bias.clone()

        # Quantize weight to uint8 (Vectorized)
        scale_reshaped = self.scales.unsqueeze(1)
        zeros_reshaped = self.zeros.unsqueeze(1)
        # Ensure original weight is on CPU for packing if it isn't already
        weight_fp = linear.weight.data.to(device='cpu', non_blocking=True)

        intweight = torch.clamp(torch.round(weight_fp / scale_reshaped) + zeros_reshaped, 0, 255).to(torch.uint8)

        # Transpose to [infeatures, outfeatures]
        intweight = intweight.t().contiguous() # Shape: [infeatures, outfeatures]

        # --- Vectorized Packing ---
        # Ensure intweight is on CPU for the view/reshape operations
        intweight = intweight.cpu()

        # Check dimensions
        if self.infeatures % 4 != 0:
             raise ValueError("infeatures must be divisible by 4 for vectorized packing")
        if intweight.shape != (self.infeatures, self.outfeatures):
             raise ValueError(f"Unexpected intweight shape: {intweight.shape}. Expected: {(self.infeatures, self.outfeatures)}")

        # Reshape to group elements that will be packed together
        # Shape: [infeatures / 4, 4, outfeatures]
        intweight_reshaped = intweight.view(self.infeatures // 4, 4, self.outfeatures)

        # Cast components to uint32 *before* shifting
        w1 = intweight_reshaped[:, 0, :].to(torch.int64)
        w2 = intweight_reshaped[:, 1, :].to(torch.int64)
        w3 = intweight_reshaped[:, 2, :].to(torch.int64)
        w4 = intweight_reshaped[:, 3, :].to(torch.int64)

        # Perform bitwise operations on the entire tensors
        # Order: w4 w3 w2 w1 (assuming little-endian interpretation by kernel)
        packed_val_tensor_int64 = w1 | (w2 << 8) | (w3 << 16) | (w4 << 24)

        # Assign the resulting tensor directly to the buffer
        self.qweight = packed_val_tensor_int64.to(torch.uint32).contiguous() # Shape: [infeatures / 4, outfeatures]
        # --------------------------

    def forward(self, x):
        if quant_cuda is None: raise RuntimeError("quant_cuda extension not available.")
        if x.shape[-1] != self.infeatures: raise ValueError(f"Input feature mismatch: {self.infeatures} vs {x.shape[-1]}")
        original_shape = None
        if len(x.shape) > 2: original_shape = x.shape; x = x.reshape(-1, self.infeatures)
        out_shape_flat = list(x.shape[:-1]) + [self.outfeatures]
        y = torch.zeros(out_shape_flat, dtype=x.dtype, device=x.device)
        if self.bias is not None and self.bias.numel() > 0: y += self.bias # Check bias has elements
        quant_cuda.vecquant8matmul(x, self.qweight, y, self.scales, self.zeros)
        if original_shape is not None: y = y.reshape(list(original_shape[:-1]) + [self.outfeatures])
        return y

def make_quant3(module, names_dict, name=''): # Changed `names` to `names_dict`
    if isinstance(module, Quant3Linear): return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        # Check if the *specific layer name* (relative to parent) is in the dict keys for replacement
        if attr in names_dict: # Check relative name `attr`
            if isinstance(tmp, nn.Linear):
                 setattr(module, attr, Quant3Linear(tmp.in_features, tmp.out_features))
            # else: print(f"Warning: Layer {name1} targeted but not nn.Linear.") # Optional warning
    for name1, child in module.named_children():
        # Pass the original full names dict down for checking deeper levels
        make_quant3(child, names_dict, name + '.' + name1 if name != '' else name1)