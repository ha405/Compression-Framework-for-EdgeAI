from ..base import BaseGPTQModel


class QwenGPTQ(BaseGPTQModel):
    base_modules = [
        "transformer.wte",
        "transformer.wpe",
        "transformer.ln_f",
        "transformer.visual",
    ]
    pre_lm_head_norm_module = "transformer.ln_f"

    layers_node = ["transformer.h"]
    layer_type = "QWenBlock"
    layer_modules = [
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.w1", "mlp.w2"],
        ["mlp.c_proj"],
    ]
