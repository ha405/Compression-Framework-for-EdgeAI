from ..base import BaseGPTQModel


class LlamaGPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = ["model.layers"]
    layer_type = "LlamaDecoderLayer"
    layers_modules_tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("k_proj", "v_proj", "q_proj", "o_proj"),
            "mlp": ("up_proj", "gate_proj", "down_proj"),
        }
    ]

    # Inside each `LlamaDecoderLayer` layer are many internal modules
    # List them in the order executed in model forward() code
    # Many models have same execution order of: attention (q_k_v) projection, attention (output) projection, mlp (n) projections
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj",],
        ["mlp.down_proj"],
    ]
