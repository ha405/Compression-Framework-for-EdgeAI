from .._const import EXPERT_INDEX_PLACEHOLDER
from ..base import BaseGPTQModel


class Phi3GPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens", "embed_dropout", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = "model.layers"
    layer_type = ["Phi3DecoderLayer"]
    layer_modules = [
        ["self_attn.qkv_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_up_proj"],
        ["mlp.down_proj"],
    ]

class PhiMoEGPTQForCausalLM(BaseGPTQModel):
    require_pkgs_version = ["transformers<=4.44.2"]

    layer_type = "PhiMoEDecoderLayer"
    layers_node = "model.layers"
    base_modules = ["model.embed_tokens", "model.norm"]

    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        [f"block_sparse_moe.experts.{EXPERT_INDEX_PLACEHOLDER}.w1"],
        [f"block_sparse_moe.experts.{EXPERT_INDEX_PLACEHOLDER}.w2"],
    ]

__all__ = ["Phi3GPTQ", "PhiMoEGPTQForCausalLM"]
