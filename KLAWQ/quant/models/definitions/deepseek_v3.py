from .._const import EXPERT_INDEX_PLACEHOLDER
from ..base import BaseGPTQModel


class DeepSeekV3GPTQ(BaseGPTQModel):
    # deepseek_v3 requires custom model code
    require_trust_remote_code = True

    require_fast_init = False

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "n_routed_experts"

    base_modules = ["model.embed_tokens", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = ["model.layers"]
    layer_type = "DeepseekV3DecoderLayer"

    # DeepSeek V3 uses dynamic modules based on lora(rank):
    layer_modules_strict = False

    layer_modules = [
        ["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"],
        ["self_attn.q_b_proj", "self_attn.kv_b_proj"],
        ["self_attn.o_proj"],

        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],

        # included in layer 3-61, uses dynamic_expert_index
        # DeepSeek-V3 uses 256 experts
        # for quantization on A100, don't merge gate_proj and up_proj
        # if you have enough vram to process 256 * 2 module inputs, then you can merge gate_proj and up_proj
        # into single stage which will make the quantization process faster
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_proj"],
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.up_proj"],
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.down_proj"],

        # included in layer 3-61
        ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"],
        ["mlp.shared_experts.down_proj"],
    ]
