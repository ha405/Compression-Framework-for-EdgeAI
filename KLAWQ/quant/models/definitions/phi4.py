from ..base import BaseGPTQModel


class Phi4MMGPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens", "model.embed_tokens_extend", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = "model.layers"
    layer_type = "Phi4MMDecoderLayer"

    # text modules only
    layer_modules = [
        ["self_attn.qkv_proj.base_layer"],
        ["self_attn.o_proj.base_layer"],
        ["mlp.gate_up_proj.base_layer"],
        ["mlp.down_proj.base_layer"],
    ]

    require_monkeypatch = True

    def monkey_patch(self):
        if not self.quantized:
            original_forward = self.model.forward

            # patch so input_mode is default to 0 (InputMode.LANGUAGE) if not passed
            # phi4mm default is None which causes quant error as it expects it to be always passed
            def patched_forward(self, **kwargs):
                if "input_mode" not in kwargs:
                    kwargs["input_mode"] = 0
                return original_forward(**kwargs)

            # bind forward to instance
            self.model.forward = patched_forward.__get__(self.model, type(self.model))

__all__ = ["Phi4MMGPTQ"]
