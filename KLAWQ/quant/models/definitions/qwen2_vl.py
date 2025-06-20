from .base_qwen2_vl import BaseQwen2VLGPTQ


class Qwen2VLGPTQ(BaseQwen2VLGPTQ):
    layer_type = "Qwen2VLDecoderLayer"
