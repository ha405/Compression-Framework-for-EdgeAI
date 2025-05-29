import os
from .models import GPTQModel, get_best_device
from .quantization import BaseQuantizeConfig, QuantizeConfig
from .utils import BACKEND
from .version import __version__

if os.getenv('GPTQMODEL_USE_MODELSCOPE', 'False').lower() in ['true', '1']:
    try:
        from modelscope.utils.hf_util.patcher import patch_hub
        patch_hub()
    except Exception:
        raise ModuleNotFoundError("you have set GPTQMODEL_USE_MODELSCOPE env, but doesn't have modelscope? install it with `pip install modelscope`")
