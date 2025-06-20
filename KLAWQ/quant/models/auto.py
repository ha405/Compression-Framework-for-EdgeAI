# auto_gptq/modeling/__init__.py

import os

import threadpoolctl

from ..utils.logger import setup_logger

log = setup_logger()

if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF", None):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'
    log.info("ENV: Auto setting PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' for memory saving.")

if not os.environ.get("CUDA_DEVICE_ORDER", None):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    log.info("ENV: Auto setting CUDA_DEVICE_ORDER=PCI_BUS_ID for correctness.")

if 'CUDA_VISIBLE_DEVICES' in os.environ and 'ROCR_VISIBLE_DEVICES' in os.environ:
    del os.environ['ROCR_VISIBLE_DEVICES']

import sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import os.path
import random
from os.path import isdir, join
from typing import Any, Dict, List, Optional, Type, Union

import numpy
import torch
from huggingface_hub import list_repo_files
from tokenicer import Tokenicer
# MODIFIED: Import nn to check for non-HF models
import torch.nn as nn
from transformers import AutoConfig, GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase

from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..quantization import QUANT_CONFIG_FILENAME
from ..quantization.gptq import CPU
from ..utils import BACKEND
from ..utils.eval import EVAL
from ..utils.model import copy_py_files, find_modules, get_model_files_size, get_moe_layer_modules, get_state_dict_for_save, load_checkpoint_in_model_then_tie_weights, make_quant
from ..utils.torch import torch_empty_cache
from .base import BaseGPTQModel, QuantizeConfig
from ..version import __version__
from transformers.modeling_utils import no_init_weights
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils.generic import ContextManagers

from ..quantization.config import (FORMAT, META_FIELD_DAMP_AUTO_INCREMENT, META_FIELD_DAMP_PERCENT, META_FIELD_MSE,
                                   META_FIELD_QUANTIZER, META_FIELD_STATIC_GROUPS, META_FIELD_TRUE_SEQUENTIAL,
                                   META_FIELD_URI, META_QUANTIZER_GPTQMODEL, META_VALUE_URI)

from .definitions.deepseek_v2 import DeepSeekV2GPTQ
from .definitions.deepseek_v3 import DeepSeekV3GPTQ
from .definitions.gpt2 import GPT2GPTQ
from .definitions.llama import LlamaGPTQ
from .definitions.mistral import MistralGPTQ
from .definitions.mixtral import MixtralGPTQ
from .definitions.mllama import MLlamaGPTQ
from .definitions.mobilellm import MobileLLMGPTQ
from .definitions.phi import PhiGPTQ
from .definitions.phi3 import Phi3GPTQ, PhiMoEGPTQForCausalLM
from .definitions.phi4 import Phi4MMGPTQ
from .definitions.qwen import QwenGPTQ
from .definitions.qwen2 import Qwen2GPTQ
from .definitions.qwen2_5_vl import Qwen2_5_VLGPTQ
from .definitions.qwen2_moe import Qwen2MoeGPTQ
from .definitions.qwen2_vl import Qwen2VLGPTQ
from .definitions.qwen3 import Qwen3GPTQ
from .definitions.qwen3_moe import Qwen3MoeGPTQ
from .definitions.resnet import ResNet50GPTQ
torch.manual_seed(787)
random.seed(787)
numpy.random.seed(787)

MODEL_MAP = {
    "gpt2": GPT2GPTQ,
    "llama": LlamaGPTQ,
    "qwen": QwenGPTQ,
    "mistral": MistralGPTQ,
    "mixtral": MixtralGPTQ,
    "qwen2": Qwen2GPTQ,
    "qwen3": Qwen3GPTQ,
    "phi": PhiGPTQ,
    "phi3": Phi3GPTQ,
    "phi4mm": Phi4MMGPTQ,
    "phimoe": PhiMoEGPTQForCausalLM,
    "qwen2_moe": Qwen2MoeGPTQ,
    "qwen3_moe": Qwen3MoeGPTQ,
    "qwen2_vl": Qwen2VLGPTQ,
    "qwen2_5_vl": Qwen2_5_VLGPTQ,
    "deepseek_v2": DeepSeekV2GPTQ,
    "deepseek_v3": DeepSeekV3GPTQ,
    "mllama": MLlamaGPTQ,
    "mobilellm": MobileLLMGPTQ,
    "resnet50": ResNet50GPTQ, # This line from you is correct and necessary
}

SUPPORTED_MODELS = list(MODEL_MAP.keys())


def check_and_get_model_type(model_dir, trust_remote_code=False):
    # This function is now only called for Hugging Face models.
    try:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")
        model_type = config.model_type
        return model_type.lower()
    except Exception as e:
        raise ValueError(f"Could not determine model type for {model_dir}. Please provide `model_type` argument. Error: {e}")


class GPTQModel:
    def __init__(self):
        raise EnvironmentError()

    @classmethod
    def load(
            cls,
            model_id_or_path: Optional[str],
            quantize_config: Optional[QuantizeConfig | Dict] = None,
            device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
            device: Optional[Union[str, torch.device]] = None,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            trust_remote_code: bool = False,
            verify_hash: Optional[Union[str, List[str]]] = None,
            **kwargs,
    ):
        if isinstance(model_id_or_path, str):
            model_id_or_path = model_id_or_path.strip()

        if isinstance(quantize_config, Dict):
            quantize_config = QuantizeConfig(**quantize_config)

        if isinstance(backend, str):
            backend = BACKEND(backend)

        is_quantized = False
        try:
            # MODIFIED: Wrap in try-except for non-HF models
            if hasattr(AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code),
                       "quantization_config"):
                is_quantized = True
        except Exception:
            pass # Fails for models like resnet50, which is expected.

        if not is_quantized:
            for name in [QUANT_CONFIG_FILENAME, "quant_config.json"]:
                if isdir(model_id_or_path):
                    if os.path.exists(join(model_id_or_path, name)):
                        is_quantized = True
                        break

                else:
                    # MODIFIED: Handle cases where model_id_or_path is not a repo
                    try:
                        files = list_repo_files(repo_id=model_id_or_path)
                        for f in files:
                            if f == name:
                                is_quantized = True
                                break
                    except Exception:
                        pass # Fails for local paths or non-repo models, which is fine.

        if is_quantized:
            return cls.from_quantized(
                model_id_or_path=model_id_or_path,
                device_map=device_map,
                device=device,
                backend=backend,
                trust_remote_code=trust_remote_code,
                verify_hash=verify_hash,
                **kwargs,
            )
        else:
            return cls.from_pretrained(
                model_id_or_path=model_id_or_path,
                quantize_config=quantize_config,
                device_map=device_map,
                device=device,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

    @classmethod
    def from_pretrained(
            cls,
            model_id_or_path: str,
            quantize_config: QuantizeConfig,
            trust_remote_code: bool = False,
            # ADDED: New argument to manually specify the model type
            model_type: Optional[str] = None,
            **model_init_kwargs,
    ) -> BaseGPTQModel:
        # MODIFIED: Check if model is a HF model before checking for quantization config
        try:
            config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
            if hasattr(config, "quantization_config"):
                log.warn("Model is already quantized, will use `from_quantized` to load quantized model.\n"
                               "If you want to quantize the model, please pass un_quantized model path or id, and use "
                               "`from_pretrained` with `quantize_config`.")
                return cls.from_quantized(model_id_or_path, trust_remote_code=trust_remote_code, model_type=model_type)
        except Exception:
            pass # Not a Hugging Face model, proceed.


        if quantize_config and quantize_config.dynamic:
            log.warn(
                "GPTQModel's per-module `dynamic` quantization feature is fully supported in latest vLLM and SGLang but not yet available in hf transformers.")
        
        # MODIFIED: Use the user-provided model_type if available, otherwise detect it.
        if not model_type:
            log.info("model_type not specified, trying to auto-detect from config...")
            model_type = check_and_get_model_type(model_id_or_path, trust_remote_code)
        
        log.info(f"Using model type: {model_type}")
        if model_type not in MODEL_MAP:
            raise ValueError(f"Unsupported model type '{model_type}'. Supported types: {list(MODEL_MAP.keys())}")

        return MODEL_MAP[model_type].from_pretrained(
            pretrained_model_id_or_path=model_id_or_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            **model_init_kwargs,
        )

    @classmethod
    def from_quantized(
            cls,
            model_id_or_path: Optional[str],
            device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
            device: Optional[Union[str, int]] = None,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            trust_remote_code: bool = False,
            verify_hash: Optional[Union[str, List[str]]] = None,
            # ADDED: New argument to manually specify the model type
            model_type: Optional[str] = None,
            **kwargs,
    ) -> BaseGPTQModel:
        # MODIFIED: Use the user-provided model_type if available, otherwise detect it.
        if not model_type:
            log.info("model_type not specified, trying to auto-detect from config...")
            model_type = check_and_get_model_type(model_id_or_path, trust_remote_code)
        
        log.info(f"Using model type: {model_type}")
        if model_type not in MODEL_MAP:
            raise ValueError(f"Unsupported model type '{model_type}'. Supported types: {list(MODEL_MAP.keys())}")

        if isinstance(backend, str):
            backend = BACKEND(backend)

        return MODEL_MAP[model_type].from_quantized(
            model_id_or_path=model_id_or_path,
            device_map=device_map,
            device=device,
            backend=backend,
            trust_remote_code=trust_remote_code,
            verify_hash=verify_hash,
            **kwargs,
        )

    @classmethod
    def eval(
            cls,
            model_or_id_or_path: str=None,
            tokenizer: Union[PreTrainedTokenizerBase, Tokenicer]=None,
            tasks: Union[EVAL.LM_EVAL, EVAL.EVALPLUS, List[EVAL.LM_EVAL], List[EVAL.EVALPLUS], EVAL.MMLU_PRO, List[EVAL.MMLU_PRO]] = None,
            framework: Union[Type[EVAL.LM_EVAL],Type[EVAL.EVALPLUS],Type[EVAL.MMLU_PRO]] = EVAL.LM_EVAL,
            batch_size: Union[int, str] = 1,
            trust_remote_code: bool = False,
            output_path: Optional[str] = None,
            llm_backend: str = 'gptqmodel',
            backend: BACKEND = BACKEND.AUTO,
            random_seed: int = 1234,
            model_args: Dict[str, Any] = None,
            ntrain: int = 1,
            **args
    ):
        from peft import PeftModel
        if model_args is None:
            model_args = {}
        if tasks is None:
            if framework == EVAL.LM_EVAL:
                tasks = [EVAL.LM_EVAL.ARC_CHALLENGE]
            if framework == EVAL.MMLU_PRO:
                tasks = [EVAL.MMLU_PRO.MATH]
            else:
                tasks = [EVAL.EVALPLUS.HUMAN]

        elif not isinstance(tasks, List):
            tasks = [tasks]

        if framework is None:
            raise ValueError("Eval parameter: `framework` cannot be set to None")

        if not isinstance(tasks, list):
            raise ValueError("Eval parameter: `tasks` must be of List type")

        if llm_backend not in ['gptqmodel', 'vllm']:
            raise ValueError('Eval framework support llm_backend: [gptqmodel, vllm]')

        if isinstance(model_or_id_or_path, str):
            log.info(f"Eval: loading using backend = `{backend}`")
            model = GPTQModel.load(model_id_or_path=model_or_id_or_path, backend=backend)
            model_id_or_path = model_or_id_or_path
        elif isinstance(model_or_id_or_path, BaseGPTQModel) or isinstance(model_or_id_or_path, (PreTrainedModel, PeftModel)):
            model = model_or_id_or_path
            model_id_or_path = model.config.name_or_path
        else:
            raise ValueError(f"`model_or_id_or_path` is invalid. expected: `model instance or str` actual: `{model_or_id_or_path}`")

        if tokenizer is None:
            if isinstance(model, BaseGPTQModel):
                tokenizer = model.tokenizer
            elif isinstance(model, PreTrainedModel) or model_id_or_path.strip():
                tokenizer = Tokenicer.load(model_id_or_path)

        if tokenizer is None:
            raise ValueError("Tokenizer: Auto-loading of tokenizer failed with `model_or_id_or_path`. Please pass in `tokenizer` as argument.")


        if backend=="gptqmodel":
            model_args["tokenizer"] = tokenizer

        if framework == EVAL.LM_EVAL:
            from lm_eval.utils import make_table

            for task in tasks:
                if task not in EVAL.get_task_enums():
                    raise ValueError(f"Eval.lm_eval supported `tasks`: `{EVAL.get_all_tasks_string()}`, actual = `{task}`")

            model_name = "hf" if llm_backend == "gptqmodel" else llm_backend

            if llm_backend == "gptqmodel":
                model_args["gptqmodel"] = True
            model_args["pretrained"] = model_id_or_path

            try:
                from lm_eval import simple_evaluate
                from lm_eval.models.huggingface import HFLM
            except BaseException:
                raise ValueError("lm_eval is not installed. Please install via `pip install gptqmodel[eval]`.")

            if llm_backend == "gptqmodel" and model is not None:
                model_name = HFLM(
                    pretrained=model,
                    batch_size=batch_size,
                    trust_remote_code=trust_remote_code,
                )

            gen_kwargs = args.pop("gen_kwargs", None)

            if gen_kwargs is None:
                if hasattr(model, "generation_config") and isinstance(model.generation_config, GenerationConfig):
                    gen_dict = {
                        "do_sample": model.generation_config.do_sample,
                        "temperature": model.generation_config.temperature,
                        "top_k": model.generation_config.top_k,
                        "top_p": model.generation_config.top_p,
                        "min_p": model.generation_config.min_p,

                    }
                    gen_kwargs = ','.join(f"{key}={value}" for key, value in gen_dict.items() if value not in ["", {}, None, []])
                else:
                    gen_kwargs = "temperature=0.0,top_k=50"

            log.info(f"LM-EVAL: `gen_kwargs` = `{gen_kwargs}`")

            apply_chat_template = args.pop("apply_chat_template", False)
            log.info(f"LM-EVAL: `apply_chat_template` = `{apply_chat_template}`")

            results = simple_evaluate(
                model=model_name,
                model_args=model_args,
                tasks=[task.value for task in tasks],
                batch_size=batch_size,
                apply_chat_template=apply_chat_template,
                gen_kwargs=gen_kwargs,
                random_seed=random_seed,
                numpy_random_seed=random_seed,
                torch_random_seed=random_seed,
                fewshot_random_seed=random_seed,
                **args,
            )

            if results is None:
                raise ValueError('lm_eval run fail, check your code!!!')

            print('--------lm_eval Eval Result---------')
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))
            print('--------lm_eval Result End---------')
            return results
        elif framework == EVAL.EVALPLUS:
            for task in tasks:
                if task not in EVAL.get_task_enums():
                    raise ValueError(f"evalplus support tasks: {EVAL.get_all_tasks_string()}")
            from ..utils.eval import evalplus, evalplus_make_table

            results = {}
            for task in tasks:
                base_formatted, plus_formatted, result_path = evalplus(
                    model=model_id_or_path,
                    dataset=task.value,
                    batch=batch_size,
                    trust_remote_code=trust_remote_code,
                    output_file=output_path,
                    backend=llm_backend
                )
                results[task.value] = {"base tests": base_formatted, "base + extra tests": plus_formatted,
                                       "results_path": result_path}
            print('--------evalplus Eval Result---------')
            evalplus_make_table(results)
            print('--------evalplus Result End---------')
            return results
        elif framework == EVAL.MMLU_PRO:
            for task in tasks:
                if task not in EVAL.get_task_enums():
                    raise ValueError(f"eval support tasks: {EVAL.get_all_tasks_string()}")
            from ..utils.mmlupro import mmlupro
            selected_subjects = ",".join(tasks)
            results = mmlupro(model,
                              tokenizer,
                              save_dir=output_path,
                              seed=random_seed,
                              selected_subjects=selected_subjects,
                              ntrain=ntrain,
                              batch_size=batch_size)

            print('--------MMLUPro Eval Result---------')
            print(results)
            print('--------MMLUPro Result End---------')
            return results
        else:
            raise ValueError("Eval framework support: EVAL.LM_EVAL, EVAL.EVALPLUS, EVAL.MMLUPRO")

    @staticmethod
    def export(model_id_or_path: str, target_path: str, format: str, trust_remote_code: bool = False):
        config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)

        if not config.quantization_config:
            raise ValueError("Model is not quantized")

        gptq_config = config.quantization_config

        gptq_model = GPTQModel.load(model_id_or_path, backend=BACKEND.TORCH)

        if format == "mlx":
            raise ValueError("MLX export is not supported.")
        elif format == "hf":
            from ..nn_modules.qlinear.torch import dequantize_model

            dequantized_model = dequantize_model(gptq_model.model)
            dequantized_model.save_pretrained(target_path)

        gptq_model.tokenizer.save_pretrained(target_path)

    @staticmethod
    def push_to_hub(repo_id: str,
                    quantized_path: str,
                    private: bool = False,
                    exists_ok: bool = False,
                    token: Optional[str] = None,
                    ):

        if not quantized_path:
            raise RuntimeError("You must pass quantized model path as str to push_to_hub.")

        if not repo_id:
            raise RuntimeError("You must pass repo_id as str to push_to_hub.")

        from huggingface_hub import HfApi
        repo_type = "model"

        api = HfApi()
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
        except Exception:
            api.create_repo(repo_id=repo_id, repo_type=repo_type, token=token, private=private, exist_ok=exists_ok)

        api.upload_large_folder(
            folder_path=quantized_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )