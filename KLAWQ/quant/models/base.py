# auto_gptq/modeling/base.py

from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch._dynamo
import torch.nn as nn
from packaging.version import Version
from tokenicer import Tokenicer
# MODIFIED: Added check for PreTrainedModel to handle different model types
from transformers import (AutoModelForCausalLM, AutoProcessor, PreTrainedModel,
                          PreTrainedTokenizerBase, ProcessorMixin, modeling_utils)

from ..nn_modules.hooked_linear import replace_module_with_hooked_legacy, replace_module_with_hooked_tree
from ..nn_modules.qlinear import BaseQuantLinear
from ..quantization import GPTQ, QuantizeConfig
from ..quantization.config import FORMAT, QUANT_METHOD, QUANTIZE_BLACK_LIST
from ..utils.backend import BACKEND
from ..utils.data import collate_data
from ..utils.device import get_cpu_usage_memory, get_gpu_usage_memory
from ..utils.hf import autofix_hf_model_config
from ..utils.importer import select_quant_linear
from ..utils.logger import setup_logger
from ..utils.model import (MODALITY, find_modules, get_device, get_module,
                           get_module_by_name_prefix, get_moe_layer_modules, move_to, nested_move_to, pack_model)
from ..utils.torch import torch_compile, torch_empty_cache
from ._const import CALIBRATION_DATASET_CONCAT_CHAR, CPU, DEFAULT_MAX_SHARD_SIZE, DEVICE, SUPPORTS_MODULE_TYPES
from .loader import ModelLoader
from .writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, PROCESS_LOG_TIME,
                     QUANT_LOG_DAMP, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES, ModelWriter)

TORCH_MIN_VERSION_STR = '2.6.0'
PYTORCH_MIN_VERSION_WITH_COMPILE = Version(TORCH_MIN_VERSION_STR)

def check_support_param_buffer_assignment(*args, **kwargs):
    return False


modeling_utils.check_support_param_buffer_assignment = check_support_param_buffer_assignment

log = setup_logger()

class BaseGPTQModel(nn.Module):
    base_modules: List[str] = None

    lm_head: str = "lm_head"

    layers_node: str = None
    layer_type: Union[List[str], str] = None
    layer_modules: List[List[str]] = None
    layers_modules_tree: List[str] = None

    layer_modules_strict = True

    pre_lm_head_norm_module: str = None

    require_trust_remote_code = None
    require_pkgs_version: Optional[List[str]] = None
    require_dtype: Optional[str|torch.dtype] = None
    require_fast_init: bool = True

    require_load_processor = False

    dynamic_expert_index: Optional[str] = None

    loader = AutoModelForCausalLM

    require_monkeypatch = False

    support_batch_quantize = True

    info: Dict[str, str] = {}

    supports_desc_act = [True, False]

    modality: List[MODALITY] = [MODALITY.TEXT]

    quant_override_files: Dict[str, Union[str | Dict[str, Any]]] = {}

    server = None

    def __init__(
        self,
        # MODIFIED: Changed type hint to be more generic
        model: nn.Module,
        quantized: bool,
        quantize_config: QuantizeConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        qlinear_kernel: nn.Module = None,
        load_quantized_model: bool = False,
        trust_remote_code: bool = False,
        model_local_path: str = None,
    ):
        super().__init__()

        self.model = model

        self.compiled = False
        self.quantized = quantized
        self.qlinear_kernel = qlinear_kernel
        self.load_quantized_model = load_quantized_model
        
        # MODIFIED: Added a check to handle models without tokenizers (e.g., vision models)
        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                self.tokenizer = Tokenicer.load(tokenizer, trust_remote_code=trust_remote_code)
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")
            # MODIFIED: Ensure model is a PreTrainedModel before assigning tokenizer to it
            if isinstance(self.model, PreTrainedModel):
                self.model.tokenizer = self.tokenizer.tokenizer
        else:
            self.tokenizer = tokenizer
            if isinstance(self.model, PreTrainedModel):
                self.model.tokenizer = tokenizer

        if isinstance(self.model, PreTrainedModel):
            autofix_hf_model_config(self.model, path=model_local_path)

        self.quantize_config = quantize_config

        self.trust_remote_code = trust_remote_code
        self.model_local_path = model_local_path
        self.quant_log = []

        self.processor: ProcessorMixin = None
        if self.require_load_processor:
            self.processor = AutoProcessor.from_pretrained(model_local_path)

        if self.require_monkeypatch:
            self.monkey_patch()

        log.info(f"Kernel: loaded -> `[{', '.join(cls.__name__ for cls in self.kernels())}]`")

    def prepare_dataset(
        self,
        calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        calibration_dataset_concat_size: Optional[int] = None,
        batch_size: int = 1,
    ):
        if isinstance(calibration_dataset[0], (str, list)) or (isinstance(calibration_dataset[0], list) and all(isinstance(x, int) for x in calibration_dataset[0])):
            if self.tokenizer is None:
                raise ValueError(f"tokenizer must be provided when calibration_dataset is List[str] or List[int], type: {type(calibration_dataset[0])}")

            new_calibration_dataset = []
            for data in calibration_dataset:
                if isinstance(data, list) and all(isinstance(x, int) for x in data):
                    input_ids = torch.tensor([data], dtype=torch.long)
                    attention_mask = torch.ones_like(input_ids)
                    new_calibration_dataset.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask
                    })
                else:
                    tokenized = self.tokenizer(data, return_tensors="pt")
                    new_calibration_dataset.append({
                        "input_ids": tokenized["input_ids"],
                        "attention_mask": tokenized["attention_mask"]
                    })
            calibration_dataset = new_calibration_dataset

        def _convert_tensor_to_list(tensor):
            if isinstance(tensor, torch.Tensor):
                if len(tensor.shape) == 1:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.long()
                return tensor.cpu().numpy().tolist()
            return [tensor]

        new_calibration_dataset = []
        for example in calibration_dataset:
            input_ids = _convert_tensor_to_list(example["input_ids"])
            attention_mask = _convert_tensor_to_list(example["attention_mask"])

            new_calibration_dataset.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )

        if calibration_dataset_concat_size:
            concatenated_data = []
            input_ids_buff = []
            attention_mask_buff = []
            current_length = 0

            new_line = self.tokenizer(CALIBRATION_DATASET_CONCAT_CHAR, return_tensors="pt")
            new_line_input_ids = _convert_tensor_to_list(new_line["input_ids"])[0]
            new_line_attention_mask = _convert_tensor_to_list(new_line["attention_mask"])[0]
            new_line_input_ids_len = len(new_line_input_ids)

            for example in new_calibration_dataset:
                input_ids = example["input_ids"][0]
                attention_mask = example["attention_mask"][0]

                if current_length + len(input_ids) + new_line_input_ids_len >= calibration_dataset_concat_size:
                    if len(input_ids_buff) > 0:
                        remaining_space = calibration_dataset_concat_size - current_length
                        if remaining_space > 0:
                            input_ids_buff.extend(new_line_input_ids)
                            input_ids_buff.extend(input_ids[:remaining_space - new_line_input_ids_len])
                            attention_mask_buff.extend(new_line_attention_mask)
                            attention_mask_buff.extend(attention_mask[:remaining_space - new_line_input_ids_len])

                            concatenated_data.append({
                                "input_ids": [input_ids_buff],
                                "attention_mask": [attention_mask_buff]
                            })
                        else:
                            concatenated_data.append({
                                "input_ids": [input_ids_buff],
                                "attention_mask": [attention_mask_buff]
                            })

                        input_ids_buff = input_ids[:calibration_dataset_concat_size]
                        attention_mask_buff = attention_mask[:calibration_dataset_concat_size]
                        current_length = len(input_ids_buff)
                    else:
                        input_ids_buff = input_ids[:calibration_dataset_concat_size]
                        attention_mask_buff = attention_mask[:calibration_dataset_concat_size]
                        current_length = len(input_ids_buff)
                else:
                    if len(input_ids_buff) > 0:
                        input_ids_buff.extend(new_line_input_ids)
                        attention_mask_buff.extend(new_line_attention_mask)
                        current_length += new_line_input_ids_len

                    input_ids_buff.extend(input_ids)
                    attention_mask_buff.extend(attention_mask)
                    current_length += len(input_ids)


            if input_ids_buff:
                padding_length = calibration_dataset_concat_size - len(input_ids_buff)
                if padding_length > 0:
                    input_ids_buff.extend([self.tokenizer.pad_token_id] * padding_length)
                    attention_mask_buff.extend([0] * padding_length)
                concatenated_data.append({
                    "input_ids": [input_ids_buff],
                    "attention_mask": [attention_mask_buff]
                })

            new_calibration_dataset = concatenated_data

        if self.support_batch_quantize:
            new_calibration_dataset_batched = [
                collate_data(new_calibration_dataset[start: start + batch_size], self.tokenizer.pad_token_id)
                for start in range(0, len(new_calibration_dataset), batch_size)
            ]
        else:
            new_calibration_dataset_batched = [
                {"input_ids": torch.tensor(block["input_ids"], dtype=torch.long)}
                for block in new_calibration_dataset
            ]

        return new_calibration_dataset_batched

    def quantize(
        self,
        calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        calibration_dataset_concat_size: Optional[int] = None,
        batch_size: int = 1,
        calibration_enable_gpu_cache: bool = True,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        logger_board: Optional[str] = None,
        buffered_fwd: bool = False,
        auto_gc: bool = True,
    ) -> Dict[str, List[Dict[str, str]]]:
        if self.quantized:
            raise EnvironmentError("quantize() is called a model that is already quantized")

        if self.quantize_config.quant_method in QUANTIZE_BLACK_LIST:
            raise ValueError(
                f"Unsupported quantization operation for quant method: {self.quantize_config.quant_method}"
            )

        if not self.support_batch_quantize:
            log.warn("Quantize: batch_size overriden by model class definition to `disabled`")
            batch_size = 1

        _ = select_quant_linear(
            bits=self.quantize_config.bits,
            dynamic=self.quantize_config.dynamic,
            group_size=self.quantize_config.group_size,
            desc_act=self.quantize_config.desc_act,
            sym=self.quantize_config.sym,
            backend=BACKEND.TORCH,
            device=DEVICE(self.quantize_config.device),
            pack=True,
            format=self.quantize_config.format,
            pack_dtype=self.quantize_config.pack_dtype,
        )

        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                self.tokenizer = Tokenicer.load(tokenizer, trust_remote_code=self.trust_remote_code)
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")

        from ..looper.gptq_processor import GPTQProcessor
        from ..looper.module_looper import ModuleLooper

        args = {
            "tokenizer": self.tokenizer,
            "qcfg": self.quantize_config,
            "calibration_dataset": calibration_dataset,
            "prepare_dataset_func": self.prepare_dataset,
            "calibration_dataset_concat_size": calibration_dataset_concat_size,
            "batch_size": batch_size,
            "logger_board": logger_board,
        }

        processors = [GPTQProcessor(**args)]

        module_looper = ModuleLooper(self, processors=processors)

        return module_looper.loop(
            calibration_enable_gpu_cache=calibration_enable_gpu_cache,
            buffered_fwd=buffered_fwd,
            auto_gc=auto_gc,
            backend=BACKEND.TORCH,
        )

    def to(self, device: Union[str, torch.device]):
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
            return self
        else:
            raise f"{self.model.__class__.__name__} does not support the to() method"

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, inputs=None, **kwargs):
        with torch.inference_mode():
            # MODIFIED: Check if model supports the 'generate' method.
            if not hasattr(self.model, 'generate'):
                log.error(f"The loaded model ({self.model.__class__.__name__}) does not support the `.generate()` method. Please use a direct forward pass, e.g., `model(inputs)`.")
                return

            pad_token_id = kwargs.get("pad_token_id", None)
            if pad_token_id is None and self.tokenizer:
                kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            if isinstance(inputs, str) or (isinstance(inputs, list) and all(isinstance(x, str) for x in inputs)):
                if self.tokenizer is None:
                    raise ValueError("You passed in an `input` to `generate()` of type `str` but model is missing a tokenizer.")
                inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, padding_side="left").to(self.model.device)
                return self.model.generate(**inputs, **kwargs)

            return self.model.generate(inputs=inputs, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def push_to_hub(self,
                    repo_id: str,
                    quantized_path: str,
                    private: bool = False,
                    exists_ok: bool = False,
                    token: Optional[str] = None):

        log.error("`push_to_hub()` api cannot be used on the model instance. Please use `GPTQModel.push_to_hub()` static api instead.")

    def save(
            self,
            save_dir: str,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
            meta_quantizer: Optional[str] = None,
            **kwargs,
    ):
        if self.quantized:
            self.save_quantized(
                save_dir=save_dir,
                safetensors_metadata=safetensors_metadata,
                max_shard_size=max_shard_size,
                meta_quantizer=meta_quantizer)

            for name, value in self.quant_override_files.items():
                json_path = os.path.join(save_dir, name)
                with open(json_path, "w", encoding="utf-8") as f:
                    if isinstance(value, str):
                        f.write(value)
                    else:
                        f.write(json.dumps(value))
        else:
            # MODIFIED: Handle saving for both HF models and generic nn.Module models.
            if isinstance(self.model, PreTrainedModel):
                self.save_pretrained(save_dir=save_dir, **kwargs)
            else:
                # ADDED: Logic to save generic torch models.
                log.info(f"Model is not a PreTrainedModel. Saving state_dict to `{os.path.join(save_dir, 'model.pth')}`.")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pth"))


    def kernels(self) -> List[Type[BaseQuantLinear]]:
        if not isinstance(self.model, nn.Module):
            return []
        loaded_kernels = set()
        modules = find_modules(self.model, layers=[BaseQuantLinear])
        for k, v in modules.items():
            loaded_kernels.add(v.__class__)

        return list(loaded_kernels)

    def compile(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        log.warn("Deprecation: `model.compile()` is deprecated. Please use `model.optimize()` instead.")
        return self.optimize(backend=backend, mode=mode, fullgraph=fullgraph)

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        if not self.quantized:
            log.warn("model is not quantized, skip compiling...")
            return self

        if Version(torch.__version__) < PYTORCH_MIN_VERSION_WITH_COMPILE:
            self.compiled = False
            log.warn(f"To use compile(), you need to have torch version >= {TORCH_MIN_VERSION_STR}, please "
                           f"upgrade it by `pip install -U torch torchaudio torchvision`")
            return self

        log.info(f"Compiling qlinear modules with backend: `{backend}`, mode: `{mode}`")
        modules = find_modules(self.model, layers=[BaseQuantLinear])
        for name in modules.keys():
            modules[name].optimize(fullgraph=False, backend=backend, mode=mode)

        log.info(f"Compiling model with backend: `{backend}`, mode: `{mode}`")

        self.model = torch_compile(self.model, fullgraph=fullgraph, backend=backend, mode=mode)

        return self

    def pre_quantize_generate_hook_start(self):
        pass

    def pre_quantize_generate_hook_end(self):
        pass

    def lm_head_pre_quantize_generate_hook(self, inputs: List[List[torch.tensor]]) -> List[List[torch.tensor]]:
        if self.pre_lm_head_norm_module:
            norm, _ = get_module_by_name_prefix(self.model, [self.pre_lm_head_norm_module])
            self.pre_quantize(norm)

            for element in inputs:
                for i in range(len(element)):
                    element[i] = norm(element[i])

            self.post_quantize(norm)
        return inputs

    def pre_quantize(self, module: nn.Module) -> nn.Module:
        if get_device(module) == CPU and self.quantize_config.device != CPU:
            return move_to(module, device=self.quantize_config.device)
        return module

    def post_quantize(self, module: nn.Module) -> nn.Module:
        return move_to(module, device=CPU)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except Exception:
            return getattr(self.model, item)

__all__ = ["BaseGPTQModel"]

BaseGPTQModel = ModelLoader(ModelWriter(BaseGPTQModel))