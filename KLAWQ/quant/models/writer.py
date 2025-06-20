import copy
import csv
import json
import os
import re
from os.path import isfile, join
from typing import Dict, Optional, Union

import torch
import transformers
from huggingface_hub import split_torch_state_dict_into_shards
from huggingface_hub.constants import SAFETENSORS_WEIGHTS_FILE_PATTERN
from safetensors.torch import save_file
from safetensors.torch import save_file as safe_save
from transformers import AutoConfig, PreTrainedTokenizerFast, ProcessorMixin
from transformers.modeling_utils import no_init_weights
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils.generic import ContextManagers

from ..quantization.config import (FORMAT, META_FIELD_DAMP_AUTO_INCREMENT, META_FIELD_DAMP_PERCENT, META_FIELD_MSE,
                                   META_FIELD_QUANTIZER, META_FIELD_STATIC_GROUPS, META_FIELD_TRUE_SEQUENTIAL,
                                   META_FIELD_URI, META_QUANTIZER_GPTQMODEL, META_VALUE_URI)
from ..utils.backend import BACKEND
from ..utils.logger import setup_logger
from ..utils.model import (copy_py_files, find_modules,
                           get_model_files_size, get_moe_layer_modules, get_state_dict_for_save,
                           load_checkpoint_in_model_then_tie_weights, make_quant)
from ..utils.torch import torch_empty_cache
from ..version import __version__
from ._const import CPU, DEFAULT_MAX_SHARD_SIZE

log = setup_logger()

PROCESS_LOG_NAME = "process"
PROCESS_LOG_LAYER = "layer"
PROCESS_LOG_MODULE = "module"
QUANT_LOG_LOSS = "loss"
QUANT_LOG_NSAMPLES = "samples"
QUANT_LOG_DAMP = "damp"
PROCESS_LOG_TIME = "time"
PROCESS_LOG_FWD_TIME = "fwd_time"
PROCESS_MAX_MEMORY = "max_vram"


def ModelWriter(cls):
    def save_pretrained(
            self,
            save_dir: str,
            **kwargs,
    ):
        log.warn("You are using save_pretrained, which will re-direct to save_quantized.")
        self.save_quantized(save_dir=save_dir, **kwargs)

    cls.save_pretrained = save_pretrained

    def save_quantized(
            self,
            save_dir: str,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
            meta_quantizer: Optional[str] = None,
    ):
        os.makedirs(save_dir, exist_ok=True)

        if self.quant_log:
            with open(os.path.join(save_dir, "quant_log.csv"), mode='w', newline='') as file:
                w = csv.writer(file)
                w.writerow([PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES, QUANT_LOG_DAMP, PROCESS_LOG_TIME])
                w.writerows([[entry.get(PROCESS_LOG_LAYER), entry.get(PROCESS_LOG_MODULE), entry.get(QUANT_LOG_LOSS),
                              entry.get(QUANT_LOG_DAMP), entry.get(PROCESS_LOG_TIME)] for entry in self.quant_log])

        pre_quantized_size_mb = get_model_files_size(self.model_local_path)
        # This line is safe because 0.0 / 1024 is 0.0
        pre_quantized_size_gb = pre_quantized_size_mb / 1024

        quantizers = [f"{META_QUANTIZER_GPTQMODEL}:{__version__}"]
        if meta_quantizer:
            if len(meta_quantizer.split(":")) == 2:
                quantizers.append(meta_quantizer.replace(" ",""))
            else:
                log.warn(f"meta_quantizer: '{meta_quantizer}' format is invalid, expected: 'quantizer_name:version'")

        self.quantize_config.meta_set_versionable(
            key=META_FIELD_QUANTIZER,
            value=quantizers
        )

        self.quantize_config.meta_set(
            key=META_FIELD_URI,
            value=META_VALUE_URI,
        )

        self.quantize_config.meta_set(
            key=META_FIELD_DAMP_PERCENT,
            value=self.quantize_config.damp_percent
        )

        self.quantize_config.meta_set(
            key=META_FIELD_DAMP_AUTO_INCREMENT,
            value=self.quantize_config.damp_auto_increment
        )

        self.quantize_config.meta_set(
            key=META_FIELD_STATIC_GROUPS,
            value=self.quantize_config.static_groups
        )

        self.quantize_config.meta_set(
            key=META_FIELD_TRUE_SEQUENTIAL,
            value=self.quantize_config.true_sequential
        )

        self.quantize_config.meta_set(
            key=META_FIELD_MSE,
            value=self.quantize_config.mse
        )
        
        is_hf_model = getattr(self, "is_hf_model", True)
        
        quantize_config = copy.deepcopy(self.quantize_config)

        if not self.quantized:
            raise ValueError("Save aborted as model is not quantized. Please call `quantize()` first.")

        if not self.load_quantized_model:
            model = self.model
        else:
            model = self.get_model_with_quantize(
                qcfg=quantize_config,
                model_id_or_path=self.model_local_path,
            )

        if is_hf_model:
            config = copy.deepcopy(self.model.config)
            config.quantization_config = quantize_config.to_dict()
            self.model.config = config
            self.model.save_pretrained(save_dir, state_dict={}, is_main_process=True)
        else:
            log.info("Skipping model.config and model.save_pretrained for non-HuggingFace model.")
        
        quantize_config.save_pretrained(save_dir)

        if hasattr(self,"processor") and isinstance(self.processor, ProcessorMixin):
            self.processor.save_pretrained(save_dir)

        model.to(CPU)
        state_dict = get_state_dict_for_save(model)

        model_base_name = "model"

        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        model_save_name = model_base_name + ".safetensors"

        if hasattr(self, 'qlinear_kernel') and self.qlinear_kernel and not self.qlinear_kernel.SUPPORTS_SHARDS and max_shard_size is not None:
            log.warn("Sharding is not supported for this quant. Disabling sharding.")
            max_shard_size = None

        if max_shard_size is None:
            if safetensors_metadata is None:
                safetensors_metadata = {}
            elif not isinstance(safetensors_metadata, dict):
                raise TypeError("safetensors_metadata must be a dictionary.")
            else:
                log.debug(f"Received safetensors_metadata: {safetensors_metadata}")
                new_safetensors_metadata = {}
                converted_keys = False
                for key, value in safetensors_metadata.items():
                    if not isinstance(key, str) or not isinstance(value, str):
                        converted_keys = True
                        try:
                            new_key = str(key)
                            new_value = str(value)
                        except Exception as e:
                            raise TypeError(
                                f"safetensors_metadata: both keys and values must be strings and an error occured when trying to convert them: {e}"
                            )
                        if new_key in new_safetensors_metadata:
                            log.warn(
                                f"After converting safetensors_metadata keys to strings, the key '{new_key}' is duplicated. Ensure that all your metadata keys are strings to avoid overwriting."
                            )
                        new_safetensors_metadata[new_key] = new_value
                safetensors_metadata = new_safetensors_metadata
                if converted_keys:
                    log.debug(
                        f"One or more safetensors_metadata keys or values had to be converted to str(). Final safetensors_metadata: {safetensors_metadata}"
                    )

            safetensors_metadata["format"] = "pt"
            safe_save(state_dict, join(save_dir, model_save_name), safetensors_metadata)
            total_size_mb = os.path.getsize(join(save_dir, model_save_name)) / (1024 * 1024)
        else:
            file_name_pattern = SAFETENSORS_WEIGHTS_FILE_PATTERN

            state_dict_split= split_torch_state_dict_into_shards(state_dict, max_shard_size=max_shard_size, filename_pattern=file_name_pattern)

            for filename in os.listdir(save_dir):
                full_filename = join(save_dir, filename)

                filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
                reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

                if (
                        filename.startswith(model_base_name)
                        and isfile(full_filename)
                        and filename not in state_dict_split.filename_to_tensors.keys()
                        and reg.fullmatch(filename_no_suffix) is not None
                ):
                    os.remove(full_filename)

            total_size_mb = 0
            for filename, tensors in state_dict_split.filename_to_tensors.items():
                shard = {tensor: state_dict[tensor] for tensor in tensors}
                if safetensors_metadata is None:
                    safetensors_metadata = {}
                elif not isinstance(safetensors_metadata, dict):
                    raise TypeError("safetensors_metadata must be a dictionary.")
                else:
                    log.debug(f"Received safetensors_metadata: {safetensors_metadata}")
                    new_safetensors_metadata = {}
                    converted_keys = False
                    for key, value in safetensors_metadata.items():
                        if not isinstance(key, str) or not isinstance(value, str):
                            converted_keys = True
                            try:
                                new_key = str(key)
                                new_value = str(value)
                            except Exception as e:
                                raise TypeError(
                                    f"safetensors_metadata: both keys and values must be strings and an error occured when trying to convert them: {e}")
                            if new_key in new_safetensors_metadata:
                                log.warn(
                                    f"After converting safetensors_metadata keys to strings, the key '{new_key}' is duplicated. Ensure that all your metadata keys are strings to avoid overwriting.")
                            new_safetensors_metadata[new_key] = new_value
                    safetensors_metadata = new_safetensors_metadata
                    if converted_keys:
                        log.debug(
                            f"One or more safetensors_metadata keys or values had to be converted to str(). Final safetensors_metadata: {safetensors_metadata}")

                safetensors_metadata["format"] = "pt"

                safe_save(shard, join(save_dir, filename), safetensors_metadata)
                shard_size_mb = os.path.getsize(join(save_dir, filename)) / (1024 * 1024)
                total_size_mb += shard_size_mb

            if state_dict_split.is_sharded:
                index = {
                    "metadata": state_dict_split.metadata,
                    "weight_map": state_dict_split.tensor_to_filename,
                }

                index_save_name = model_save_name + ".index.json"
                index_save_path = join(save_dir, index_save_name)
                with open(index_save_path, "w", encoding="utf-8") as f:
                    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                    f.write(content)

        if not self.load_quantized_model:
            # --- START OF THE DEFINITIVE FIX ---
            # This block prevents the ZeroDivisionError by checking the original model size.
            if pre_quantized_size_mb > 0:
                total_size_gb = total_size_mb / 1024
                size_diff_mb = pre_quantized_size_mb - total_size_mb
                size_diff_gb = size_diff_mb / 1024
                percent_diff = (size_diff_mb / pre_quantized_size_mb) * 100
                log.info(f"Pre-Quantized model size: {pre_quantized_size_mb:.2f}MB, {pre_quantized_size_gb:.2f}GB")
                log.info(f"Quantized model size: {total_size_mb:.2f}MB, {total_size_gb:.2f}GB")
                log.info(f"Size difference: {size_diff_mb:.2f}MB, {size_diff_gb:.2f}GB - {percent_diff:.2f}%")
            else:
                # If original size is 0, just print the new size without comparison.
                total_size_gb = total_size_mb / 1024
                log.info(f"Quantized model size: {total_size_mb:.2f}MB, {total_size_gb:.2f}GB")
                log.info("Could not determine original model size. Skipping size difference calculation.")
            # --- END OF THE DEFINITIVE FIX ---

        if self.trust_remote_code and is_hf_model: 
            copy_py_files(save_dir, model_id_or_path=self.model_local_path)

        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)

            saved_tokenizer_config = get_tokenizer_config(save_dir)
            if "tokenizer_class" in saved_tokenizer_config:
                config_tokenizer_class = saved_tokenizer_config.get("tokenizer_class")
                if (not config_tokenizer_class.endswith("Fast")) and (
                    isinstance(self.tokenizer.tokenizer, PreTrainedTokenizerFast)
                    ):
                    saved_tokenizer_config["tokenizer_class"] = saved_tokenizer_config["tokenizer_class"] + "Fast"
                    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
                        json.dump(saved_tokenizer_config, f, indent=2, ensure_ascii=False)


    cls.save_quantized = save_quantized

    def get_model_with_quantize(self, qcfg, model_id_or_path):
        is_hf_model = getattr(self, "is_hf_model", True)

        if not is_hf_model:
            raise NotImplementedError(
                "The `get_model_with_quantize` method (used when saving a model loaded from a quantized state) "
                "is not supported for non-HuggingFace models."
            )
        
        config = AutoConfig.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
        )

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        transformers.modeling_utils._init_weights = False
        init_contexts = [no_init_weights()]
        with ContextManagers(init_contexts):
            model = cls.loader.from_config(
                config, torch_dtype=torch.float16
            )

            if hasattr(self, 'dynamic_expert_index') and self.dynamic_expert_index is not None:
                num_experts = getattr(config, self.dynamic_expert_index)
                _ = get_moe_layer_modules(layer_modules=self.layer_modules,
                                                      num_experts=num_experts)

            modules = find_modules(model)
            ignore_modules = [self.lm_head] + self.base_modules

            for name in list(modules.keys()):
                if qcfg.lm_head and name == self.lm_head:
                    continue

                if any(name.startswith(ignore_module) for ignore_module in ignore_modules) or all(
                        not name.endswith(ignore_module) for sublist in self.layer_modules for ignore_module in sublist
                ):
                    if name is not self.lm_head:
                        log.info(f"The layer {name} is not quantized.")
                    del modules[name]

            make_quant(
                model,
                quant_result=modules,
                qcfg=qcfg,
                backend=BACKEND.AUTO,
                lm_head_name=cls.lm_head,
                pack=True,
            )

        load_checkpoint_in_model_then_tie_weights(
            model,
            dtype=torch.float16,
            checkpoint=self.checkpoint_file_name,
        )
        torch_empty_cache()
        return model
        # --- END OF ORIGINAL HUGGING FACE LOGIC ---

    cls.get_model_with_quantize = get_model_with_quantize

    return cls