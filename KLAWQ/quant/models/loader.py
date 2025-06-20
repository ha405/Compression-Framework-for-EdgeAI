from __future__ import annotations

import importlib.util
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, List, Optional, Union

import torch
import transformers

if os.getenv('GPTQMODEL_USE_MODELSCOPE', 'False').lower() in ['true', '1']:
    try:
        from modelscope import snapshot_download
    except Exception:
        raise ModuleNotFoundError("env `GPTQMODEL_USE_MODELSCOPE` used but modelscope pkg is not found: please install with `pip install modelscope`.")
else:
    from huggingface_hub import snapshot_download

from packaging.version import InvalidVersion, Version
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from transformers.modeling_utils import no_init_weights
from transformers.utils import is_flash_attn_2_available
from transformers.utils.generic import ContextManagers

from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..quantization import QuantizeConfig
from ..quantization.config import FORMAT
from ..utils.backend import BACKEND
from ..utils.importer import auto_select_device, normalize_device_device_map, select_quant_linear
from ..utils.logger import setup_logger
from ..utils.model import (auto_dtype, find_modules, get_checkpoints,
                           get_moe_layer_modules, gptqmodel_post_init, load_checkpoint_in_model_then_tie_weights,
                           make_quant, simple_dispatch_model, verify_model_hash, verify_sharded_model_hashes)
from ._const import DEVICE, normalize_device

log = setup_logger()

ATTN_IMPLEMENTATION = "attn_implementation"
USE_FLASH_ATTENTION_2 = "use_flash_attention_2"
def parse_version_string(version_str: str):
    try:
        return Version(version_str)
    except InvalidVersion:
        raise ValueError(f"Invalid version format: {version_str}")


def parse_requirement(req):
    for op in [">=", "<=", ">", "<", "=="]:
        if op in req:
            pkg, version_required = req.split(op, 1)
            return pkg.strip(), op, version_required.strip()
    raise ValueError(f"Unsupported version constraint in: {req}")


def compare_versions(installed_version, required_version, operator):
    installed = parse_version_string(installed_version)
    required = parse_version_string(required_version)
    if operator == ">":
        return installed > required
    elif operator == ">=":
        return installed >= required
    elif operator == "<":
        return installed < required
    elif operator == "<=":
        return installed <= required
    elif operator == "==":
        return installed == required
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def check_versions(model_class, requirements: List[str]):
    if requirements is None:
        return
    for req in requirements:
        pkg, operator, version_required = parse_requirement(req)
        try:
            installed_version = version(pkg)
            if not compare_versions(installed_version, version_required, operator):
                raise ValueError(f"{model_class} requires version {req}, but current {pkg} version is {installed_version} ")
        except PackageNotFoundError:
            raise ValueError(f"{model_class} requires version {req}, but {pkg} not installed.")


def get_model_local_path(pretrained_model_id_or_path, **kwargs):
    # MODIFIED: Wrap in try-except for non-string paths (which can happen with generic models)
    try:
        is_local = os.path.isdir(pretrained_model_id_or_path)
    except TypeError:
        is_local = False
        
    if is_local:
        return pretrained_model_id_or_path
    else:
        # MODIFIED: Wrap in try-except to handle cases where the ID is not on the Hub (e.g., "resnet50")
        try:
            download_kwargs = kwargs.copy()
            download_kwargs.pop("max_memory", None)
            download_kwargs.pop("attn_implementation", None)
            download_kwargs.pop("use_flash_attention_2", None)
            return snapshot_download(pretrained_model_id_or_path, **download_kwargs)
        except Exception:
            log.info(f"Could not download '{pretrained_model_id_or_path}' from hub. Assuming it's a local model type name.")
            return pretrained_model_id_or_path


def ModelLoader(cls):
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_id_or_path: str,
            quantize_config: QuantizeConfig,
            trust_remote_code: bool = False,
            torch_dtype: str | torch.dtype = "auto",
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            device: Optional[Union[str, int]] = None,
            **model_init_kwargs,
    ):
        # --- Common setup ---
        if quantize_config is None or not isinstance(quantize_config, QuantizeConfig):
            raise AttributeError("`quantize_config` must be passed and be an instance of QuantizeConfig.")

        quantize_config.calculate_bits_per_weight()

        if cls.require_trust_remote_code and not trust_remote_code:
            raise ValueError(
                f"{pretrained_model_id_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        check_versions(cls, cls.require_pkgs_version)

        # ADDED: Check if the model class is a Hugging Face model. Default to True if not specified.
        is_hf_model = getattr(cls, "is_hf_model", True)
        
        if is_hf_model:
            # --- START OF ORIGINAL HUGGING FACE LOGIC (UNTOUCHED) ---
            cpu_device_map = {"": "cpu"}

            if quantize_config.device is not None:
                if device is not None or device_map is not None:
                    raise AttributeError("Passing device and device_map is not allowed when QuantizeConfig.device is set. Non-quantized model is always loaded as cpu. Please set QuantizeConfig.device for accelerator used in quantization or do not set for auto-selection.")

            if quantize_config.desc_act not in cls.supports_desc_act:
                raise ValueError(f"{cls} only supports desc_act={cls.supports_desc_act}, "
                                f"but quantize_config.desc_act is {quantize_config.desc_act}.")

            model_local_path = get_model_local_path(pretrained_model_id_or_path, **model_init_kwargs)

            if quantize_config.device is None:
                quantize_config.device = auto_select_device(None, None)
            else:
                quantize_config.device = normalize_device(quantize_config.device)

            def skip(*args, **kwargs):
                pass

            torch.nn.init.kaiming_uniform_ = skip
            torch.nn.init.uniform_ = skip
            torch.nn.init.normal_ = skip

            model_init_kwargs["trust_remote_code"] = trust_remote_code

            config = AutoConfig.from_pretrained(model_local_path, **model_init_kwargs)

            atten_impl = model_init_kwargs.get("attn_implementation", None)

            if atten_impl is not None and atten_impl != "auto":
                log.info(f"Loader: overriding attn_implementation in config to `{atten_impl}`")
                config._attn_implementation = atten_impl

            if quantize_config.device is None:
                quantize_config.device = auto_select_device(None, None)
            else:
                quantize_config.device = normalize_device(quantize_config.device)

            if cls.require_dtype:
                torch_dtype = cls.require_dtype

            if torch_dtype is None or torch_dtype == "auto" or not isinstance(torch_dtype, torch.dtype):
                torch_dtype = auto_dtype(config=config, device=quantize_config.device, quant_inference=False)

            model_init_kwargs["device_map"] = cpu_device_map
            model_init_kwargs["torch_dtype"] = torch_dtype
            model_init_kwargs["_fast_init"] = cls.require_fast_init

            model = cls.loader.from_pretrained(model_local_path, config=config, **model_init_kwargs)

            model_config = model.config.to_dict()
            seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions", "multimodal_max_length"]
            if any(k in model_config for k in seq_len_keys):
                for key in seq_len_keys:
                    if key in model_config:
                        model.seqlen = model_config[key]
                        break
            else:
                log.warn("Model: can't get model's sequence length from model config, will set to 4096.")
                model.seqlen = 4096
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id_or_path, trust_remote_code=trust_remote_code)
            # --- END OF ORIGINAL HUGGING FACE LOGIC ---
        
        else:
            # ADDED: Generic Model (e.g., torchvision) Loading Logic
            log.info(f"Loading non-HuggingFace model '{pretrained_model_id_or_path}' using the specified loader.")
            model_local_path = pretrained_model_id_or_path

            if quantize_config.device is None:
                quantize_config.device = auto_select_device(None, None)
            else:
                quantize_config.device = normalize_device(quantize_config.device)
            
            def skip(*args, **kwargs):
                pass
            torch.nn.init.kaiming_uniform_ = skip
            torch.nn.init.uniform_ = skip
            torch.nn.init.normal_ = skip
            
            # Use the loader directly, assuming it takes a `weights` argument for pre-trained versions.
            model = cls.loader(weights="DEFAULT")
            model.seqlen = None # No concept of seqlen for most vision models
            
            if torch_dtype == "auto" or not isinstance(torch_dtype, torch.dtype):
                torch_dtype = torch.float32 # A safe default for vision models
            model = model.to(dtype=torch_dtype)

            tokenizer = None # No tokenizer for vision models
            model.eval()

        return cls(
            model,
            quantized=False,
            quantize_config=quantize_config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            model_local_path=model_local_path,
        )

    cls.from_pretrained = from_pretrained

    @classmethod
    def from_quantized(
            cls,
            model_id_or_path: Optional[str],
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            device: Optional[Union[str, int]] = None,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            torch_dtype: str | torch.dtype = "auto",
            trust_remote_code: bool = False,
            verify_hash: Optional[Union[str, List[str]]] = None,
            **kwargs,
    ):
        # ADDED: Check if the model class is a Hugging Face model.
        is_hf_model = getattr(cls, "is_hf_model", True)
        
        if is_hf_model:
            # --- START OF ORIGINAL HUGGING FACE LOGIC (UNTOUCHED) ---
            device = normalize_device_device_map(device, device_map)

            if isinstance(backend, str):
                backend = BACKEND(backend)
            device = auto_select_device(device, backend)
            device_map = device.to_device_map()

            if cls.require_trust_remote_code and not trust_remote_code:
                raise ValueError(
                    f"{model_id_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
                )

            check_versions(cls, cls.require_pkgs_version)

            model_local_path = get_model_local_path(model_id_or_path, **kwargs)

            cache_dir = kwargs.pop("cache_dir", None)
            force_download = kwargs.pop("force_download", False)
            resume_download = kwargs.pop("resume_download", False)
            proxies = kwargs.pop("proxies", None)
            local_files_only = kwargs.pop("local_files_only", False)
            use_auth_token = kwargs.pop("use_auth_token", None)
            revision = kwargs.pop("revision", None)
            subfolder = kwargs.pop("subfolder", "")
            commit_hash = kwargs.pop("_commit_hash", None)

            cached_file_kwargs = {
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "resume_download": resume_download,
                "local_files_only": local_files_only,
                "use_auth_token": use_auth_token,
                "revision": revision,
                "subfolder": subfolder,
                "_raise_exceptions_for_missing_entries": False,
                "_commit_hash": commit_hash,
            }

            config: PretrainedConfig = AutoConfig.from_pretrained(
                model_local_path,
                trust_remote_code=trust_remote_code,
                **cached_file_kwargs,
            )

            if cls.require_dtype:
                torch_dtype = cls.require_dtype

            if torch_dtype is None or torch_dtype == "auto" or not isinstance(torch_dtype, torch.dtype) :
                torch_dtype = auto_dtype(config=config, device=device, quant_inference=True)

            qcfg = QuantizeConfig.from_pretrained(model_local_path, **cached_file_kwargs, **kwargs)

            qcfg.calculate_bits_per_weight()

            if qcfg.format != FORMAT.GPTQ:
                raise ValueError(f"Only FORMAT.GPTQ is supported for loading, actual = {qcfg.format}")

            possible_model_basenames = [
                f"gptq_model-{qcfg.bits}bit-{qcfg.group_size}g",
                "model",
            ]

            extensions = [".safetensors"]

            model_local_path = str(model_local_path)

            is_sharded, resolved_archive_file, true_model_basename = get_checkpoints(
                model_id_or_path=model_local_path,
                extensions=extensions,
                possible_model_basenames=possible_model_basenames,
                **cached_file_kwargs,
            )

            if ".bin" in resolved_archive_file:
                raise ValueError(
                    "Loading of .bin files are not allowed due to safety. Please convert your model to safetensor or pytorch format."
                )

            model_save_name = resolved_archive_file
            if verify_hash:
                if is_sharded:
                    verified = verify_sharded_model_hashes(model_save_name, verify_hash)
                else:
                    verified = verify_model_hash(model_save_name, verify_hash)
                if not verified:
                    raise ValueError(f"Hash verification failed for {model_save_name}")
                log.info(f"Hash verification succeeded for {model_save_name}")

            def skip(*args, **kwargs):
                pass

            torch.nn.init.kaiming_uniform_ = skip
            torch.nn.init.uniform_ = skip
            torch.nn.init.normal_ = skip

            transformers.modeling_utils._init_weights = False

            init_contexts = [no_init_weights()]

            with ContextManagers(init_contexts):
                if config.architectures:
                    model_class = getattr(transformers, config.architectures[0], None)
                    if model_class is not None and hasattr(model_class, "_supports_flash_attn_2"):
                        supports_flash_attn = model_class._supports_flash_attn_2
                    else:
                        supports_flash_attn = None
                else:
                    supports_flash_attn = None

                args = {}
                if supports_flash_attn and device in [DEVICE.CUDA, DEVICE.ROCM]:
                    if ATTN_IMPLEMENTATION in kwargs:
                        args[ATTN_IMPLEMENTATION] = kwargs.pop(ATTN_IMPLEMENTATION, None)
                    if USE_FLASH_ATTENTION_2 in kwargs:
                        args[USE_FLASH_ATTENTION_2] = kwargs.pop(USE_FLASH_ATTENTION_2, None)
                    if not args and importlib.util.find_spec("flash_attn") is not None:
                        has_attn_implementation = Version(transformers.__version__) >= Version("4.46.0")
                        if is_flash_attn_2_available() and has_attn_implementation:
                            args = {ATTN_IMPLEMENTATION: "flash_attention_2"}
                        elif is_flash_attn_2_available() and not has_attn_implementation:
                            args = {USE_FLASH_ATTENTION_2: True}

                        log.info("Optimize: Auto enabling flash attention2")

                model = cls.loader.from_config(
                    config, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype, **args
                )
                model.checkpoint_file_name = model_save_name

                if cls.dynamic_expert_index is not None:
                    num_experts = getattr(config, cls.dynamic_expert_index)
                    cls.layer_modules = get_moe_layer_modules(layer_modules=cls.layer_modules,
                                                            num_experts=num_experts)

                modules = find_modules(model)
                ignore_modules = [cls.lm_head] + cls.base_modules

                for name in list(modules.keys()):
                    if qcfg.lm_head and name == cls.lm_head:
                        continue
                    
                    # MODIFIED: layers_node can be None for generic models, add a check
                    if cls.layers_node and not any(name.startswith(prefix) for prefix in cls.layers_node):
                        del modules[name]
                        continue
                        
                    if any(name.startswith(ignore_module) for ignore_module in ignore_modules):
                        del modules[name]
                        continue

                    if all(not name.endswith(ignore_module) for sublist in cls.layer_modules for ignore_module in sublist):
                        if name is not cls.lm_head:
                            log.info(f"The layer {name} is not quantized.")
                        del modules[name]

                preload_qlinear_kernel = make_quant(
                    model,
                    quant_result=modules,
                    qcfg=qcfg,
                    backend=BACKEND.TORCH,
                    lm_head_name=cls.lm_head,
                    device=device,
                )
            
            tokenizer = AutoTokenizer.from_pretrained(model_local_path, trust_remote_code=trust_remote_code)
            # --- END OF ORIGINAL HUGGING FACE LOGIC ---
        else:
            # ADDED: Generic Model (e.g., torchvision) Loading Logic
            device = normalize_device_device_map(device, device_map)

            if isinstance(backend, str):
                backend = BACKEND(backend)
            device = auto_select_device(device, backend)
            device_map = device.to_device_map()

            model_local_path = get_model_local_path(model_id_or_path, **kwargs)
            qcfg = QuantizeConfig.from_pretrained(model_local_path, **kwargs)
            qcfg.calculate_bits_per_weight()

            is_sharded, resolved_archive_file, _ = get_checkpoints(
                model_id_or_path=model_local_path,
                extensions=[".safetensors"],
                possible_model_basenames=[f"gptq_model-{qcfg.bits}bit-{qcfg.group_size}g", "model"],
                **kwargs,
            )
            model_save_name = resolved_archive_file

            with no_init_weights():
                model = cls.loader(weights=None)
                if torch_dtype is None or torch_dtype == "auto" or not isinstance(torch_dtype, torch.dtype):
                    torch_dtype = torch.float32
                model = model.to(dtype=torch_dtype)

                modules = find_modules(model, [torch.nn.Linear, torch.nn.Conv2d])
                make_quant(model, modules, qcfg, device=device)
            
            tokenizer = None

        load_checkpoint_in_model_then_tie_weights(
            model,
            dtype=torch_dtype,
            checkpoint=model_save_name,
            device_map=device_map,
            offload_state_dict=True,
            offload_buffers=True,
        )

        model = simple_dispatch_model(model, device_map)

        qlinear_kernel = select_quant_linear(
            bits=qcfg.bits,
            dynamic=qcfg.dynamic,
            group_size=qcfg.group_size,
            desc_act=qcfg.desc_act,
            sym=qcfg.sym,
            backend=backend,
            format=qcfg.format,
            device=device,
            pack_dtype=qcfg.pack_dtype,
        )
        
        if is_hf_model:
            model_config = model.config.to_dict()
            seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions", "multimodal_max_length"]
            if any(k in model_config for k in seq_len_keys):
                for key in seq_len_keys:
                    if key in model_config:
                        model.seqlen = model_config[key]
                        break
            else:
                log.warn("can't get model's sequence length from model config, will set to 4096.")
                model.seqlen = 4096
        else:
            model.seqlen = None

        model = gptqmodel_post_init(model, use_act_order=qcfg.desc_act, quantize_config=qcfg)
        model.eval()

        return cls(
            model,
            quantized=True,
            quantize_config=qcfg,
            tokenizer=tokenizer,
            qlinear_kernel=qlinear_kernel,
            load_quantized_model=True,
            trust_remote_code=trust_remote_code,
            model_local_path=model_local_path,
        )

    cls.from_quantized = from_quantized

    return cls