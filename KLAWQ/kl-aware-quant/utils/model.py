from __future__ import annotations

import collections
import functools
import hashlib
import json
import operator
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import accelerate
import threadpoolctl as tctl
import torch
import torch.nn as nn
import transformers
from huggingface_hub import HfApi, hf_hub_download
from packaging import version
from torch.nn.modules.conv import _ConvNd
from transformers import PretrainedConfig
from transformers.pytorch_utils import id_tensor_storage
from transformers.utils.hub import cached_file

from ..looper.named_module import NamedModule
from ..models._const import (CPU, EXPERT_INDEX_PLACEHOLDER, SUPPORTS_MODULE_TYPES)
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..quantization import QuantizeConfig
from ..quantization.config import FORMAT, FORMAT_FIELD_JSON, QUANT_METHOD, dynamic_get
from .backend import BACKEND
from .importer import select_quant_linear
from .logger import setup_logger
from .torch import torch_empty_cache, torch_new_stream_ctx
from ..models._const import DEVICE

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
    is_local = os.path.isdir(pretrained_model_id_or_path)
    if is_local:
        return pretrained_model_id_or_path
    else:
        download_kwargs = kwargs.copy()
        download_kwargs.pop("max_memory", None)
        download_kwargs.pop("attn_implementation", None)
        download_kwargs.pop("use_flash_attention_2", None)
        return snapshot_download(pretrained_model_id_or_path, **download_kwargs)


def ModelLoader(cls):
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_id_or_path: str,
            quantize_config: QuantizeConfig,
            trust_remote_code: bool = False,
            torch_dtype: [str | torch.dtype] = "auto",
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            device: Optional[Union[str, int]] = None,
            **model_init_kwargs,
    ):
        cpu_device_map = {"": "cpu"}

        if quantize_config is None or not isinstance(quantize_config, QuantizeConfig):
            raise AttributeError("`quantize_config` must be passed and be an instance of QuantizeConfig.")

        quantize_config.calculate_bits_per_weight()

        if quantize_config.device is not None:
            if device is not None or device_map is not None:
                raise AttributeError("Passing device and device_map is not allowed when QuantizeConfig.device is set. Non-quantized model is always loaded as cpu. Please set QuantizeConfig.device for accelerator used in quantization or do not set for auto-selection.")

        if quantize_config.desc_act not in cls.supports_desc_act:
            raise ValueError(f"{cls} only supports desc_act={cls.supports_desc_act}, "
                             f"but quantize_config.desc_act is {quantize_config.desc_act}.")

        if cls.require_trust_remote_code and not trust_remote_code:
            raise ValueError(
                f"{pretrained_model_id_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        check_versions(cls, cls.require_pkgs_version)

        model_local_path = get_model_local_path(pretrained_model_id_or_path, **model_init_kwargs)

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
            torch_dtype: [str | torch.dtype] = "auto",
            trust_remote_code: bool = False,
            verify_hash: Optional[Union[str, List[str]]] = None,
            **kwargs,
    ):
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
                    from transformers.utils import is_flash_attn_2_available
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

                if not any(name.startswith(prefix) for prefix in cls.layers_node) or any(name.startswith(ignore_module) for ignore_module in ignore_modules) or all(
                        not name.endswith(ignore_module) for sublist in cls.layer_modules for ignore_module in sublist
                ):
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

        model = gptqmodel_post_init(model, use_act_order=qcfg.desc_act, quantize_config=qcfg)

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)

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

def recurse_getattr(obj, attr: str):
    def _getattr(obj, attr):
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def recurse_setattr(module, name, value):
    if "." not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split(".", 1)
        recurse_setattr(getattr(module, name), rest, value)


def get_device(obj: torch.Tensor | nn.Module):
    if isinstance(obj, torch.Tensor):
        return obj.device

    params = list(obj.parameters())
    if len(params) > 0:
        return params[0].device
    else:
        log.warn(f"Quantize: Unable to determine device of `{obj}`. default to `cpu`")
        return torch.device('cpu')

def move_to(obj: torch.Tensor | nn.Module, device: torch.device, dtype: torch.dtype = None, stream: bool = False):
    if get_device(obj) != device:
        if stream:
            assert dtype is None, f"streaming does not support changing dtype: actual = `{dtype}"
            if not isinstance(obj, torch.Tensor):
                raise NotImplementedError(
                    f"Streaming `move_to` is not supported for non-Tensors: actual = `{obj.__class__.__name__}`")

            if device == CPU:
                obj_copy = torch.zeros_like(obj, device=CPU, pin_memory=True)
                streamCtx = torch_new_stream_ctx()
                if streamCtx:
                    with streamCtx:
                        obj_copy.copy_(obj, non_blocking=True)
                    return obj_copy
                else:
                    obj = obj.to(device=device, non_blocking=True)
            else:
                obj = obj.to(device=device, non_blocking=True)
        else:
            obj = obj.to(device=device, dtype=dtype, non_blocking=False)

    return obj


def nested_move_to(v, device, dtype: torch.dtype = None, stream: bool = False):
    if isinstance(v, torch.Tensor):
        return move_to(v, device=device, dtype=dtype, stream=stream)
    elif isinstance(v, (list, tuple)):
        return type(v)([nested_move_to(e, device=device, dtype=dtype, stream=stream) for e in v])
    else:
        return v


def find_modules(module: nn.Module, layers=None, name: str="") -> Dict[str, nn.Module]:
    if not layers:
        layers = SUPPORTS_MODULE_TYPES

    if isinstance(module, tuple(layers)):
       return {name: module}

    res = {}
    for name1, child in module.named_children():
        res.update(find_modules(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def get_module_by_name_prefix(model, module_name: List[str]):
    for name, module in model.named_modules():
        for prefix in module_name:
            if name.startswith(prefix):
                return module, prefix


def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module

def get_module(module, key):
    name_list = key.split(".")
    for name in name_list:
        module = getattr(module, name, None)
    return module

def make_quant(
    module,
    quant_result: Dict[str, Dict[str, Any]],
    qcfg: QuantizeConfig,
    backend: BACKEND,
    lm_head_name: str,
    pack: bool = False,
    device: DEVICE = None,
    from_quantized: bool = False,
) -> Type[BaseQuantLinear]:

    bits = qcfg.bits
    group_size =qcfg.group_size
    format = qcfg.format
    desc_act = qcfg.desc_act
    sym = qcfg.sym
    dynamic = qcfg.dynamic
    pack_dtype = qcfg.pack_dtype

    quant_linear_candidates = select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        backend=backend,
        format=format,
        pack=pack,
        dynamic=dynamic,
        device=device,
        pack_dtype=pack_dtype,
        multi_select=True,
    )

    log.info(f"Kernel: candidates -> `[{', '.join(cls.__name__ for cls in quant_linear_candidates)}]`")

    for cls in quant_linear_candidates:
        try:
            linear_cls = create_quant_layer(
                linear_cls=cls,
                bits=bits,
                desc_act=desc_act,
                dynamic=dynamic,
                group_size=group_size,
                module=module,
                quant_result=quant_result,
                sym=sym,
                device=device,
                lm_head_name=lm_head_name,
                pack_dtype=pack_dtype,
                backend=backend,
            )
            log.info(f"Kernel: selected -> `{linear_cls.__name__}`.")
            return linear_cls
        except NotImplementedError as e:
            log.info(f"Kernel: skipped -> `{cls}`.")

            if backend not in [BACKEND.AUTO, BACKEND.AUTO_TRAINABLE]:
                raise e

    raise ValueError(f"No compatible quant linear was found for this module: {module.__class__.__name__}")


def create_quant_layer(
        linear_cls: Type[BaseQuantLinear],
        bits: int,
        desc_act: bool,
        dynamic,
        group_size: int,
        module,
        quant_result: Dict[str, Dict[str, Any]],
        sym: bool,
        device: DEVICE,
        lm_head_name: str,
        pack_dtype: torch.dtype,
        backend: BACKEND,
) -> Type[BaseQuantLinear]:
    if isinstance(module, linear_cls):
        return linear_cls
    for name, submodule in module.named_modules():
        if name not in quant_result:
            continue

        if not isinstance(submodule, BaseQuantLinear):
            ori_layer_device = next(submodule.parameters()).device
        else:
            ori_layer_device = submodule.list_buffers()[0].device

        if isinstance(submodule, NamedModule):
            in_features = submodule.state.get("in_features")
            out_features = submodule.state.get("out_features")
        elif isinstance(submodule, nn.Linear):
            in_features = submodule.in_features
            out_features = submodule.out_features
        elif isinstance(submodule, _ConvNd):
            in_features = submodule.in_channels
            out_features = submodule.out_channels
        elif isinstance(submodule, transformers.Conv1D):
            in_features = submodule.weight.shape[0]
            out_features = submodule.weight.shape[1]
        elif isinstance(submodule, BaseQuantLinear):
            in_features = submodule.in_features
            out_features = submodule.out_features
        else:
            raise NotImplementedError(f"Unsupported module {submodule}")

        bias = submodule.bias is not None

        tmp_bits = bits
        tmp_group_size = group_size
        tmp_desc_act = desc_act
        tmp_sym = sym
        tmp_pack_dtype = pack_dtype

        if dynamic is not None:
            overrides = dynamic_get(dynamic=dynamic, module_name=name)
            if overrides == False:
                continue

            if overrides:
                tmp_bits = overrides.get("bits", bits)
                tmp_group_size = overrides.get("group_size", group_size)
                tmp_desc_act = overrides.get("desc_act", desc_act)
                tmp_sym = overrides.get("sym", sym)
                tmp_pack_dtype = overrides.get("pack_dtype", pack_dtype)

        _, err = linear_cls.validate(
            bits=tmp_bits,
            group_size=tmp_group_size,
            desc_act=tmp_desc_act,
            sym=tmp_sym,
            pack_dtype=tmp_pack_dtype,
            in_features=in_features,
            out_features=out_features,
            device=device,
        )
        if err is not None:
            raise err

        new_layer = linear_cls(
            bits=tmp_bits,
            group_size=tmp_group_size,
            desc_act=tmp_desc_act,
            sym=tmp_sym,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=tmp_pack_dtype,
            bias=bias,
            name=name,
            lm_head_name=lm_head_name,
            backend=backend,
        )
        new_layer.device = ori_layer_device
        recurse_setattr(module, name, new_layer.to(ori_layer_device))
    return linear_cls


def pack_module(name, qModules, quant_result: Dict[str, Dict[str, Any]], layers, quant_linear_cls):
    with tctl.threadpool_limits(limits=1):
        r = quant_result[name]
        scale, zero, g_idx = r["scale"], r["zero"], r["g_idx"]
        qModules[name] = qModules[name].to(CPU)
        layers[name], scale, zero, g_idx = (
            layers[name].to(CPU),
            scale.to(CPU),
            zero.to(CPU),
            g_idx.to(CPU) if g_idx is not None else None,
        )
        qModules[name].pack(linear=layers[name], scales=scale, zeros=zero, g_idx=g_idx)


def pack_model(
    model,
    quant_result: Dict[str, Dict[str, Any]],
    bits,
    group_size,
    backend: BACKEND,
    format: str | FORMAT,
    quant_method: str | QUANT_METHOD,
    lm_head_name: str,
    desc_act=False,
    sym: bool = True,
    dynamic=None,
    parallel_packing: bool = True,
    pack_dtype: torch.dtype = None,
):
    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        format=format,
        quant_method=quant_method,
        desc_act=desc_act,
        sym=sym,
        dynamic=dynamic,
        pack_dtype=pack_dtype,
    )

    model.to(CPU)

    log.info("Packing model...")

    modules = find_modules(model)

    modules = {n: modules[n] for n in quant_result}
    quant_linear_cls = make_quant(
        model,
        quant_result=quant_result,
        qcfg=qcfg,
        backend=backend,
        lm_head_name=lm_head_name,
        pack=True,
    )

    qModules = find_modules(model, [quant_linear_cls])

    assert len(qModules) > 0, f"No quantizeed modules[{quant_linear_cls}] found in the model."

    names = list(qModules.keys())

    if parallel_packing:
        max_workers = 2
    else:
        max_workers = 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with log.pb(names).manual() as pb:
            def wrapper(name):
                pb.next()
                pb.title(f"Packing {name}").draw()
                pack_module(name=name, qModules=qModules, quant_result=quant_result, layers=modules,
                            quant_linear_cls=quant_linear_cls)

            for _ in executor.map(wrapper, names):
                pass

    log.info("Model packed.")
    return quant_linear_cls


def verify_model_hash(file_path: str, verify_hash: str):
    if not isinstance(verify_hash, str):
        raise ValueError("model verify_hash must be a string")
    if ':' not in verify_hash:
        raise ValueError("verify_hash must be in the format 'hash_type:hash_value'")
    hash_type, hash_value = verify_hash.split(':', 1)
    hash_func = getattr(hashlib, hash_type, None)
    if not hash_func:
        raise ValueError(f"No hash function found for type: {hash_type}")
    with open(file_path, "rb") as f:
        file_hash = hash_func(f.read()).hexdigest()
    return file_hash == hash_value


def verify_sharded_model_hashes(jsonPath: str, verify_hash: List[str]):
    if not isinstance(verify_hash, list):
        raise ValueError("sharded model verify_hash must be a list")

    with open(jsonPath, 'r') as f:
        index_data = json.load(f)
    weight_map = index_data['weight_map']
    shard_files = set(weight_map.values())
    if len(shard_files) != len(verify_hash):
        raise ValueError("Number of shards and number of hash values do not match.")

    for shard_file, expected_hash in zip(shard_files, verify_hash):
        if not verify_model_hash(shard_file, expected_hash):
            log.info(f"Hash verification failed for {shard_file}")
            return False
    return True

def simple_dispatch_model(model, device_map):

    if "" in device_map:
        d = device_map[""]
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model
    else:
        raise ValueError("internal device_map must contain an empty string")

    return model


def hf_gptqmodel_post_init(model, use_act_order: bool, quantize_config: QuantizeConfig = None):
    return gptqmodel_post_init(model, use_act_order, quantize_config)


def gptqmodel_post_init(model, use_act_order: bool, quantize_config: QuantizeConfig = None):
    for _, submodule in model.named_modules():
        if isinstance(submodule, BaseQuantLinear):
            submodule.post_init()

    torch_empty_cache()

    return model


def get_checkpoints(model_id_or_path: str, extensions: List[str], possible_model_basenames: List[str], **cached_file_kwargs):
    searched_files = []
    resolved_archive_file = None
    true_model_basename = None

    if os.path.isdir(model_id_or_path):
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                shard_index_name = possible_model_basename + ext + ".index.json"
                searched_files.append(shard_index_name)
                possible_index_file = os.path.join(model_id_or_path, shard_index_name)
                if os.path.isfile(possible_index_file):
                    possible_model_basename = possible_index_file.replace(ext + ".index.json", "")
                    return True, possible_index_file, possible_model_basename
                else:
                    model_save_name = os.path.join(model_id_or_path, possible_model_basename)
                    searched_files.append(possible_model_basename + ext)
                    if os.path.isfile(model_save_name + ext):
                        resolved_archive_file = model_save_name + ext
                        return False, resolved_archive_file, possible_model_basename
    else:
        temp = None
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                shard_index_name = possible_model_basename + ext + ".index.json"
                shard_index = cached_file(
                    model_id_or_path,
                    shard_index_name,
                    **cached_file_kwargs,
                )
                searched_files.append(shard_index_name)
                if shard_index is not None:
                    with open(str(shard_index)) as f:
                        index_json = json.load(f)
                        shards = list(set(index_json["weight_map"].values()))
                        for shard in shards:
                            resolved_archive_file = cached_file(
                                model_id_or_path,
                                shard,
                                **cached_file_kwargs,
                            )
                        return True, shard_index, possible_model_basename
                else:
                    resolved_archive_file = cached_file(
                        model_id_or_path,
                        possible_model_basename + ext,
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is None:
                        resolved_archive_file = temp
                    searched_files.append(possible_model_basename + ext)
                    if resolved_archive_file is not None:
                        temp = resolved_archive_file
                        return False, resolved_archive_file, possible_model_basename

    if resolved_archive_file is None:
        raise FileNotFoundError(
            f"Could not find a model in {model_id_or_path} with a name in {', '.join(searched_files)}. Please specify the argument model_basename to use a custom file name."
        )

    return False, resolved_archive_file, true_model_basename


def auto_dtype(config: PretrainedConfig,
               device: DEVICE,
               quant_inference: bool = False) -> torch.dtype:

    assert isinstance(device, DEVICE)

    if device in [DEVICE.MPS, DEVICE.XPU]:
        log.info("Loader: Auto dtype (MPS or XPU): `torch.float16`")
        return torch.float16

    if device in [DEVICE.CPU]:
        log.info("Loader: Auto dtype (CPU): `torch.bfloat16`")
        return torch.bfloat16

    dtype = getattr(config, "torch_dtype")
    if dtype and not isinstance(dtype, torch.dtype):
        raise ValueError(f"torch_dtype in config must be a torch.dtype, but got {dtype}")

    if dtype in [torch.float32, torch.float64]:
        log.info("Loader: Auto dtype (float32 down-cast): `torch.bfloat16`")
        return torch.bfloat16
    elif dtype == torch.float16:
        log.info("Loader: Auto dtype (native float16): `torch.float16`")
        return torch.float16
    elif dtype == torch.bfloat16:
        log.info("Loader: Auto dtype (native bfloat16): `torch.bfloat16`")
        return torch.bfloat16
    else:
        log.info(f"Loader: Auto dtype (native = `{dtype}`): `torch.bfloat16`")
        return torch.bfloat16


def get_moe_layer_modules(layer_modules: List, num_experts: int) -> List:
    new_inside_layer_modules = []
    for names in layer_modules:
        new_inside_layer_modules.append([])
        for n in names:
            if EXPERT_INDEX_PLACEHOLDER in n:
                for index in range(num_experts):
                    new_inside_layer_modules[-1].append(n.replace(EXPERT_INDEX_PLACEHOLDER, str(index)))
            else:
                new_inside_layer_modules[-1].append(n)

    return new_inside_layer_modules


def check_to_quantized(config):
    if isinstance(config, dict):
        if config["bits"] > 8 or "fp" in config["data_type"] or "float" in config["data_type"]:
            return False
        return True
    else:
        if config.bits > 8 or "fp" in config.data_type or "float" in config.data_type:
            return False
        return True


def copy_py_files(save_dir, file_extension=".py", model_id_or_path=""):
    os.makedirs(save_dir, exist_ok=True)

    if os.path.isdir(model_id_or_path):
        py_files = [f for f in os.listdir(model_id_or_path) if f.endswith('.py')]
        for file in py_files:
            shutil.copy2(os.path.join(model_id_or_path, file), save_dir)
    else:
        api = HfApi()
        model_info = api.model_info(model_id_or_path)
        for file in model_info.siblings:
            if file.rfilename.endswith(file_extension):
                _ = hf_hub_download(repo_id=model_id_or_path, filename=file.rfilename,
                                                  local_dir=save_dir)

def get_model_files_size(pre_quantized_model_path, file_extension=['.bin', '.safetensors', '.pth', '.pt', '.ckpt', '.h5', '.pb', '.onnx']):
    if os.path.isdir(pre_quantized_model_path):
        pre_quantized_size_bytes = sum(
            os.path.getsize(os.path.join(pre_quantized_model_path, f))
            for f in os.listdir(pre_quantized_model_path)
            if os.path.isfile(os.path.join(pre_quantized_model_path, f)) and os.path.splitext(f)[
                1] in file_extension
        )
    else:
        api = HfApi()
        files_data = api.list_repo_files(pre_quantized_model_path)
        pre_quantized_size_bytes = 0
        for file_info in files_data:
            if any(file_info.endswith(ext) for ext in file_extension):
                file_metadata = api.model_info(pre_quantized_model_path, files_metadata=True)
                for file_data in file_metadata.siblings:
                    if file_data.rfilename == file_info:
                        pre_quantized_size_bytes += file_data.size
    pre_quantized_size_mb = pre_quantized_size_bytes / (1024 * 1024)
    return pre_quantized_size_mb

def check_requires_version(requires_version, current_version):
    OPERATOR_MAP = {
        "<=": operator.le,
        ">=": operator.ge,
        "==": operator.eq,
        "<": operator.lt,
        ">": operator.gt,
    }
    match = re.match(r"(<=|>=|==|<|>)\s*([\d\.]+)", requires_version)
    if match:
        op_symbol, required_version = match.groups()
        current_version = version.parse(current_version)
        required_version = version.parse(required_version)
        return OPERATOR_MAP[op_symbol](current_version, required_version)
    else:
        return None


class MODALITY(str, Enum):
    TEXT = "text"
    IMAGE_TO_TEXT = "image_to_text"


def get_state_dict_for_save(model: nn.Module) -> Dict:
    state_dict = model.state_dict()

    ptrs = collections.defaultdict(list)
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            ptrs[id_tensor_storage(tensor)].append(name)
        else:
            ptrs[id(tensor)].append(name)

    shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
    warn_names = set()
    for names in shared_ptrs.values():
        if model._tied_weights_keys is not None:
            found = 0
            for name in sorted(names):
                matches_pattern = any(re.search(pat, name) for pat in model._tied_weights_keys)
                if matches_pattern and name in state_dict:
                    found += 1
                    if found < len(names):
                        del state_dict[name]

        found = 0
        for name in names:
            if name in state_dict:
                found += 1
                if found > 1:
                    del state_dict[name]
                    warn_names.add(name)
    if len(warn_names) > 0:
        log.warn.once(
            f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading",
        )
    return state_dict

def load_checkpoint_in_model_then_tie_weights(model, *args, **kwargs):
    accelerate.load_checkpoint_in_model(model, *args, **kwargs)
    model.tie_weights()