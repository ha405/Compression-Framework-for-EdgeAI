import copy
import json
import os.path
import re
from dataclasses import dataclass, field, fields
from os.path import join
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version

from ..utils.logger import setup_logger

log = setup_logger()

FORMAT_FIELD_CODE = "format"
FORMAT_FIELD_JSON = "checkpoint_format"
QUANT_METHOD_FIELD = "quant_method"
PACK_DTYPE_FIELD = "pack_dtype"
QUANT_CONFIG_FILENAME = "quantize_config.json"
QUANT_CONFIG_FILENAME_COMPAT = [QUANT_CONFIG_FILENAME, "quant_config.json", "config.json"]

META_FIELD = "meta"
META_FIELD_QUANTIZER = "quantizer"

META_QUANTIZER_GPTQMODEL = "gptqmodel"

META_FIELD_URI = "uri"
META_VALUE_URI = "https://github.com/modelcloud/gptqmodel"

META_FIELD_DAMP_PERCENT = "damp_percent"
META_FIELD_DAMP_AUTO_INCREMENT = "damp_auto_increment"

META_FIELD_STATIC_GROUPS = "static_groups"
META_FIELD_TRUE_SEQUENTIAL = "true_sequential"

META_FIELD_MSE = "mse"


# saved formats
class FORMAT:
    GPTQ = "gptq"


# quant methods
class QUANT_METHOD:
    GPTQ = "gptq"


QUANT_METHOD_FORMAT_MAPPING = {
    QUANT_METHOD.GPTQ: {
        FORMAT.GPTQ,
    },
}

QUANTIZE_BLACK_LIST = {}

# compat
QUANT_CONFIG_ARG_SYNONYMS = {
    "w_bit": "bits",
    "q_group_size": "group_size",
    FORMAT_FIELD_JSON: FORMAT_FIELD_CODE,
}

def dict_scale_dtype_to_str(d: Dict[str, Any]) -> None:
    """
    Checks whether the passed dictionary and its nested dicts have a *scale_dtype* key and if it's not None,
    converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
    string, which can then be stored in the json format.
    """
    if d.get("scale_dtype", None) is not None and not isinstance(d["scale_dtype"], str):
        d["scale_dtype"] = str(d["scale_dtype"]).split(".")[1]
    for value in d.values():
        if isinstance(value, dict):
            dict_scale_dtype_to_str(value)

def dynamic_get(dynamic: Dict[str, Dict[str, Union[int, bool]]], module_name: str, key: str = None,
                default: Union[int, bool] = None, sub_key: str = None) -> Union[Dict, int, bool]:

    if dynamic is None:
        return default

    for pattern, overrides in dynamic.items():
        if pattern.startswith("-:"):
            if re.match(pattern.removeprefix("-:"), module_name):
                return False
        elif re.match(pattern.removeprefix("+:"), module_name):
            if key is None:
                return overrides
            else:
                if sub_key:
                    sub_value = overrides.get(key, None)
                    if isinstance(sub_value, Dict):
                        return sub_value.get(sub_key, default)
                    else:
                        log.info(f"QuantConfig: Dynamic `sub_key`: `{sub_key}` failed extraction from  `sub_value`: `{sub_value}`")
                else:
                    return overrides.get(key, default)
    return default

@dataclass
class QuantizeConfig():
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})

    dynamic: Optional[Dict[str, Dict[str, Union[int, bool]]]] = field(default=None)

    group_size: int = field(default=128)

    damp_percent: float = field(default=0.01)
    damp_auto_increment: float = field(default=0.0025)

    desc_act: bool = field(default=True)
    static_groups: bool = field(default=False)
    sym: bool = field(default=True)
    true_sequential: bool = field(default=True)

    lm_head: bool = field(default=False)

    quant_method: QUANT_METHOD = field(default=QUANT_METHOD.GPTQ)

    format: FORMAT = field(default=FORMAT.GPTQ)

    mse: float = field(default=0.0)

    parallel_packing: bool = field(default=True)

    meta: Optional[Dict] = field(default=None)

    device: Optional[Union[str, torch.device]] = field(default=None)

    pack_dtype: Optional[Union[str, torch.dtype]] = field(default=torch.int32)

    rotation: Optional[str] = field(default=None, metadata={"choices": ["hadamard", "random"]})

    beta: float = field(default=0.0, metadata={"help": "Strength of KL divergence term (beta=0 recovers vanilla GPTQ)"})
    tau: float = field(default=1.0, metadata={"help": "Temperature for softmax in KL divergence calculation (must be > 0)"})

    def __post_init__(self):
        fields_info = fields(self)

        if self.pack_dtype is None:
            self.pack_dtype = torch.int32
        else:
            if isinstance(self.pack_dtype, str):
                self.pack_dtype = self.pack_dtype.lower()
                if self.pack_dtype not in ["int64", "int32", "int16", "int8"]:
                    raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {self.pack_dtype}")
                self.pack_dtype = getattr(torch, self.pack_dtype)
            elif isinstance(self.pack_dtype, torch.dtype):
                if self.pack_dtype not in [torch.int64, torch.int32, torch.int16, torch.int8]:
                    raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {self.pack_dtype}")
            else:
                raise ValueError(f"QuantizeConfig: Unsupported `pack_dtype`: {self.pack_dtype}")

        valid_formats = QUANT_METHOD_FORMAT_MAPPING.get(self.quant_method, None)
        if valid_formats is None:
            raise ValueError(f"QuantizeConfig: Unsupported `quant_method`: {self.quant_method}")

        if self.format not in valid_formats:
            raise ValueError(
                f"QuantizeConfig: checkpoint `format` used is {self.format}, and the quantization method is {self.quant_method}. "
            )

        if self.bits not in fields_info[0].metadata["choices"]:
            raise ValueError(f"QuantizeConfig: `bits` must be in the set of `{fields_info[0].metadata['choices']}`.")

        if self.dynamic is not None:
            self.dynamic = {
                **{k: v for k, v in self.dynamic.items() if k.startswith('-')},
                **{k: v for k, v in self.dynamic.items() if not k.startswith('-')}
            }

            for layer, layer_dict in self.dynamic.items():
                for key, value in layer_dict.items():
                    if key == "bits" and value not in fields_info[0].metadata["choices"]:
                        raise ValueError(f"QuantizeConfig: Layer `{layer}` only support quantization of  `{fields_info[0].metadata['choices']}` bits.")
                    elif key == "group_size" and value != -1 and value <= 0:
                        raise ValueError("QuantizeConfig: `group_size` must in the value set of `[-1, 16, 32, 64, 128]`.")

        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("QuantizeConfig: `group_size` must in the value set of `[-1, 16, 32, 64, 128]`.")

        if not (0 < self.damp_percent < 1):
            raise ValueError("QuantizeConfig: `damp_percent` must between 0 and 1.")

        if self.damp_auto_increment < 0:
            raise ValueError("QuantizeConfig:: `damp_auto_increment` must greater than 0.")

        if self.meta is not None:
            if not isinstance(self.meta, dict):
                raise ValueError("QuantizeConfig: `meta` must be a dictionary")
            for key, value in self.meta.items():
                if not isinstance(key, str):
                    raise ValueError("QuantizeConfig: `meta` keys must be strings")
        else:
            self.meta = {}

        if self.beta < 0:
            raise ValueError("QuantizeConfig: `beta` must be non-negative (>= 0).")
        if self.tau <= 0:
            raise ValueError("QuantizeConfig: `tau` (temperature) must be positive (> 0).")

    def meta_set(self, key: str, value: Any):
        self.meta[key] = value

    def meta_get(self, key: str) -> Any:
        return self.meta.get(key)

    def dynamic_get(self, layer_name: str, key: str = None, default: Union[int, bool, float] = None, sub_key: str = None
                    ) -> Union[Dict, int, bool, float]:
        return dynamic_get(self.dynamic, layer_name, key, default, sub_key)
    def meta_set_versionable(self, key: str, value: List[str]):
        self.meta_set(key, value)
    def meta_get_versionable(self, key: str) -> List[Tuple[str, str]]:
        values = self.meta_get(key)
        if values is None:
            return []
        if not isinstance(values, list):
            values = [values]
        result = []
        for val in values:
            parts = val.split(":")
            if len(parts) >= 2:
                result.append((parts[0].lower(), parts[1].lower()))
        return result

    def save_pretrained(self, save_dir: str, **kwargs):
        with open(join(save_dir, QUANT_CONFIG_FILENAME), "w", encoding="utf-8") as f:
            d = self.to_dict()
            json_str = json.dumps(d, indent=2)
            log.info(f"Saved Quantize Config: \n{json_str}")
            f.write(json_str)

    @classmethod
    def from_quant_config(cls, quantize_cfg, format: str = None):
        valid_formats = {FORMAT.GPTQ}
        format_auto_inferred = False
        if format:
            if format not in valid_formats:
                raise ValueError(f"QuantizeConfig: Unknown quantization checkpoint format: {format}.")
            if quantize_cfg.get(FORMAT_FIELD_JSON):
                raise ValueError("QuantizeConfig: Conflicting quantization format passed in manually and also exists in model config.")
        elif quantize_cfg.get(FORMAT_FIELD_JSON) is None:
            format_auto_inferred = True

        field_names = [field.name for field in fields(cls)]

        normalized = {
            QUANT_METHOD_FIELD: QUANT_METHOD.GPTQ,
            FORMAT_FIELD_CODE: format if format else FORMAT.GPTQ,
        }
        for key, val in quantize_cfg.items():
            key = key.lower()
            if key in QUANT_CONFIG_ARG_SYNONYMS and QUANT_CONFIG_ARG_SYNONYMS[key] in field_names:
                key = QUANT_CONFIG_ARG_SYNONYMS[key]

            if key == FORMAT_FIELD_JSON:
                val = val.lower()

                if val == FORMAT.GPTQ:
                    normalized[key] = val
                else:
                    raise ValueError(f"QuantizeConfig: Unknown quantization format: `{val}`.")
            elif key == QUANT_METHOD_FIELD:
                val = val.lower()
                if val != QUANT_METHOD.GPTQ:
                    raise ValueError(f"QuantizeConfig: Unknown quantization method: `{val}`.")
                else:
                    normalized[QUANT_METHOD_FIELD] = val
            elif key in field_names:
                normalized[key] = val
            else:
                log.info(f"QuantizeConfig: Ignoring unknown parameter in the quantization configuration: {key}.")

        if format_auto_inferred:
            log.info(f"QuantizeConfig: `{FORMAT_FIELD_JSON}` is missing from the quantization configuration and is automatically inferred to {normalized[FORMAT_FIELD_CODE]}")

        if "sym" not in normalized:
            log.warn(
                "QuantizeConfig: config does not contain `sym` (symmetric quantization). This may result in silent errors. Defaulting to `sym=True`."
            )

        return cls(**normalized)

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        format = kwargs.pop("format", None)

        transformers_config = False
        resolved_config_file = None
        for quantize_config_filename in QUANT_CONFIG_FILENAME_COMPAT:
            resolved_config_file = join(save_dir, quantize_config_filename)
            if os.path.exists(resolved_config_file):
                if quantize_config_filename == "config.json":
                    transformers_config = True
                break

        if resolved_config_file is None:
            raise ValueError(
                "QuantizeConfig: No quantize_config.json, quant_config.json or config.json file was found in the model repository."
            )

        with open(resolved_config_file, "r", encoding="utf-8") as f:
            args_from_json = json.load(f)

            if transformers_config:
                args_from_json = args_from_json["quantization_config"]

            return cls.from_quant_config(args_from_json, format)

    def to_dict(self):
        out = {
            "bits": self.bits,
            "dynamic": self.dynamic,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "sym": self.sym,
            "lm_head": self.lm_head,
            QUANT_METHOD_FIELD:self.quant_method,
            FORMAT_FIELD_JSON: self.format,
            PACK_DTYPE_FIELD: str(self.pack_dtype).split(".")[-1],
            META_FIELD: self.meta,
            "beta": self.beta,
            "tau": self.tau,
        }

        out = {k: v for k, v in out.items() if v is not None and (v not in [None, {}])}

        dict_scale_dtype_to_str(out)
        return out
    def calculate_bits_per_weight(self):
        if self.group_size != -1:
            per_group_bits = self.group_size * self.bits
            per_group_bits += 16
            per_group_bits += self.bits
            per_group_bits += 4
            bpw = per_group_bits / self.group_size
            bpw += 0.1
        else:
            bpw = self.bits
        log.info(f"Estimated Quantization BPW (bits per weight): {bpw} bpw, based on [bits: {self.bits}, group_size: {self.group_size}]")

@dataclass
class BaseQuantizeConfig(QuantizeConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        log.warn("QuantizeConfig: BaseQuantizeConfig is re-named and pending deprecation. Please use `QuantizeConfig` instead.")