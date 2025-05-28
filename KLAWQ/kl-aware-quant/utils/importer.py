import os
from collections import OrderedDict
from typing import Dict, List, Optional, Type, Union

import torch

from ..models._const import DEVICE
from ..nn_modules.qlinear import BaseQuantLinear, PackableQuantLinear
from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..quantization import FORMAT
from ..utils.logger import setup_logger
from . import BACKEND
from .rocm import IS_ROCM
from .torch import HAS_CUDA, HAS_MPS, HAS_XPU, auto_select_device, normalize_device

message_logged = False
log = setup_logger()

AUTO_SELECT_BACKEND_ORDER = OrderedDict({
    BACKEND.TORCH: TorchQuantLinear,
})

FORMAT_DICT = {
    FORMAT.GPTQ: [BACKEND.TORCH],
}

def normalize_device_device_map(device: Optional[Union[str, torch.device]], device_map: Optional[Union[str, Dict]]) -> Optional[DEVICE]:
    normalized_device = None
    if device is None:
        if device_map is not None:
            devices = {device_map} if isinstance(device_map, str) else set(device_map.values())
            normalized_devices = set()
            for device in devices:
                if isinstance(device, str) and device == "auto":
                    return None
                normalized_devices.add(normalize_device(device))
            if len(normalized_devices) == 1:
                d = normalized_devices.pop()
                if d in DEVICE:
                    normalized_device = d
            elif len(normalized_devices) > 1:
                normalized_devices.discard(DEVICE.CPU)
                normalized_device = normalized_devices.pop()
    else:
        if isinstance(device, str):
            normalized_device = normalize_device(device)
        elif isinstance(device, torch.device):
            normalized_device = DEVICE(device.type)
        else:
            raise ValueError(f"device must be a string or torch.device, got {type(device)}")

    if normalized_device == DEVICE.CUDA and IS_ROCM:
        normalized_device = DEVICE.ROCM
    return normalized_device


def auto_select_device(device: Optional[DEVICE], backend: Optional[BACKEND]) -> DEVICE:
    assert device is None or isinstance(device, DEVICE)
    assert backend is None or isinstance(backend, BACKEND)

    if device is None:
        if HAS_CUDA:
            device = DEVICE.CUDA
        elif HAS_MPS:
            device = DEVICE.MPS
        else:
            device = DEVICE.CPU
    return device

def hf_select_quant_linear(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        checkpoint_format: str,
        meta: Optional[Dict[str, any]] = None,
        pack: Optional[bool] = True,
        device_map: Optional[Union[str, dict]] = None,
        backend: Optional[Union[str, BACKEND]] = None,
) -> Type[BaseQuantLinear]:
    if isinstance(backend, str):
        backend = BACKEND(backend.lower())

    if device_map is not None:
        device = normalize_device_device_map(None, device_map)
    else:
        device = DEVICE.CPU

    return select_quant_linear(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
        backend=backend,
        device=device,
        format=FORMAT.GPTQ,
        pack=pack,
        dynamic=None,
        pack_dtype=torch.int32,
    )


def select_quant_linear(
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        device: Optional[DEVICE] = None,
        backend: BACKEND = BACKEND.AUTO,
        format: FORMAT = FORMAT.GPTQ,
        pack: bool = False,
        dynamic=None,
        pack_dtype: torch.dtype = None,
        multi_select: bool = False,
) -> Union[Type[BaseQuantLinear], List[Type[BaseQuantLinear]]]:
    if device is None:
        device = DEVICE.CUDA

    backend = BACKEND.AUTO if backend is None else backend

    trainable = backend == BACKEND.AUTO_TRAINABLE

    validated_qlinears = []
    if backend in [BACKEND.AUTO, BACKEND.AUTO_TRAINABLE]:
        allow_quant_linears = [(k, v) for k,v in AUTO_SELECT_BACKEND_ORDER.items() if k in FORMAT_DICT[format]]
        err = None
        global message_logged
        for k, cls in allow_quant_linears:
            validate, err = cls.validate(
                bits=bits,
                group_size=group_size,
                desc_act=desc_act,
                sym=sym,
                pack_dtype=pack_dtype,
                dynamic=dynamic,
                device=device,
                trainable=trainable,
            )
            if os.environ.get("DEBUG") and not validate:
                log.info(f"skip {k} for {str(err)}")
            if validate:
                if pack:
                    check_pack_func = issubclass(cls, PackableQuantLinear)
                    if check_pack_func:
                        log.info(f"{'Packing' if pack else ''} Kernel: Auto-selection: adding candidate `{cls.__name__}`")
                        validated_qlinears.append(cls)
                        if not multi_select:
                            return cls
                else:
                    log.info(f"{'Packing' if pack else ''} Kernel: Auto-selection: adding candidate `{cls.__name__}`")
                    validated_qlinears.append(cls)
                    if not multi_select:
                        return cls

        if err:
            raise err

        return validated_qlinears

    if backend == BACKEND.TORCH:
        qlinear = TorchQuantLinear
    else:
        qlinear = TorchQuantLinear

    validate, err = qlinear.validate(bits=bits, group_size=group_size, desc_act=desc_act, sym=sym, pack_dtype=pack_dtype, dynamic=dynamic, device=device, trainable=trainable)

    log.info(f"{'Packing' if pack else ''} Kernel: selected: `{qlinear.__name__}`")
    if not validate:
        raise ValueError(err)
    else:
        if multi_select:
            return [qlinear]
        else:
            return qlinear