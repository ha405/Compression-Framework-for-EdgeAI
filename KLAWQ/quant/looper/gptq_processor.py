import copy
import time
from typing import Callable, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseGPTQModel
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, PROCESS_LOG_NAME,
                             PROCESS_LOG_TIME, PROCESS_MAX_MEMORY, QUANT_LOG_DAMP, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES)
from ..quantization import GPTQ
from ..quantization.config import QUANT_METHOD, QuantizeConfig
from ..quantization.gptq import CPU, DEVICE_0, DEVICE_1
from ..utils.logger import setup_logger
from ..utils.model import move_to, pack_model
from ..utils.torch import torch_empty_cache, torch_sync

log = setup_logger()

class GPTQProcessor(LoopProcessor):
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration_dataset, prepare_dataset_func,
                 calibration_dataset_concat_size: Optional[int], batch_size: int,
                 logger_board: str = "", require_fwd: bool = True, retain_w: bool = False):

        super().__init__(tokenizer=tokenizer, qcfg=qcfg, calibration_dataset=calibration_dataset,
                         calibration_dataset_concat_size=calibration_dataset_concat_size,
                         prepare_dataset_func=prepare_dataset_func, batch_size=batch_size,
                         logger_board=logger_board, require_fwd=require_fwd)

        self.retain_w = retain_w
        self.avg_losses = []

    def log_plotly(self):
        task = self.logger_task
        if task is not None:
            from ..utils.plotly import create_plotly
            x = list(range(self.layer_count))
            gpu_fig = create_plotly(x=x, y=self.gpu_memorys, xaxis_title="layer", yaxis_title="GPU usage (GB)")
            cpu_fig = create_plotly(x=x, y=self.cpu_memorys, xaxis_title="layer", yaxis_title="CPU usage (GB)")
            loss_fig = create_plotly(x=self.module_names, y=self.avg_losses, xaxis_title="layer", yaxis_title="loss")
            time_fig = create_plotly(x=self.module_names, y=self.durations, xaxis_title="layer", yaxis_title="time")
            task.get_logger().report_plotly('GPU Memory', 'GPU Memory', gpu_fig)
            task.get_logger().report_plotly('CPU Memory', 'CPU Memory', cpu_fig)
            task.get_logger().report_plotly('avg_loss', 'avg_loss', loss_fig)
            task.get_logger().report_plotly('quant_time', 'quant_time', time_fig)

    def set_calibration_dataset(self, calibration_dataset):
        raise NotImplementedError("GPTQProcessor's calibration_dataset cannot be modified")

    def preprocess(self, module: NamedModule, buffered_fwd: bool):
        if self.qcfg.dynamic_get(layer_name=module.full_name) == False:
            return

        if not isinstance(module.instance, (torch.nn.Linear, torch.nn.Conv2d)):
            log.info(f"Skipping module {module.full_name} of type {type(module.instance)} as it is not a Linear or Conv2d layer.")
            return

        qcfg_clone = copy.deepcopy(self.qcfg)

        if self.qcfg.dynamic is not None:
            qcfg_clone.bits = self.qcfg.dynamic_get(module.full_name, "bits", qcfg_clone.bits)
            qcfg_clone.sym = self.qcfg.dynamic_get(module.full_name, "sym", qcfg_clone.sym)
            qcfg_clone.mse = self.qcfg.dynamic_get(module.full_name, "mse", qcfg_clone.mse)
            qcfg_clone.group_size = self.qcfg.dynamic_get(module.full_name, "group_size", qcfg_clone.group_size)
            qcfg_clone.desc_act = self.qcfg.dynamic_get(module.full_name, "desc_act", qcfg_clone.desc_act)
            qcfg_clone.damp_percent = self.qcfg.dynamic_get(module.full_name, "damp_percent", qcfg_clone.damp_percent)
            qcfg_clone.static_groups = self.qcfg.dynamic_get(module.full_name, "static_groups", qcfg_clone.static_groups)

        self.qcfg_dynamic = qcfg_clone
        tmp = GPTQ(module=module, qcfg=qcfg_clone)

        if buffered_fwd:
            log.info(f"Experimental: enabling fwd buffered mode for: `{module.name}`")
            tmp.fwd_inputs_buffered = True

        tmp.quantizer.configure(perchannel=True)
        self.tasks[module.name] = tmp

    def is_skipped(self, module: NamedModule) -> bool:
        t = self.tasks.get(module.name, False)
        if t == False:
            return True
        else:
            return False

    def preprocess_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def tmp(_, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            g = self.tasks.get(name)
            if g:
                g.add_batch(inp[0].data, out.data)
            del inp, out
        return tmp

    def process(self, module: NamedModule, auto_gc: bool = True):
        if torch.cuda.device_count() > 1:
            torch.cuda.synchronize()

        self.pb.title(f"Quantizing {module.name} in layer ").draw()

        g = self.tasks[module.name]
        wq_2d, scale, zero, g_idx, duration, avg_loss, damp_percent, nsamples = g.quantize()

        self.durations.append(duration)
        self.avg_losses.append(avg_loss)
        self.module_names.append(f"layer-{module.layer_index}-{module.name}")

        stats_0 = torch.cuda.memory_stats(DEVICE_0)
        active_0 = stats_0.get("active_bytes.all.current", 0) / 1024 ** 2
        peak_active_0 = stats_0.get("active_bytes.all.peak", 0) / 1024 ** 2

        if torch.cuda.device_count() > 1:
            stats_1 = torch.cuda.memory_stats(DEVICE_1)
            active_1 = stats_1.get("active_bytes.all.current", 0) / 1024 ** 2
            peak_active_1 = stats_1.get("active_bytes.all.peak", 0) / 1024 ** 2
            max_memory = f"{active_0:.2f}MB, {active_1:.2f}MB"
        else:
            max_memory = f"{active_0:.2f}MB"

        stat = {
            PROCESS_LOG_NAME:  self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            QUANT_LOG_LOSS: f"{avg_loss:.10f}",
            QUANT_LOG_NSAMPLES: f"{nsamples}",
            QUANT_LOG_DAMP: f"{damp_percent:.5f}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: f"{self.fwd_time:.3f}",
            PROCESS_MAX_MEMORY: max_memory,
        }

        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        self.log.append(stat)
        self.log_new_row(stat)

        self.result_save(module.full_name, {
            "scale": move_to(scale, device=CPU, stream=self.stream),
            "zero": move_to(zero, device=CPU, stream=self.stream),
            "g_idx": move_to(g_idx, device=CPU, stream=self.stream),
        })

        if self.retain_w:
            w = module.weight.data
            module.state.update({"w": w})

        self.tasks[module.name].free()

        if hasattr(module, 'original_shape'):
            wq = wq_2d.reshape(module.original_shape)
        else:
            wq = wq_2d

        wq = wq.to(device=DEVICE_0)
        module.state.update({"wq": wq})
        module.weight.data = wq

        if auto_gc:
            torch_empty_cache()

    def submodule_finalize(self, module: NamedModule):
        if "wq" in module.state:
            module.weight.data = move_to(module.state.pop("wq"), device=CPU, stream=self.stream)
            module.state.pop("w", None)

    def finalize(self, model: BaseGPTQModel, **kwargs):
        if self.stream:
            torch_sync()

        backend = kwargs.pop("backend")
        model.qlinear_kernel = pack_model(
            model=model.model,
            quant_result=self.results(),
            bits=self.qcfg.bits,
            group_size=self.qcfg.group_size,
            backend=backend,
            desc_act=self.qcfg.desc_act,
            format=self.qcfg.format,
            quant_method=self.qcfg.quant_method,
            lm_head_name=model.lm_head,
            dynamic=self.qcfg.dynamic,
            parallel_packing=self.qcfg.parallel_packing,
            pack_dtype=self.qcfg.pack_dtype,
        )
        model.quantized = True
        model.quantize_config.quant_method = QUANT_METHOD.GPTQ
        super().finalize(model=model, **kwargs)

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        if self.calibration_dataset is None:
            raise ValueError("GPTQProcessor's calibration_dataset must be provided.")
        else:
            return True

    def name(self) -> str:
        return "gptq"