import copy
import json
import os
import queue
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from random_word import RandomWords
from torch import Tensor
from torch.nn import Module

from ..looper.input_cache import InputCache
from ..looper.named_module import NamedModule
from ..models import BaseGPTQModel
from ..quantization.config import QuantizeConfig
from ..utils.device import get_cpu_usage_memory, get_gpu_usage_memory
from ..utils.logger import setup_logger

log = setup_logger()


class LoopProcessor:
    def __init__(
            self,
            tokenizer, qcfg: QuantizeConfig,
            calibration_dataset,
            prepare_dataset_func,
            calibration_dataset_concat_size: Optional[int],
            batch_size: int = 1,
            logger_board: str = "",
            require_fwd: bool = True,
            fwd_after_process: bool = True,
            fwd_all_modules_in_single_pass: bool = False,
    ):

        self._results: Dict[str, Any] = {}

        self.stream = False

        self.tokenizer = tokenizer
        self.qcfg = qcfg
        self.qcfg_dynamic = None

        self.require_fwd = require_fwd
        self.fwd_after_process = fwd_after_process
        self.fwd_all_modules_in_single_pass = fwd_all_modules_in_single_pass

        self.inputs_cache: InputCache = InputCache(None, None, None, None)
        self.tasks = {}

        self.pb = None
        self.logger_task = None
        self.fwd_time = None
        self.layer_count = None


        self.gpu_memorys = []
        self.cpu_memorys = []
        self.durations = []
        self.module_names = []

        self.log = []
        self.logger_board = logger_board
        self.log_max_widths = {}
        self.log_call_count = 0
        current_time = datetime.now().strftime("%m_%d_%Y_%Hh_%Mm_%Ss")
        self.log_tmp_log_file_name = f"{self.name()}_log_{RandomWords().get_random_word()}_time_{current_time}.log"
        self.log_worker_queue = queue.Queue()
        self.log_worker: threading.Thread = None

        if self.logger_board == "clearml":
            try:
                from clearml import Task
            except ImportError as _:
                raise ImportError(
                    "The logger_board is set to 'clearml', but required dependencies are missing. "
                    "Please install them by running: pip install gptqmodel[logger]"
                )
            self.logger_task = Task.init(project_name='GPTQModel',
                                         task_name=f'{self.__class__.__name__}-{RandomWords().get_random_word()}',
                                         task_type=Task.TaskTypes.optimizer)
        else:
            self.logger_task = None


        if calibration_dataset is not None:
            if len(calibration_dataset) == 0:
                raise ValueError("Calibration dataset must not be empty.")

            min_calibration_dataset_size = 256
            # MODIFIED: Text-specific warnings are not always relevant
            # min_calibration_dataset_input_ids_avg_length = 256 
            if len(calibration_dataset) < min_calibration_dataset_size:
                log.warn(f"Calibration dataset size should be more than {min_calibration_dataset_size}. "
                               f"Current: {len(calibration_dataset)}.")

            calibration_dataset = prepare_dataset_func(calibration_dataset=calibration_dataset,
                                                            calibration_dataset_concat_size=calibration_dataset_concat_size,
                                                            batch_size=batch_size)
            
            # --- START OF MODIFIED VALIDATION BLOCK ---
            # This logic is now generic and works for both text and image models.
            total_elements = 0
            is_text_data = True # Assume text by default unless we find other data types
            for row in calibration_dataset:
                if "input_ids" in row:
                    data = row["input_ids"]
                    if isinstance(data, torch.Tensor):
                        # For text, we are interested in the sequence length
                        total_elements += data.shape[-1]
                    else:
                        total_elements += len(data)
                else:
                    is_text_data = False
                    # For other modalities, we count the number of samples in the batch
                    found_tensor = False
                    for key, value in row.items():
                        if isinstance(value, torch.Tensor):
                            total_elements += value.shape[0] 
                            found_tensor = True
                            break 
                    if not found_tensor:
                        log.warn(f"Could not find a tensor in calibration data batch to verify size. Keys: {list(row.keys())}")
            
            avg = total_elements / len(calibration_dataset)
            
            # Only show the text-specific warning if we detected text data
            if is_text_data:
                min_calibration_dataset_input_ids_avg_length = 256
                if avg < min_calibration_dataset_input_ids_avg_length:
                    log.warn(f"The average length of input_ids of calibration_dataset should be greater than "
                                   f"{min_calibration_dataset_input_ids_avg_length}: actual avg: {avg}.")
            # --- END OF MODIFIED VALIDATION BLOCK ---

            self.num_batches = len(calibration_dataset)

        self.calibration_dataset = calibration_dataset

    def log_save_async(self, stat):
        if self.log_worker is None:
            log.info(f"Process: progress logs for `{self.name()}` will be streamed to file: `{self.log_tmp_log_file_name}`")
            def _thread_log_worker():
                while True:
                    data = self.log_worker_queue.get()
                    if data == False:
                        return
                    with open(self.log_tmp_log_file_name, 'a') as f:
                        json.dump(data, f, indent=4)
                        f.write("\n")

                    self.log_worker_queue.task_done()

            self.log_worker = threading.Thread(target=_thread_log_worker, daemon=True)
            self.log_worker.start()

        self.log_worker_queue.put(stat)

    def loss_color(self, loss_value):
        if loss_value <= 0.1:
            return "\033[92m"
        elif loss_value <= 1:
            return "\033[96m"
        elif loss_value <= 5:
            return "\033[93m"
        elif loss_value <= 20:
            return "\033[33m"
        else:
            return "\033[91m"

    def log_new_row(self, stat):
        self.log_call_count += 1
        self.log_save_async(stat)

        for key, value in stat.items():
            current_width = max(len(str(key)), len(str(value))) + 4
            if key not in self.log_max_widths or current_width > self.log_max_widths[key]:
                self.log_max_widths[key] = current_width

        if self.log_call_count % 20 == 1:
            header_row = "| " + " | ".join(
                str(key).ljust(self.log_max_widths[key]) for key in self.log_max_widths.keys()) + " |"

            if self.log_call_count == 1:
                log.info(len(header_row) * "-")
            log.info(header_row)
            log.info(len(header_row) * "-")

        formatted_row = "| "
        for key in self.log_max_widths.keys():
            value = stat.get(key, "")
            if key == "loss":
                color_code = self.loss_color(float(value))
                formatted_value = f"{color_code}{value}\033[0m"
            else:
                formatted_value = str(value)
            formatted_row += formatted_value.ljust(self.log_max_widths[key]) + " | "

        log.info(formatted_row)
        log.info(len(formatted_row) * "-")


    def result_save(self, key: str, value: Any):
        assert self.result_get(key) is None, f"key: {key} already exists in `self.result`"
        self._results[key] = value

    def result_get(self, key: str, default: Any = None) -> Any:
        return self._results.get(key, default)

    def results(self):
        return self._results

    def collect_memory_info(self, layer_index: int):
        if self.logger_task is not None:
            gpu_memory = get_gpu_usage_memory()
            cpu_memory = get_cpu_usage_memory()
            self.logger_task.get_logger().report_scalar(
                title='GPU Memory',
                series='GPU Memory',
                value=gpu_memory,
                iteration=layer_index,
            )

            self.logger_task.get_logger().report_scalar(
                title='CPU Memory',
                series='CPU Memory',
                value=cpu_memory,
                iteration=layer_index,
            )
            self.gpu_memorys.append(gpu_memory)
            self.cpu_memorys.append(cpu_memory)

    def log_plotly(self):
        pass

    def set_calibration_dataset(self, calibration_dataset):
        pass



    def set_fwd_time(self, fwd_time: float):
        self.fwd_time = fwd_time

    def preprocess(self, module: NamedModule, **kwargs):
        pass

    def is_skipped(self, module: NamedModule) -> bool:
        pass

    def receive_input_cache(self, input_cache: InputCache):
        self.inputs_cache = input_cache

    def receive_layer_inputs(self, layer_inputs: List[List[Tensor]]):
        self.inputs_cache.layer_inputs = layer_inputs

    def clear_cache_data(self):
        self.tasks = {}
        self.inputs_cache.layer_inputs = []

    def preprocess_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        pass

    def process(self, module: NamedModule):
        pass

    def submodule_finalize(self, module: NamedModule):
        pass

    def finalize(self, model: BaseGPTQModel, **kwargs):
        del self.inputs_cache
        del self._results

        if self.log_worker is not None:
            self.log_worker_queue.put(False)

    def release_calibration_dataset(self):
        del self.calibration_dataset

    def number_batches(self) -> int:
        return self.num_batches

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        pass

    def name(self) -> str:
        pass