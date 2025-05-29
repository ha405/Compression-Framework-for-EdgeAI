import copy
import time
from typing import List

import torch

from ..looper.gptq_processor import GPTQProcessor
from ..looper.input_cache import InputCache
from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseGPTQModel
from ..models._const import SUPPORTS_MODULE_TYPES
from ..nn_modules.hooked_linear import replace_module_with_hooked_legacy, replace_module_with_hooked_tree
from ..quantization.gptq import CPU
from ..utils.logger import setup_logger
from ..utils.model import (find_modules, get_device, get_module, get_module_by_name_prefix,
                           get_moe_layer_modules, move_to, nested_move_to)
from ..utils.torch import torch_empty_cache

log = setup_logger()

class ModuleLooper():
    def __init__(self, model: BaseGPTQModel, processors: List[LoopProcessor]):
        self.processors = processors
        self.gptq_model = model
        self.support_batch_quantize = model.support_batch_quantize

    def cache_inputs(self, layers, auto_gc, calibration_data, calibration_enable_gpu_cache):
        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []

        cur_layer_device = get_device(layers[0])
        data_device = cur_layer_device if calibration_enable_gpu_cache else CPU

        def store_input_hook(_, args, kwargs):
            layer_input = []
            for inp in args:
                layer_input.append(move_to(inp, device=data_device))
            if len(layer_input) == 0:
                if kwargs.get("hidden_states") is not None:
                    layer_input.append(move_to(kwargs["hidden_states"], device=data_device))

            layer_inputs.append(layer_input)

            if kwargs.get("attention_mask") is not None:
                attention_masks.append(kwargs["attention_mask"].to(device=data_device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to(pos_ids, device=data_device))
            one_kwargs = {}
            for (k, v) in kwargs.items():
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to(v, device=data_device)
            layer_input_kwargs.append(one_kwargs)

            raise ValueError

        layers[0] = layers[0].to(self.gptq_model.quantize_config.device)
        ori_outside_layer_module_devices = {}
        for module_name in self.gptq_model.base_modules:
            module, _ = get_module_by_name_prefix(self.gptq_model.model, [module_name])

            if module is None:
                continue

            ori_outside_layer_module_devices[module_name] = get_device(module)
            if module is not None:
                move_to(module, cur_layer_device)
        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
        is_ovis = self.gptq_model.__class__.__name__ == "OvisGPTQ"
        self.gptq_model.pre_quantize_generate_hook_start()
        for example in calibration_data:
            for k, v in example.items():
                data_device = self.gptq_model.quantize_config.device if k == "pixel_values" else cur_layer_device
                if isinstance(v, list):
                    for index in range(len(v)):
                        if len(v[index].shape) == 1:
                            v[index] = v[index].unsqueeze(0)
                        v[index] = move_to(v[index].to(self.gptq_model.model.visual_tokenizer.dtype) if is_ovis else v[index],
                                                  device=data_device)
                else:
                    if len(v.shape) == 1:
                        v = v.unsqueeze(0)
                    example[k] = move_to(v, device=data_device)
            try:
                self.gptq_model.model(**example)
            except ValueError:
                pass
        self.gptq_model.pre_quantize_generate_hook_end()
        handle.remove()
        move_to(layers[0], device=CPU)
        for module_name in self.gptq_model.base_modules:
            module, _ = get_module_by_name_prefix(self.gptq_model.model, [module_name])
            if module is not None:
                move_to(module, device=ori_outside_layer_module_devices[module_name])
        if auto_gc:
            torch_empty_cache()
        return InputCache(layer_inputs=layer_inputs, layer_input_kwargs=layer_input_kwargs, position_ids=position_ids,
                          attention_masks=attention_masks)

    @torch.no_grad()
    def loop(self, auto_gc=True, calibration_enable_gpu_cache=True, buffered_fwd=False, **kwargs):
        if self.gptq_model.quantize_config.lm_head:
            if self.gptq_model.model.config.tie_word_embeddings and hasattr(self.gptq_model.model.model, "_tied_weights_keys"):
                tied_keys = self.gptq_model.model._tied_weights_keys
                for item in tied_keys:
                    if self.gptq_model.lm_head in item:
                        raise NotImplementedError("quantization of `lm_head` layer with `tied_weights=True` model state is not supported. Please check model has `tied_weights=False`.")

            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if get_module(self.gptq_model.model, key=self.gptq_model.lm_head) is None:
                raise ValueError(f"could not find layer {self.gptq_model.lm_head} in the model, exit...")

            if not isinstance(lm_head_module, tuple(SUPPORTS_MODULE_TYPES)):
                raise NotImplementedError(f"This type({type(lm_head_module)}) of lm_head quantization is currently not "
                                          f"supported. SUPPORTS_MODULE_TYPES is {SUPPORTS_MODULE_TYPES}")

            lm_head_quant_config = {"bits": 8, "group_size": 32, "sym": True, "desc_act": False, "mse": 2.4}
            if self.gptq_model.quantize_config.dynamic is None:
                self.gptq_model.quantize_config.dynamic = {self.gptq_model.lm_head: lm_head_quant_config}
            elif self.gptq_model.quantize_config.dynamic_get(self.gptq_model.lm_head, default=None) is None:
                self.gptq_model.quantize_config.dynamic[self.gptq_model.lm_head] = lm_head_quant_config

        forward_pass_use_cache = self.gptq_model.model.config.use_cache if hasattr(self.gptq_model.model.config, "use_cache") else False
        self.gptq_model.model.config.use_cache = False

        layers, layers_prefix = get_module_by_name_prefix(self.gptq_model.model, self.gptq_model.layers_node)

        for p_index, processor in enumerate(self.processors):
            if not processor.verify_calibration_dataset(p_index):
                if isinstance(processor, GPTQProcessor):
                    prev_processor = self.processors[p_index - 1]
                    processor.set_calibration_dataset(copy.copy(prev_processor.calibration_dataset))
                    processor.receive_input_cache(copy.copy(prev_processor.inputs_cache))
                continue

            input_cache = self.cache_inputs(layers=layers, auto_gc=auto_gc,
                                            calibration_data=processor.calibration_dataset,
                                            calibration_enable_gpu_cache=calibration_enable_gpu_cache)
            processor.receive_input_cache(input_cache)

        for processor in self.processors:
            processor.release_calibration_dataset()

        layer_modules = self.gptq_model.layer_modules

        if not self.gptq_model.quantize_config.true_sequential:
            layer_modules = [sum(layer_modules, [])]

        if self.gptq_model.dynamic_expert_index is not None:
            num_experts = getattr(self.gptq_model.model.config, self.gptq_model.dynamic_expert_index)
            layer_modules = get_moe_layer_modules(layer_modules=self.gptq_model.layer_modules,
                                                  num_experts=num_experts)

        layer_count = len(layers)
        quant_modules_pb = (log.pb(layer_count + 1 if self.gptq_model.quantize_config.lm_head else layer_count)
                            .manual()
                            .set(left_steps_offset=1))

        for processor in self.processors:
            processor.layer_count = layer_count
            processor.pb = quant_modules_pb

        shared_kv_cache_dict = {}

        if self.gptq_model.layers_modules_tree:
            replace_module_with_hooked_tree(self.gptq_model.model, self.gptq_model.layers_modules_tree, debug=False)
        else:
            replace_module_with_hooked_legacy(self.gptq_model.model)

        for layer_index in quant_modules_pb:
            is_lm_head_module = layer_index >= layer_count

            if is_lm_head_module:
                quant_modules_pb.title("Quantizing lm_head").draw()
                module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
                layer_inputs = self.gptq_model.lm_head_pre_quantize_generate_hook(layer_inputs)
            else:
                quant_modules_pb.title(f"Quantizing layer {layer_index} of {layer_count - 1}").draw()
                module = layers[layer_index]

            if module.__class__.__name__.lower() == "MllamaCrossAttentionDecoderLayer".lower():
                continue

            self.gptq_model.pre_quantize(module)

            cur_layer_device = get_device(module)
            full = find_modules(module, name=self.gptq_model.lm_head if is_lm_head_module else "")

            for p_index, processor in enumerate(self.processors):
                processor.log_call_count = 0
                processor.collect_memory_info(layer_index)

                layer_inputs = processor.inputs_cache.layer_inputs
                layer_input_kwargs = processor.inputs_cache.layer_input_kwargs
                position_ids = processor.inputs_cache.position_ids
                attention_masks = processor.inputs_cache.attention_masks

                processed_subset = {}

                modules_to_process = [[self.gptq_model.lm_head]] if is_lm_head_module else layer_modules

                if processor.fwd_all_modules_in_single_pass:
                    modules_to_process = [sum(modules_to_process, [])]

                for index, names in enumerate(modules_to_process):
                    subset = {}
                    for n in names:
                        if n in full:
                            subset[n] = full[n]
                        elif self.gptq_model.layer_modules_strict:
                            raise ValueError(f"layer module item `{n}` not found in model, please check your model config.")

                    skipped_modules = []

                    for name in subset:
                        layer_name = self.gptq_model.lm_head if is_lm_head_module else f"{layers_prefix}.{layer_index}.{name}"

                        if not isinstance(subset[name], NamedModule):
                            named_module = NamedModule(subset[name], name=name, full_name=layer_name,
                                                      layer_index=layer_index)
                            subset[name] = named_module
                            full[name] = named_module

                        processor.preprocess(subset[name], buffered_fwd=buffered_fwd)
                        if processor.is_skipped(subset[name]):
                            skipped_modules.append(name)

                    for name in skipped_modules:
                        subset.pop(name)

                    if len(subset) == 0:
                        continue

                    handle = []
                    for name in subset:
                        if hasattr(subset[name], 'forward_hook'):
                            subset[name].forward_hook = processor.preprocess_fwd_hook(name)
                        else:
                            assert (f"forward_hook missing for module name: `{name}`, layer name: {layer_name}")
                            handle.append(subset[name].register_forward_hook(processor.preprocess_fwd_hook(name)))

                    fwd_start = time.time()
                    layer_outputs = []
                    for j in range(processor.num_batches):
                        layer_input = []
                        for k, layer_inp in enumerate(layer_inputs[j]):
                            layer_input.append(move_to(layer_inp, device=cur_layer_device))

                        mask = attention_masks[j]
                        layer_attention_mask = mask if mask is None else move_to(mask, device=cur_layer_device)

                        additional_layer_inputs = {"attention_mask": layer_attention_mask} if self.support_batch_quantize else {}
                        layer_position_ids = (
                            None if not position_ids else move_to(position_ids[j], device=cur_layer_device)
                        )
                        if layer_position_ids is not None:
                            additional_layer_inputs["position_ids"] = layer_position_ids
                        for k, v in layer_input_kwargs[j].items():
                            additional_layer_inputs[k] = nested_move_to(v, device=cur_layer_device)

                        if hasattr(module, "reuse_kv"):
                            if module.reuse_kv:
                                additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(
                                    layer_index - 1)

                            layer_output = module(*layer_input) if is_lm_head_module else module(*layer_input,
                                                                                                 **additional_layer_inputs)
                            if shared_kv_cache_dict.get(layer_index) is None:
                                shared_kv_cache_dict[layer_index] = layer_output[-1]
                        else:
                            layer_output = module(*layer_input) if is_lm_head_module else module(*layer_input,
                                                                                  **additional_layer_inputs)

                        if not processor.fwd_after_process:
                            layer_outputs.append([layer_output[0]])

                        del layer_input
                        del additional_layer_inputs

                    if not processor.fwd_after_process:
                        processor.receive_layer_inputs(layer_outputs)
                        del layer_outputs

                    fwd_end = time.time()
                    fwd_time = fwd_end - fwd_start

                    processor.set_fwd_time(fwd_time)

                    for h in handle:
                        h.remove()

                    for name in subset:
                        if hasattr(subset[name], 'forward_hook'):
                            subset[name].forward_hook = None

                    if isinstance(processor, GPTQProcessor):
                        moe_skip_modules = []
                        for name in subset :
                            if processor.tasks[name].fwd_counter == 0:
                                log.error(f"`{name}` was not invoked, if it is a MoE module, it may lack sufficient calibration data routed to it.")
                                moe_skip_modules.append(name)

                        for name in moe_skip_modules:
                            subset.pop(name)
                    
                    for name_index, name in enumerate(subset):
                        m = subset[name]
                        processor.process(module=m, auto_gc=auto_gc)
                        processed_subset[name] = m

                    if index == len(modules_to_process) - 1:
                        if auto_gc:
                            torch_empty_cache()

                is_last_module = layer_index == len(quant_modules_pb) - 1
                if not is_last_module and processor.fwd_after_process:
                    layer_outputs = []
                    for j in range(processor.num_batches):
                        layer_input = []
                        for k, layer_inp in enumerate(layer_inputs[j]):
                            layer_input.append(move_to(layer_inp, device=cur_layer_device))

                        mask = attention_masks[j]
                        layer_attention_mask = mask if mask is None else move_to(mask, device=cur_layer_device)

                        additional_layer_inputs = {"attention_mask": layer_attention_mask} if self.support_batch_quantize else {}
                        layer_position_ids = None if not position_ids else move_to(position_ids[j], device=cur_layer_device)
                        if layer_position_ids is not None:
                            additional_layer_inputs["position_ids"] = layer_position_ids
                        for k, v in layer_input_kwargs[j].items():
                            additional_layer_inputs[k] = nested_move_to(v, device=cur_layer_device)

                        if hasattr(module, "reuse_kv"):
                            if module.reuse_kv:
                                additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(layer_index - 1)

                        layer_output = move_to(
                            module(*layer_input)[0] if is_lm_head_module else
                            module(*layer_input, **additional_layer_inputs)[0],
                            device=cur_layer_device if calibration_enable_gpu_cache else CPU,
                        )
                        layer_outputs.append([layer_output])

                        del layer_input
                        del additional_layer_inputs
                        if processor.num_batches > 1 and j == processor.num_batches - 1:
                            if auto_gc:
                                torch_empty_cache()

                if p_index == len(self.processors) - 1:
                    if not is_lm_head_module:
                        layers[layer_index] = self.gptq_model.post_quantize(module)
                    else:
                        self.gptq_model.post_quantize(module)

                if processor.fwd_after_process:
                    processor.clear_cache_data()
                    processor.receive_layer_inputs(layer_outputs)

                if p_index == len(self.processors) - 1:
                    for reverse_p in reversed(self.processors):
                        for name in processed_subset:
                            reverse_p.submodule_finalize(processed_subset[name])
                    del module

                if auto_gc:
                    torch_empty_cache()

        total_log = {}

        for reverse_p in reversed(self.processors):
            if isinstance(reverse_p, GPTQProcessor):
                pass
            
            processor_name = reverse_p.name()
            total_log[processor_name] = reverse_p.log
            if processor_name in ["gptq"]:
                self.gptq_model.quant_log = reverse_p.log

            for module_log in reverse_p.log:
                log.info(module_log)
            reverse_p.log_plotly()

            reverse_p.finalize(model=self.gptq_model, **kwargs)

        self.gptq_model.model.config.use_cache = forward_pass_use_cache


        if auto_gc:
            torch_empty_cache()

        return total_log