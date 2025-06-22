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
        if hasattr(self.gptq_model, 'base_modules') and self.gptq_model.base_modules:
            for module_name in self.gptq_model.base_modules:
                module, _ = get_module_by_name_prefix(self.gptq_model.model, [module_name])
                if module is None:
                    continue
                ori_outside_layer_module_devices[module_name] = get_device(module)
                if module is not None:
                    move_to(module, cur_layer_device)
        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
        
        self.gptq_model.pre_quantize_generate_hook_start()
        for example in calibration_data:
            for k, v in example.items():
                if isinstance(v, torch.Tensor):
                    example[k] = move_to(v, device=cur_layer_device)
                elif isinstance(v, list) and all(isinstance(i, torch.Tensor) for i in v):
                    example[k] = [move_to(item, device=cur_layer_device) for item in v]
            try:
                self.gptq_model.model(**example)
            except ValueError:
                pass
        self.gptq_model.pre_quantize_generate_hook_end()
        handle.remove()
        move_to(layers[0], device=CPU)
        if hasattr(self.gptq_model, 'base_modules') and self.gptq_model.base_modules:
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
        is_hf_model = getattr(self.gptq_model, "is_hf_model", True)

        if is_hf_model and self.gptq_model.quantize_config.lm_head:
            if hasattr(self.gptq_model.model, 'config') and hasattr(self.gptq_model.model.config, 'tie_word_embeddings') and self.gptq_model.model.config.tie_word_embeddings and hasattr(self.gptq_model.model, "_tied_weights_keys"):
                tied_keys = self.gptq_model.model._tied_weights_keys
                for item in tied_keys:
                    if self.gptq_model.lm_head in item:
                        raise NotImplementedError("quantization of `lm_head` layer with `tied_weights=True` model state is not supported.")
            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if get_module(self.gptq_model.model, key=self.gptq_model.lm_head) is None:
                raise ValueError(f"could not find layer {self.gptq_model.lm_head} in the model, exit...")
            if not isinstance(lm_head_module, tuple(SUPPORTS_MODULE_TYPES)):
                raise NotImplementedError(f"This type({type(lm_head_module)}) of lm_head quantization is currently not supported.")
            lm_head_quant_config = {"bits": 8, "group_size": 32, "sym": True, "desc_act": False, "mse": 2.4}
            if self.gptq_model.quantize_config.dynamic is None:
                self.gptq_model.quantize_config.dynamic = {self.gptq_model.lm_head: lm_head_quant_config}
            elif self.gptq_model.quantize_config.dynamic_get(self.gptq_model.lm_head, default=None) is None:
                self.gptq_model.quantize_config.dynamic[self.gptq_model.lm_head] = lm_head_quant_config
        
        if is_hf_model:
            forward_pass_use_cache = self.gptq_model.model.config.use_cache
            self.gptq_model.model.config.use_cache = False
        else:
            forward_pass_use_cache = False

        if hasattr(self.gptq_model, "get_layers") and self.gptq_model.get_layers is not None:
            layers_with_names = self.gptq_model.get_layers(self.gptq_model.model)
            layers = [module for _, module in layers_with_names]
        else:
            layers, layers_prefix = get_module_by_name_prefix(self.gptq_model.model, self.gptq_model.layers_node)
            layers_with_names = [(f"{layers_prefix}.{i}", layer) for i, layer in enumerate(layers)]

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
        
        if is_hf_model and self.gptq_model.dynamic_expert_index is not None:
            num_experts = getattr(self.gptq_model.model.config, self.gptq_model.dynamic_expert_index)
            layer_modules = get_moe_layer_modules(layer_modules=self.gptq_model.layer_modules, num_experts=num_experts)

        layer_count = len(layers)
        quant_modules_pb = (log.pb(layer_count + 1 if (is_hf_model and self.gptq_model.quantize_config.lm_head) else layer_count)
                            .manual().set(left_steps_offset=1))

        for processor in self.processors:
            processor.layer_count = layer_count
            processor.pb = quant_modules_pb

        shared_kv_cache_dict = {}
        if self.gptq_model.layers_modules_tree:
            replace_module_with_hooked_tree(self.gptq_model.model, self.gptq_model.layers_modules_tree, debug=False)
        else:
            replace_module_with_hooked_legacy(self.gptq_model.model)

        for layer_index, (layer_name, module) in enumerate(layers_with_names):
            quant_modules_pb.next()
            is_lm_head_module = is_hf_model and self.gptq_model.quantize_config.lm_head and layer_index >= layer_count
            if is_lm_head_module:
                quant_modules_pb.title("Quantizing lm_head").draw()
                module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            else:
                quant_modules_pb.title(f"Quantizing layer {layer_name}").draw()
            
            self.gptq_model.pre_quantize(module)
            cur_layer_device = get_device(module)

            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                full = {layer_name: module}
                modules_to_process_names = [[layer_name]]
            else:
                full = find_modules(module, name=self.gptq_model.lm_head if is_lm_head_module else "")
                modules_to_process_names = [[self.gptq_model.lm_head]] if is_lm_head_module else layer_modules

            for p_index, processor in enumerate(self.processors):
                processor.log_call_count = 0
                processor.collect_memory_info(layer_index)
                layer_inputs = processor.inputs_cache.layer_inputs
                layer_input_kwargs = processor.inputs_cache.layer_input_kwargs
                position_ids = processor.inputs_cache.position_ids
                attention_masks = processor.inputs_cache.attention_masks
                processed_subset = {}
                
                current_modules_to_process = modules_to_process_names
                if processor.fwd_all_modules_in_single_pass and not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                     current_modules_to_process = [sum(current_modules_to_process, [])]
                
                for index, names in enumerate(current_modules_to_process):
                    subset = {}
                    for n in names:
                        if n in full:
                            subset[n] = full[n]
                        elif not self.gptq_model.layer_modules_strict:
                            continue
                        else:
                            raise ValueError(f"layer module item `{n}` not found in model, please check your model config.")

                    skipped_modules = []
                    for name in subset:
                        submodule_instance = subset[name]
                        if not isinstance(submodule_instance, NamedModule):
                            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                                full_submodule_name = name
                            else:
                                full_submodule_name = f"{layer_name}.{name}"
                            short_name_for_hook = name.split('.')[-1]
                            named_module = NamedModule(submodule_instance, name=short_name_for_hook,
                                                      full_name=full_submodule_name, layer_index=layer_index)
                            subset[name] = named_module
                        
                        processor.preprocess(subset[name], buffered_fwd=buffered_fwd)
                        if processor.is_skipped(subset[name]):
                            skipped_modules.append(name)
                    
                    for name in skipped_modules:
                        subset.pop(name)

                    if len(subset) == 0:
                        continue

                    handle = []
                    for name in subset:
                        hook_name = subset[name].name
                        if hasattr(subset[name], 'forward_hook'):
                            subset[name].forward_hook = processor.preprocess_fwd_hook(hook_name)
                        else:
                            handle.append(subset[name].register_forward_hook(processor.preprocess_fwd_hook(hook_name)))
                    
                    fwd_start = time.time()
                    layer_outputs = []
                    for j in range(processor.num_batches):
                        layer_input = []
                        for k, layer_inp in enumerate(layer_inputs[j]):
                            layer_input.append(move_to(layer_inp, device=cur_layer_device))

                        mask = attention_masks[j] if attention_masks and j < len(attention_masks) else None
                        layer_attention_mask = mask if mask is None else move_to(mask, device=cur_layer_device)
                        additional_layer_inputs = {"attention_mask": layer_attention_mask} if (is_hf_model and self.support_batch_quantize) else {}
                        layer_position_ids = (None if not position_ids or not j < len(position_ids) else move_to(position_ids[j], device=cur_layer_device))
                        if layer_position_ids is not None:
                            additional_layer_inputs["position_ids"] = layer_position_ids
                        if layer_input_kwargs and j < len(layer_input_kwargs):
                            for k, v in layer_input_kwargs[j].items():
                                additional_layer_inputs[k] = nested_move_to(v, device=cur_layer_device)

                        layer_output = module(*layer_input, **additional_layer_inputs)
                        if not processor.fwd_after_process:
                            output_to_store = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                            layer_outputs.append([output_to_store])
                        del layer_input, additional_layer_inputs

                    if not processor.fwd_after_process:
                        processor.receive_layer_inputs(layer_outputs)
                        del layer_outputs

                    fwd_end = time.time()
                    processor.set_fwd_time(fwd_end - fwd_start)

                    for h in handle:
                        h.remove()
                    for name in subset:
                        if hasattr(subset[name], 'forward_hook'):
                            subset[name].forward_hook = None

                    if isinstance(processor, GPTQProcessor):
                        moe_skip_modules = []
                        for name in subset:
                            if processor.tasks[name].fwd_counter == 0:
                                log.error(f"`{name}` was not invoked, if it is a MoE module, it may lack sufficient calibration data routed to it.")
                                moe_skip_modules.append(name)
                        for name in moe_skip_modules:
                            subset.pop(name)
                    
                    for name_index, name in enumerate(subset):
                        m = subset[name]
                        processor.process(module=m, auto_gc=auto_gc)
                        processed_subset[name] = m

                    if index == len(current_modules_to_process) - 1 and auto_gc:
                        torch_empty_cache()

                is_last_module = layer_index == len(layers_with_names) - 1
                if not is_last_module and processor.fwd_after_process:
                    layer_outputs = []
                    for j in range(processor.num_batches):
                        layer_input = []
                        for k, layer_inp in enumerate(layer_inputs[j]):
                            layer_input.append(move_to(layer_inp, device=cur_layer_device))
                        
                        mask = attention_masks[j] if attention_masks and j < len(attention_masks) else None
                        layer_attention_mask = mask if mask is None else move_to(mask, device=cur_layer_device)
                        additional_layer_inputs = {"attention_mask": layer_attention_mask} if (is_hf_model and self.support_batch_quantize) else {}
                        layer_position_ids = None if not position_ids or not j < len(position_ids) else move_to(position_ids[j], device=cur_layer_device)
                        if layer_position_ids is not None:
                            additional_layer_inputs["position_ids"] = layer_position_ids
                        if layer_input_kwargs and j < len(layer_input_kwargs):
                            for k, v in layer_input_kwargs[j].items():
                                additional_layer_inputs[k] = nested_move_to(v, device=cur_layer_device)
                        
                        output_result = module(*layer_input, **additional_layer_inputs)
                        output_to_store = output_result[0] if isinstance(output_result, tuple) else output_result
                        layer_output = move_to(output_to_store, device=cur_layer_device if calibration_enable_gpu_cache else CPU)
                        layer_outputs.append([layer_output])
                        del layer_input, additional_layer_inputs
                        if processor.num_batches > 1 and j == processor.num_batches - 1 and auto_gc:
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
            processor_name = reverse_p.name()
            total_log[processor_name] = reverse_p.log
            if processor_name in ["gptq"]:
                self.gptq_model.quant_log = reverse_p.log

            for module_log in reverse_p.log:
                log.info(module_log)
            reverse_p.log_plotly()
            reverse_p.finalize(model=self.gptq_model, **kwargs)
        
        if is_hf_model:
            self.gptq_model.model.config.use_cache = forward_pass_use_cache

        if auto_gc:
            torch_empty_cache()

        return total_log