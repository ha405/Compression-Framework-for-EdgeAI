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
            for k, v in kwargs.items():
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to(v, device=data_device)
            layer_input_kwargs.append(one_kwargs)
            raise ValueError

        layers[0] = layers[0].to(self.gptq_model.quantize_config.device)
        ori_outside_layer_module_devices = {}
        if hasattr(self.gptq_model, 'base_modules') and self.gptq_model.base_modules:
            for module_name in self.gptq_model.base_modules:
                module, _ = get_module_by_name_prefix(self.gptq_model.model, [module_name])
                if module is None: continue
                ori_outside_layer_module_devices[module_name] = get_device(module)
                if module is not None: move_to(module, cur_layer_device)
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
        if auto_gc: torch_empty_cache()
        return InputCache(layer_inputs=layer_inputs, layer_input_kwargs=layer_input_kwargs, position_ids=position_ids, attention_masks=attention_masks)

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
            
        main_processor = self.processors[0]
        calibration_data = main_processor.calibration_dataset
        if calibration_data is None:
            raise ValueError("Calibration dataset is not prepared. It must be loaded before calling loop.")
        
        model_device = self.gptq_model.quantize_config.device
        for i, data in enumerate(calibration_data):
            calibration_data[i] = nested_move_to(data, model_device)
        
        # Get all layers to be processed to set up the progress bar correctly
        base_module_names = self.gptq_model.base_modules if hasattr(self.gptq_model, 'base_modules') else []
        if hasattr(self.gptq_model, "get_layers") and self.gptq_model.get_layers is not None:
            layers_with_names = self.gptq_model.get_layers(self.gptq_model.model)
            layers = [module for _, module in layers_with_names]
        else:
            layers, layers_prefix = get_module_by_name_prefix(self.gptq_model.model, self.gptq_model.layers_node)
            layers_with_names = [(f"{layers_prefix}.{i}", layer) for i, layer in enumerate(layers)]
        
        total_steps = len(base_module_names) + len(layers_with_names)
        quant_modules_pb = (log.pb(total_steps).manual().set(left_steps_offset=1))
        for processor in self.processors:
            processor.pb = quant_modules_pb

        # A. DEDICATED LOOP FOR BASE_MODULES (conv1, fc)
        if base_module_names:
            log.info(f"Processing {len(base_module_names)} base modules first...")
            for module_name in base_module_names:
                quant_modules_pb.title(f"Quantizing base module: {module_name}").draw()
                module = get_module(self.gptq_model.model, module_name)
                if not module:
                    log.warning(f"Could not find base module: {module_name}")
                    quant_modules_pb.next()
                    continue
                
                inputs_cache = []
                data_device = model_device if calibration_enable_gpu_cache else CPU
                def store_input_hook(_, args, kwargs):
                    inputs_cache.append(args[0].to(data_device, non_blocking=True))
                    raise ValueError("Input captured for base module")

                handle = module.register_forward_pre_hook(store_input_hook, with_kwargs=True)
                for data in calibration_data:
                    try: self.gptq_model.model(**data)
                    except ValueError: pass
                handle.remove()

                named_module = NamedModule(module, name=module_name, full_name=module_name, layer_index=-1)
                for processor in self.processors:
                    if isinstance(processor, GPTQProcessor):
                        processor.clear_cache_data()
                        processor.receive_layer_inputs(inputs_cache)
                        processor.preprocess(named_module, buffered_fwd=buffered_fwd)
                        if not processor.is_skipped(named_module):
                            processor.process(named_module)
                            processor.submodule_finalize(named_module)
                
                quant_modules_pb.next()
                if auto_gc: torch_empty_cache()

        # B. ORIGINAL LOGIC FOR MAIN BLOCKS (layer1, layer2, etc.)
        for p_index, processor in enumerate(self.processors):
            if not processor.verify_calibration_dataset(p_index):
                if isinstance(processor, GPTQProcessor):
                    prev_processor = self.processors[p_index - 1]
                    processor.set_calibration_dataset(copy.copy(prev_processor.calibration_dataset))
                    processor.receive_input_cache(copy.copy(prev_processor.inputs_cache))
                continue
            input_cache = self.cache_inputs(layers=layers, auto_gc=auto_gc,
                                            calibration_data=main_processor.calibration_dataset,
                                            calibration_enable_gpu_cache=calibration_enable_gpu_cache)
            processor.receive_input_cache(input_cache)

        for processor in self.processors:
            processor.release_calibration_dataset()

        layer_modules = self.gptq_model.layer_modules
        if not self.gptq_model.quantize_config.true_sequential:
            layer_modules = [sum(layer_modules, [])]
        
        for processor in self.processors:
            processor.layer_count = len(layers_with_names)

        if self.gptq_model.layers_modules_tree:
            replace_module_with_hooked_tree(self.gptq_model.model, self.gptq_model.layers_modules_tree, debug=False)
        else:
            replace_module_with_hooked_legacy(self.gptq_model.model)

        log.info(f"Processing {len(layers_with_names)} main layer blocks...")
        for layer_index, (layer_name, module) in enumerate(layers_with_names):
            quant_modules_pb.title(f"Quantizing block {layer_name}").draw()
            
            self.gptq_model.pre_quantize(module)
            cur_layer_device = get_device(module)
            full = find_modules(module)

            for p_index, processor in enumerate(self.processors):
                processor.log_call_count = 0
                layer_inputs = processor.inputs_cache.layer_inputs
                layer_input_kwargs = processor.inputs_cache.layer_input_kwargs
                processed_subset = {}
                
                for index, names in enumerate(layer_modules):
                    subset = {n: full[n] for n in names if n in full}
                    skipped_modules = []
                    for name in list(subset.keys()):
                        if not isinstance(subset[name], NamedModule):
                            named_module = NamedModule(subset[name], name=name, full_name=f"{layer_name}.{name}", layer_index=layer_index)
                            subset[name] = named_module
                        
                        processor.preprocess(subset[name], buffered_fwd=buffered_fwd)
                        if processor.is_skipped(subset[name]):
                            skipped_modules.append(name)
                    
                    for name in skipped_modules: subset.pop(name)
                    if len(subset) == 0: continue

                    handle = []
                    for name in subset:
                        handle.append(subset[name].register_forward_hook(processor.preprocess_fwd_hook(subset[name].name)))
                    
                    for j in range(processor.num_batches):
                        layer_input_batch = [move_to(inp, device=cur_layer_device) for inp in layer_inputs[j]]
                        additional_kwargs = {k: nested_move_to(v, cur_layer_device) for k, v in layer_input_kwargs[j].items()}
                        module(*layer_input_batch, **additional_kwargs)
                    
                    for h in handle: h.remove()
                    
                    for name in subset:
                        processor.process(module=subset[name], auto_gc=auto_gc)
                        processed_subset[name] = subset[name]

                if not is_hf_model:
                     layer_outputs_next = []
                     for j in range(processor.num_batches):
                        layer_input_batch = [move_to(inp, device=cur_layer_device) for inp in layer_inputs[j]]
                        additional_kwargs = {k: nested_move_to(v, cur_layer_device) for k, v in layer_input_kwargs[j].items()}
                        output = module(*layer_input_batch, **additional_kwargs)
                        output_to_store = output[0] if isinstance(output, tuple) else output
                        layer_outputs_next.append([move_to(output_to_store, CPU if not calibration_enable_gpu_cache else cur_layer_device)])
                     processor.clear_cache_data()
                     processor.receive_layer_inputs(layer_outputs_next)

                if p_index == len(self.processors) - 1:
                    layers[layer_index] = self.gptq_model.post_quantize(module)
                    for reverse_p in reversed(self.processors):
                        for name in processed_subset:
                            reverse_p.submodule_finalize(processed_subset[name])
                    del module

            quant_modules_pb.next()
            if auto_gc: torch_empty_cache()

        for reverse_p in reversed(self.processors):
            reverse_p.finalize(model=self.gptq_model, **kwargs)
        
        if is_hf_model:
            self.gptq_model.model.config.use_cache = forward_pass_use_cache

        if auto_gc:
            torch_empty_cache()

        return {}