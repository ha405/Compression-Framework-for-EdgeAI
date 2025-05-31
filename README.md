# GPTQModel

GPTQModel is a Python library designed to facilitate the quantization of large language models (LLMs) using the GPTQ algorithm. It offers tools for loading, quantizing, evaluating, and saving quantized models. This README provides a comprehensive guide to using the library, including installation instructions, basic usage examples, and detailed explanations of key features.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Loading a Model](#loading-a-model)
  - [Quantizing a Model](#quantizing-a-model)
  - [Evaluating a Model](#evaluating-a-model)
  - [Saving a Model](#saving-a-model)
  - [Pushing to Hub](#pushing-to-hub)
- [Quantization Configuration](#quantization-configuration)
- [Backend Support](#backend-support)
- [Logging](#logging)
- [Environment Variables](#environment-variables)
- [Memory Management](#memory-management)
- [Model Definitions](#model-definitions)
- [Known Issues and Limitations](#known-issues-and-limitations)
- [Contributing](#contributing)
- [License](#license)

## Features

- **GPTQ Quantization:** Implements the GPTQ algorithm for quantizing LLMs, reducing their size and improving inference speed.
- **Model Loading:** Supports loading models from Hugging Face Hub or local paths.
- **Evaluation:** Provides tools for evaluating model performance using standard benchmarks.
- **Saving:** Allows saving quantized models in a compact format.
- **Backend Flexibility:** Supports multiple backends, including `torch`, `triton`, `exllama`, `exllama_v2`, `marlin`, `bitblas`, and `ipex` for optimized performance on different hardware.
- **Configuration:** Offers a flexible quantization configuration system with dynamic per-module settings.
- **Logging:** Integrates with `logbar` for detailed logging and progress tracking.
- **Integration with External Inference Engines**: Seamlessly integrates with popular external inference engines like `vllm`, `sglang`, and `mlx` for accelerated deployment and inference.
- **Support for Mixture of Experts**: Easily quantize MoE models such as Mixtral, Qwen2-MoE, DeepSeek V2/V3 with dynamic expert indexing for proper routing of quantizable modules.

## Installation

```bash
pip install gptqmodel
```

For extended features and dependencies (like evaluation or specific loggers):

```bash
pip install gptqmodel[all]
```

or

```bash
pip install gptqmodel[eval]
```

## Usage

### Loading a Model

```python
from KLAWQ.quant.models import GPTQModel
from KLAWQ.quant.quantization import QuantizeConfig

model_id = "facebook/opt-125m" # Replace with your model ID

# Load a pre-trained model with a quantization configuration
model = GPTQModel.load(model_id)

# Alternatively, load the model with custom configurations
qconfig = QuantizeConfig(bits=4, group_size=128, desc_act=True)
model = GPTQModel.load(model_id, quantize_config=qconfig)
```

### Quantizing a Model

```python
from KLAWQ.quant.models import GPTQModel
from KLAWQ.quant.quantization import QuantizeConfig
from transformers import AutoTokenizer

model_id = "facebook/opt-125m"
model = GPTQModel.from_pretrained(model_id, quantize_config=qconfig)

# Prepare a dataset for calibration (replace with your data)
tokenizer = AutoTokenizer.from_pretrained(model_id)
calibration_data = ["This is a sample sentence.", "Another example for calibration."]

# Quantize the model using a calibration dataset
quant_log = model.quantize(calibration_data, batch_size=1)
```

### Evaluating a Model

```python
from KLAWQ.quant.models import GPTQModel
from KLAWQ.quant.utils import EVAL
from transformers import AutoTokenizer

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GPTQModel.load(model_id)

# Evaluate the model using a specified task
results = GPTQModel.eval(model_or_id_or_path=model, tokenizer=tokenizer, tasks=[EVAL.LM_EVAL.HELLASWAG])
print(results)
```

### Saving a Model

```python
from KLAWQ.quant.models import GPTQModel

model_id = "facebook/opt-125m"
model = GPTQModel.load(model_id)
output_dir = "path/to/save/quantized/model"

# Save the quantized model
model.save(output_dir)
```

### Pushing to Hub

```python
from KLAWQ.quant.models import GPTQModel

quantized_path = "path/to/save/quantized/model"
repo_id = "your_username/your_quantized_model" #replace with your actual repo name

# Push the quantized model to the Hugging Face Hub
GPTQModel.push_to_hub(repo_id, quantized_path, private=True, exists_ok=True)
```

## Quantization Configuration

The `QuantizeConfig` class allows you to specify various parameters for the quantization process.

```python
from KLAWQ.quant.quantization import QuantizeConfig

qconfig = QuantizeConfig(
    bits=4,                      # Quantization bits
    group_size=128,               # Group size for quantization
    desc_act=True,                 # Use descriptor-aware quantization
    sym=True,                      # Symmetric quantization
    damp_percent=0.01,             # Damping percentage for the Hessian inverse
    damp_auto_increment=0.0015,    # Increment value for damping
    dynamic=None,              # Per-module configuration (Dynamic Quantization)

    #Quant Method and Format parameters
    quant_method = 'gptq',
    format = 'gptq',
    pack_dtype = torch.int32,   # Data type to pack the quantized weights
    lm_head = False # whether to quantize lm_head

)
```

**Dynamic Quantization**

The `dynamic` attribute allows you to specify different quantization parameters for different modules in the model. This can be useful for optimizing the quantization process for specific layers.

```python
qconfig = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    sym=True,
    dynamic={
        "+:.*attention.*": {"bits": 8},  # Set attention layers to 8 bits
        "-:.*ffn.*": False           # Skip ffn layers
    }
)
```

**Custom Module Configuration with EXPERT_INDEX_PLACEHOLDER**
  
For mixture of experts models it can be neccesary to specify expert modules dynamically. This can be done using `EXPERT_INDEX_PLACEHOLDER` constant.

```python
from KLAWQ.quant.models._const import EXPERT_INDEX_PLACEHOLDER

qconfig = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    sym=True,
    dynamic={
        f"+:.*experts.{EXPERT_INDEX_PLACEHOLDER}.*": {"bits": 4},
    }
)
```

## Backend Support

GPTQModel supports multiple backends for different hardware/software configurations.

- `TORCH`: For compatibility and general purpose.
- `TRITON`: For faster execution on NVIDIA GPUs.
- `EXLLAMA_V1/V2`: Optimized for specific batch sizes.
- `MARLIN/MARLIN_FP16`: Experimental kernels for further performance gains.
- `BITBLAS`: Higher speed at the cost of pre-compilation.
- `IPEX`: Kernel Optimized for Intel XPU and CPU.
- `VLLM`, `SGLANG`, and `MLX`: for seamless integration with external inference engines.

Example specifying backend:

```python
model = GPTQModel.load(model_id, backend="triton")
```

## Logging

The library uses the `logbar` library for logging. You can control the logging level and output format using the standard Python logging configuration.

## Environment Variables

- `PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'`: Optimizes memory allocation for CUDA.
- `CUDA_DEVICE_ORDER='PCI_BUS_ID'`: Ensures correct device ordering.
- `GPTQMODEL_USE_MODELSCOPE='True'`: to download from modelscope instead of huggingface hub.
  - If used, the library will attempt to download from modelscope, install it using `pip install modelscope`.
- `DEBUG`: enable debug logs by setting env.

## Memory Management

- The library uses `torch_empty_cache()` to free up GPU memory during the quantization process.
- It is recommended to have sufficient GPU memory to load and quantize the model.

## Model Definitions

The library includes pre-defined configurations for various model architectures, such as:

- `LlamaGPTQ`
- `MistralGPTQ`
- `MixtralGPTQ`
- `QwenGPTQ`
- `Qwen2GPTQ`
- `PhiGPTQ`
- `DeepSeekV2GPTQ`
- `DeepSeekV3GPTQ`

You can extend the library to support additional model architectures by creating new model definition classes (see existing definitions in `KLAWQ/quant/models/definitions/`).

## Known Issues and Limitations

- Quantization process can be memory intensive. It might require a significant amount of VRAM.
- Not all models and configurations are supported. Check compatibility before using.
- Tied weights with `lm_head` are not fully supported with quantization.

## Contributing

Contributions to GPTQModel are welcome! To contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes and add appropriate tests.
4.  Submit a pull request with a clear description of your changes.

## License

GPTQModel is licensed under the Apache License 2.0.
```
