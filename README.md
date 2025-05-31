```markdown
# KLAWQ/quant

[![PyPI Version](https://img.shields.io/pypi/v/gptqmodel)](https://pypi.org/project/gptqmodel/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/gptqmodel)](https://pypi.org/project/gptqmodel/)

**KLAWQ/quant** is a high-performance model quantization library designed to enable efficient deployment of large language models (LLMs).  It offers state-of-the-art quantization techniques, with a focus on GPTQ (Generative Post-training Quantization), to reduce model size and accelerate inference, while maintaining acceptable accuracy.

## Features

*   **GPTQ Quantization:** Primary support for GPTQ algorithm, a leading quantization method for LLMs.
*   **Wide Model Support:**  A growing list of predefined configurations for popular model architectures (LLaMA, Mistral, Qwen, Phi, and others), simplifying the quantization process.
*   **Hardware Acceleration:** Optimized kernels for CUDA, XPU, MPS, and CPU (with IPEX) to leverage hardware acceleration.
*   **Dynamic Quantization:** Flexible quantization configurations, allowing for different quantization settings per module.
*   **Integration with Hugging Face Transformers:** Seamless integration with the Hugging Face Transformers library, enabling easy loading and saving of quantized models.
*   **Evaluation Tools**: Integrated tooling to easily evaluate performance after quantization with LM-Eval, EvalPlus and MMLU Pro.
*   **Logging and Monitoring**: Extensive logging capabilities for monitoring the quantization process, including layer-wise loss, memory usage, and quantization time.
*   **ClearML Integration**: Built-in support for ClearML for experiment tracking and management.

## Installation

```bash
pip install gptqmodel
```

For specific features or logging:

```bash
pip install gptqmodel[logger]  # For clearml support
pip install gptqmodel[eval] # For EvalPlus
```

## Usage

### Quantizing a Model

```python
from KLAWQ.quant.models import GPTQModel
from KLAWQ.quant.quantization import QuantizeConfig

# Load a model
model_id = "facebook/opt-125m"

# Define a quantization configuration
quantize_config = QuantizeConfig(bits=4, group_size=128, desc_act=True)

# Load and quantize the model
gptq_model = GPTQModel.from_pretrained(model_id, quantize_config)

# Perform quantization (calibration dataset required)
calibration_dataset = ["This is a sample sentence.", "Another example for calibration."]  # Replace with a real dataset
quant_log = gptq_model.quantize(calibration_dataset)

# Save the quantized model
gptq_model.save_quantized("path/to/save/quantized/model")
```

### Loading a Quantized Model

```python
from KLAWQ.quant.models import GPTQModel

# Load a quantized model
model_path = "path/to/save/quantized/model"
quantized_model = GPTQModel.load(model_path)

# Use the quantized model for inference
text = "Generate an introduction to large language models"
output = quantized_model.generate(text)
print(output)
```

### Evaluating a Quantized Model

```python
from KLAWQ.quant.models import GPTQModel
from KLAWQ.quant.utils.eval import EVAL

# Load the quantized model
model_path = "path/to/save/quantized/model"
quantized_model = GPTQModel.load(model_path)

# Evaluate on a specific task
results = GPTQModel.eval(
    model_or_id_or_path=quantized_model,
    tasks=[EVAL.LM_EVAL.HELLASWAG]
)

print(results)
```

## Repository Structure

```
KLAWQ/quant/
├── __init__.py         # Main entry point for the library
├── looper/             # Module looping and processing logic
│   ├── __init__.py
│   ├── dequantize_processor.py # Processor for dequantizing models.
│   ├── gptq_processor.py   # Processor for GPTQ quantization.
│   ├── input_cache.py      # Data structures for caching layer inputs.
│   ├── loop_processor.py   # Base class for loop processors.
│   └── module_looper.py    # Main loop logic for processing modules.
├── models/             # Definitions and auto-loading of quantized models
│   ├── __init__.py
│   ├── _const.py          # Constants used in model definitions.
│   ├── auto.py           # Auto-loading logic for GPTQ models.
│   ├── base.py           # Base class for GPTQ models.
│   ├── definitions/      # Model specific configurations
│   │   ├── __init__.py
│   │   ├── deepseek_v2.py  # Model definition for DeepSeek V2.
│   │   ├── deepseek_v3.py  # Model definition for DeepSeek V3.
│   │   ├── gpt2.py         # Model definition for GPT2.
│   │   ├── llama.py        # Model definition for LLaMA.
│   │   ├── mistral.py      # Model definition for Mistral.
│   │   ├── mixtral.py      # Model definition for Mixtral.
│   │   ├── mllama.py       # Model definition for MLlama.
│   │   ├── mobilellm.py    # Model definition for MobileLLM.
│   │   ├── phi.py          # Model definition for Phi.
│   │   ├── phi3.py         # Model definition for Phi-3 family
│   │   ├── phi4.py         # Model definition for Phi-4 family
│   │   ├── qwen.py         # Model definition for Qwen.
│   │   ├── qwen2.py        # Model definition for Qwen2.
│   │   ├── qwen2_5_vl.py    # Model definition for Qwen2.5-VL.
│   │   ├── qwen2_moe.py    # Model definition for Qwen2-MoE.
│   │   └── qwen2_vl.py     # Model definition for Qwen2-VL.
│   ├── loader.py         # Model loading logic
│   └── writer.py         # Model writing logic
├── nn_modules/         # Custom neural network modules (QLinear)
│   ├── __init__.py
│   ├── hooked_linear.py  # Hooked Linear modules for forwarding processing
│   └── qlinear/          # Quantized Linear layers
│       ├── __init__.py
│       ├── bitblas_target_detector.py # Helper file for detecting the correct bitblas target.
│       ├── torch.py        # PyTorch implementation of quantized linear layers.
│       └── utils.py        # Utility functions for quantized linear layers.
├── quantization/       # Quantization algorithms and configurations
│   ├── __init__.py
│   ├── config.py         # Configuration classes for quantization.
│   ├── gptq.py           # Implementation of the GPTQ quantization algorithm.
│   └── quantizer.py      # Base classes for quantizers.
├── utils/              # Utility functions (logger, data, model...)
│   ├── __init__.py
│   ├── backend.py        # Backend-related enums and utilities.
│   ├── calibration.py   # Helper functions for calibration process.
│   ├── data.py           # Data loading and processing utilities.
│   ├── device.py         # Device utilities (GPU memory usage).
│   ├── eval.py           # Evaluation utilities.
│   ├── evalplus.py        # EvalPlus evaluation setup.
│   ├── hf.py             # Hugging Face model helper functions.
│   ├── image.py          # Image processing utilities.
│   ├── importer.py       # Module importing and selection utilities.
│   ├── logger.py         # Logging setup.
│   ├── mmlupro.py        # Helper functions to run mmlu pro eval
│   ├── model.py          # Model manipulation utilities.
│   ├── plotly.py         # Plotly helpers.
│   ├── rocm.py           # ROCm-specific utilities.
│   ├── safetensor.py     # Utilities for working with safetensors.
│   ├── tensor.py         # Utilities for model parameters count and size calculation
│   ├── terminal.py       # Terminal utilities for size calculation
│   └── torch.py          # PyTorch-related utilities.
└── version.py          # Library version information
```

## Model Definitions
The `/KLAWQ/quant/models/definitions` directory contains model-specific quantization configurations that are dynamically loaded by the library. Each file in this directory defines the architecture and quantization settings for a supported model. Key elements typically found in these definitions:
* `base_modules`: Modules residing outside the primary layer structure (e.g., input embeddings, final normalization layers).
* `layers_node`: The location of the repeating model layers (e.g., decoder blocks).
* `layer_type`: The class of the repeating model layer.
* `layer_modules`: List of quantized operations within each repeating layer, specified by module name.
* `pre_lm_head_norm_module`:  The final normalization layer that is quantized before the `lm_head` layer.

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](link-to-contributing-guide) for guidelines.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
```
