# KLAWQ: KL-Aware Weight Quantization for Efficient Large Language Models on Edge Devices

This repository contains the code for the paper "KLAWQ: KL-Aware Weight Quantization for Efficient Large Language Models on Edge Devices."

**Paper Abstract:**
Large Language Models (LLMs) exhibit remarkable capabilities but their substantial size and computational demands impede deployment on resource-constrained edge devices. Post-Training Quantization (PTQ) offers an efficient compression pathway, yet state-of-the-art methods like GPTQ, while effective, primarily optimize for weight reconstruction and can suffer accuracy degradation, especially at very low bit-widths. Knowledge Distillation (KD) aims to preserve functional fidelity but is typically a separate, costly training process. We propose KLAWQ (KL-Aware Weight Quantization), a novel PTQ framework that enhances GPTQ by directly integrating a Kullback-Leibler (KL) divergence objective into its second-order, layer-wise quantization procedure. By making the quantization process explicitly aware of the full-precision teacher model’s output distribution, KLAWQ seeks to preserve both structural weight fidelity and crucial functional characteristics. Experimental evaluation on the GPT-2 model at 8-bit precision demonstrates that KLAWQ can achieve improved perplexity compared to vanilla GPTQ through its KL-aware optimization. This approach offers a promising direction for developing more robust and accurate quantization techniques, facilitating the deployment of LLMs on edge platforms.

## Core Idea

KLAWQ extends the GPTQ post-training quantization algorithm. The primary modification involves incorporating a Kullback-Leibler (KL) divergence term into GPTQ's layer-wise, second-order quantization objective. This makes the weight quantization process aware of the full-precision teacher model's output distribution, aiming to preserve functional fidelity alongside weight reconstruction accuracy.

The KLAWQ objective (from Equation 6 in the paper) is:
`L(Q) = L_MSE(Q) + β * L_KL(Q) + γ * L_CE(Q)`

In the current implementation (and the experiments in the paper focusing on GPT-2 8-bit), `γ` is effectively zero, so the Cross-Entropy (CE) term is not used.

## Key Implementation Details

The KLAWQ logic is integrated into a modified version of the `gptqmodel` library. The core changes are located within the `KLAWQ/gptqmodel/` directory (relative to the repository root `Compression-Framework-for-EdgeAI/`).

1.  **KLAWQ Hyperparameters (`β`, `τ`):**
    *   Defined in `KLAWQ/gptqmodel/quantization/config.py`.
    *   The `QuantizeConfig` class is extended with `beta` (KL divergence strength) and `tau` (distillation temperature).
        ```python
        # In QuantizeConfig:
        beta: float = field(default=0.0, metadata={"help": "Strength of KL divergence term (beta=0 recovers vanilla GPTQ)"})
        tau: float = field(default=1.0, metadata={"help": "Temperature for softmax in KL divergence calculation (must be > 0)"})
        ```

2.  **KL Hessian Approximation (`A`) Calculation:**
    *   Implemented in `KLAWQ/kl-aware-quant//quantization/gptq.py` within the `GPTQ.process_batch` method.
    *   When `qcfg.beta > 0`, this section computes an approximation of the Hessian related to the KL divergence term. It uses the current layer's full-precision weights (`self.W_orig`) to get outputs, calculates soft probabilities (`pt`) using the configured `tau`, and then forms the matrix `A` (Equation 7 in the paper, with some approximations).
        ```python
        # Snippet from GPTQ.process_batch:
        if self.qcfg.beta > 0:
            if self.A is None: self.A = torch.zeros(...) # Initialize A if needed
            output = reshaped_inp @ self.W_orig.T # Get layer output with FP weights
            # Apply temperature scaling and softmax to get soft probabilities
            pt = F.softmax(output / self.qcfg.tau, dim=-1)
            # Approximate diagonal of (diag(pt) - pt*pt.T) part of KL Hessian
            kl_weights = torch.sum(pt * (1.0 - pt), dim=-1)
            # Form the batch contribution to A
            weighted_inp = reshaped_inp * torch.sqrt(kl_weights).unsqueeze(1)
            self.A.addmm_(weighted_inp.T, weighted_inp, beta=beta_scale, alpha=alpha_scale)
        ```

3.  **Combined Hessian (`H_tot`):**
    *   Formed in `KLAWQ/kl-aware-quant/quantization/gptq.py` at the beginning of the `GPTQ.quantize` method.
    *   The standard Hessian `H` (from reconstruction error) and the KL Hessian `A` are combined:
        ```python
        # Snippet from GPTQ.quantize:
        if self.qcfg.beta > 0 and self.A is not None:
            H_tot = self.H + self.qcfg.beta * self.A # H_tot = H + βA
        elif self.qcfg.beta > 0 and self.A is None:
             # Handle case where A might not have been computed
             H_tot = self.H
        else: # Vanilla GPTQ if beta is 0
            H_tot = self.H
        # ...
        H = H_tot.clone() # H now represents the (potentially KLAWQ-modified) Hessian
        ```

4.  **Quantization with Modified Hessian:**
    *   The combined and damped Hessian `(H'_tot)^-1` (referred to as `Hinv` in the code after `self.hessian_inverse(H)`) is then used for the column-wise quantization and error propagation steps. This follows the standard GPTQ procedure but utilizes the KLAWQ-modified Hessian. This occurs within `GPTQ.hessian_inverse` and the main quantization loop in `GPTQ.quantize`.

## Project Structure Highlight

The repository `Compression-Framework-for-EdgeAI` contains the KLAWQ project.

*   `Compression-Framework-for-EdgeAI/` (Repository Root)
    *   `KLAWQ`
        *   `Kl-aware-quant/`: This directory contains the modified `gptqmodel` library where KLAWQ is integrated.
            *   `quantization/`
                *   `config.py`: Defines the `QuantizeConfig` class, extended to include KLAWQ's `beta` and `tau` hyperparameters.
                *   `gptq.py`: Implements the core KLAWQ algorithm modifications within the GPTQ framework. This includes the KL Hessian (`A`) calculation, the combination `H_tot = H + βA`, and the subsequent use of this modified Hessian in the quantization process.
                *   `quantizer.py`: Contains standard quantization functions used by GPTQ.
            *   (Other subdirectories like `models/`, `looper/`, `utils/` are part of the base `gptqmodel` library.)
        *   `kl-hessian-gptq-analysis.ipynb`: A Jupyter notebook demonstrating KLAWQ quantization and evaluation on the GPT-2 model (text-based LLM). This serves as the primary example for understanding KLAWQ usage.
        *   `kl-hessian-gptq-main.ipynb`: A Jupyter notebook showcasing KLAWQ applied to a multi-modal model (Qwen-VL).
    *   `requirements.txt`: Lists necessary Python packages for the project.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ha405/Compression-Framework-for-EdgeAI
    cd Compression-Framework-for-EdgeAI
    ```

2.  **Install Dependencies:**
    Install the required packages using the `requirements.txt` file located in the repository root.
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include PyTorch, Transformers, Datasets, and NumPy.

3.  **Python Path Configuration (for Notebooks/Scripts):**
    To ensure that Python can import your KLAWQ-modified `gptqmodel` library from within the `KLAWQ` subdirectory, you need to add the `KLAWQ` directory to the Python path. This is typically done at the beginning of your Jupyter notebooks or Python scripts:
    ```python
    import sys
    import os

    # Assuming your script/notebook is run from the repository root ('Compression-Framework-for-EdgeAI')
    # or you provide an absolute path to the 'KLAWQ' directory.
    klawq_library_parent_path = os.path.abspath("./KLAWQ") # Path to the KLAWQ directory

    if klawq_library_parent_path not in sys.path:
        sys.path.insert(0, klawq_library_parent_path)
        print(f"Added '{klawq_library_parent_path}' to sys.path")

    # Now you can import from your modified library, e.g.:
    # from gptqmodel import GPTQModel, QuantizeConfig
    ```
    This allows `from gptqmodel import ...` to correctly import your KLAWQ-specific modifications.

## Running KLAWQ: Example (GPT-2 Quantization)

The following example is based on `KLAWQ/kl-hessian-gptq-analysis.ipynb` and shows how to quantize GPT-2 using KLAWQ. Ensure you have performed the setup steps, especially Python path configuration.

```python
import torch
from datasets import load_dataset
from itertools import islice
from transformers import AutoTokenizer
import os
import sys

# --- Python Path Setup (if not done elsewhere or for standalone script) ---
# This section should be adapted based on where your script is located
# relative to the 'KLAWQ' directory. If running a notebook from the repo root,
# the os.path.abspath("./KLAWQ") method shown in Setup is generally good.
# For this example, we'll assume it's handled as per the Setup section.
# klawq_library_parent_path = os.path.abspath("./KLAWQ") # Or your specific path
# if klawq_library_parent_path not in sys.path:
#    sys.path.insert(0, klawq_library_parent_path)
# Ensure this path is correct for your environment before running.

# Import from your KLAWQ-modified library
from gptqmodel import GPTQModel, QuantizeConfig

# 1. Model and Data Configuration
model_id = "gpt2"
quant_output_path = "./gpt2_klawq_8bit_g128" # Desired save path for quantized model

# KLAWQ-specific parameters
beta_klawq = 2.0  # Strength of KL divergence term
tau_klawq = 0.7   # Distillation temperature

# 2. Load Calibration Data (e.g., a subset of C4)
num_calibration_samples = 1024
try:
    c4_stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
    calibration_texts_raw = [
        sample["text"] for sample in islice(c4_stream, num_calibration_samples) 
        if sample.get("text", "").strip()
    ]
except Exception as e:
    print(f"Failed to load C4 dataset via streaming: {e}. Using dummy calibration data.")
    calibration_texts_raw = [f"This is sample calibration sentence number {i+1}." for i in range(num_calibration_samples)]
print(f"Using {len(calibration_texts_raw)} samples for calibration.")

# 3. Initialize QuantizeConfig with KLAWQ parameters
quant_config = QuantizeConfig(
    bits=8,
    group_size=128,
    beta=beta_klawq,
    tau=tau_klawq
)

# 4. Load Model with QuantizeConfig
print(f"Loading model: {model_id} with KLAWQ config.")
model = GPTQModel.load(model_id, quant_config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 5. Prepare Calibration Data for the Model
max_len = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length is not None else 512
calibration_data_for_quantizer = []
print(f"Tokenizing {len(calibration_texts_raw)} calibration samples with max_length={max_len}...")
for text_sample in calibration_texts_raw:
    tokenized_sample = tokenizer(
        text_sample, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
    )
    calibration_data_for_quantizer.append({
        "input_ids": tokenized_sample["input_ids"].squeeze(0).tolist(),
        "attention_mask": tokenized_sample["attention_mask"].squeeze(0).tolist()
    })

if not calibration_data_for_quantizer:
    raise ValueError("No calibration data was prepared. Check data loading and tokenization.")

# 6. Run KLAWQ Quantization
print(f"Starting KLAWQ quantization for {model_id}...")
model.quantize(calibration_data_for_quantizer, batch_size=8)
print("KLAWQ quantization finished.")

# 7. Save Quantized Model
os.makedirs(os.path.dirname(quant_output_path), exist_ok=True)
model.save(quant_output_path)
print(f"KLAWQ-quantized model saved to {quant_output_path}")

# 8. (Optional) Evaluate the quantized model
# Refer to `KLAWQ/kl-hessian-gptq-analysis.ipynb` for detailed evaluation steps.

```
## Expected Outcome

As demonstrated in the paper, KLAWQ aims to achieve improved perplexity compared to vanilla GPTQ for models like GPT-2 at 8-bit precision. This is achieved by making the quantization process aware of the teacher model's output distribution through the integrated KL divergence term.

## Requirements
The necessary Python packages are listed in the requirements.txt file in the root of this repository (Compression-Framework-for-EdgeAI/requirements.txt).

