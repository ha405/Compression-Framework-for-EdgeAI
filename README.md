# KLAWQ: KL-Aware Weight Quantization for Efficient Large Language Models on Edge Devices

This repository contains the code for the paper "KLAWQ: KL-Aware Weight Quantization for Efficient Large Language Models on Edge Devices."

**Paper Abstract:**  
Large Language Models (LLMs) are powerful but their size and computational demands hinder deployment on edge devices. Post-Training Quantization (PTQ) offers efficient compression, but methods like GPTQ focus on weight reconstruction, potentially losing accuracy at low bit-widths. Knowledge Distillation (KD) preserves functional fidelity but is costly. KLAWQ enhances GPTQ by integrating a Kullback-Leibler (KL) divergence objective into its quantization process, aligning it with the full-precision model’s output distribution. Experiments on GPT-2 at 8-bit precision show improved perplexity over vanilla GPTQ.

## Core Idea
KLAWQ extends GPTQ by adding a KL divergence term to the quantization objective:  
`L(Q) = L_MSE(Q) + β * L_KL(Q) + γ * L_CE(Q)`  
In this implementation, `γ = 0`, focusing on MSE and KL terms.

## Key Implementation Details
- **Hyperparameters (`β`, `τ`):** Defined in `KLAWQ/gptqmodel/quantization/config.py`.
- **KL Hessian (`A`):** Computed in `KLAWQ/kl-aware-quant/quantization/gptq.py`.
- **Combined Hessian (`H_tot`):** Formed as `H_tot = H + βA` in `GPTQ.quantize`.
- **Quantization:** Uses the modified Hessian for column-wise quantization.

## Project Structure
- `Compression-Framework-for-EdgeAI/`
  - `KLAWQ/`
    - `Kl-aware-quant/`
      - `quantization/`
        - `config.py`
        - `gptq.py`
        - `quantizer.py`
    - `kl-hessian-gptq-analysis.ipynb`
    - `kl-hessian-gptq-main.ipynb`
  - `requirements.txt`

## Setup
1. Clone the repository:  
   ```bash
   git clone https://github.com/ha405/Compression-Framework-for-EdgeAI
   cd Compression-Framework-for-EdgeAI