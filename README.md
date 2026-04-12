# Autoregressive-Protein-Modelling

# PAR: Protein Autoregressive Modeling via Multiscale Structure Generation

Unofficial PyTorch implementation of [PAR](https://arxiv.org/abs/2602.04883) (Qu et al., 2026, ByteDance Seed / UIUC).

> **Protein Autoregressive Modeling via Multiscale Structure Generation**  
> Yanru Qu*, Cheng-Yen Hsieh*, Zaixiang Zheng, Ge Liu, Quanquan Gu  
> arXiv:2602.04883 (Feb 2026)

---

## Overview

PAR is the **first multi-scale autoregressive framework** for protein backbone generation. It generates Cα backbone structures coarse-to-fine via next-scale prediction, analogous to sculpting a statue: first a rough topology, then progressively finer structural details.

### Key Components

1. **Multi-scale Downsampling** — Hierarchically downsamples protein structures into *n* coarse-to-fine scales during training (non-parametric, deterministic).
2. **Autoregressive Transformer** — Non-equivariant transformer that encodes prior scales and produces per-scale conditional embeddings `z^i`.
3. **Flow-based Backbone Decoder** — Flow matching network that generates Cα coordinates conditioned on `z^i`, operating directly in continuous space (no discretization).
4. **Exposure Bias Mitigation** — Noisy Context Learning (NCL) + Scheduled Sampling (SS) to bridge the training/inference gap.

### Highlights

- **Zero-shot generalization**: prompted generation (16 3D points → full backbone) and motif scaffolding without fine-tuning
- **Efficient sampling**: SDE at coarse scales + ODE at fine scales → **2.5× speedup**
- **Favorable scaling**: FPSD improves with model size and compute
- **96.6% designability** after PDB fine-tuning (400M model)

---

## Project Structure

```
par_protein/
├── models/
│   ├── downsampling.py       # Multi-scale downsampling (Eq. 2)
│   ├── ar_transformer.py     # Autoregressive transformer T_θ (Eq. 4)
│   ├── flow_decoder.py       # Flow-based backbone decoder v_θ (Eq. 5)
│   └── par.py                # Full PAR model
├── training/
│   ├── loss.py               # Flow matching objective (Eq. 5)
│   └── exposure_bias.py      # NCL + Scheduled Sampling (Sec. 3.3)
├── data/
│   └── dataset.py            # Protein backbone dataset utilities
├── utils/
│   ├── sampling.py           # ODE/SDE sampling (Eq. 6)
│   └── interpolation.py      # Upsampling / positional encoding helpers
├── configs/
│   ├── par_60m.yaml
│   ├── par_200m.yaml
│   └── par_400m.yaml
├── train.py
├── sample.py
└── README.md
```

---

## Installation

```bash
pip install torch torchvision einops
pip install biotite  # for secondary structure annotation (P-SEA)
# optional: pip install wandb  for experiment tracking
```

---

## Quick Start

### Training

```bash
python train.py --config configs/par_60m.yaml --data_dir /path/to/afdb_structures
```

### Unconditional Sampling

```bash
python sample.py --checkpoint checkpoints/par_400m.pt --lengths 50 100 150 200 250 --n_samples 100
```

### Prompted Generation (zero-shot)

```python
from models.par import PAR
import torch

model = PAR.from_pretrained("checkpoints/par_400m.pt")
prompt = torch.randn(1, 16, 3)   # 16-point coarse layout
structures = model.prompted_generation(prompt, target_length=200, n_samples=4)
```

### Zero-shot Motif Scaffolding

```python
motif_coords = torch.load("motif.pt")       # (M, 3) Cα coordinates
motif_mask   = torch.load("motif_mask.pt")  # (L,) boolean mask
structures   = model.motif_scaffolding(motif_coords, motif_mask, target_length=200, n_samples=4)
```

---

## Model Configurations

| Model   | AR (T_θ) | Decoder (v_θ) | Designability | FPSD vs PDB |
|---------|----------|---------------|---------------|-------------|
| PAR-60M | 60M      | 60M           | 85%           | ~280        |
| PAR-200M| 60M      | 200M          | 87%           | 252         |
| PAR-400M| 60M      | 400M          | 96%           | 231         |
| PAR-PDB | 60M      | 400M (FT)     | **96.6%**     | **161**     |

Default scale config: **S = {64, 128, 256}** (3 scales, defined by length).

---

## Training Details

- **Dataset**: AFDB representative set (~600K structures, pLDDT > 80, length 32–256)
- **Optimizer**: Adam, lr=1e-4, no warmup
- **Steps**: 200K (AFDB pre-training) + 5K (PDB fine-tuning)
- **Hardware**: 8× H100 GPUs, batch size 15/GPU
- **Sampling at eval**: 1000 ODE/SDE steps (PAR), γ=0.30 default

---

## Method Details

### Multi-scale Decomposition (Eq. 1–2)

```
x  (full backbone, L×3)
  └─ Down(x, 64)   → x¹  (64×3)
  └─ Down(x, 128)  → x²  (128×3)
  └─ Down(x, 256)  → x³  (256×3) = x  (if L≤256)
```
Downsampling is 1D linear interpolation along the sequence dimension.

### AR Transformer (Eq. 4)

Inputs are all prior scales upsampled to the current scale size, concatenated along the sequence dim:

```
z^i = T_θ( bos, Up(x¹, size(2)), ..., Up(x^{i-1}, size(i)) )
```

### Flow Matching Objective (Eq. 5)

```
L(θ) = E_x [ (1/n) Σ_i (1/size(i)) E_{t,ε} [ ||v_θ(x^i_t, t, z^i) - (x^i - ε)||² ] ]
```

where `x^i_t = t·x^i + (1-t)·ε`.

### SDE Sampling (Eq. 6)

```
dx_t = v_θ(x_t, t) dt + g(t) s_θ(x_t, t) dt + √(2g(t)γ) dW_t
```

Score function: `s_θ(x_t, t) = (t·v_θ(x_t,t) - x_t) / (1-t)`

---

## Citation

```bibtex
@article{qu2026par,
  title   = {Protein Autoregressive Modeling via Multiscale Structure Generation},
  author  = {Qu, Yanru and Hsieh, Cheng-Yen and Zheng, Zaixiang and Liu, Ge and Gu, Quanquan},
  journal = {arXiv preprint arXiv:2602.04883},
  year    = {2026}
}
```
