# ECDP: Evidence-Calibrated Displacement Pyramids for Deformable Medical Image Registration

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Parameters](https://img.shields.io/badge/Parameters-3.232M-green.svg)]()
[![Inference Time](https://img.shields.io/badge/Inference-393.13ms-brightgreen.svg)]()

This repository provides the official implementation and optimal pretrained model weights for:
> **Evidence-Calibrated Displacement Pyramids for Deformable Medical Image Registration**

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Method Highlights](#-method-highlights)
- [Pretrained Checkpoints & Benchmark Results](#-pretrained-checkpoints--benchmark-results)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Quick Start Demo](#-quick-start-demo)
- [Python API Usage](#-python-api-usage)
- [Benchmarking & Evaluation](#-benchmarking--evaluation)
- [Training & Fine-Tuning](#-training--fine-tuning)
- [Citation](#-citation)
- [License](#-license)

---

## 🌟 Overview

Deformable medical image registration is central to image-guided radiotherapy, surgical navigation, and longitudinal disease monitoring. While recent deep learning frameworks demonstrate remarkable registration speed, fine-scale deformations frequently suffer from **Scale-Inconsistent Coordinate Drift (SICD)**: localized updates that improve apparent photometric similarity while inadvertently causing anatomical distortion or catastrophic topological folding.

**ECDP** addresses this challenge through:
1. **Multi-Scale Correlation-Driven Displacement Pyramids**: A 5-scale coarse-to-fine registration backbone driven by 3D local correlation volumes and recurrent ConvGRU velocity updates.
2. **Prospective Evidence-based Local Evidence Routing (PE-LER)**: Evaluates candidate residual updates on full-resolution local descriptor similarity prior to velocity compounding, dynamically arbitrating updates via signed identity routing:
   $$\Delta S = S_{\text{cand}} - S_{\text{base}}$$
   $$G = \text{clamp}(1 + \alpha \cdot g_{\text{error}} \cdot \tanh(\gamma \cdot \Delta S), 0, 2)$$
   where $\alpha = 0.20$ and $\gamma = 10.0$.
3. **Stationary Velocity Field (SVF) Integration**: Scaling-and-squaring Lie group integration guarantees diffeomorphic properties and preserves topological regularity.

---

## 📊 Pretrained Checkpoints & Benchmark Results

The pretrained model checkpoints can be downloaded from [Google Drive (Coming Soon)] or [GitHub Releases](https://github.com/goghfan/ECDP/releases), and placed into the corresponding `checkpoints/` directories:

| Dataset | Modality | Anatomy | Model | DSC (%) ↑ | $\text{Neg}(|J_\phi| \le 0)$ (%) ↓ | Parameters | Checkpoint Path |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Abdomen CT-CT** | CT | Abdomen (13 organs) | **ECDP (Ours)** | **52.56** | **0.0466** | 3.24M | `checkpoints/abdomen_ctct/ecdp_best.pt` |
| Abdomen CT-CT | CT | Abdomen (13 organs) | EDDP Baseline | 51.86 | 0.0320 | 3.23M | `checkpoints/abdomen_ctct/eddp_baseline.pt` |
| **IXI** | MRI-T1 | Brain (30 structures) | **ECDP (Ours)** | **75.34** | **0.00026** | 3.24M | `checkpoints/ixi/ecdp_best.pt` |
| IXI | MRI-T1 | Brain (30 structures) | EDDP Baseline | 74.80 | 0.00017 | 3.23M | `checkpoints/ixi/eddp_baseline.pt` |
| **LPBA40** | MRI-T1 | Brain (54 structures) | **ECDP (Ours)** | **73.84** | **0.0024** | 3.24M | `checkpoints/lpba40/ecdp_best.pt` |
| LPBA40 | MRI-T1 | Brain (54 structures) | EDDP Baseline | 73.20 | 0.0038 | 3.23M | `checkpoints/lpba40/eddp_baseline.pt` |

*Note: All checkpoints are clean PyTorch serialized files strictly verified against official benchmark numbers. See [checkpoints/README.md](checkpoints/README.md) for detailed download and placement instructions.*

---

## 📁 Repository Structure

```
ECDP/
├── configs/                  # Dataset and model configurations
│   ├── abdomen_ctct.yaml     # Abdomen CT-CT configuration
│   ├── ixi.yaml              # IXI brain MRI configuration
│   └── lpba40.yaml           # LPBA40 brain MRI configuration
├── checkpoints/              # Optimal pretrained model checkpoints
│   ├── abdomen_ctct/         # Best ECDP & EDDP baseline for Abdomen CT-CT
│   ├── ixi/                  # Best ECDP & EDDP baseline for IXI
│   └── lpba40/               # Best ECDP & EDDP baseline for LPBA40
├── ecdp/                     # Core ECDP library
│   ├── core/                 # Mathematical operations
│   │   ├── geometry.py       # Warp, compose_flows, resize_flow, scaling-and-squaring, Jacobian
│   │   ├── losses.py         # Local NCC loss, diffusion regularizer, Dice loss
│   │   └── metrics.py        # DSC, HD95, Jacobian summary statistics
│   ├── models/               # Neural network architectures
│   │   ├── backbone.py       # EDDP multi-scale recurrent correlation backbone
│   │   ├── peler.py          # PE-LER prospective evidence gating module
│   │   └── ecdp_model.py     # High-level ECDP and EDDP model interfaces
│   └── data/                 # Data loading utilities
│       └── dataset.py        # 3D medical volume dataset loader (supports .nii.gz, .npz)
├── scripts/                  # Standalone execution scripts
│   ├── demo_register.py      # Quick-start pairwise registration CLI
│   ├── evaluate.py           # Benchmark evaluation script
│   └── train.py              # End-to-end training and fine-tuning script
├── tests/                    # Unit test suite
│   └── test_inference.py     # Verifies parameter counts, forward passes, and weights
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation script
└── LICENSE                   # Apache 2.0 open-source license
```

---

## 🛠️ Installation
 
```bash
git clone https://github.com/goghfan/ECDP.git
cd ECDP

# Install dependencies
pip install -r requirements.txt

# Install ECDP package in editable mode
pip install -e .
```

### System Requirements
- Python >= 3.8
- PyTorch >= 2.0.0 (with CUDA support recommended)
- 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM (RTX 3090 / 4090 / A100 recommended)

---

## 🚀 Quick Start Demo

Register two 3D medical images with a single command line:

```bash
python scripts/demo_register.py \
    --config configs/abdomen_ctct.yaml \
    --checkpoint checkpoints/abdomen_ctct/ecdp_best.pt \
    --fixed path/to/fixed_volume.nii.gz \
    --moving path/to/moving_volume.nii.gz \
    --moving_seg path/to/moving_seg.nii.gz \
    --output_dir results/
```

Results are saved to `results/`:
- `warped_moving.nii.gz`: Deformed moving image aligned to the fixed target.
- `warped_seg.nii.gz`: Transformed segmentation mask (using nearest-neighbor interpolation).
- `deformation_flow_xyz.npy`: 3D deformation displacement field $[3, D, H, W]$ in voxel units.

---

## 💻 Python API Usage

```python
import torch
import yaml
from ecdp import ECDPModel, warp

# 1. Load configuration
with open("configs/abdomen_ctct.yaml") as f:
    cfg = yaml.safe_load(f)

# 2. Instantiate model and load pretrained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECDPModel(cfg["model"], alpha=cfg.get("alpha", 0.20), gain_scale=cfg.get("gamma", 10.0)).to(device)

state = torch.load("checkpoints/abdomen_ctct/ecdp_best.pt", map_location=device)
model.load_state_dict(state["model"], strict=True)
model.eval()

# 3. Perform registration
# Expected input shape: [B, 1, D, H, W], normalized to [0, 1]
fixed = torch.randn(1, 1, 160, 192, 160, device=device)
moving = torch.randn(1, 1, 160, 192, 160, device=device)

with torch.inference_mode():
    warped, flow_zyx, diagnostics = model(fixed, moving, return_diagnostics=True)
    flow_xyz = diagnostics["final_flow_xyz"]

print(f"Deformed image shape: {warped.shape}")
print(f"Displacement field shape: {flow_xyz.shape}")
```

---

## 🧪 Unit Tests

Run the test suite to verify parameter count, forward inference, and checkpoint loading:

```bash
python -m unittest tests/test_inference.py
```

Expected output:
```
ECDP trainable parameters: 3,240,661 (~3.24M)
EDDP trainable parameters: 3,232,420 (~3.23M)
Verified checkpoint: checkpoints/abdomen_ctct/ecdp_best.pt (strict=True)
...
Ran 3 tests in 1.48s
OK
```

---

## 📜 Citation

If you use ECDP or this codebase in your research, please cite:

```bibtex
@article{ecdp2026,
  title={Evidence-Calibrated Displacement Pyramids for Deformable Medical Image Registration},
  author={ECDP Authors},
  year={2026}
}
```

---

## 📄 License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
