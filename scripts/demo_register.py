#!/usr/bin/env python3
"""
Quick-start pairwise 3D registration demo using pretrained ECDP.

Usage:
    python scripts/demo_register.py \
        --config configs/abdomen_ctct.yaml \
        --checkpoint checkpoints/abdomen_ctct/ecdp_best.pt \
        --fixed data/sample_fixed.nii.gz \
        --moving data/sample_moving.nii.gz \
        --output_dir results/
"""

import argparse
import pathlib
import yaml
import torch
import nibabel as nib
import numpy as np

from ecdp import ECDPModel, warp
from ecdp.data.dataset import load_nifti_or_npz


def main():
    parser = argparse.ArgumentParser(description="ECDP 3D Medical Image Registration Demo")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--fixed", type=str, required=True, help="Fixed 3D volume (.nii.gz or .npz)")
    parser.add_argument("--moving", type=str, required=True, help="Moving 3D volume (.nii.gz or .npz)")
    parser.add_argument("--fixed_seg", type=str, default=None, help="Optional fixed segmentation")
    parser.add_argument("--moving_seg", type=str, default=None, help="Optional moving segmentation")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save registered results")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading ECDP model for {cfg['dataset_name']}...")
    model = ECDPModel(cfg["model"], alpha=cfg.get("alpha", 0.20), gain_scale=cfg.get("gamma", 10.0)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval()

    fixed_np = load_nifti_or_npz(args.fixed).astype(np.float32)
    moving_np = load_nifti_or_npz(args.moving).astype(np.float32)

    f_min, f_max = fixed_np.min(), fixed_np.max()
    m_min, m_max = moving_np.min(), moving_np.max()
    fixed_norm = (fixed_np - f_min) / (f_max - f_min) if f_max > f_min else fixed_np
    moving_norm = (moving_np - m_min) / (m_max - m_min) if m_max > m_min else moving_np

    fixed_t = torch.from_numpy(fixed_norm).unsqueeze(0).unsqueeze(0).to(device)
    moving_t = torch.from_numpy(moving_norm).unsqueeze(0).unsqueeze(0).to(device)

    print(f"Running registration on volume shape {list(fixed_norm.shape)}...")
    with torch.inference_mode():
        moved_t, flow_zyx, diag = model(fixed_t, moving_t, return_diagnostics=True)
        flow_xyz = diag["final_flow_xyz"]

    moved_np = moved_t.squeeze().cpu().numpy()
    flow_xyz_np = flow_xyz.squeeze().cpu().numpy()

    # Save warped volume as NIfTI
    nib.save(nib.Nifti1Image(moved_np, np.eye(4)), str(out_dir / "warped_moving.nii.gz"))
    np.save(str(out_dir / "deformation_flow_xyz.npy"), flow_xyz_np)

    if args.moving_seg is not None:
        m_seg = load_nifti_or_npz(args.moving_seg).astype(np.int64)
        m_seg_t = torch.from_numpy(m_seg).unsqueeze(0).unsqueeze(0).to(device).float()
        warped_seg_t = warp(m_seg_t, flow_xyz, mode="nearest")
        warped_seg_np = warped_seg_t.squeeze().cpu().numpy().astype(np.int16)
        nib.save(nib.Nifti1Image(warped_seg_np, np.eye(4)), str(out_dir / "warped_seg.nii.gz"))
        print(f"Warped segmentation saved to {out_dir / 'warped_seg.nii.gz'}")

    print(f"Registration finished! Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
