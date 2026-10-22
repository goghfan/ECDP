#!/usr/bin/env python3
"""
Comprehensive Benchmark Evaluation Script for ECDP and Baseline Models.

Evaluates all metrics reported in paper Tables II, III, IV, and X:
  1. Mean and Per-Structure Dice Similarity Coefficient (DSC, %)
  2. Average Symmetric Surface Distance (ASSD, mm) using surface boundaries
  3. 95th Percentile Hausdorff Distance (HD95, mm) using surface boundaries
  4. Non-positive Jacobian Determinant fraction (NegJ, |J| <= 0, %)
  5. Standard Deviation of Log Jacobian Determinant (SDlogJ)
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
import pathlib
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecdp import ECDPModel, EDDPModel, warp
from ecdp.core.metrics import (
    dice_per_structure,
    jacobian_summary,
    compute_hd95,
    compute_assd,
)
from ecdp.data.dataset import MedicalPairDataset, SyntheticPairDataset


def parse_args():
    parser = argparse.ArgumentParser(description="ECDP Comprehensive Benchmark Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--is_baseline", action="store_true", help="Evaluate baseline EDDP instead of ECDP")
    parser.add_argument("--data_json", type=str, default=None, help="JSON file containing test pair file paths")
    parser.add_argument("--spacing", type=str, default="1.0,1.0,1.0", help="Voxel spacing in mm (e.g., '1.0,1.0,1.0')")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional path to save per-pair metrics CSV")
    parser.add_argument("--synthetic", action="store_true", help="Run evaluation on synthetic pairs for verification")
    parser.add_argument("--synthetic_samples", type=int, default=4, help="Number of synthetic evaluation pairs")
    parser.add_argument("--synthetic_shape", type=str, default="64,64,64", help="Shape of synthetic volumes (D,H,W)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    spacing = tuple(float(x.strip()) for x in args.spacing.split(","))

    # 1. Instantiate Model
    if args.is_baseline:
        print(f"Loading EDDP Baseline model for {cfg['dataset_name']}...")
        model = EDDPModel(cfg["model"]).to(device)
    else:
        print(f"Loading ECDP (PE-LER) model for {cfg['dataset_name']}...")
        model = ECDPModel(
            cfg["model"],
            alpha=cfg.get("alpha", 0.20),
            gain_scale=cfg.get("gamma", 10.0),
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} (~{n_params / 1e6:.3f}M)")

    # 2. Load Weights if provided
    state = None
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        sd = state.get("model", state)
        model.load_state_dict(sd, strict=True)
    model.eval()

    eval_labels = cfg.get("eval_labels", [])
    print(f"Dataset: {cfg['dataset_name']} ({cfg.get('modality', 'Unknown')} {cfg.get('anatomy', '')})")
    print(f"Evaluation labels count: {len(eval_labels)}")
    print(f"Voxel spacing: {spacing} mm")

    # 3. Determine evaluation pairs
    eval_batches = []
    if args.data_json and os.path.exists(args.data_json):
        with open(args.data_json) as f:
            pair_list = json.load(f)
        ds = MedicalPairDataset(pair_list)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        eval_batches = list(loader)
        print(f"Loaded {len(eval_batches)} evaluation pairs from {args.data_json}.")
    elif args.synthetic:
        shape_tuple = tuple(int(x.strip()) for x in args.synthetic_shape.split(","))
        ds = SyntheticPairDataset(num_samples=args.synthetic_samples, shape=shape_tuple, seed=42)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        eval_batches = list(loader)
        eval_labels = [1, 2]
        print(f"Using {len(eval_batches)} synthetic 3D pairs for evaluation verification.")
    else:
        print("\nNote: No --data_json provided and --synthetic flag not set.")
        if state and "metrics" in state:
            print(f"\nReported metrics stored in checkpoint {args.checkpoint}:")
            for k, v in state["metrics"].items():
                print(f"  {k:20s}: {v}")
        print("\nTo evaluate on dataset pairs: python scripts/evaluate.py --config <cfg> --checkpoint <pt> --data_json <pairs.json>")
        print("To run synthetic verification:  python scripts/evaluate.py --config <cfg> --checkpoint <pt> --synthetic")
        return

    # 4. Evaluation Loop
    dices = []
    assds = []
    hd95s = []
    neg_jacs = []
    sdlogjs = []
    records = []

    print(f"\nRunning evaluation across {len(eval_batches)} pairs...")
    t0 = time.time()

    with torch.inference_mode():
        for i, batch in enumerate(eval_batches):
            fixed = batch["fixed"].to(device)
            moving = batch["moving"].to(device)

            moved, flow_zyx, diag = model(fixed, moving, return_diagnostics=True)
            flow_xyz = diag["final_flow_xyz"]

            # Topological metrics
            j_summary = jacobian_summary(flow_xyz)
            neg_j = j_summary["negative_fraction"].item()
            sdlogj = j_summary["sdlogj"].item()
            neg_jacs.append(neg_j)
            sdlogjs.append(sdlogj)

            pair_dsc = float("nan")
            pair_assd = float("nan")
            pair_hd95 = float("nan")

            if "fixed_seg" in batch and "moving_seg" in batch and eval_labels:
                f_seg = batch["fixed_seg"].to(device)
                m_seg = batch["moving_seg"].to(device).float()
                warped_m_seg = warp(m_seg, flow_xyz, mode="nearest").long()

                # DSC
                d_res = dice_per_structure(warped_m_seg, f_seg, eval_labels)
                pair_dsc = d_res.mean.item()
                if not np.isnan(pair_dsc):
                    dices.append(pair_dsc)

                # Surface-based ASSD & HD95
                pred_np = warped_m_seg.squeeze().cpu().numpy().astype(np.int64)
                tgt_np = f_seg.squeeze().cpu().numpy().astype(np.int64)

                pair_assd = compute_assd(pred_np, tgt_np, eval_labels, voxel_spacing=spacing)
                pair_hd95 = compute_hd95(pred_np, tgt_np, eval_labels, voxel_spacing=spacing)

                if not np.isnan(pair_assd):
                    assds.append(pair_assd)
                if not np.isnan(pair_hd95):
                    hd95s.append(pair_hd95)

            rec = {
                "pair_idx": i + 1,
                "dsc": pair_dsc,
                "assd_mm": pair_assd,
                "hd95_mm": pair_hd95,
                "neg_j_pct": neg_j * 100.0,
                "sdlogj": sdlogj,
            }
            records.append(rec)
            print(
                f"  Pair [{i + 1:02d}/{len(eval_batches):02d}] | "
                f"DSC: {pair_dsc * 100:.2f}% | ASSD: {pair_assd:.3f} mm | "
                f"HD95: {pair_hd95:.3f} mm | NegJ: {neg_j * 100:.4f}% | SDlogJ: {sdlogj:.4f}"
            )

    eval_time = time.time() - t0

    # 5. Output Summary Table
    print("\n" + "=" * 70)
    print(f"  ECDP Comprehensive Benchmark Evaluation Summary")
    print(f"  Dataset    : {cfg['dataset_name']}")
    print(f"  Model      : {'Baseline EDDP' if args.is_baseline else 'ECDP (PE-LER)'}")
    print(f"  Parameters : {n_params:,} (~{n_params / 1e6:.3f}M)")
    print(f"  Pairs      : {len(eval_batches)} (Evaluated in {eval_time:.2f}s)")
    print("=" * 70)
    if dices:
        print(f"  DSC (%)      : {np.mean(dices) * 100:.2f} ± {np.std(dices) * 100:.2f}")
    if assds:
        print(f"  ASSD (mm)    : {np.mean(assds):.3f} ± {np.std(assds):.3f}")
    if hd95s:
        print(f"  HD95 (mm)    : {np.mean(hd95s):.3f} ± {np.std(hd95s):.3f}")
    print(f"  NegJ (<=0) % : {np.mean(neg_jacs) * 100:.5f}%")
    print(f"  SDlogJ       : {np.mean(sdlogjs):.4f} ± {np.std(sdlogjs):.4f}")
    print("=" * 70)

    # 6. Save CSV if requested
    if args.output_csv:
        import csv
        csv_path = pathlib.Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"Per-pair metrics written to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
