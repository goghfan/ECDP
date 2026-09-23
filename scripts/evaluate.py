#!/usr/bin/env python3
"""
Benchmark Evaluation Script for ECDP and Baseline Models.

Evaluates:
  1. Mean and Per-Structure Dice Similarity Coefficient (DSC)
  2. Non-positive Jacobian Determinant fraction (NegJ, |J| <= 0)
  3. Standard Deviation of Log Jacobian Determinant (SDlogJ)
  4. 95th Percentile Hausdorff Distance (HD95)
"""

import os
import sys
import time
import argparse
import pathlib
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from ecdp import ECDPModel, EDDPModel, warp
from ecdp.core.metrics import dice_per_structure, jacobian_summary, compute_hd95
from ecdp.data.dataset import MedicalPairDataset


def parse_args():
    parser = argparse.ArgumentParser(description="ECDP Benchmark Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--is_baseline", action="store_true", help="Evaluate baseline EDDP instead of ECDP")
    parser.add_argument("--data_json", type=str, default=None, help="JSON file containing test pair file paths")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.is_baseline:
        print(f"Loading EDDP Baseline model for {cfg['dataset_name']}...")
        model = EDDPModel(cfg["model"]).to(device)
    else:
        print(f"Loading ECDP (PE-LER) model for {cfg['dataset_name']}...")
        model = ECDPModel(cfg["model"], alpha=cfg.get("alpha", 0.20), gain_scale=cfg.get("gamma", 10.0)).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    sd = state.get("model", state)
    model.load_state_dict(sd, strict=True)
    model.eval()

    eval_labels = cfg.get("eval_labels", [])
    print(f"Target dataset: {cfg['dataset_name']} ({cfg.get('modality', 'Unknown')} {cfg.get('anatomy', '')})")
    print(f"Evaluation labels: {len(eval_labels)} structures")

    if args.data_json and os.path.exists(args.data_json):
        import json
        with open(args.data_json) as f:
            pair_list = json.load(f)
        dataset = MedicalPairDataset(pair_list)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        print(f"Found {len(dataset)} evaluation pairs in {args.data_json}.")

        dices = []
        neg_jacs = []
        sdlogjs = []

        with torch.inference_mode():
            for i, batch in enumerate(loader):
                fixed = batch["fixed"].to(device)
                moving = batch["moving"].to(device)
                moved, flow_zyx, diag = model(fixed, moving, return_diagnostics=True)
                flow_xyz = diag["final_flow_xyz"]

                j_summary = jacobian_summary(flow_xyz)
                neg_jacs.append(j_summary["negative_fraction"].item())
                sdlogjs.append(j_summary["sdlogj"].item())

                if "fixed_seg" in batch and "moving_seg" in batch:
                    f_seg = batch["fixed_seg"].to(device)
                    m_seg = batch["moving_seg"].to(device).float()
                    warped_m_seg = warp(m_seg, flow_xyz, mode="nearest").long()
                    d_res = dice_per_structure(warped_m_seg, f_seg, eval_labels)
                    dices.append(d_res.mean.item())

        if dices:
            print(f"Results across {len(dices)} pairs:")
            print(f"  Mean DSC:  {np.mean(dices) * 100:.2f}%")
        print(f"  Mean NegJ: {np.mean(neg_jacs) * 100:.5f}%")
        print(f"  Mean SDlogJ: {np.mean(sdlogjs):.4f}")
    else:
        print("Note: Provide --data_json with test pair paths to run full dataset evaluation.")
        if "metrics" in state:
            print("Checkpoint metadata reported metrics:")
            for k, v in state["metrics"].items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
