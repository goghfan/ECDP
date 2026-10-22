#!/usr/bin/env python3
"""
ECDP Training and Fine-Tuning Pipeline.

Supports:
  1. Training ECDP end-to-end or warm-starting from baseline EDDP.
  2. Training baseline EDDP.
  3. Mixed precision (AMP) with torch.cuda.amp.GradScaler.
  4. Local Normalized Cross-Correlation (LNCC) and spatial diffusion regularization.
  5. Validation with multi-structure DSC and Jacobian determinant regularity.
  6. Synthetic 3D volume pair generation for rapid pipeline verification.
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
from torch.utils.data import Dataset, DataLoader

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecdp import ECDPModel, EDDPModel, warp
from ecdp.core.losses import LocalNCCLoss, DiffusionLoss
from ecdp.core.metrics import dice_per_structure, jacobian_summary
from ecdp.data.dataset import MedicalPairDataset, SyntheticPairDataset


def parse_args():
    parser = argparse.ArgumentParser(description="ECDP Training & Fine-Tuning Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data_json", type=str, default=None, help="Path to training pairs JSON")
    parser.add_argument("--val_data_json", type=str, default=None, help="Path to validation pairs JSON")
    parser.add_argument("--warmstart", type=str, default=None, help="Path to checkpoint for warm-start initialization")
    parser.add_argument("--output_dir", type=str, default="experiments/ecdp_run", help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1 for 3D volumes)")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--val_freq", type=int, default=1, help="Validation frequency in epochs")
    parser.add_argument("--save_freq", type=int, default=5, help="Checkpoint save frequency in epochs")
    parser.add_argument("--is_baseline", action="store_true", help="Train baseline EDDP instead of full ECDP")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP mixed precision")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic 3D pairs for verification")
    parser.add_argument("--synthetic_samples", type=int, default=8, help="Number of synthetic samples")
    parser.add_argument("--synthetic_shape", type=str, default="64,64,64", help="Shape of synthetic volumes (D,H,W)")
    return parser.parse_args()


def build_dataloaders(args, cfg) -> tuple[DataLoader, DataLoader | None]:
    if args.synthetic or (args.data_json is None):
        if args.data_json is None and not args.synthetic:
            print("Note: --data_json not provided; running in --synthetic mode for pipeline verification.")
        shape_tuple = tuple(int(x.strip()) for x in args.synthetic_shape.split(","))
        train_ds = SyntheticPairDataset(num_samples=args.synthetic_samples, shape=shape_tuple, seed=42)
        val_ds = SyntheticPairDataset(num_samples=max(2, args.synthetic_samples // 4), shape=shape_tuple, seed=123)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        return train_loader, val_loader

    with open(args.data_json) as f:
        train_pairs = json.load(f)
    train_ds = MedicalPairDataset(train_pairs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    val_loader = None
    if args.val_data_json and os.path.exists(args.val_data_json):
        with open(args.val_data_json) as f:
            val_pairs = json.load(f)
        val_ds = MedicalPairDataset(val_pairs)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader


def validate(model, val_loader, eval_labels, device) -> dict[str, float]:
    model.eval()
    dices = []
    neg_fractions = []
    sdlogjs = []

    with torch.inference_mode():
        for batch in val_loader:
            fixed = batch["fixed"].to(device)
            moving = batch["moving"].to(device)
            moved, flow_zyx, diag = model(fixed, moving, return_diagnostics=True)
            flow_xyz = diag["final_flow_xyz"]

            js = jacobian_summary(flow_xyz)
            neg_fractions.append(js["negative_fraction"].item())
            sdlogjs.append(js["sdlogj"].item())

            if "fixed_seg" in batch and "moving_seg" in batch and eval_labels:
                f_seg = batch["fixed_seg"].to(device)
                m_seg = batch["moving_seg"].to(device).float()
                warped_m_seg = warp(m_seg, flow_xyz, mode="nearest").long()
                d_res = dice_per_structure(warped_m_seg, f_seg, eval_labels)
                if not torch.isnan(d_res.mean):
                    dices.append(d_res.mean.item())

    model.train()
    return {
        "mean_dsc": float(np.mean(dices)) if dices else 0.0,
        "neg_fraction": float(np.mean(neg_fractions)) if neg_fractions else 0.0,
        "sdlogj": float(np.mean(sdlogjs)) if sdlogjs else 0.0,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    use_amp = (not args.no_amp) and (device.type == "cuda")

    print(f"============================================================")
    print(f"  ECDP Training Pipeline")
    print(f"  Dataset       : {cfg.get('dataset_name', 'Custom')}")
    print(f"  Model Type    : {'Baseline EDDP' if args.is_baseline else 'Full ECDP (PE-LER)'}")
    print(f"  Device        : {device} (AMP: {use_amp})")
    print(f"  Output Dir    : {out_dir.resolve()}")
    print(f"============================================================")

    # 1. Instantiate Model
    if args.is_baseline:
        model = EDDPModel(cfg["model"]).to(device)
    else:
        model = ECDPModel(
            cfg["model"],
            alpha=cfg.get("alpha", 0.20),
            gain_scale=cfg.get("gamma", 10.0),
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model trainable parameters: {n_params:,} (~{n_params / 1e6:.3f}M)")

    # 2. Warm-start weights if provided
    if args.warmstart:
        print(f"Loading warm-start weights from: {args.warmstart}")
        state = torch.load(args.warmstart, map_location=device)
        sd = state.get("model", state)
        model.load_state_dict(sd, strict=False)

    # 3. DataLoaders
    train_loader, val_loader = build_dataloaders(args, cfg)
    print(f"Training batches per epoch: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")

    # 4. Optimization & Losses
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    sim_loss_fn = LocalNCCLoss(window=9)
    smooth_loss_fn = DiffusionLoss("l2")
    lambda_grad = float(cfg.get("lambda_grad", 0.5))
    eval_labels = cfg.get("eval_labels", [1, 2] if args.synthetic else [])

    best_val_dsc = -1.0
    best_epoch = 0

    # 5. Training Loop
    print("\nStarting training loop...")
    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_sim = 0.0
        running_smooth = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            fixed = batch["fixed"].to(device)
            moving = batch["moving"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                moved, flow_zyx, diag = model(fixed, moving, return_diagnostics=True)
                loss_sim = sim_loss_fn(moved, fixed)
                loss_smooth = smooth_loss_fn(flow_zyx)
                loss = (loss_sim + lambda_grad * loss_smooth) / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * args.accum_steps
            running_sim += loss_sim.item()
            running_smooth += loss_smooth.item()

        scheduler.step()
        epoch_time = time.time() - epoch_start
        n_steps = len(train_loader)
        avg_loss = running_loss / n_steps
        avg_sim = running_sim / n_steps
        avg_smooth = running_smooth / n_steps

        msg = (
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | Time: {epoch_time:.1f}s | "
            f"Loss: {avg_loss:.4f} (Sim: {avg_sim:.4f}, Smooth: {avg_smooth:.4f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # 6. Periodic Validation
        val_metrics = {}
        if val_loader and (epoch % args.val_freq == 0 or epoch == args.epochs):
            val_metrics = validate(model, val_loader, eval_labels, device)
            msg += f" | Val DSC: {val_metrics['mean_dsc'] * 100:.2f}%, NegJ: {val_metrics['neg_fraction'] * 100:.4f}%"

            if val_metrics["mean_dsc"] > best_val_dsc:
                best_val_dsc = val_metrics["mean_dsc"]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metrics": val_metrics,
                        "config": cfg,
                    },
                    out_dir / "best_model.pt",
                )
                msg += " [BEST SAVED]"

        print(msg)

        # 7. Regular Checkpoint Saving
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "config": cfg,
                },
                out_dir / f"checkpoint_epoch_{epoch:03d}.pt",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "config": cfg,
                },
                out_dir / "latest_model.pt",
            )

    total_time = time.time() - total_start_time
    print(f"\n============================================================")
    print(f"Training completed in {total_time / 60:.2f} minutes.")
    if best_epoch > 0:
        print(f"Best Validation DSC: {best_val_dsc * 100:.2f}% (Epoch {best_epoch})")
        print(f"Best model saved to: {out_dir / 'best_model.pt'}")
    print(f"Latest model saved to: {out_dir / 'latest_model.pt'}")
    print(f"============================================================")


if __name__ == "__main__":
    main()
