"""Evaluation metrics: Dice Similarity Coefficient, Hausdorff Distance, and Jacobian regularity."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
from torch import Tensor
from scipy.ndimage import distance_transform_edt
from .geometry import jacobian_determinant


@dataclass(frozen=True)
class DiceResult:
    mean: Tensor
    per_structure: Tensor


def dice_per_structure(
    prediction: Tensor,
    target: Tensor,
    labels: Sequence[int],
    *,
    epsilon: float = 1e-5,
) -> DiceResult:
    """Compute per-structure Dice Similarity Coefficient (DSC)."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must share [B, 1, D, H, W] shape")
    if prediction.shape[1] != 1:
        raise ValueError("label tensors must have one channel")
    if not labels:
        raise ValueError("labels must not be empty")

    scores = []
    for label in labels:
        predicted = prediction == int(label)
        expected = target == int(label)
        intersection = (predicted & expected).flatten(1).sum(dim=1).double()
        union = predicted.flatten(1).sum(dim=1).double() + expected.flatten(1).sum(dim=1).double()
        scores.append(2.0 * intersection / (union + epsilon))
    per_structure = torch.stack(scores, dim=1)
    return DiceResult(mean=per_structure.mean(), per_structure=per_structure)


def jacobian_summary(flow_xyz: Tensor, epsilon: float = 1e-6) -> dict[str, Tensor]:
    """Compute Jacobian summary statistics: NegJ (fraction of |J| <= 0), SDlogJ, and mean det."""
    determinant = jacobian_determinant(flow_xyz)
    positive = determinant.clamp_min(epsilon)
    log_jacobian = positive.log()
    return {
        "negative_fraction": (determinant <= 0).float().mean(),
        "sdlogj": log_jacobian.std(unbiased=False),
        "determinant_mean": determinant.mean(),
        "determinant_min": determinant.min(),
    }


def compute_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    labels: Sequence[int],
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute 95th percentile Hausdorff Distance (HD95) in mm across structures."""
    hd95_list = []
    for label in labels:
        pred_bin = (pred == label)
        tgt_bin = (target == label)
        if not np.any(pred_bin) or not np.any(tgt_bin):
            continue

        pred_dist = distance_transform_edt(~pred_bin, sampling=voxel_spacing)
        tgt_dist = distance_transform_edt(~tgt_bin, sampling=voxel_spacing)

        pred_surf_dist = pred_dist[tgt_bin]
        tgt_surf_dist = tgt_dist[pred_bin]

        all_dists = np.concatenate([pred_surf_dist, tgt_surf_dist])
        hd95_list.append(np.percentile(all_dists, 95))

    return float(np.mean(hd95_list)) if hd95_list else float("nan")
