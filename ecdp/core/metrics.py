"""Evaluation metrics: Dice Similarity Coefficient, Surface Distances (ASSD, HD95), and Jacobian regularity."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
from torch import Tensor
from scipy.ndimage import distance_transform_edt, binary_erosion
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


def extract_surface_border(mask: np.ndarray) -> np.ndarray:
    """Extract 1-voxel binary surface boundary of a 3D binary volume."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask)
    return mask ^ eroded


def compute_surface_distances(
    pred_bin: np.ndarray,
    target_bin: np.ndarray,
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute symmetric surface point-to-point distances in millimeters.

    Uses the standard surface contour extraction (mask XOR eroded_mask) followed
    by Euclidean Distance Transform on the complementary volume.
    """
    pred_border = extract_surface_border(pred_bin)
    target_border = extract_surface_border(target_bin)

    if not np.any(pred_border) or not np.any(target_border):
        return np.array([]), np.array([])

    dt_target = distance_transform_edt(~target_border, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)

    d_pred_to_target = dt_target[pred_border]
    d_target_to_pred = dt_pred[target_border]
    return d_pred_to_target, d_target_to_pred


def compute_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    labels: Sequence[int],
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute standard 95th percentile Hausdorff Distance (HD95) in mm across structures.

    Strictly evaluates distances on the surface boundary points, consistent with MedIA / IEEE TMI.
    """
    hd95_list = []
    for label in labels:
        pred_bin = (pred == int(label))
        tgt_bin = (target == int(label))
        if not np.any(pred_bin) or not np.any(tgt_bin):
            continue

        d1, d2 = compute_surface_distances(pred_bin, tgt_bin, voxel_spacing)
        if len(d1) == 0 or len(d2) == 0:
            continue

        all_dists = np.concatenate([d1, d2])
        hd95_list.append(np.percentile(all_dists, 95))

    return float(np.mean(hd95_list)) if hd95_list else float("nan")


def compute_assd(
    pred: np.ndarray,
    target: np.ndarray,
    labels: Sequence[int],
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """Compute Average Symmetric Surface Distance (ASSD) in mm across structures.

    Strictly evaluates symmetric distances on the surface boundary points.
    """
    assd_list = []
    for label in labels:
        pred_bin = (pred == int(label))
        tgt_bin = (target == int(label))
        if not np.any(pred_bin) or not np.any(tgt_bin):
            continue

        d1, d2 = compute_surface_distances(pred_bin, tgt_bin, voxel_spacing)
        if len(d1) == 0 or len(d2) == 0:
            continue

        assd_val = float((d1.sum() + d2.sum()) / (len(d1) + len(d2)))
        assd_list.append(assd_val)

    return float(np.mean(assd_list)) if assd_list else float("nan")
