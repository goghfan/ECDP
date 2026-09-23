"""Label-free image similarity, deformation regularity, and segmentation overlap losses."""

from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LocalNCCLoss(nn.Module):
    """Negative squared local normalized cross correlation for 3D images."""

    def __init__(
        self,
        window: int = 9,
        epsilon: float = 1e-5,
        *,
        mask_invalid: bool = True,
    ) -> None:
        super().__init__()
        if window <= 0 or window % 2 == 0:
            raise ValueError("window must be a positive odd integer")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.window = int(window)
        self.epsilon = float(epsilon)
        self.mask_invalid = bool(mask_invalid)

    def forward(self, fixed: Tensor, moved: Tensor) -> Tensor:
        if fixed.shape != moved.shape or fixed.ndim != 5 or fixed.shape[1] != 1:
            raise ValueError("fixed and moved must share [B, 1, D, H, W] shape")
        with torch.autocast(device_type=fixed.device.type, enabled=False):
            return self._forward_float(fixed.float(), moved.float())

    def _forward_float(self, work_fixed: Tensor, work_moved: Tensor) -> Tensor:
        kernel = torch.ones(
            1, 1, self.window, self.window, self.window,
            device=work_fixed.device, dtype=torch.float32,
        )
        padding = self.window // 2
        count = float(self.window ** 3)

        fixed_sum = F.conv3d(work_fixed, kernel, padding=padding)
        moved_sum = F.conv3d(work_moved, kernel, padding=padding)
        fixed_sq_sum = F.conv3d(work_fixed.square(), kernel, padding=padding)
        moved_sq_sum = F.conv3d(work_moved.square(), kernel, padding=padding)
        cross_sum = F.conv3d(work_fixed * work_moved, kernel, padding=padding)

        cross = cross_sum - fixed_sum * moved_sum / count
        fixed_var = (fixed_sq_sum - fixed_sum.square() / count).clamp_min(0.0)
        moved_var = (moved_sq_sum - moved_sum.square() / count).clamp_min(0.0)
        valid = (fixed_var > self.epsilon) & (moved_var > self.epsilon)
        score = cross.square() / (fixed_var * moved_var + self.epsilon)
        if not self.mask_invalid:
            return -score.mean()
        valid_float = valid.to(score.dtype)
        mean_score = (score * valid_float).sum() / valid_float.sum().clamp_min(1.0)
        return -mean_score


class DiffusionLoss(nn.Module):
    """Spatial flow gradient derivative penalty (diffusion regularizer)."""

    def __init__(self, penalty: str = "l2") -> None:
        super().__init__()
        if penalty not in ("l1", "l2"):
            raise ValueError("penalty must be 'l1' or 'l2'")
        self.penalty = penalty

    def forward(self, flow: Tensor) -> Tensor:
        if flow.ndim != 5 or flow.shape[1] != 3:
            raise ValueError("flow must have shape [B, 3, D, H, W]")
        differences = (
            flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :],
            flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :],
            flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1],
        )
        if self.penalty == "l2":
            differences = tuple(d.square() for d in differences)
        else:
            differences = tuple(d.abs() for d in differences)
        return sum(d.mean() for d in differences) / 3.0


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss for segmentation supervision."""

    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        return 1.0 - dice.mean()
