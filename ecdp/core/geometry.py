"""Displacement geometry and stationary velocity field (SVF) integration for 3D registration.

Coordinate Convention:
    Flows and velocities have shape [B, 3, D, H, W] with channels (x, y, z) in voxel units.
    warp(source, flow) samples source at x + flow(x), using align_corners=True.
"""

from __future__ import annotations
from typing import Sequence
import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_flow(flow: Tensor) -> None:
    if flow.ndim != 5 or flow.shape[1] != 3:
        raise ValueError(f"flow must have shape [B, 3, D, H, W], got {tuple(flow.shape)}")
    if not flow.is_floating_point():
        raise TypeError("flow must be floating point")


def _base_grid(
    spatial_shape: Sequence[int], *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    depth, height, width = spatial_shape
    z = torch.linspace(-1.0, 1.0, depth, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)


def warp(
    source: Tensor,
    flow: Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> Tensor:
    """Warp a 3D tensor by an x/y/z voxel displacement field."""
    _validate_flow(flow)
    if source.ndim != 5:
        raise ValueError(f"source must have shape [B, C, D, H, W], got {tuple(source.shape)}")
    if source.shape[0] != flow.shape[0] or source.shape[2:] != flow.shape[2:]:
        raise ValueError("source and flow batch/spatial shapes must match")
    if source.device != flow.device:
        raise ValueError("source and flow must be on the same device")

    depth, height, width = flow.shape[2:]
    denominators = flow.new_tensor(
        (max(width - 1, 1), max(height - 1, 1), max(depth - 1, 1))
    ).view(1, 3, 1, 1, 1)
    normalized_flow = 2.0 * flow / denominators
    grid = _base_grid(flow.shape[2:], device=flow.device, dtype=flow.dtype)
    grid = grid + normalized_flow.permute(0, 2, 3, 4, 1)
    return F.grid_sample(
        source,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


def resize_flow(flow: Tensor, size: Sequence[int]) -> Tensor:
    """Resize a flow while preserving normalized sampling displacement."""
    _validate_flow(flow)
    target = tuple(int(value) for value in size)
    if len(target) != 3 or any(value <= 0 for value in target):
        raise ValueError(f"size must contain three positive integers, got {size}")
    old_depth, old_height, old_width = flow.shape[2:]
    new_depth, new_height, new_width = target
    resized = F.interpolate(flow, size=target, mode="trilinear", align_corners=True)
    scales = flow.new_tensor(
        (
            max(new_width - 1, 1) / max(old_width - 1, 1),
            max(new_height - 1, 1) / max(old_height - 1, 1),
            max(new_depth - 1, 1) / max(old_depth - 1, 1),
        )
    ).view(1, 3, 1, 1, 1)
    return resized * scales


def compose_flows(first: Tensor, second: Tensor) -> Tensor:
    """Return the flow equivalent to warp(warp(x, first), second)."""
    _validate_flow(first)
    _validate_flow(second)
    if first.shape != second.shape:
        raise ValueError("flows must have identical shapes")
    transported_first = warp(first, second, padding_mode="border")
    return second + transported_first


def scaling_and_squaring(
    velocity: Tensor,
    integration_steps: int = 7,
    *,
    padding_mode: str = "border",
) -> Tensor:
    """Compute displacement field exp(v) from stationary velocity field v via scaling-and-squaring."""
    _validate_flow(velocity)
    steps = int(integration_steps)
    if steps < 1:
        raise ValueError("integration_steps must be at least 1")
    scale = 2.0 ** steps
    with torch.autocast(device_type=velocity.device.type, enabled=False):
        disp = velocity.float() / scale
        for _ in range(steps):
            disp = disp + warp(disp, disp, mode="bilinear", padding_mode=padding_mode)
    return disp.to(velocity.dtype)


def jacobian_determinant(flow: Tensor) -> Tensor:
    """Compute det(I + grad(flow)) at every voxel."""
    _validate_flow(flow)
    work = flow.float() if flow.dtype in (torch.float16, torch.bfloat16) else flow
    grad_z, grad_y, grad_x = torch.gradient(work, dim=(2, 3, 4), edge_order=1)

    j00 = 1.0 + grad_x[:, 0]
    j01 = grad_y[:, 0]
    j02 = grad_z[:, 0]
    j10 = grad_x[:, 1]
    j11 = 1.0 + grad_y[:, 1]
    j12 = grad_z[:, 1]
    j20 = grad_x[:, 2]
    j21 = grad_y[:, 2]
    j22 = 1.0 + grad_z[:, 2]

    return (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )


def negative_jacobian_fraction(flow: Tensor) -> Tensor:
    """Fraction of voxels with non-positive transformation Jacobian."""
    return (jacobian_determinant(flow) <= 0).float().mean()


def xyz_to_zyx(flow: Tensor) -> Tensor:
    """Convert flow from (x, y, z) channel order to (z, y, x)."""
    if flow.ndim != 5 or flow.shape[1] != 3:
        raise ValueError("flow must have shape [B, 3, D, H, W]")
    return flow[:, (2, 1, 0)]


def zyx_to_xyz(flow: Tensor) -> Tensor:
    """Convert flow from (z, y, x) channel order to (x, y, z)."""
    if flow.ndim != 5 or flow.shape[1] != 3:
        raise ValueError("flow must have shape [B, 3, D, H, W]")
    return flow[:, (2, 1, 0)]
