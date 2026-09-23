"""EDDP Multi-scale Recurrent Correlation Backbone for Deformable Registration."""

from __future__ import annotations
import math
from typing import Mapping, Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ecdp.core.geometry import compose_flows, resize_flow, scaling_and_squaring, warp, jacobian_determinant


class ConvInsBlock(nn.Module):
    """3D Conv3d + InstanceNorm3d + LeakyReLU."""
    def __init__(
        self,
        in_c: int,
        out_c: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, k, s, p)
        self.norm = nn.InstanceNorm3d(out_c, affine=True, track_running_stats=False)
        self.act = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvNoNorm(nn.Module):
    """3D Conv3d + LeakyReLU only."""
    def __init__(
        self,
        in_c: int,
        out_c: int,
        k: int = 3,
        s: int = 1,
        p: int = 1,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, k, s, p)
        self.act = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


class UNetEncoderV2(nn.Module):
    """5-stage multi-scale UNet encoder for 3D medical images."""
    def __init__(self, in_channels: int = 1, base_c: int = 4) -> None:
        super().__init__()
        c = base_c
        self.out_channels = [2 * c, 4 * c, 8 * c, 16 * c, 32 * c]
        self.conv0 = nn.Sequential(
            ConvNoNorm(in_channels, c),
            ConvInsBlock(c, 2 * c),
            ConvInsBlock(2 * c, 2 * c),
        )
        self.conv1 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(2 * c, 4 * c),
            ConvInsBlock(4 * c, 4 * c),
        )
        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(4 * c, 8 * c),
            ConvInsBlock(8 * c, 8 * c),
        )
        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(8 * c, 16 * c),
            ConvInsBlock(16 * c, 16 * c),
        )
        self.conv4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(16 * c, 32 * c),
            ConvInsBlock(32 * c, 32 * c),
        )

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("image must have shape [B, 1, D, H, W]")
        out0 = self.conv0(image)
        out1 = self.conv1(out0)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        return out0, out1, out2, out3, out4


class FeatureProjector(nn.Module):
    """1x1 Conv3d projection of pyramidal features to embedding space."""
    def __init__(self, in_c: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_c, dim, kernel_size=1, bias=True)

    def forward(self, feat: Tensor) -> Tensor:
        return self.proj(feat)


def offsets_cube_zyx(
    radius: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Return the (2r+1)^3 offset grid in (z, y, x) spatial ordering."""
    values = [
        (dz, dy, dx)
        for dz in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    return torch.tensor(values, device=device, dtype=dtype)


def full_3d_correlation_volume(
    fixed_embed: Tensor,
    moving_embed: Tensor,
    radius: int,
) -> tuple[Tensor, Tensor]:
    """Compute (2r+1)^3 cosine correlation cost volume."""
    fixed_n = F.normalize(fixed_embed.float(), dim=1, eps=1e-6)
    moving_n = F.normalize(moving_embed.float(), dim=1, eps=1e-6)

    offsets = offsets_cube_zyx(radius, device=fixed_embed.device, dtype=torch.float32)
    max_r = radius
    padded = F.pad(moving_n, (max_r, max_r, max_r, max_r, max_r, max_r))
    depth, height, width = fixed_n.shape[2:]

    corrs = []
    for dz, dy, dx in offsets.tolist():
        dz_i, dy_i, dx_i = int(dz), int(dy), int(dx)
        shifted = padded[
            :,
            :,
            max_r + dz_i : max_r + dz_i + depth,
            max_r + dy_i : max_r + dy_i + height,
            max_r + dx_i : max_r + dx_i + width,
        ]
        corrs.append((fixed_n * shifted).sum(dim=1, keepdim=True))
    corr = torch.cat(corrs, dim=1)
    return corr, offsets


class ResidualConvBlock(nn.Module):
    """3D residual conv block without normalization."""
    def __init__(self, channels: int, dilation: int = 1, alpha: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, 1, dilation, dilation=dilation)
        self.act1 = nn.LeakyReLU(alpha, inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, 3, 1, dilation, dilation=dilation)
        self.act2 = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        hidden = self.act1(self.conv1(x))
        hidden = self.conv2(hidden)
        return self.act2(hidden + residual)


class CorrelationEncoder(nn.Module):
    """Compress a (2r+1)^3 local correlation volume to dense features (K -> 64 -> 32)."""
    def __init__(self, num_candidates: int, hidden_channels: int = 64, out_channels: int = 32) -> None:
        super().__init__()
        self.num_candidates = int(num_candidates)
        self.conv1 = nn.Conv3d(num_candidates, hidden_channels, 1)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv3d(hidden_channels, out_channels, 3, 1, 1)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, corr: Tensor) -> Tensor:
        return self.act2(self.conv2(self.act1(self.conv1(corr))))


class ContextEncoder(nn.Module):
    """Fixed-image context encoder for recurrent GRU updater."""
    def __init__(self, in_channels: int, out_channels: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, feat: Tensor) -> Tensor:
        return self.act(self.conv(feat))


class ConvGRUCell3D(nn.Module):
    """3D Recurrent Convolutional Gated Recurrent Unit."""
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.conv_z = nn.Conv3d(in_channels + hidden_channels, hidden_channels, 3, 1, 1)
        self.conv_r = nn.Conv3d(in_channels + hidden_channels, hidden_channels, 3, 1, 1)
        self.conv_h = nn.Conv3d(in_channels + hidden_channels, hidden_channels, 3, 1, 1)

    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
        combined = torch.cat((x, hidden), dim=1)
        z = torch.sigmoid(self.conv_z(combined))
        r = torch.sigmoid(self.conv_r(combined))
        candidate = torch.tanh(self.conv_h(torch.cat((x, r * hidden), dim=1)))
        return (1.0 - z) * hidden + z * candidate


class DeltaVelocityHead(nn.Module):
    """Delta-velocity head predicting bounded recurrent velocity updates."""
    def __init__(self, in_channels: int, hidden_channels: int = 24, velocity_limit: float = 2.0) -> None:
        super().__init__()
        self.velocity_limit = float(velocity_limit)
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, 3, 1, 1)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, 3, 1, 1)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        self.head = nn.Conv3d(hidden_channels, 3, 3, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.act2(self.conv2(self.act1(self.conv1(hidden))))
        return self.velocity_limit * torch.tanh(self.head(hidden))


class CoarseVelocityHead(nn.Module):
    """Non-recurrent coarse velocity estimator: [corr, context, flow] -> velocity."""
    def __init__(self, in_channels: int, hidden_channels: int = 48) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, 3, 1, 1)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.block1 = ResidualConvBlock(hidden_channels, dilation=1)
        self.block2 = ResidualConvBlock(hidden_channels, dilation=2)
        self.conv3 = nn.Conv3d(hidden_channels, hidden_channels // 2, 3, 1, 1)
        self.act3 = nn.LeakyReLU(0.1, inplace=True)
        self.head = nn.Conv3d(hidden_channels // 2, 3, 3, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, corr_feat: Tensor, context: Tensor, flow: Tensor) -> Tensor:
        x = torch.cat((corr_feat, context, flow), dim=1)
        hidden = self.act1(self.conv1(x))
        hidden = self.block1(hidden)
        hidden = self.block2(hidden)
        hidden = self.act3(self.conv3(hidden))
        return self.head(hidden)


class FineVelocityHead(nn.Module):
    """Finest-level velocity refinement head (L1, 1:1 scale)."""
    def __init__(self, in_channels: int, hidden_channels: int = 16, velocity_limit: float = 1.0) -> None:
        super().__init__()
        self.velocity_limit = float(velocity_limit)
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, 3, 1, 1)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, 3, 1, 1)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)
        self.head = nn.Conv3d(hidden_channels, 3, 3, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, context: Tensor) -> Tensor:
        hidden = self.act2(self.conv2(self.act1(self.conv1(context))))
        return self.velocity_limit * torch.tanh(self.head(hidden))


class EDDPBackbone(nn.Module):
    """Complete 5-scale coarse-to-fine correlation-driven recurrent registration backbone."""
    _SCALE_NAMES = ("L1", "L2", "L3", "L4", "L5")
    _VEL_LEVEL_WEIGHTS = (0.25, 0.25, 1.0, 1.0, 0.5)

    def __init__(
        self,
        *,
        base_c: int = 4,
        projection_dims: Sequence[int] = (16, 48, 48, 48, 48),
        radii: Sequence[int] = (2, 2, 2, 3, 3),
        corr_hidden: int = 64,
        corr_out: int = 48,
        context_dim: int = 16,
        gru_hidden: int = 48,
        recurrent_levels: Sequence[int] = (1, 2),
        recurrent_iters: Sequence[int] = (3, 2),
        alpha_init: float = 0.5,
        delta_velocity_limit: float = 2.0,
        coarse_hidden: int = 48,
        negative_jacobian_limit: float = 0.000135637526659,
        l3_context_mode: str = "raw",
        offset_aware_correlation: bool = False,
        direct_offset_levels: Sequence[int] = (),
        l2_topology_gate: bool = False,
        fine_hidden: int = 16,
        fine_velocity_limit: float = 1.0,
        integration_steps: int = 7,
        fine_levels: Sequence[int] = (0,),
    ) -> None:
        super().__init__()
        self.fine_levels = tuple(int(i) for i in fine_levels)
        self.recurrent_levels = tuple(int(i) for i in recurrent_levels)
        self.level_iters = dict(zip(recurrent_levels, recurrent_iters))
        self.integration_steps = int(integration_steps)
        self.l3_context_mode = str(l3_context_mode).lower()
        self.l3_raw_dim = 16
        self.offset_aware = bool(offset_aware_correlation)
        self.direct_offset_levels = tuple(int(i) for i in direct_offset_levels)
        self.l2_topology_gate = bool(l2_topology_gate)
        self.negative_jacobian_limit = float(negative_jacobian_limit)
        self.gru_hidden = int(gru_hidden)
        self.radii = tuple(int(r) for r in radii)
        self.projection_dims = tuple(int(d) for d in projection_dims)

        self.encoder = UNetEncoderV2(in_channels=1, base_c=base_c)
        ch = self.encoder.out_channels
        self.projections = nn.ModuleList(
            FeatureProjector(ch[i], self.projection_dims[i]) for i in range(5)
        )
        num_cands = [(2 * r + 1) ** 3 for r in self.radii]
        self.correlation_encoders = nn.ModuleList(
            (CorrelationEncoder(num_cands[i], corr_hidden, corr_out) if i not in fine_levels else nn.Identity())
            for i in range(5)
        )
        self.offset_statistics = nn.ModuleList(nn.Identity() for _ in range(5))
        self.offset_fusions = nn.ModuleList(nn.Identity() for _ in range(5))

        self.context_encoders = nn.ModuleList(
            (ContextEncoder(ch[i], context_dim) if i not in fine_levels else nn.Identity())
            for i in range(5)
        )

        gru_in = {}
        for level in recurrent_levels:
            if level == 2 and self.l3_context_mode == "raw":
                gru_in[level] = corr_out + context_dim + self.l3_raw_dim + 3 + 3
            else:
                gru_in[level] = corr_out + context_dim + 3 + 3

        self.convgrus = nn.ModuleList(
            (ConvGRUCell3D(gru_in[i], gru_hidden) if i in recurrent_levels else nn.Identity())
            for i in range(5)
        )
        self.l3_raw_projectors = nn.ModuleList(
            (
                nn.Sequential(nn.Conv3d(ch[i], self.l3_raw_dim, 1), nn.LeakyReLU(0.1, inplace=True))
                if i in recurrent_levels and i == 2 and self.l3_context_mode == "raw"
                else nn.Identity()
            )
            for i in range(5)
        )
        self.direct_decoders = nn.ModuleList(nn.Identity() for _ in range(5))
        self.delta_heads = nn.ModuleList(
            (DeltaVelocityHead(gru_hidden, gru_hidden // 2, delta_velocity_limit) if i in recurrent_levels else nn.Identity())
            for i in range(5)
        )
        self.alphas = nn.ParameterList(
            (
                nn.Parameter(torch.full((), float(math.log(alpha_init / (1.0 - alpha_init)))))
                if i in recurrent_levels else nn.Parameter(torch.zeros(()))
            )
            for i in range(5)
        )
        self.coarse_heads = nn.ModuleList(
            (CoarseVelocityHead(corr_out + context_dim + 3, coarse_hidden) if i not in fine_levels and i not in recurrent_levels else nn.Identity())
            for i in range(5)
        )
        self.fine_heads = nn.ModuleList(
            (FineVelocityHead(2 * self.projection_dims[i] + 3, fine_hidden, fine_velocity_limit) if i in fine_levels else nn.Identity())
            for i in range(5)
        )

    def _fine_update(
        self,
        level: int,
        feat_fixed: Tensor,
        feat_moving: Tensor,
        flow_resized: Tensor,
        diagnostics: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        warped_moving = warp(feat_moving, flow_resized)
        q = self.projections[level](feat_fixed)
        k_proj = self.projections[level](warped_moving)
        context = torch.cat((q, k_proj, flow_resized), dim=1)
        velocity = self.fine_heads[level](context)
        increment = scaling_and_squaring(velocity.float(), self.integration_steps)
        flow_at_level = compose_flows(flow_resized, increment)
        return flow_at_level, velocity

    def _coarse_update(
        self,
        level: int,
        feat_fixed: Tensor,
        feat_moving: Tensor,
        flow_resized: Tensor,
        diagnostics: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        warped_moving = warp(feat_moving, flow_resized)
        q = self.projections[level](feat_fixed)
        k_proj = self.projections[level](warped_moving)
        corr, offsets = full_3d_correlation_volume(q, k_proj, self.radii[level])
        corr_feat = self.correlation_encoders[level](corr)
        context = self.context_encoders[level](feat_fixed)
        velocity = self.coarse_heads[level](corr_feat, context, flow_resized)
        increment = scaling_and_squaring(velocity.float(), self.integration_steps)
        flow_at_level = compose_flows(flow_resized, increment)
        return flow_at_level, velocity

    def _recurrent_update(
        self,
        level: int,
        feat_fixed: Tensor,
        feat_moving: Tensor,
        flow_resized: Tensor,
        diagnostics: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        iters = self.level_iters[level]
        q = self.projections[level](feat_fixed)
        context = self.context_encoders[level](feat_fixed)
        raw_l3 = self.l3_raw_projectors[level](feat_fixed) if level == 2 and self.l3_context_mode == "raw" else None
        velocity = flow_resized.new_zeros(flow_resized.shape)
        hidden: Tensor | None = None
        iter_velocities: list[Tensor] = []

        for it in range(iters):
            target = compose_flows(flow_resized, velocity)
            k = self.projections[level](warp(feat_moving, target))
            corr, offsets = full_3d_correlation_volume(q, k, self.radii[level])
            corr_feat = self.correlation_encoders[level](corr)

            if level == 2 and self.l3_context_mode == "raw":
                x = torch.cat((corr_feat, context, raw_l3, flow_resized, velocity), dim=1)
            else:
                x = torch.cat((corr_feat, context, flow_resized, velocity), dim=1)

            if hidden is None:
                hidden = x.new_zeros(x.shape[0], self.gru_hidden, *x.shape[2:])

            hidden = self.convgrus[level](x, hidden)
            delta_v = self.delta_heads[level](hidden)
            alpha = torch.sigmoid(self.alphas[level]).to(velocity.dtype)
            velocity = velocity + alpha * delta_v
            iter_velocities.append(velocity)

        increment = scaling_and_squaring(velocity.float(), self.integration_steps)
        flow_at_level = compose_flows(flow_resized, increment)
        return flow_at_level, velocity, iter_velocities

    def forward(self, fixed: Tensor, moving: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if fixed.ndim != 5 or moving.ndim != 5:
            raise ValueError("images must have shape [B, 1, D, H, W]")
        fixed_feats = self.encoder(fixed)
        moving_feats = self.encoder(moving)
        diagnostics: dict[str, Tensor] = {}
        flow: Tensor | None = None
        scale_flows: list[Tensor] = []
        scale_velocities: list[Tensor] = []

        for level in range(4, -1, -1):
            feat_fixed = fixed_feats[level]
            feat_moving = moving_feats[level]
            target_shape = feat_fixed.shape[2:]
            flow_ = fixed.new_zeros(fixed.shape[0], 3, *target_shape) if flow is None else resize_flow(flow, target_shape)

            if level in self.fine_levels:
                flow, velocity = self._fine_update(level, feat_fixed, feat_moving, flow_, diagnostics)
            elif level in self.recurrent_levels:
                flow, velocity, iter_velocities = self._recurrent_update(level, feat_fixed, feat_moving, flow_, diagnostics)
            else:
                flow, velocity = self._coarse_update(level, feat_fixed, feat_moving, flow_, diagnostics)

            scale_flows.append(flow)
            scale_velocities.append(velocity)

        warped_moving = warp(moving, flow)
        diagnostics["scale_flows"] = scale_flows
        diagnostics["scale_velocities"] = scale_velocities
        diagnostics["final_flow_xyz"] = flow
        return flow, warped_moving, diagnostics
