"""PE-LER: Preview-Evaluated Local Evidence Routing Module.

Evaluates candidate residual updates on full-resolution local descriptor similarity
prior to velocity compounding, arbitrating update magnitudes via signed identity routing:
    Delta s(x) = <q_1(x), k_c(x)> - <q_1(x), k_1(x)>                   (Eq. 17)
    G(x)       = 1 + alpha * tanh(gamma * Delta s(x))                   (Eq. 18)
    v_gated(x) = G(x) * v_raw(x)                                        (Eq. 19)

As described in the paper, this mechanism introduces ZERO additional learnable parameters
over the EDDP backbone (both EDDP and ECDP have exactly 3,232,420 parameters / ~3.232M).
"""

from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ecdp.core.geometry import compose_flows, resize_flow, scaling_and_squaring, warp
from ecdp.models.backbone import EDDPBackbone


class PE_LER_Backbone(EDDPBackbone):
    """EDDP Backbone with PE-LER prospective evidence gating at the finest scale (L1)."""
    def __init__(self, alpha: float = 0.20, gain_scale: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.gain_scale = float(gain_scale)

    def _fine_update_with_images(
        self,
        level: int,
        feat_fixed: Tensor,
        feat_moving: Tensor,
        flow_resized: Tensor,
        fixed_img: Tensor,
        moving_img: Tensor,
        diagnostics: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        name = self._SCALE_NAMES[level]

        # 1. Base prediction
        warped_moving_feat = warp(feat_moving, flow_resized)
        q = self.projections[level](feat_fixed)
        k_base = self.projections[level](warped_moving_feat)
        context = torch.cat((q, k_base, flow_resized), dim=1)
        velocity_raw = self.fine_heads[level](context)

        # 2. Candidate preview before integration
        inc_raw = scaling_and_squaring(velocity_raw.float(), self.integration_steps)
        cand_flow = compose_flows(flow_resized, inc_raw)
        warped_moving_cand_feat = warp(feat_moving, cand_flow)
        k_cand = self.projections[level](warped_moving_cand_feat)

        # 3. Prospective descriptor similarity gain: Delta s(x) = S_cand - S_base (Eq. 17)
        q_n = F.normalize(q.float(), dim=1, eps=1e-6)
        k_base_n = F.normalize(k_base.float(), dim=1, eps=1e-6)
        k_cand_n = F.normalize(k_cand.float(), dim=1, eps=1e-6)

        sim_base = (q_n * k_base_n).sum(dim=1, keepdim=True)
        sim_cand = (q_n * k_cand_n).sum(dim=1, keepdim=True)
        gain = sim_cand - sim_base

        # 4. Residual identity gating: G(x) = 1 + alpha * tanh(gamma * Delta s(x)) (Eq. 18)
        delta_gate = self.alpha * torch.tanh(self.gain_scale * gain)
        gate = 1.0 + delta_gate
        acceptance_rate = (gain > 0.0).float().mean()

        # 5. Apply modulation to velocity residual (Eq. 19)
        velocity_gated = velocity_raw * gate
        increment_gated = scaling_and_squaring(velocity_gated.float(), self.integration_steps)
        flow_at_level = compose_flows(flow_resized, increment_gated)

        diagnostics[f"{name}_velocity_rms"] = velocity_gated.float().square().mean().sqrt()
        diagnostics[f"{name}_increment_rms"] = increment_gated.float().square().mean().sqrt()
        diagnostics[f"{name}_evidence_acceptance"] = acceptance_rate
        diagnostics[f"{name}_evidence_gate_mean"] = gate.mean()
        diagnostics[f"{name}_relative_gain_mean"] = gain.mean()
        diagnostics["gate"] = gate
        diagnostics["delta_gate"] = delta_gate
        diagnostics["evidence"] = gain

        return flow_at_level, velocity_gated

    def forward(self, fixed: Tensor, moving: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        diagnostics: dict[str, Tensor] = {}
        fixed_feats = self.encoder(fixed)
        moving_feats = self.encoder(moving)

        flow: Tensor | None = None
        scale_flows: list[Tensor] = []
        scale_velocities: list[Tensor] = []

        # Coarse-to-fine levels (L5 -> L4 -> L3 -> L2)
        for level in range(len(fixed_feats) - 1, 0, -1):
            feat_f = fixed_feats[level]
            feat_m = moving_feats[level]
            target_shape = feat_f.shape[2:]
            flow_ = fixed.new_zeros(fixed.shape[0], 3, *target_shape) if flow is None else resize_flow(flow, target_shape)

            if level in self.recurrent_levels:
                flow, vel, _ = self._recurrent_update(level, feat_f, feat_m, flow_, diagnostics)
            else:
                flow, vel = self._coarse_update(level, feat_f, feat_m, flow_, diagnostics)

            scale_flows.append(flow)
            scale_velocities.append(vel)

        # Finest scale (L1, Level 0) with PE-LER routing
        level = 0
        feat_f = fixed_feats[0]
        feat_m = moving_feats[0]
        flow_ = resize_flow(flow, feat_f.shape[2:])
        final_flow, fine_vel = self._fine_update_with_images(level, feat_f, feat_m, flow_, fixed, moving, diagnostics)

        scale_flows.append(final_flow)
        scale_velocities.append(fine_vel)

        moved = warp(moving, final_flow)
        diagnostics["scale_flows"] = scale_flows
        diagnostics["scale_velocities"] = scale_velocities
        diagnostics["final_flow_xyz"] = final_flow
        return final_flow, moved, diagnostics
