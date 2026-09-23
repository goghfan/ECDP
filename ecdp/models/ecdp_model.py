"""High-level ECDP and EDDP registration model interfaces."""

from __future__ import annotations
from typing import Any, Mapping
import torch
from torch import Tensor, nn

from ecdp.core.geometry import xyz_to_zyx
from ecdp.models.backbone import EDDPBackbone
from ecdp.models.peler import PE_LER_Backbone


class ECDPModel(nn.Module):
    """Evidence-Calibrated Displacement Pyramids (ECDP) Model.

    Proposed full architecture incorporating PE-LER prospective evidence gating.
    """
    def __init__(self, cfg_model: Mapping[str, Any], alpha: float = 0.20, gain_scale: float = 10.0):
        super().__init__()
        m_cfg = dict(cfg_model)
        m_cfg.pop("architecture", None)
        m_cfg.pop("safe_acceptance", None)

        self.backbone = PE_LER_Backbone(
            alpha=alpha,
            gain_scale=gain_scale,
            base_c=int(m_cfg.get("base_c", 4)),
            projection_dims=tuple(int(v) for v in m_cfg.get("projection_dims", [16, 48, 48, 48, 48])),
            radii=tuple(int(v) for v in m_cfg.get("radii", [2, 2, 2, 3, 3])),
            corr_hidden=int(m_cfg.get("corr_hidden", 64)),
            corr_out=int(m_cfg.get("corr_out", 48)),
            context_dim=int(m_cfg.get("context_dim", 16)),
            gru_hidden=int(m_cfg.get("gru_hidden", 48)),
            recurrent_levels=tuple(int(v) for v in m_cfg.get("recurrent_levels", [1, 2])),
            recurrent_iters=tuple(int(v) for v in m_cfg.get("recurrent_iters", [3, 2])),
            alpha_init=float(m_cfg.get("alpha_init", 0.5)),
            delta_velocity_limit=float(m_cfg.get("delta_velocity_limit", 2.0)),
            coarse_hidden=int(m_cfg.get("coarse_hidden", 48)),
            negative_jacobian_limit=float(m_cfg.get("negative_jacobian_limit", 0.000135637526659)),
            l3_context_mode="raw",
            offset_aware_correlation=False,
            direct_offset_levels=(),
            l2_topology_gate=False,
            fine_hidden=int(m_cfg.get("fine_hidden", 16)),
            fine_velocity_limit=float(m_cfg.get("fine_velocity_limit", 1.0)),
            integration_steps=int(m_cfg.get("integration_steps", 7)),
            fine_levels=(0,),
        )

    def forward(self, fixed: Tensor, moving: Tensor, *, return_diagnostics: bool = False):
        final_flow_xyz, moved, diag = self.backbone(fixed, moving)
        flow_zyx = xyz_to_zyx(final_flow_xyz)
        diag = dict(diag)
        diag["final_flow_xyz"] = final_flow_xyz
        diag["flow_zyx"] = flow_zyx
        if not return_diagnostics:
            return moved, flow_zyx
        return moved, flow_zyx, diag


class EDDPModel(nn.Module):
    """Explicit Displacement Diffusion Pyramids (EDDP) Baseline Model.

    Recurrent correlation architecture without PE-LER routing.
    """
    def __init__(self, cfg_model: Mapping[str, Any]):
        super().__init__()
        m_cfg = dict(cfg_model)
        m_cfg.pop("architecture", None)
        m_cfg.pop("safe_acceptance", None)

        self.backbone = EDDPBackbone(
            base_c=int(m_cfg.get("base_c", 4)),
            projection_dims=tuple(int(v) for v in m_cfg.get("projection_dims", [16, 48, 48, 48, 48])),
            radii=tuple(int(v) for v in m_cfg.get("radii", [2, 2, 2, 3, 3])),
            corr_hidden=int(m_cfg.get("corr_hidden", 64)),
            corr_out=int(m_cfg.get("corr_out", 48)),
            context_dim=int(m_cfg.get("context_dim", 16)),
            gru_hidden=int(m_cfg.get("gru_hidden", 48)),
            recurrent_levels=tuple(int(v) for v in m_cfg.get("recurrent_levels", [1, 2])),
            recurrent_iters=tuple(int(v) for v in m_cfg.get("recurrent_iters", [3, 2])),
            alpha_init=float(m_cfg.get("alpha_init", 0.5)),
            delta_velocity_limit=float(m_cfg.get("delta_velocity_limit", 2.0)),
            coarse_hidden=int(m_cfg.get("coarse_hidden", 48)),
            negative_jacobian_limit=float(m_cfg.get("negative_jacobian_limit", 0.000135637526659)),
            l3_context_mode="raw",
            offset_aware_correlation=False,
            direct_offset_levels=(),
            l2_topology_gate=False,
            fine_hidden=int(m_cfg.get("fine_hidden", 16)),
            fine_velocity_limit=float(m_cfg.get("fine_velocity_limit", 1.0)),
            integration_steps=int(m_cfg.get("integration_steps", 7)),
            fine_levels=(0,),
        )

    def forward(self, fixed: Tensor, moving: Tensor, *, return_diagnostics: bool = False):
        final_flow_xyz, moved, diag = self.backbone(fixed, moving)
        flow_zyx = xyz_to_zyx(final_flow_xyz)
        diag = dict(diag)
        diag["final_flow_xyz"] = final_flow_xyz
        diag["flow_zyx"] = flow_zyx
        if not return_diagnostics:
            return moved, flow_zyx
        return moved, flow_zyx, diag
