#!/usr/bin/env python3
"""Unit tests to verify model instantiation, parameter count, forward pass, and checkpoint loading."""

import unittest
import torch
import yaml
import pathlib

from ecdp import ECDPModel, EDDPModel, warp, compose_flows, jacobian_determinant

ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestECDP(unittest.TestCase):
    def setUp(self):
        with open(ROOT / "configs" / "abdomen_ctct.yaml") as f:
            self.cfg = yaml.safe_load(f)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_parameter_count(self):
        ecdp = ECDPModel(self.cfg["model"]).to(self.device)
        eddp = EDDPModel(self.cfg["model"]).to(self.device)
        n_ecdp = sum(p.numel() for p in ecdp.parameters() if p.requires_grad)
        n_eddp = sum(p.numel() for p in eddp.parameters() if p.requires_grad)
        print("ECDP trainable parameters:", n_ecdp, "(~3.24M)")
        print("EDDP trainable parameters:", n_eddp, "(~3.23M)")
        self.assertEqual(n_ecdp, 3240661)
        self.assertEqual(n_eddp, 3232420)

    def test_forward_pass_and_diagnostics(self):
        model = ECDPModel(self.cfg["model"]).to(self.device)
        model.eval()
        fixed = torch.randn(1, 1, 64, 64, 64, device=self.device)
        moving = torch.randn(1, 1, 64, 64, 64, device=self.device)

        with torch.inference_mode():
            moved, flow_zyx, diag = model(fixed, moving, return_diagnostics=True)

        self.assertEqual(moved.shape, (1, 1, 64, 64, 64))
        self.assertEqual(flow_zyx.shape, (1, 3, 64, 64, 64))
        self.assertIn("gate", diag)
        self.assertIn("delta_gate", diag)
        self.assertIn("final_flow_xyz", diag)

        det = jacobian_determinant(diag["final_flow_xyz"])
        self.assertEqual(det.shape, (1, 64, 64, 64))

    def test_load_all_pretrained_weights(self):
        ckpts = [
            ("checkpoints/abdomen_ctct/ecdp_best.pt", "configs/abdomen_ctct.yaml", True),
            ("checkpoints/abdomen_ctct/eddp_baseline.pt", "configs/abdomen_ctct.yaml", False),
            ("checkpoints/ixi/ecdp_best.pt", "configs/ixi.yaml", True),
            ("checkpoints/ixi/eddp_baseline.pt", "configs/ixi.yaml", False),
            ("checkpoints/lpba40/ecdp_best.pt", "configs/lpba40.yaml", True),
            ("checkpoints/lpba40/eddp_baseline.pt", "configs/lpba40.yaml", False),
        ]
        for ckpt_rel, cfg_rel, is_ecdp in ckpts:
            ckpt_path = ROOT / ckpt_rel
            cfg_path = ROOT / cfg_rel
            with open(cfg_path) as f:
                c = yaml.safe_load(f)

            m = ECDPModel(c["model"]).to(self.device) if is_ecdp else EDDPModel(c["model"]).to(self.device)
            state = torch.load(ckpt_path, map_location=self.device)
            sd = state.get("model", state)
            res = m.load_state_dict(sd, strict=True)
            self.assertEqual(len(res.missing_keys), 0)
            self.assertEqual(len(res.unexpected_keys), 0)
            print("Verified checkpoint:", ckpt_rel, "(strict=True)")


if __name__ == "__main__":
    unittest.main()
