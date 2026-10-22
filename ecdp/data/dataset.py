"""PyTorch Dataset loader for 3D medical volume pairs."""

from __future__ import annotations
import pathlib
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib


def load_nifti_or_npz(path: str | pathlib.Path) -> np.ndarray:
    path_str = str(path)
    if path_str.endswith(".npz"):
        data = np.load(path_str)
        return data["vol"] if "vol" in data else data[list(data.keys())[0]]
    elif path_str.endswith((".nii", ".nii.gz")):
        img = nib.load(path_str)
        return np.asanyarray(img.dataobj)
    raise ValueError(f"Unsupported file format: {path_str}")


class MedicalPairDataset(Dataset):
    """Dataset providing (fixed, moving) image and segmentation pairs for 3D registration."""

    def __init__(
        self,
        pair_list: Sequence[dict[str, str]],
        normalize_intensities: bool = True,
    ):
        self.pairs = list(pair_list)
        self.normalize_intensities = normalize_intensities

    def __len__(self) -> int:
        return len(self.pairs)

    def _normalize(self, vol: np.ndarray) -> np.ndarray:
        vol = vol.astype(np.float32)
        min_v, max_v = np.min(vol), np.max(vol)
        if max_v > min_v:
            vol = (vol - min_v) / (max_v - min_v)
        return vol

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        info = self.pairs[idx]
        fixed = load_nifti_or_npz(info["fixed"])
        moving = load_nifti_or_npz(info["moving"])

        if self.normalize_intensities:
            fixed = self._normalize(fixed)
            moving = self._normalize(moving)

        item = {
            "fixed": torch.from_numpy(fixed).unsqueeze(0).float(),
            "moving": torch.from_numpy(moving).unsqueeze(0).float(),
        }

        if "fixed_seg" in info and "moving_seg" in info:
            f_seg = load_nifti_or_npz(info["fixed_seg"])
            m_seg = load_nifti_or_npz(info["moving_seg"])
            item["fixed_seg"] = torch.from_numpy(f_seg.astype(np.int64)).unsqueeze(0)
            item["moving_seg"] = torch.from_numpy(m_seg.astype(np.int64)).unsqueeze(0)

        return item


class SyntheticPairDataset(Dataset):
    """In-memory synthetic 3D volume pairs for pipeline verification and tests."""

    def __init__(self, num_samples: int = 8, shape: tuple[int, int, int] = (64, 64, 64), seed: int = 42):
        super().__init__()
        self.num_samples = num_samples
        self.shape = shape
        rng = np.random.RandomState(seed)
        self.samples = []

        d, h, w = shape
        zz, yy, xx = np.meshgrid(
            np.linspace(-1, 1, d),
            np.linspace(-1, 1, h),
            np.linspace(-1, 1, w),
            indexing="ij",
        )

        for i in range(num_samples):
            bg = 0.2 + 0.1 * np.sin(zz * 3.0) * np.cos(yy * 3.0)

            c1 = rng.uniform(-0.3, 0.3, size=3)
            r1 = rng.uniform(0.2, 0.35, size=3)
            e1 = ((zz - c1[0]) / r1[0]) ** 2 + ((yy - c1[1]) / r1[1]) ** 2 + ((xx - c1[2]) / r1[2]) ** 2 < 1.0

            c2 = rng.uniform(-0.3, 0.3, size=3)
            r2 = rng.uniform(0.15, 0.25, size=3)
            e2 = ((zz - c2[0]) / r2[0]) ** 2 + ((yy - c2[1]) / r2[1]) ** 2 + ((xx - c2[2]) / r2[2]) ** 2 < 1.0

            fixed_vol = bg + 0.5 * e1.astype(np.float32) + 0.3 * e2.astype(np.float32)
            fixed_vol = np.clip(fixed_vol + rng.normal(0, 0.02, size=shape), 0.0, 1.0).astype(np.float32)

            fixed_seg = np.zeros(shape, dtype=np.int64)
            fixed_seg[e1] = 1
            fixed_seg[e2] = 2

            shift = rng.uniform(-0.08, 0.08, size=3)
            e1_m = ((zz - c1[0] - shift[0]) / r1[0]) ** 2 + ((yy - c1[1] - shift[1]) / r1[1]) ** 2 + ((xx - c1[2] - shift[2]) / r1[2]) ** 2 < 1.0
            e2_m = ((zz - c2[0] - shift[0]) / r2[0]) ** 2 + ((yy - c2[1] - shift[1]) / r2[1]) ** 2 + ((xx - c2[2] - shift[2]) / r2[2]) ** 2 < 1.0

            moving_vol = bg + 0.5 * e1_m.astype(np.float32) + 0.3 * e2_m.astype(np.float32)
            moving_vol = np.clip(moving_vol + rng.normal(0, 0.02, size=shape), 0.0, 1.0).astype(np.float32)

            moving_seg = np.zeros(shape, dtype=np.int64)
            moving_seg[e1_m] = 1
            moving_seg[e2_m] = 2

            self.samples.append({
                "fixed": torch.from_numpy(fixed_vol).unsqueeze(0).float(),
                "moving": torch.from_numpy(moving_vol).unsqueeze(0).float(),
                "fixed_seg": torch.from_numpy(fixed_seg).unsqueeze(0),
                "moving_seg": torch.from_numpy(moving_seg).unsqueeze(0),
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]
