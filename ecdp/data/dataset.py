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
