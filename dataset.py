"""
Protein backbone dataset utilities for PAR training.

Supports:
  - Loading Cα coordinates from PDB / mmCIF / numpy files
  - Length filtering (32–256 residues for AFDB pre-training)
  - Padding + length tracking for batching
  - AFDB representative set convention (used in Table 1)
"""

import os
import glob
from typing import List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Abstract backbone dataset
# ---------------------------------------------------------------------------

class BackboneDataset(Dataset):
    """
    Dataset of protein Cα backbone coordinates.

    Each item is a dict:
        'coords':  (L, 3) float32 tensor  — Cα positions
        'length':  int                    — number of residues
        'name':    str                    — identifier (filename / PDB id)

    Subclass and implement `_load_coords` for your file format.
    """

    def __init__(
        self,
        data_dir: str,
        file_pattern: str = "*.npy",
        min_length: int = 32,
        max_length: int = 256,
        max_samples: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        self.data_dir   = data_dir
        self.min_length = min_length
        self.max_length = max_length
        self.transform  = transform

        # Discover files
        pattern = os.path.join(data_dir, "**", file_pattern)
        all_files = sorted(glob.glob(pattern, recursive=True))

        if max_samples is not None:
            all_files = all_files[:max_samples]

        self.files = all_files
        print(f"[Dataset] Found {len(self.files)} files in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Optional[dict]:
        path = self.files[idx]
        try:
            coords = self._load_coords(path)            # (L, 3) numpy or tensor
            if not isinstance(coords, torch.Tensor):
                coords = torch.tensor(coords, dtype=torch.float32)
            L = coords.shape[0]

            if L < self.min_length or L > self.max_length:
                return None

            if self.transform is not None:
                coords = self.transform(coords)

            return {
                "coords": coords,
                "length": L,
                "name":   os.path.basename(path),
            }
        except Exception as e:
            print(f"[Dataset] Warning: could not load {path}: {e}")
            return None

    def _load_coords(self, path: str) -> torch.Tensor:
        """Override in subclass. Return (L, 3) Cα coordinates."""
        import numpy as np
        return torch.from_numpy(np.load(path)).float()


# ---------------------------------------------------------------------------
# Numpy backbone dataset  (.npy files, shape (L, 3))
# ---------------------------------------------------------------------------

class NumpyBackboneDataset(BackboneDataset):
    """Loads Cα coordinates stored as .npy arrays of shape (L, 3)."""
    pass   # _load_coords already handles .npy in base class


# ---------------------------------------------------------------------------
# Collate function  (handles variable-length + None items)
# ---------------------------------------------------------------------------

def collate_backbones(
    batch: List[Optional[dict]],
    pad_value: float = 0.0,
) -> Optional[dict]:
    """
    Pad Cα coordinate tensors to the maximum length in the batch.

    Filters out None items (failed loads or length-filtered samples).

    Returns None if the entire batch is empty.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    B       = len(batch)

    coords_padded = torch.full((B, max_len, 3), pad_value, dtype=torch.float32)
    for i, item in enumerate(batch):
        L = item["length"]
        coords_padded[i, :L] = item["coords"]

    return {
        "coords":  coords_padded,    # (B, L_max, 3)
        "lengths": lengths,          # (B,)
        "names":   [item["name"] for item in batch],
    }


def build_dataloader(
    data_dir: str,
    batch_size: int = 15,
    num_workers: int = 4,
    min_length: int = 32,
    max_length: int = 256,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a DataLoader for PAR training."""
    dataset = NumpyBackboneDataset(
        data_dir   = data_dir,
        min_length = min_length,
        max_length = max_length,
        max_samples= max_samples,
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        collate_fn  = collate_backbones,
        pin_memory  = pin_memory,
        drop_last   = True,
    )


# ---------------------------------------------------------------------------
# Random backbone dataset (for unit tests / smoke tests without real data)
# ---------------------------------------------------------------------------

class RandomBackboneDataset(Dataset):
    """Generates random Cα coordinates for testing without real protein data."""

    def __init__(
        self,
        n_samples: int = 1000,
        min_length: int = 32,
        max_length: int = 256,
    ):
        self.n_samples  = n_samples
        self.min_length = min_length
        self.max_length = max_length

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        L = torch.randint(self.min_length, self.max_length + 1, (1,)).item()
        # Simulate a random walk (rough backbone geometry)
        steps  = torch.randn(L, 3) * 3.8    # ~3.8 Å per residue
        coords = steps.cumsum(dim=0)
        return {"coords": coords.float(), "length": L, "name": f"random_{idx}"}
