"""
Multi-scale protein backbone downsampling (Section 3.1, Eq. 2).

Given a backbone x ∈ R^{L×3}, produces a hierarchy of coarse-to-fine
representations via 1D linear interpolation along the sequence dimension.

    qdecompose: x → {x¹, x², ..., xⁿ}
    xⁱ = Down(x, size(i))  ∈ R^{size(i)×3}

Key property: 1D sequence downsampling preserves pairwise spatial
relationships (LDDT=1 across all scales; see Appendix C.8).
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Core interpolation helpers
# ---------------------------------------------------------------------------

def interpolate_coords(coords: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Linearly interpolate Cα coordinate sequence to a different length.

    Args:
        coords: (B, L, 3) or (L, 3)  backbone Cα coordinates
        target_size: desired output sequence length

    Returns:
        (B, target_size, 3) or (target_size, 3)
    """
    squeeze = coords.dim() == 2
    if squeeze:
        coords = coords.unsqueeze(0)                   # (1, L, 3)

    B, L, _ = coords.shape
    if L == target_size:
        out = coords.clone()
    else:
        # F.interpolate expects (B, C, L)
        x = coords.permute(0, 2, 1)                    # (B, 3, L)
        x = F.interpolate(x, size=target_size, mode="linear", align_corners=True)
        out = x.permute(0, 2, 1)                       # (B, target_size, 3)

    return out.squeeze(0) if squeeze else out


def interpolate_positions(length: int, target_size: int,
                          device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Compute interpolated position IDs for scale i.

    Uniformly samples target_size positions from [1, length] so that the
    relative positions reflect the original sequence layout at every scale.
    (Section 3.2, "Interpolated Position Embedding")

    Returns:
        (target_size,) float tensor of position indices in [1, length]
    """
    return torch.linspace(1, length, steps=target_size, device=device)


# ---------------------------------------------------------------------------
# Scale configuration helpers
# ---------------------------------------------------------------------------

def scales_by_length(scale_sizes: List[int], protein_length: int) -> List[int]:
    """
    Filter fixed scale sizes to those ≤ protein_length, always appending L.

    Example: scale_sizes=[64,128,256], L=200 → [64, 128, 200]
    """
    sizes = [s for s in scale_sizes if s < protein_length]
    sizes.append(protein_length)
    return sizes


def scales_by_ratio(ratios: List[float], protein_length: int) -> List[int]:
    """
    Compute scale sizes as fractions of protein length.

    Example: ratios=[0.25, 0.5, 1.0], L=200 → [50, 100, 200]
    """
    sizes = [max(1, int(r * protein_length)) for r in ratios]
    sizes[-1] = protein_length  # ensure final scale equals full length
    return sizes


# ---------------------------------------------------------------------------
# Main downsampling function
# ---------------------------------------------------------------------------

def multiscale_downsample(
    backbone: torch.Tensor,
    scale_sizes: List[int],
    protein_lengths: torch.Tensor,
    mode: str = "by_length",
) -> List[torch.Tensor]:
    """
    Produce multi-scale representations of a batch of protein backbones.

    Args:
        backbone:        (B, L_max, 3) padded Cα coordinates
        scale_sizes:     list of target sizes (for by_length mode) or
                         ratios in (0,1] (for by_ratio mode)
        protein_lengths: (B,) actual lengths before padding
        mode:            "by_length" (default, better empirically) or "by_ratio"

    Returns:
        List of n tensors, each (B, size_i, 3).
        The last element always corresponds to the full (but possibly
        length-normalised) backbone.

    Notes:
        - Padding is handled per-sample; only valid residues are interpolated.
        - This function is non-parametric and deterministic (no learned params).
    """
    B, L_max, _ = backbone.shape
    scales_batch: List[List[torch.Tensor]] = [[] for _ in range(B)]

    for b in range(B):
        L = int(protein_lengths[b].item())
        coords = backbone[b, :L, :]  # (L, 3) — strip padding

        if mode == "by_length":
            sizes = scales_by_length(scale_sizes, L)
        elif mode == "by_ratio":
            sizes = scales_by_ratio(scale_sizes, L)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for s in sizes:
            scales_batch[b].append(interpolate_coords(coords, s))  # (s, 3)

    # Pad each scale to the maximum size across the batch so we can stack
    n_scales = len(scales_batch[0])
    result: List[torch.Tensor] = []
    for i in range(n_scales):
        max_size = max(scales_batch[b][i].shape[0] for b in range(B))
        padded = torch.zeros(B, max_size, 3,
                             device=backbone.device, dtype=backbone.dtype)
        for b in range(B):
            xi = scales_batch[b][i]          # (size_i_b, 3)
            padded[b, : xi.shape[0]] = xi
        result.append(padded)

    return result


# ---------------------------------------------------------------------------
# Upsampling (used during AR inference, Eq. 4)
# ---------------------------------------------------------------------------

def upsample_coords(coords: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Upsample Cα coordinate sequence to target_size via linear interpolation.

    Equivalent to interpolate_coords; provided as a named alias to match
    the paper's notation:  Up(x^{i-1}, size(i))

    Args:
        coords: (B, S, 3) or (S, 3)
        target_size: desired output length

    Returns:
        (B, target_size, 3) or (target_size, 3)
    """
    return interpolate_coords(coords, target_size)


# ---------------------------------------------------------------------------
# Spatial relationship verification (Appendix C.8)
# ---------------------------------------------------------------------------

def pairwise_distance_map(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute all-pair Cα distance matrix.

    Args:
        coords: (L, 3)
    Returns:
        (L, L) distance matrix
    """
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)   # (L, L, 3)
    return diff.norm(dim=-1)


def verify_downsampling_preserves_spatial_relations(
    coords: torch.Tensor, size: int
) -> Tuple[float, float]:
    """
    Reproduce Appendix C.8 sanity check.

    Compares pairwise distances computed from:
      (a) downsampled 1D coordinate sequence
      (b) bicubic-downsampled full-resolution 2D distance map

    Returns (rmse, lddt) — lower RMSE / LDDT=1.0 means good preservation.
    """
    L = coords.shape[0]
    # (a) downsample coordinate sequence, then compute distances
    ds_coords = interpolate_coords(coords, size)        # (size, 3)
    dist_a = pairwise_distance_map(ds_coords)           # (size, size)

    # (b) compute full dist map, then bicubic-downsample it
    dist_full = pairwise_distance_map(coords)           # (L, L)
    dist_b = F.interpolate(
        dist_full.unsqueeze(0).unsqueeze(0),            # (1, 1, L, L)
        size=(size, size),
        mode="bicubic",
        align_corners=True,
    ).squeeze()                                         # (size, size)

    rmse = (dist_a - dist_b).pow(2).mean().sqrt().item()

    # lDDT: fraction of pairs whose distance error < 0.5 Å
    lddt = ((dist_a - dist_b).abs() < 0.5).float().mean().item()
    return rmse, lddt
