"""
Flow Matching training objective for PAR  (Section 3.2, Eq. 5).

L(θ) = E_x [ (1/n) Σ_i (1/size(i)) E_{t,ε} ||v_θ(x^i_t, t^i, z^i) − (x^i − ε^i)||² ]

where:
  x^i_t = t^i · x^i + (1 − t^i) · ε^i      (linear interpolation)
  ε^i ~ N(0, I)
  t^i ~ p(t)   (uniform in [0,1] following Geffner et al. / Proteina)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time sampling
# ---------------------------------------------------------------------------

def sample_times(
    batch_size: int,
    n_per_sample: int = 1,
    device: torch.device = torch.device("cpu"),
    t_min: float = 1e-4,
    t_max: float = 1.0 - 1e-4,
) -> torch.Tensor:
    """
    Sample t ~ Uniform(t_min, t_max) for each item in the batch.

    Args:
        batch_size:    B
        n_per_sample:  draw n_per_sample times per item (default 1)
        device:        torch device

    Returns:
        (B,) or (B * n_per_sample,) tensor of time values
    """
    t = torch.rand(batch_size * n_per_sample, device=device)
    return t * (t_max - t_min) + t_min


# ---------------------------------------------------------------------------
# Linear interpolation (forward process)
# ---------------------------------------------------------------------------

def linear_interpolate(
    x_clean: torch.Tensor,
    epsilon: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    x_t = t · x + (1 − t) · ε    (conditional flow matching interpolation)

    Args:
        x_clean: (B, S, 3) clean backbone coordinates
        epsilon:  (B, S, 3) standard Gaussian noise
        t:        (B,) time values in [0, 1]

    Returns:
        (B, S, 3) interpolated noisy sample
    """
    t_bc = t.view(-1, 1, 1)   # broadcast over S and 3
    return t_bc * x_clean + (1.0 - t_bc) * epsilon


# ---------------------------------------------------------------------------
# Per-scale flow matching loss
# ---------------------------------------------------------------------------

def flow_matching_loss_scale(
    velocity_pred: torch.Tensor,
    x_clean: torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss for a single scale: ||v_θ − (x − ε)||²

    The target is (x − ε), the direction from noise to clean data.

    Args:
        velocity_pred: (B, S, 3) predicted velocity
        x_clean:       (B, S, 3) clean coordinates at this scale
        epsilon:       (B, S, 3) noise sample

    Returns:
        scalar loss (mean over B, S, 3)
    """
    target = x_clean - epsilon                             # (B, S, 3)
    return F.mse_loss(velocity_pred, target, reduction="mean")


# ---------------------------------------------------------------------------
# Full PAR training loss  (Eq. 5)
# ---------------------------------------------------------------------------

def par_flow_matching_loss(
    velocity_preds: List[torch.Tensor],
    multi_scale_clean: List[torch.Tensor],
    epsilons: List[torch.Tensor],
    length_weights: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Aggregate flow matching loss across all n scales.

    L = (1/n) Σ_i  (1/size(i)) · ||v_θ^i − (x^i − ε^i)||²_mean

    The (1/size(i)) normalisation keeps the gradient scale comparable
    across scales of very different lengths.

    Args:
        velocity_preds:      list of n tensors (B, size_i, 3) — decoder outputs
        multi_scale_clean:   list of n tensors (B, size_i, 3) — GT at each scale
        epsilons:            list of n tensors (B, size_i, 3) — noise samples
        length_weights:      optional per-scale weights [1/size(i)]; computed
                             automatically from scale sizes if None.

    Returns:
        total_loss:   scalar
        per_scale_losses: list of n scalar tensors (for logging)
    """
    n = len(velocity_preds)
    assert len(multi_scale_clean) == n and len(epsilons) == n

    per_scale: List[torch.Tensor] = []

    for i in range(n):
        scale_i  = multi_scale_clean[i].shape[1]       # size(i)
        w_i      = (length_weights[i]
                    if length_weights is not None
                    else 1.0 / scale_i)
        loss_i   = flow_matching_loss_scale(
            velocity_preds[i], multi_scale_clean[i], epsilons[i]
        )
        per_scale.append(w_i * loss_i)

    total_loss = sum(per_scale) / n
    return total_loss, per_scale


# ---------------------------------------------------------------------------
# Convenience: sample noise + build training batch for one forward pass
# ---------------------------------------------------------------------------

def build_noisy_batch(
    multi_scale_clean: List[torch.Tensor],
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    For each scale, sample (epsilon, t) and compute x_t.

    Returns:
        x_t_list:  list of n (B, size_i, 3) noisy inputs
        epsilons:  list of n (B, size_i, 3)
        t_list:    list of n (B,) time values
    """
    x_t_list, epsilons, t_list = [], [], []
    for x_clean in multi_scale_clean:
        B, S, _ = x_clean.shape
        eps = torch.randn_like(x_clean)
        t   = sample_times(B, device=device)
        x_t = linear_interpolate(x_clean, eps, t)
        x_t_list.append(x_t)
        epsilons.append(eps)
        t_list.append(t)
    return x_t_list, epsilons, t_list
