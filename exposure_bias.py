"""
Exposure Bias Mitigation  (Section 3.3)

Two complementary techniques to bridge the training / inference gap in PAR:

1. Noisy Context Learning (NCL)
   ─────────────────────────────
   Prior-scale inputs x^i are corrupted with Gaussian noise during training:
       x^i_ncl = w^i_ncl · x^i + (1 − w^i_ncl) · ε^i_ncl
   where w^i_ncl ~ Uniform[0,1] and ε^i_ncl ~ N(0,I).
   The AR transformer is trained on these corrupted inputs, making it robust
   to imperfect context at inference time.

2. Scheduled Sampling (SS)
   ────────────────────────
   With probability 0.5, the ground-truth context x^i is replaced by the
   decoder's own prediction x^i_pred at later scales:
       x^i_pred = x^i_t + (1−t^i) · v_θ(x^i_t, t^i, z^i)
   This exposes the model to its own outputs during training.
   NCL can be combined with SS by additionally noising x^i_pred.

Reference: Table 3 — NCL alone reduces sc-RMSD from 2.20 → 1.58;
           NCL + SS further reduces it to 1.48.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Noisy Context Learning (NCL)
# ---------------------------------------------------------------------------

def noisy_context_learning(
    coords: torch.Tensor,
    w_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    Corrupt a coordinate tensor with Gaussian noise for NCL.

    x_ncl = w · x + (1 − w) · ε,   w ~ Uniform(w_range),  ε ~ N(0,I)

    Args:
        coords:   (B, S, 3) clean Cα coordinates
        w_range:  range for noise weight w; (0,1) means full range

    Returns:
        (B, S, 3) corrupted coordinates (same shape)
    """
    w = torch.rand(coords.shape[0], 1, 1,
                   device=coords.device, dtype=coords.dtype)
    w = w * (w_range[1] - w_range[0]) + w_range[0]     # rescale to w_range
    eps = torch.randn_like(coords)
    return w * coords + (1.0 - w) * eps


def apply_ncl_to_scales(
    multi_scale_coords: List[torch.Tensor],
    w_range: Tuple[float, float] = (0.0, 1.0),
) -> List[torch.Tensor]:
    """
    Apply NCL independently to each scale's coordinate tensor.

    Args:
        multi_scale_coords: list of n tensors (B, size_i, 3)
        w_range:            noise weight range

    Returns:
        list of n corrupted tensors, same shapes
    """
    return [noisy_context_learning(x, w_range) for x in multi_scale_coords]


# ---------------------------------------------------------------------------
# Scheduled Sampling (SS)
# ---------------------------------------------------------------------------

def scheduled_sampling_context(
    ground_truth: torch.Tensor,
    model_prediction: torch.Tensor,
    p_replace: float = 0.5,
    apply_ncl: bool = True,
    ncl_w_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    Replace ground-truth context with model predictions with probability p_replace.

    Optionally applies NCL noise to the model predictions (combining SS + NCL).

    Args:
        ground_truth:      (B, S, 3) true coordinates at scale i
        model_prediction:  (B, S, 3) decoder prediction x^i_pred
        p_replace:         probability of using model prediction (default 0.5)
        apply_ncl:         if True, also corrupt the model prediction with NCL
        ncl_w_range:       noise weight range for NCL on the prediction

    Returns:
        (B, S, 3) context to feed into the AR transformer
    """
    use_pred = torch.rand(1).item() < p_replace

    if use_pred:
        ctx = model_prediction
        if apply_ncl:
            ctx = noisy_context_learning(ctx, ncl_w_range)
    else:
        ctx = ground_truth

    return ctx


# ---------------------------------------------------------------------------
# Full exposure-bias-aware training context builder
# ---------------------------------------------------------------------------

class ExposureBiasMitigation(nn.Module):
    """
    Manages NCL + SS during PAR training.

    Usage (inside training loop):
        ebm = ExposureBiasMitigation(use_ncl=True, use_ss=True)
        context_scales = ebm(
            clean_scales,       # list of ground-truth scale tensors
            predicted_scales,   # list of model predictions (or None before generation)
        )
        # feed context_scales to AR transformer
    """

    def __init__(
        self,
        use_ncl: bool = True,
        use_ss:  bool = True,
        ncl_w_range: Tuple[float, float] = (0.0, 1.0),
        ss_p_replace: float = 0.5,
        ss_apply_ncl: bool = True,
    ):
        super().__init__()
        self.use_ncl      = use_ncl
        self.use_ss       = use_ss
        self.ncl_w_range  = ncl_w_range
        self.ss_p_replace = ss_p_replace
        self.ss_apply_ncl = ss_apply_ncl

    def forward(
        self,
        clean_scales: List[torch.Tensor],
        predicted_scales: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        """
        Build the context list for the AR transformer with exposure-bias
        mitigation applied.

        Args:
            clean_scales:     list of n tensors (B, size_i, 3) — ground truth
            predicted_scales: list of n tensors (or None entries) — decoder preds.
                              If None or entry is None, falls back to ground truth.

        Returns:
            list of n context tensors (B, size_i, 3)
        """
        n = len(clean_scales)
        context: List[torch.Tensor] = []

        for i in range(n):
            gt   = clean_scales[i]
            pred = (predicted_scales[i]
                    if predicted_scales is not None and predicted_scales[i] is not None
                    else None)

            if self.use_ss and pred is not None:
                # Scheduled sampling: maybe replace gt with pred
                ctx = scheduled_sampling_context(
                    gt, pred,
                    p_replace=self.ss_p_replace,
                    apply_ncl=self.ss_apply_ncl and self.use_ncl,
                    ncl_w_range=self.ncl_w_range,
                )
            elif self.use_ncl:
                # NCL only
                ctx = noisy_context_learning(gt, self.ncl_w_range)
            else:
                # Teacher forcing (baseline)
                ctx = gt

            context.append(ctx)

        return context
