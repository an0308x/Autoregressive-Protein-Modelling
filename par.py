"""
PAR: Protein Autoregressive Model  (full model, Section 3).

Combines:
  1. Multi-scale downsampling (Section 3.1)
  2. AR Transformer T_θ       (Section 3.2, Eq. 4)
  3. Flow-based Decoder v_θ   (Section 3.2, Eq. 5)
  4. Exposure bias mitigation (Section 3.3)

Plus zero-shot capabilities:
  - Prompted generation        (Section 4.2)
  - Motif scaffolding          (Section 4.2)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.downsampling import (
    multiscale_downsample,
    upsample_coords,
    scales_by_length,
)
from models.ar_transformer import ARTransformer
from models.flow_decoder    import FlowDecoder
from training.loss          import build_noisy_batch, par_flow_matching_loss
from training.exposure_bias import ExposureBiasMitigation
from utils.sampling         import multiscale_sample, ode_sample, sde_sample


# ---------------------------------------------------------------------------
# PAR
# ---------------------------------------------------------------------------

class PAR(nn.Module):
    """
    Protein Autoregressive Model (PAR).

    Args:
        scale_sizes:    Fixed scale sizes S = {size(1), ..., size(n)}.
                        Default [64, 128, 256] (3-scale, by-length, best from Tab. 4).
        ar_d_model:     AR transformer hidden dim
        ar_d_cond:      AR transformer conditioning output dim
        ar_n_heads:     AR transformer attention heads
        ar_n_layers:    AR transformer depth
        dec_d_model:    Decoder hidden dim
        dec_n_heads:    Decoder attention heads
        dec_n_layers:   Decoder depth
        dropout:        Dropout for both modules
        use_ncl:        Use Noisy Context Learning
        use_ss:         Use Scheduled Sampling
    """

    def __init__(
        self,
        scale_sizes: List[int] = (64, 128, 256),
        # AR Transformer
        ar_d_model:  int = 512,
        ar_d_cond:   int = 128,
        ar_n_heads:  int = 12,
        ar_n_layers: int = 12,
        # Flow Decoder
        dec_d_model: int = 512,
        dec_n_heads: int = 12,
        dec_n_layers: int = 12,
        # Shared
        dropout:     float = 0.0,
        n_scales:    int = 3,
        # Exposure bias
        use_ncl: bool = True,
        use_ss:  bool = True,
    ):
        super().__init__()

        self.scale_sizes = list(scale_sizes)
        self.n_scales    = n_scales
        self.bos_size    = scale_sizes[0]  # size(1) = first scale

        # AR Transformer T_θ
        self.ar_transformer = ARTransformer(
            d_model  = ar_d_model,
            d_cond   = ar_d_cond,
            n_heads  = ar_n_heads,
            n_layers = ar_n_layers,
            n_scales = n_scales,
            dropout  = dropout,
        )

        # Flow-based backbone decoder v_θ
        self.flow_decoder = FlowDecoder(
            d_model  = dec_d_model,
            d_cond   = ar_d_cond,   # conditioning dim matches AR output
            n_heads  = dec_n_heads,
            n_layers = dec_n_layers,
            n_scales = n_scales,
            dropout  = dropout,
        )

        # Exposure bias mitigation
        self.ebm = ExposureBiasMitigation(
            use_ncl = use_ncl,
            use_ss  = use_ss,
        )

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        backbone: torch.Tensor,
        protein_lengths: torch.Tensor,
        use_self_conditioning: bool = True,
        sc_prob: float = 0.5,
    ) -> Dict:
        """
        Full training forward pass.

        Args:
            backbone:        (B, L_max, 3) padded Cα coordinates
            protein_lengths: (B,) actual lengths
            use_self_conditioning: use self-conditioning during training
            sc_prob:         probability of using self-conditioning

        Returns:
            dict with 'loss', 'per_scale_losses', 'velocity_preds'
        """
        device = backbone.device
        B      = backbone.shape[0]
        L      = int(protein_lengths.max().item())

        # 1. Multi-scale downsampling
        multi_scale_gt = multiscale_downsample(
            backbone, self.scale_sizes, protein_lengths, mode="by_length"
        )  # list of n tensors (B, size_i, 3)

        # 2. Build noisy inputs x^i_t for all scales
        x_t_list, epsilons, t_list = build_noisy_batch(multi_scale_gt, device)

        # 3. Exposure-bias-aware context
        #    We need decoder predictions for SS; do a quick decode pass first
        #    (without gradients for the predicted context)
        predicted_for_ss: List[Optional[torch.Tensor]] = [None] * len(multi_scale_gt)

        if self.ebm.use_ss:
            with torch.no_grad():
                # Rough predictions for scheduled sampling context
                for i, (x_t, t, x_gt) in enumerate(
                    zip(x_t_list, t_list, multi_scale_gt)
                ):
                    # Get conditioning from AR transformer (teacher-forced GT)
                    zi, _ = self.ar_transformer.forward(
                        prior_scales     = multi_scale_gt[:i],
                        target_scale_size= x_gt.shape[1],
                        protein_length   = L,
                        scale_idx        = i,
                        bos_size         = self.bos_size,
                    )
                    v_pred = self.flow_decoder(x_t, t, zi, scale_idx=i)
                    x_pred = x_t + (1.0 - t.view(-1,1,1)) * v_pred
                    predicted_for_ss[i] = x_pred

        context_scales = self.ebm(multi_scale_gt, predicted_for_ss)

        # 4. Main forward pass: AR transformer + flow decoder
        velocity_preds: List[torch.Tensor] = []

        for i in range(len(multi_scale_gt)):
            x_t   = x_t_list[i]
            t     = t_list[i]
            x_gt  = multi_scale_gt[i]

            # z^i from AR transformer (conditioned on (possibly noisy) context)
            zi, _ = self.ar_transformer.forward(
                prior_scales     = context_scales[:i],
                target_scale_size= x_gt.shape[1],
                protein_length   = L,
                scale_idx        = i,
                bos_size         = self.bos_size,
            )

            # Self-conditioning (Appendix A.3)
            self_cond: Optional[torch.Tensor] = None
            if use_self_conditioning and torch.rand(1).item() < sc_prob:
                with torch.no_grad():
                    v_sc = self.flow_decoder(x_t, t, zi, scale_idx=i)
                    self_cond = x_t + (1.0 - t.view(-1,1,1)) * v_sc

            # Predict velocity
            v = self.flow_decoder(x_t, t, zi, scale_idx=i, self_cond=self_cond)
            velocity_preds.append(v)

        # 5. Compute flow matching loss
        total_loss, per_scale_losses = par_flow_matching_loss(
            velocity_preds, multi_scale_gt, epsilons
        )

        return {
            "loss":             total_loss,
            "per_scale_losses": per_scale_losses,
            "velocity_preds":   velocity_preds,
        }

    # ------------------------------------------------------------------
    # Unconditional generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        target_length: int,
        n_samples:     int = 1,
        device:        torch.device = torch.device("cpu"),
        gamma:         float = 0.30,
        sampling_modes:       Optional[List[str]]  = None,
        n_steps_per_scale:    Optional[List[int]]   = None,
        use_self_conditioning: bool = True,
        dtype:         torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Autoregressive multi-scale backbone generation.

        Args:
            target_length:       desired protein length L
            n_samples:           batch size B
            device:              torch device
            gamma:               SDE noise parameter (0.30 default)
            sampling_modes:      per-scale "sde"/"ode" (default: sde + ode...)
            n_steps_per_scale:   steps per scale (default: 400, 2, 2, ...)
            use_self_conditioning: use self-conditioning
            dtype:               float32 or float16

        Returns:
            backbone: (B, L, 3) final Cα coordinates
            intermediates: list of per-scale outputs
        """
        self.eval()
        actual_scales = scales_by_length(self.scale_sizes, target_length)
        n = len(actual_scales)

        if sampling_modes is None:
            sampling_modes = ["sde"] + ["ode"] * (n - 1)
        if n_steps_per_scale is None:
            n_steps_per_scale = [400 if m == "sde" else 2 for m in sampling_modes]

        generated_scales: List[torch.Tensor] = []

        for i, scale_size in enumerate(actual_scales):
            # Build velocity function for this scale conditioned on prior
            def make_vfn(scale_idx: int, prior: List[torch.Tensor]):
                def vfn(x_t, t, self_cond=None):
                    zi, _ = self.ar_transformer.forward(
                        prior_scales     = prior,
                        target_scale_size= scale_size,
                        protein_length   = target_length,
                        scale_idx        = scale_idx,
                        bos_size         = self.bos_size,
                    )
                    return self.flow_decoder(
                        x_t, t, zi,
                        scale_idx  = scale_idx,
                        self_cond  = self_cond,
                    )
                return vfn

            vfn = make_vfn(i, generated_scales[:])

            x_noise = torch.randn(n_samples, scale_size, 3, device=device, dtype=dtype)

            if sampling_modes[i] == "sde":
                x_i = sde_sample(
                    vfn, x_noise, n_steps_per_scale[i],
                    gamma=gamma,
                    use_self_conditioning=use_self_conditioning,
                )
            else:
                x_i = ode_sample(
                    vfn, x_noise, n_steps_per_scale[i],
                    use_self_conditioning=use_self_conditioning,
                )

            generated_scales.append(x_i)

        final = generated_scales[-1]   # (B, L, 3)
        return final, generated_scales

    # ------------------------------------------------------------------
    # Zero-shot prompted generation  (Section 4.2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prompted_generation(
        self,
        prompt:        torch.Tensor,
        target_length: int,
        n_samples:     int = 1,
        gamma:         float = 0.30,
        n_steps:       int = 400,
        use_self_conditioning: bool = True,
    ) -> torch.Tensor:
        """
        Generate protein backbones conditioned on a coarse point-cloud prompt.

        The prompt (e.g., 16 3D points) specifies the global structural layout.
        PAR initialises the first-scale prediction with the prompt and
        autoregressively upsamples to full resolution.

        Args:
            prompt:        (1, P, 3) or (B, P, 3) prompt coordinates
            target_length: desired output length
            n_samples:     B
            gamma / n_steps: sampling parameters for fine scales

        Returns:
            (B, target_length, 3) generated backbones
        """
        self.eval()
        device = prompt.device

        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(0)
        prompt = prompt.expand(n_samples, -1, -1)  # (B, P, 3)

        actual_scales = scales_by_length(self.scale_sizes, target_length)
        n = len(actual_scales)

        # Scale 1 is initialised from the prompt
        x1 = upsample_coords(prompt, actual_scales[0])  # (B, size(1), 3)
        generated_scales = [x1]

        for i in range(1, n):
            scale_size = actual_scales[i]
            prior = generated_scales[:]

            def make_vfn(scale_idx, prior_list):
                def vfn(x_t, t, self_cond=None):
                    zi, _ = self.ar_transformer.forward(
                        prior_scales     = prior_list,
                        target_scale_size= scale_size,
                        protein_length   = target_length,
                        scale_idx        = scale_idx,
                        bos_size         = self.bos_size,
                    )
                    return self.flow_decoder(
                        x_t, t, zi, scale_idx=scale_idx, self_cond=self_cond
                    )
                return vfn

            x_noise = torch.randn(n_samples, scale_size, 3, device=device)
            x_i = ode_sample(
                make_vfn(i, prior), x_noise, n_steps,
                use_self_conditioning=use_self_conditioning,
            )
            generated_scales.append(x_i)

        return generated_scales[-1]  # (B, target_length, 3)

    # ------------------------------------------------------------------
    # Zero-shot motif scaffolding  (Section 4.2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def motif_scaffolding(
        self,
        motif_coords:  torch.Tensor,
        motif_mask:    torch.Tensor,
        target_length: int,
        n_samples:     int = 1,
        gamma:         float = 0.30,
        n_steps:       int = 400,
    ) -> torch.Tensor:
        """
        Zero-shot motif scaffolding via teacher-forcing of motif coordinates
        at each scale (Section 4.2).

        No fine-tuning required.

        Args:
            motif_coords:  (M, 3) or (1, M, 3) motif Cα coordinates
            motif_mask:    (target_length,) boolean mask, True = motif residue
            target_length: L
            n_samples:     B
            gamma / n_steps: sampling parameters

        Returns:
            (B, target_length, 3) scaffolded backbones
        """
        self.eval()
        device = motif_coords.device

        if motif_coords.dim() == 2:
            motif_coords = motif_coords.unsqueeze(0)
        motif_coords = motif_coords.expand(n_samples, -1, -1)  # (B, M, 3)

        actual_scales = scales_by_length(self.scale_sizes, target_length)
        n = len(actual_scales)
        generated_scales: List[torch.Tensor] = []

        for i in range(n):
            scale_size = actual_scales[i]
            prior = generated_scales[:]

            def make_vfn(si, pl):
                def vfn(x_t, t, self_cond=None):
                    zi, _ = self.ar_transformer.forward(
                        prior_scales     = pl,
                        target_scale_size= si,
                        protein_length   = target_length,
                        scale_idx        = i,
                        bos_size         = self.bos_size,
                    )
                    return self.flow_decoder(
                        x_t, t, zi, scale_idx=i, self_cond=self_cond
                    )
                return vfn

            x_noise = torch.randn(n_samples, scale_size, 3, device=device)

            if i == 0 or sampling_modes is None:
                x_i = sde_sample(make_vfn(scale_size, prior), x_noise,
                                  n_steps, gamma=gamma)
            else:
                x_i = ode_sample(make_vfn(scale_size, prior), x_noise, 2)

            # Teacher-force motif coordinates at this scale
            # Downsample mask to current scale
            mask_scale = upsample_coords(
                motif_mask.float().unsqueeze(-1).unsqueeze(0).expand(n_samples,-1,-1),
                scale_size
            ).squeeze(-1) > 0.5                            # (B, scale_size)

            motif_at_scale = upsample_coords(motif_coords, scale_size)  # (B, scale_size, 3)
            x_i[mask_scale] = motif_at_scale[mask_scale]

            generated_scales.append(x_i)

        return generated_scales[-1]

    # ------------------------------------------------------------------
    # Config factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "PAR":
        return cls(**config)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "PAR":
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg  = ckpt.get("config", {})
        cfg.update(kwargs)
        model = cls.from_config(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    def save(self, path: str, config: Optional[dict] = None):
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": config or {},
        }, path)

    def count_parameters(self) -> dict:
        ar_params  = sum(p.numel() for p in self.ar_transformer.parameters())
        dec_params = sum(p.numel() for p in self.flow_decoder.parameters())
        return {
            "ar_transformer": ar_params,
            "flow_decoder":   dec_params,
            "total":          ar_params + dec_params,
        }
