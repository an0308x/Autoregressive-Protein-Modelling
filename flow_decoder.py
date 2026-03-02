"""
Flow-based Backbone Decoder v_θ  (Section 3.2, Eq. 5).

Conditioned on the scale-wise embedding z^i produced by the AR transformer,
this network directly predicts Cα backbone coordinates in continuous space
via flow matching — no discretization required.

Flow matching objective (Eq. 5):
    L(θ) = E_x [ (1/n) Σ_i (1/size(i)) E_{t,ε} ||v_θ(x^i_t, t, z^i) - (x^i - ε)||² ]

where x^i_t = t·x^i + (1-t)·ε,  ε ~ N(0,I),  t ~ p(t).

Conditioning z^i is injected via adaptive layer norm (AdaLN), following DiT.
Self-conditioning is also supported (Section A.3, Eq. 7).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Adaptive Layer Norm (AdaLN)  — conditioning injection
# ---------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
    """
    Adaptive LayerNorm: scale and shift parameters predicted from conditioning.
    Used in DiT [Peebles & Xie 2023] and here to inject z^i into v_θ.
    """

    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # Project conditioning to 2·d_model (scale + shift)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 2 * d_model),
        )
        # Initialise projection to near-zero so training is stable at init
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, S, D)
            cond: (B, S, d_cond) or (B, 1, d_cond)  per-token conditioning
        Returns:
            (B, S, D)
        """
        gamma_beta = self.proj(cond)                      # (B, S or 1, 2D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)         # each (B, S or 1, D)
        return self.norm(x) * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar time t ∈ [0,1] → (B, d_time) sinusoidal embedding."""

    def __init__(self, d_time: int):
        super().__init__()
        assert d_time % 2 == 0
        self.d_time = d_time

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (B, 1)
        Returns:
            (B, d_time)
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        half = self.d_time // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )  # (half,)
        emb = t[:, None] * freqs[None, :]                # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, d_time)


# ---------------------------------------------------------------------------
# Decoder transformer block with AdaLN conditioning
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    Transformer block for the flow decoder, conditioned via AdaLN.
    Receives per-token conditioning (z^i projected + time embedding).
    """

    def __init__(self, d_model: int, d_cond: int, n_heads: int,
                 dropout: float = 0.0, ffn_multiplier: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.adaln1 = AdaLayerNorm(d_model, d_cond)
        self.qkv    = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

        self.adaln2 = AdaLayerNorm(d_model, d_cond)
        dim_ff = d_model * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, S, D)
            cond: (B, S, d_cond)
        """
        B, S, D = x.shape
        H, d = self.n_heads, self.d_head

        # Self-attention with AdaLN
        residual = x
        x = self.adaln1(x, cond)
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, S, H, d).transpose(1, 2)
        k = k.view(B, S, H, d).transpose(1, 2)
        v = v.view(B, S, H, d).transpose(1, 2)
        attn = F.softmax(torch.matmul(q, k.transpose(-2,-1)) * self.scale, dim=-1)
        attn = self.dropout(attn)
        out  = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        x = residual + self.attn_out(out)

        # FFN with AdaLN
        residual = x
        x = self.adaln2(x, cond)
        x = residual + self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# Flow Decoder
# ---------------------------------------------------------------------------

class FlowDecoder(nn.Module):
    """
    Flow-based backbone decoder v_θ  (Eq. 5).

    Maps (noisy backbone x^i_t, time t, conditioning z^i) → velocity vector
    that predicts (x^i − ε), i.e., the direction from noise to clean data.

    Architecture: Transformer with AdaLN conditioning.
    Per-scale shared weights; scale identity supplied via scale_emb.

    Args:
        d_model:    hidden dimension
        d_cond:     dimension of z^i coming from AR transformer
        d_time:     sinusoidal time embedding dimension
        n_heads:    attention heads
        n_layers:   transformer depth
        n_scales:   number of scales (for scale embedding)
        dropout:    dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        d_cond:  int = 128,
        d_time:  int = 196,
        n_heads: int = 12,
        n_layers: int = 12,
        n_scales: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Input: noisy coords (3) → d_model
        self.input_proj = nn.Linear(3, d_model)

        # Time embedding: t ∈ [0,1] → d_time, projected to d_cond
        self.time_emb  = SinusoidalTimeEmbedding(d_time)
        self.time_proj = nn.Linear(d_time, d_cond)

        # Scale embedding (shared decoder distinguishes scales via this)
        self.scale_emb = nn.Embedding(n_scales, d_cond)

        # Project conditioning z^i to d_cond (in case AR outputs different dim)
        self.cond_proj = nn.Linear(d_cond, d_cond)

        # Positional embedding (standard learnable for the decoder)
        self.pos_emb_proj = nn.Linear(196, d_cond)  # sinusoidal pos → d_cond
        self.pos_sinusoidal = SinusoidalTimeEmbedding(196)  # reuse for positions

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, d_cond, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 3)   # → velocity in R^3

    # ------------------------------------------------------------------
    def _get_conditioning(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        scale_idx: int,
        self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build per-token conditioning vector from t, z^i, scale_id,
        and optional self-conditioning input.

        Args:
            t:          (B,) time
            z:          (B, S, d_cond) conditioning from AR transformer
            scale_idx:  current scale (int)
            self_cond:  (B, S, 3) self-conditioning estimate (optional)
        Returns:
            (B, S, d_cond) combined conditioning
        """
        B, S, _ = z.shape
        device = z.device

        # Time embedding → d_cond, broadcast over sequence
        te  = self.time_emb(t)                                   # (B, d_time)
        te  = self.time_proj(te).unsqueeze(1).expand(-1, S, -1)  # (B, S, d_cond)

        # Scale embedding
        se  = self.scale_emb(
            torch.tensor(scale_idx, device=device)
        ).unsqueeze(0).unsqueeze(0).expand(B, S, -1)             # (B, S, d_cond)

        # AR conditioning
        zc  = self.cond_proj(z)                                  # (B, S, d_cond)

        cond = te + se + zc

        # Self-conditioning (Eq. 7 / Appendix A.3)
        if self_cond is not None:
            sc_proj = self.input_proj(self_cond)                 # (B, S, d_model)
            # Reduce to d_cond via a simple mean-pool + linear if needed
            # Here we reuse input_proj size and add to conditioning
            cond = cond + sc_proj[:, :, : cond.shape[-1]]

        return cond

    # ------------------------------------------------------------------
    def forward(
        self,
        x_t: torch.Tensor,
        t:   torch.Tensor,
        z:   torch.Tensor,
        scale_idx: int,
        self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity  v_θ(x^i_t, t, z^i) ≈ x^i − ε.

        Args:
            x_t:        (B, S, 3) noisy backbone at time t
            t:          (B,) time values in [0, 1]
            z:          (B, S, d_cond) conditioning from AR transformer
            scale_idx:  current scale index (for scale embedding)
            self_cond:  (B, S, 3) optional self-conditioning estimate

        Returns:
            velocity:   (B, S, 3)
        """
        # Project noisy input to d_model
        x = self.input_proj(x_t)                  # (B, S, d_model)

        # Build conditioning
        cond = self._get_conditioning(t, z, scale_idx, self_cond)  # (B, S, d_cond)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond)

        x = self.norm_out(x)
        velocity = self.out_proj(x)               # (B, S, 3)
        return velocity

    # ------------------------------------------------------------------
    def predict_clean(
        self,
        x_t: torch.Tensor,
        t:   torch.Tensor,
        z:   torch.Tensor,
        scale_idx: int,
        self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict clean structure  x̂ = x_t + (1−t)·v_θ(x_t, t, z)
        Used for self-conditioning and scheduled sampling.   (Eq. 7 in appendix)
        """
        v = self.forward(x_t, t, z, scale_idx, self_cond)
        # x_t = t·x + (1-t)·ε  ⟹  x̂ = x_t + (1-t)·v  when v = x-ε
        return x_t + (1.0 - t.view(-1, 1, 1)) * v
