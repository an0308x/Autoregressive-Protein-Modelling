"""
Autoregressive Transformer T_θ  (Section 3.2, Eq. 4).

Encodes all prior coarse scales and produces a per-scale conditional
embedding z^i that guides the flow-based backbone decoder at scale i.

Architecture:
  - Standard non-equivariant transformer (causal across scales,
    bidirectional within each scale)
  - Inputs: upsampled prior-scale structures concatenated along seq dim
  - Outputs: conditional embedding z^i  (same shape as the current scale)
  - BOS token: learnable embedding replaces the missing x^0 context
  - Interpolated positional encoding to preserve relative residue positions
  - Scale embedding to distinguish scale-specific distributions
  - KV-cache support for efficient autoregressive inference
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.downsampling import upsample_coords, interpolate_positions


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class InterpolatedPositionEncoding(nn.Module):
    """
    Sinusoidal position encoding using interpolated position indices.

    For scale i with size s, the indices are linspace(1, L, s) so that
    coarse scales have wide spacing (global context) and fine scales have
    dense spacing (local detail).  (Section 3.2)
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (B, S) float tensor of position indices in [1, L]
        Returns:
            (B, S, d_model) positional encodings
        """
        B, S = positions.shape
        d = self.d_model
        # Standard sinusoidal: dim indices for sin/cos
        div_term = torch.exp(
            torch.arange(0, d, 2, device=positions.device, dtype=torch.float)
            * -(math.log(10000.0) / d)
        )  # (d/2,)

        pos = positions.unsqueeze(-1).float()              # (B, S, 1)
        enc = torch.zeros(B, S, d, device=positions.device)
        enc[:, :, 0::2] = torch.sin(pos * div_term)
        enc[:, :, 1::2] = torch.cos(pos * div_term[: d // 2])
        return enc


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:        (B, S, D)
            mask:     (B, 1, S, S) attention mask; True = keep
            kv_cache: cached (K, V) from previous steps; each (B, n_heads, S_past, d_head)
        Returns:
            output (B, S, D), updated (K, V) cache
        """
        B, S, D = x.shape
        H, d = self.n_heads, self.d_head

        q, k, v = self.qkv(x).split(D, dim=-1)           # each (B, S, D)
        q = q.view(B, S, H, d).transpose(1, 2)            # (B, H, S, d)
        k = k.view(B, S, H, d).transpose(1, 2)
        v = v.view(B, S, H, d).transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, S, S_full)

        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                        # (B, H, S, d)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out), new_cache


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 ffn_multiplier: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        dim_ff = d_model * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        residual = x
        x = self.norm1(x)
        x, new_cache = self.attn(x, mask=mask, kv_cache=kv_cache)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x, new_cache


# ---------------------------------------------------------------------------
# Main Autoregressive Transformer
# ---------------------------------------------------------------------------

class ARTransformer(nn.Module):
    """
    Autoregressive Transformer T_θ.

    z^i = T_θ( bos, Up(x¹, size(2)), ..., Up(x^{i-1}, size(i)) )   (Eq. 4)

    The model processes a concatenated sequence of prior-scale structures
    (all upsampled to the current scale's size) and outputs a conditional
    embedding z^i to guide the flow decoder at scale i.

    Causal masking is applied across scales (block-causal):
      - Tokens within the same scale can attend to each other (bidirectional)
      - Tokens can only attend to tokens from the same or earlier scales

    Args:
        d_model:      transformer hidden dimension
        d_cond:       output conditioning dimension (fed to flow decoder)
        n_heads:      number of attention heads
        n_layers:     number of transformer layers
        n_scales:     number of autoregressive scales
        dropout:      dropout rate
        max_seq_len:  maximum total concatenated sequence length
    """

    def __init__(
        self,
        d_model: int = 512,
        d_cond: int = 128,
        n_heads: int = 12,
        n_layers: int = 12,
        n_scales: int = 3,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_cond   = d_cond
        self.n_scales = n_scales

        # Input projection: 3D coords → d_model
        self.input_proj = nn.Linear(3, d_model)

        # BOS token: one learnable embedding per scale-1 (replaces missing x^0)
        # Shape (size(1), d_model) — size(1) is the smallest scale size.
        # We register it as a parameter; at forward time we repeat over batch.
        self.bos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Scale embedding: one vector per scale to help the model distinguish scales
        self.scale_emb = nn.Embedding(n_scales, d_model)

        # Positional encoding
        self.pos_enc = InterpolatedPositionEncoding(d_model, max_len=max_seq_len)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        # Output projection: d_model → d_cond
        self.out_proj = nn.Linear(d_model, d_cond)

    # ------------------------------------------------------------------
    def _build_block_causal_mask(
        self,
        scale_sizes: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Block-causal attention mask.

        Tokens within scale i can attend to all tokens from scales ≤ i.
        Tokens within the same scale are fully connected (bidirectional).

        Returns:
            (1, 1, total_len, total_len) boolean mask (True = attend)
        """
        total = sum(scale_sizes)
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)

        start = 0
        for i, si in enumerate(scale_sizes):
            end = start + si
            # Scale i can attend to all previous scales + itself
            prev_end = sum(scale_sizes[:i + 1])
            mask[start:end, :prev_end] = True
            start = end

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, total, total)

    # ------------------------------------------------------------------
    def forward(
        self,
        prior_scales: List[torch.Tensor],
        target_scale_size: int,
        protein_length: int,
        scale_idx: int,
        bos_size: int,
        kv_caches: Optional[List] = None,
    ) -> Tuple[torch.Tensor, List]:
        """
        Compute conditional embedding z^i for scale i.

        Args:
            prior_scales:      list of i-1 tensors (B, size_j, 3) for j < i,
                               OR empty list for the first scale (BOS only)
            target_scale_size: size(i), the current scale to condition on
            protein_length:    full protein length L (for positional encoding)
            scale_idx:         current scale index i (0-based)
            bos_size:          size of BOS (= size(1), the first scale size)
            kv_caches:         list of per-layer KV caches (None on first call)

        Returns:
            z^i: (B, target_scale_size, d_cond) conditioning embedding
            new_kv_caches: updated list of KV caches for subsequent calls
        """
        device = self.bos.device

        # ---- Build input sequence ----------------------------------------
        # Scale 0: BOS  (B, bos_size, D)
        # Scale i>0: Up(x^{i-1}, size(i)) projected to D

        tokens_list: List[torch.Tensor]  = []
        size_list:   List[int]           = []

        if scale_idx == 0 or len(prior_scales) == 0:
            # First scale — use BOS replicated to bos_size
            # bos is (1, 1, D); expand to (B, bos_size, D)
            B = 1 if len(prior_scales) == 0 else prior_scales[0].shape[0]
            bos_tok = self.bos.expand(B, bos_size, -1)             # (B, bos_size, D)
            tokens_list.append(bos_tok)
            size_list.append(bos_size)
        else:
            # Build BOS + each prior scale upsampled to current size
            B = prior_scales[0].shape[0]

            # BOS (treated as scale 0)
            bos_tok = self.bos.expand(B, bos_size, -1)
            tokens_list.append(bos_tok)
            size_list.append(bos_size)

            for j, xj in enumerate(prior_scales):
                # Upsample x^j → size(scale_idx)
                upsampled = upsample_coords(xj, target_scale_size)  # (B, size_i, 3)
                proj = self.input_proj(upsampled)                    # (B, size_i, D)
                tokens_list.append(proj)
                size_list.append(target_scale_size)

        # Add scale embeddings and positional encodings to each segment
        processed: List[torch.Tensor] = []
        for seg_idx, (tok, sz) in enumerate(zip(tokens_list, size_list)):
            # Scale embedding (broadcast over sequence)
            se = self.scale_emb(
                torch.tensor(seg_idx, device=device)
            )  # (D,)
            tok = tok + se.unsqueeze(0).unsqueeze(0)

            # Interpolated position encoding
            pos = interpolate_positions(protein_length, sz, device=device)  # (sz,)
            pos_enc = self.pos_enc(pos.unsqueeze(0).expand(B, -1))          # (B, sz, D)
            tok = tok + pos_enc
            processed.append(tok)

        x = torch.cat(processed, dim=1)  # (B, total_len, D)

        # ---- Attention mask ----------------------------------------------
        mask = self._build_block_causal_mask(size_list, device=device)

        # ---- Transformer forward -----------------------------------------
        new_caches = []
        for layer_idx, layer in enumerate(self.layers):
            cache = kv_caches[layer_idx] if kv_caches is not None else None
            x, new_cache = layer(x, mask=mask, kv_cache=cache)
            new_caches.append(new_cache)

        x = self.norm_out(x)

        # ---- Extract output for current (final) scale segment ------------
        # The last size_list[-1] tokens correspond to the current scale
        if scale_idx == 0:
            zi_raw = x  # (B, bos_size, D) — full sequence is BOS
        else:
            zi_raw = x[:, -target_scale_size:, :]  # (B, size_i, D)

        zi = self.out_proj(zi_raw)  # (B, size_i, d_cond)
        return zi, new_caches

    # ------------------------------------------------------------------
    def forward_all_scales(
        self,
        multi_scale_coords: List[torch.Tensor],
        protein_length: int,
        bos_size: int,
    ) -> List[torch.Tensor]:
        """
        Full forward pass over all n scales during training.

        Args:
            multi_scale_coords: list of n tensors (B, size_i, 3)
            protein_length:     L
            bos_size:           size(1)

        Returns:
            list of n tensors z^i each (B, size_i, d_cond)
        """
        n = len(multi_scale_coords)
        z_list: List[torch.Tensor] = []

        for i in range(n):
            prior = multi_scale_coords[:i]  # all scales before i
            zi, _ = self.forward(
                prior_scales=prior,
                target_scale_size=multi_scale_coords[i].shape[1],
                protein_length=protein_length,
                scale_idx=i,
                bos_size=bos_size,
            )
            z_list.append(zi)

        return z_list
