"""Wan2.1 DiT forward with reference injection (G1).

Backbone = stock Wan2.1 (mlx-video `wan_2.WanModel`), UNCHANGED. Phantom's injection is
purely an *input assembly*: feed the F+K-frame latent (target ⊕ trailing refs) through the
stock patch-embed + 3D-RoPE + blocks. The extended frame grid (F+K) ropes the refs at
ordinary sequential positions F..F+K-1 (G1 §3 — no SA-3D, no DiT change). model_type='t2v'.

This module only wraps the substrate's grid/seq_len/forward plumbing for the F+K case.
"""
from __future__ import annotations

import mlx.core as mx


def prepare_grid(model, t_latent: int, h_latent: int, w_latent: int, patch_size, batch: int = 1):
    """Compute (rope_cos_sin, seq_len) for a t_latent=F+K frame grid (generate.py:517-530)."""
    f_grid = t_latent // patch_size[0]
    h_grid = h_latent // patch_size[1]
    w_grid = w_latent // patch_size[2]
    seq_len = f_grid * h_grid * w_grid
    rope_cos_sin = model.prepare_rope([(f_grid, h_grid, w_grid)] * batch)
    return rope_cos_sin, seq_len


import mlx.core as mx

def forward(model, inp, t, ctx, rope_cos_sin, seq_len, cross_kv_caches=None):
    """
    Formats S2V latents into a list [target, references] expected by mlx_video Wan2.1.
    """
    if isinstance(inp, mx.array):
        # Squeeze batch if present -> shape becomes (16, T_total, H, W)
        x_tensor = inp.squeeze(0) if (inp.ndim == 5 and inp.shape[0] == 1) else inp
        
        # Total frames T = 23 (F=21 target + K=2 refs)
        # Sequence length 11776 corresponds to 21 target + 2 ref frames at grid 32x64
        if x_tensor.ndim == 4 and x_tensor.shape[1] > 21:
            # Separate target video latent from reference latents
            target_lat = x_tensor[:, :21, :, :]      # [16, 21, 32, 64]
            ref_lats = x_tensor[:, 21:, :, :]        # [16, 2, 32, 64]
            x_in = [target_lat, ref_lats]
        else:
            x_in = [x_tensor]
            
    elif isinstance(inp, (list, tuple)):
        x_in = [x.squeeze(0) if (isinstance(x, mx.array) and x.ndim == 5) else x for x in inp]
    else:
        x_in = [inp]

    # Call mlx_video model with list of inputs
    return model(
        x_in,
        t,
        ctx,
        seq_len=seq_len,
        cross_kv_caches=cross_kv_caches,
        rope_cos_sin=rope_cos_sin
    )
