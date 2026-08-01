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


def forward(
    model,
    x: mx.array,
    t: mx.array,
    context: mx.array,
    rope=None,
    seq_len: int = None,
    teacache=None,
    teacache_mode: str = "default",
):
    """DiT forward wrapper with 5D batch expansion for WanModel."""
    
    # 1. Ensure 5D tensor shape [1, C, F, H, W] for mlx_video
    x_in = x[None] if x.ndim == 4 else x

    if teacache is not None:
        should_skip, cached_output = teacache.should_skip(
            model=model, x=x_in, t=t, context=context, mode=teacache_mode
        )
        if should_skip:
            return cached_output

    # 2. Forward pass through WanModel
    out = model(x_in, t=t, context=context, seq_len=seq_len)

    # 3. If model returns 5D [1, C, F, H, W], squeeze batch dim to return 4D [C, F, H, W]
    if out.ndim == 5 and x.ndim == 4:
        out_ret = out.squeeze(0)
    else:
        out_ret = out

    if teacache is not None:
        teacache.update_cache(out_ret, mode=teacache_mode)

    return out_ret
