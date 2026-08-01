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
    rope,
    seq_len: int,
    teacache=None,
    teacache_mode: str = "default",
):
    """
    DiT forward wrapper supporting optional TeaCache state handling.
    """
    # If using TeaCache, compute features / evaluate L1 distance before main blocks
    if teacache is not None:
        # Check if we can skip block execution using cached residual/output
        should_skip, cached_output = teacache.should_skip(
            model=model,
            x=x,
            t=t,
            context=context,
            mode=teacache_mode
        )
        if should_skip:
            return cached_output

    # Run regular forward pass if not skipped
    out = model(x, t=t, context=context, rope=rope, seq_len=seq_len)

    # Store state/output in TeaCache for current mode stream
    if teacache is not None:
        teacache.update_cache(out, mode=teacache_mode)

    return out
