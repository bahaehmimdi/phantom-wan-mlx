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
    """DiT forward wrapper handling layout transpose and text projection."""
    
    # 1. Transpose latents to match [B, F, C, H, W] expected by mlx_video
    if x.ndim == 4:
        x_in = mx.transpose(x, (1, 0, 2, 3))[None]
    elif x.ndim == 5 and x.shape[1] == 16:
        x_in = mx.transpose(x, (0, 2, 1, 3, 4))
    else:
        x_in = x

    # 2. Check and project text context if model exposes text_embedder
    ctx_in = context
    if hasattr(model, "text_embedder") and context.shape[-1] == 4096:
        ctx_in = model.text_embedder(context)
    elif hasattr(model, "text_embedding") and context.shape[-1] == 4096:
        ctx_in = model.text_embedding(context)

    # 3. Ensure context has batch dimension if needed [1, seq_len, dim]
    if ctx_in.ndim == 2:
        ctx_in = ctx_in[None]

    if teacache is not None:
        should_skip, cached_output = teacache.should_skip(
            model=model, x=x_in, t=t, context=ctx_in, mode=teacache_mode
        )
        if should_skip:
            return cached_output

    # 4. Forward pass
    if seq_len is not None:
        out = model(x_in, t=t, context=ctx_in, seq_len=seq_len)
    else:
        out = model(x_in, t=t, context=ctx_in)

    # 5. Transpose output back to sampler layout [16, F, H, W]
    if out.ndim == 5:
        out_ret = mx.transpose(out.squeeze(0), (1, 0, 2, 3))
    else:
        out_ret = out

    if teacache is not None:
        teacache.update_cache(out_ret, mode=teacache_mode)

    return out_ret
