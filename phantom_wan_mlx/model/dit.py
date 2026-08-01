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
    x: mx.array,                   # [16, F, H, W]
    t: mx.array,                   # [1] or scalar
    context: mx.array,             # [seq, 4096] or [1, seq, 4096]
    rope=None,
    seq_len: int = None,
    teacache=None,
    teacache_mode: str = "default",
):
    """DiT forward wrapper matching mlx_video WanModel.__call__(x, t, context, seq_len)."""
    
    # 1. Format input tensor layout: [16, F, H, W] -> [1, 16, F, H, W]
    if x.ndim == 4:
        x_in = x[None]
    elif x.ndim == 5 and x.shape[1] != 16 and x.shape[2] == 16:
        x_in = mx.transpose(x, (0, 2, 1, 3, 4))
    else:
        x_in = x

    # 2. Derive seq_len if not explicitly passed
    if seq_len is None:
        p_t, p_h, p_w = getattr(model.config, "patch_size", (1, 2, 2))
        _, _, F, H, W = x_in.shape
        seq_len = (F // p_t) * (H // p_h) * (W // p_w)

    # 3. Ensure context has batch dimension [1, seq_len, 4096]
    ctx_in = context
    if ctx_in.ndim == 2:
        ctx_in = ctx_in[None]

    # 4. Project umT5 context (4096 -> 1536) before cross attention
    if ctx_in.shape[-1] == 4096 and hasattr(model, "embed_text"):
        ctx_in = model.embed_text(ctx_in)

    # 5. TeaCache evaluation
    if teacache is not None and hasattr(teacache, "should_skip"):
        should_skip, cached_output = teacache.should_skip(
            model=model, x=x_in, t=t, context=ctx_in, mode=teacache_mode
        )
        if should_skip:
            return cached_output

    # 6. Forward call to WanModel with projected context (1536-dim)
    out = model(x_in, t=t, context=ctx_in, seq_len=seq_len)

    # NEW: Extract the tensor if mlx_video returns a list or tuple
    if isinstance(out, (list, tuple)):
        out = out[0]

    # 7. Return [16, F, H, W] tensor back to sampler
    out_ret = out.squeeze(0) if out.ndim == 5 else out

    # 8. Update TeaCache state
    if teacache is not None and hasattr(teacache, "update_cache"):
        teacache.update_cache(out_ret, mode=teacache_mode)

    return out_ret
