from __future__ import annotations

import mlx.core as mx


def prepare_grid(model, f_latent: int, h_latent: int, w_latent: int, patch_size: tuple[int, int, int]):
    """Prepare grid dimensions and RoPE sequence info."""
    # Compute patch sequence length
    p_t, p_h, p_w = patch_size if isinstance(patch_size, (list, tuple)) else (1, patch_size, patch_size)
    seq_len = (f_latent // p_t) * (h_latent // p_h) * (w_latent // p_w)
    rope = getattr(model, "freqs", None)
    return rope, seq_len


def forward(
    model,
    x: mx.array,                   # [16, F, H, W] from sampler
    t: mx.array,                   # [1] or scalar
    context: mx.array,             # [seq, 4096] umT5 embedding
    rope=None,
    seq_len: int = None,
    teacache=None,
    teacache_mode: str = "default",
):
    """
    DiT forward wrapper that handles:
    1. Shape layout: [16, F, H, W] -> [1, 16, F, H, W] for mlx_video
    2. Context projection: [seq, 4096] -> [1, seq, 1536] via model.text_embedding
    3. TeaCache skipping/caching
    """
    # 1. Shape check: mlx_video WanModel requires [B, C, F, H, W] with C=16
    if x.ndim == 4:
        # x is [16, F, H, W] -> [1, 16, F, H, W]
        x_in = x[None]
    elif x.ndim == 5 and x.shape[1] != 16 and x.shape[2] == 16:
        # Correct [1, F, 16, H, W] -> [1, 16, F, H, W]
        x_in = mx.transpose(x, (0, 2, 1, 3, 4))
    else:
        x_in = x

    # 2. Text Context Projection: Project 4096 -> 1536 if needed
    ctx_in = context
    if ctx_in.ndim == 2:
        ctx_in = ctx_in[None]  # [1, seq, dim]

    if ctx_in.shape[-1] == 4096 and hasattr(model, "text_embedding"):
        ctx_in = model.text_embedding(ctx_in)  # [1, seq, 1536]

    # 3. TeaCache check
    if teacache is not None and hasattr(teacache, "should_skip"):
        should_skip, cached_output = teacache.should_skip(
            model=model, x=x_in, t=t, context=ctx_in, mode=teacache_mode
        )
        if should_skip:
            return cached_output

    # 4. Model Call (mlx_video WanModel __call__(x, t, context))
    out = model(x_in, t=t, context=ctx_in)

    # 5. Output shape check: Squeeze batch [1, 16, F, H, W] -> [16, F, H, W]
    if out.ndim == 5:
        out_ret = out.squeeze(0)
    else:
        out_ret = out

    # 6. Update TeaCache
    if teacache is not None and hasattr(teacache, "update_cache"):
        teacache.update_cache(out_ret, mode=teacache_mode)

    return out_ret
