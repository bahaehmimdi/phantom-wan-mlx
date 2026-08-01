from __future__ import annotations

import mlx.core as mx


def prepare_grid(model, f_latent: int, h_latent: int, w_latent: int, patch_size: tuple[int, int, int]):
    """Prepare grid dimensions and RoPE sequence info."""
    p_t, p_h, p_w = patch_size if isinstance(patch_size, (list, tuple)) else (1, patch_size, patch_size)
    seq_len = (f_latent // p_t) * (h_latent // p_h) * (w_latent // p_w)
    rope = getattr(model, "freqs", None)
    return rope, seq_len


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
    
    # 1. Format input tensor: [16, F, H, W] -> [1, 16, F, H, W]
    if x.ndim == 4:
        x_in = x[None]
    elif x.ndim == 5 and x.shape[1] != 16 and x.shape[2] == 16:
        x_in = mx.transpose(x, (0, 2, 1, 3, 4))
    else:
        x_in = x

    # 2. Compute seq_len if not passed
    if seq_len is None:
        p_t, p_h, p_w = getattr(model.config, "patch_size", (1, 2, 2))
        _, _, F, H, W = x_in.shape
        seq_len = (F // p_t) * (H // p_h) * (W // p_w)

    # 3. Ensure context has 3D shape [1, seq_len, dim]
    ctx_in = context
    if ctx_in.ndim == 2:
        ctx_in = ctx_in[None]

    # 4. Project umT5 embeddings (4096 -> 1536) before cross attention
    if ctx_in.shape[-1] == 4096 and hasattr(model, "text_embedding"):
        ctx_in = model.text_embedding(ctx_in)

    # 5. TeaCache evaluation
    if teacache is not None and hasattr(teacache, "should_skip"):
        should_skip, cached_output = teacache.should_skip(
            model=model, x=x_in, t=t, context=ctx_in, mode=teacache_mode
        )
        if should_skip:
            return cached_output

    # 6. Call WanModel with projected context (1536) and required seq_len
    out = model(x_in, t=t, context=ctx_in, seq_len=seq_len)

    # 7. Squeeze batch dimension back to [16, F, H, W]
    out_ret = out.squeeze(0) if out.ndim == 5 else out

    # 8. Update TeaCache
    if teacache is not None and hasattr(teacache, "update_cache"):
        teacache.update_cache(out_ret, mode=teacache_mode)

    return out_ret
