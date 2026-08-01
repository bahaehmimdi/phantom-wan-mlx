"""DiT forward wrapper and grid preparation for MLX Wan2.1 S2V models."""
from __future__ import annotations

import mlx.core as mx


def prepare_grid(model, f_total: int, h_latent: int, w_latent: int, patch_size: tuple[int, int, int]):
    """Generates RoPE embeddings and computes total sequence length."""
    if hasattr(model, "prepare_grid"):
        return model.prepare_grid(f_total, h_latent, w_latent, patch_size)
    
    grid_f = f_total // patch_size[0]
    grid_h = h_latent // patch_size[1]
    grid_w = w_latent // patch_size[2]
    seq_len = grid_f * grid_h * grid_w

    if hasattr(model, "make_rope"):
        rope = model.make_rope(grid_f, grid_h, grid_w)
    else:
        rope = None

    return rope, seq_len


def forward(model, inp, t, ctx, rope_cos_sin, seq_len, cross_kv_caches=None):
    """
    Ensures input is 5D [1, 16, T, H, W] for batching while safely handling 
    mlx_video's internal patchify expectation.
    """
    # Ensure 5D tensor: [1, 16, 23, 32, 64]
    if isinstance(inp, mx.array):
        if inp.ndim == 4:
            x_in = inp[None]
        else:
            x_in = inp
    else:
        x_in = inp

    # Dynamically patch model._patchify if it strictly unpacks 4 dimensions
    orig_patchify = getattr(model, "_patchify", None)
    
    if orig_patchify is not None:
        def safe_patchify(x):
            if x.ndim == 5:
                x = x.squeeze(0)  # Squeeze batch dim -> (16, T, H, W)
            return orig_patchify(x)
        
        model._patchify = safe_patchify

    try抓:
        out = model(
            x_in,
            t,
            ctx,
            seq_len=seq_len,
            cross_kv_caches=cross_kv_caches,
            rope_cos_sin=rope_cos_sin
        )
    finally:
        if orig_patchify is not None:
            model._patchify = orig_patchify

    return out
