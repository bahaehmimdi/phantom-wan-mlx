"""DiT forward wrapper and grid preparation for MLX Wan2.1 S2V models."""
from __future__ import annotations

import mlx.core as mx


def prepare_grid(model, f_total: int, h_latent: int, w_latent: int, patch_size: tuple[int, int, int]):
    """Generates valid 3D RoPE embeddings for Wan2.1."""
    # Compute frame/height/width patch grids
    grid_f = f_total // patch_size[0]
    grid_h = h_latent // patch_size[1]
    grid_w = w_latent // patch_size[2]
    seq_len = grid_f * grid_h * grid_w

    # Priority 1: Native model helper if available
    if hasattr(model, "prepare_grid"):
        return model.prepare_grid(f_total, h_latent, w_latent, patch_size)

    # Priority 2: Native RoPE generator on the model
    if hasattr(model, "make_rope"):
        rope = model.make_rope(grid_f, grid_h, grid_w)
        return rope, seq_len

    # Priority 3: Extract pre-built RoPE / generate 3D frequency tables from mlx_video
    if hasattr(model, "rope"):
        rope = model.rope(grid_f, grid_h, grid_w)
        return rope, seq_len

    # Fallback: Let mlx_video generate RoPE internally if passed None
    return None, seq_len


def forward(model, inp, t, ctx, rope_cos_sin, seq_len, cross_kv_caches=None):
    """
    Ensures 5D tensor input and safely routes RoPE into model.__call__.
    """
    if isinstance(inp, mx.array):
        x_in = inp[None] if inp.ndim == 4 else inp
    else:
        x_in = inp

    # Patch patchify dynamically if it strictly expects 4D inputs
    orig_patchify = getattr(model, "_patchify", None)
    if orig_patchify is not None:
        def safe_patchify(x):
            if isinstance(x, mx.array) and x.ndim == 5:
                x = x.squeeze(0)
            return orig_patchify(x)
        model._patchify = safe_patchify

    # Build kwargs dictionary to prevent passing None over explicit arguments
    kwargs = {
        "seq_len": seq_len,
        "cross_kv_caches": cross_kv_caches,
    }
    
    if rope_cos_sin is not None:
        kwargs["rope_cos_sin"] = rope_cos_sin

    try:
        out = model(x_in, t, ctx, **kwargs)
    finally:
        if orig_patchify is not None:
            model._patchify = orig_patchify

    return out
