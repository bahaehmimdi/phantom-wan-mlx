"""DiT forward wrapper and grid preparation for MLX Wan2.1 S2V models."""
from __future__ import annotations

import mlx.core as mx


def prepare_grid(model, f_total: int, h_latent: int, w_latent: int, patch_size: tuple[int, int, int]):
    """Generates 3D RoPE embeddings or grid length."""
    grid_f = f_total // patch_size[0]
    grid_h = h_latent // patch_size[1]
    grid_w = w_latent // patch_size[2]
    seq_len = grid_f * grid_h * grid_w

    rope = None
    if hasattr(model, "make_rope"):
        rope = model.make_rope(grid_f, grid_h, grid_w)
    elif hasattr(model, "rope"):
        rope = model.rope(grid_f, grid_h, grid_w)

    return rope, seq_len


def forward(model, inp, t, ctx, rope_cos_sin, seq_len, cross_kv_caches=None):
    """
    Direct 5D forward pass for mlx_video Wan2.1.
    Passes (1, 16, T, H, W) straight to model.__call__ without manual patching.
    """
    # Force 5D tensor: [1, 16, 23, 32, 64]
    if isinstance(inp, mx.array):
        if inp.ndim == 4:
            x_in = inp[None]
        else:
            x_in = inp
    else:
        x_in = inp

    kwargs = {
        "seq_len": seq_len,
        "cross_kv_caches": cross_kv_caches,
    }
    
    if rope_cos_sin is not None:
        kwargs["rope_cos_sin"] = rope_cos_sin

    # Single, clean call directly into mlx_video
    return model(x_in, t, ctx, **kwargs)
