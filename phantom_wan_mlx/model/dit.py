"""DiT forward wrapper and grid preparation for MLX Wan2.1 S2V models."""
from __future__ import annotations

import mlx.core as mx


def prepare_grid(model, f_total: int, h_latent: int, w_latent: int, patch_size: tuple[int, int, int]):
    """Generates 3D RoPE embeddings directly via mlx_video model methods."""
    grid_f = f_total // patch_size[0]
    grid_h = h_latent // patch_size[1]
    grid_w = w_latent // patch_size[2]
    seq_len = grid_f * grid_h * grid_w

    rope = None
    if hasattr(model, "rope"):
        rope = model.rope(grid_f, grid_h, grid_w)
    elif hasattr(model, "make_rope"):
        rope = model.make_rope(grid_f, grid_h, grid_w)
    elif hasattr(model, "prepare_grid"):
        rope, _ = model.prepare_grid(f_total, h_latent, w_latent, patch_size)

    return rope, seq_len


def forward(model, inp, t, ctx, rope_cos_sin, seq_len, cross_kv_caches=None):
    """
    S2V Forward Pass for mlx_video Wan2.1.
    Converts inputs into pre-patchified tokens [1, seq_len, dim] to preserve batching.
    """
    # Extract tensor from batch wrapper if needed
    x = inp.squeeze(0) if (isinstance(inp, mx.array) and inp.ndim == 5 and inp.shape[0] == 1) else inp

    # Patchify through model._patchify or model.patchify
    patch_fn = getattr(model, "_patchify", getattr(model, "patchify", None))
    
    if patch_fn is not None and x.ndim == 4:
        # x is [16, 23, 32, 64] -> patchified to [11776, 1536]
        x_patched = patch_fn(x)
        # Ensure 3D shape [1, 11776, 1536] for attention projection
        if x_patched.ndim == 2:
            x_patched = x_patched[None]
    else:
        x_patched = inp if inp.ndim == 3 else inp[None]

    # Temporarily set model patchify to identity so model.__call__ accepts pre-patchified sequence
    orig_patchify = getattr(model, "_patchify", None)
    setattr(model, "_patchify", lambda v: v.squeeze(0) if v.ndim == 3 else v)

    kwargs = {
        "seq_len": seq_len,
        "cross_kv_caches": cross_kv_caches,
    }
    if rope_cos_sin is not None:
        kwargs["rope_cos_sin"] = rope_cos_sin

    try:
        out = model(x_patched, t, ctx, **kwargs)
    finally:
        if orig_patchify is not None:
            setattr(model, "_patchify", orig_patchify)

    return out
