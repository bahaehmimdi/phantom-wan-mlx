"""DiT forward wrapper and grid preparation for MLX Wan2.1 S2V models."""
from __future__ import annotations

import mlx.core as mx


def prepare_grid(model, f_total: int, h_latent: int, w_latent: int, patch_size: tuple[int, int, int]):
    """Generates 3D RoPE embeddings via mlx_video model methods or computes grid length."""
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
    S2V Forward Wrapper for mlx_video Wan2.1.
    Splits 5D latents [1, 16, F+K, H, W] into target and reference streams,
    patchifies each stream, and forwards the merged sequence tokens.
    """
    if isinstance(inp, mx.array):
        x_tensor = inp.squeeze(0) if (inp.ndim == 5 and inp.shape[0] == 1) else inp
    else:
        x_tensor = inp

    if isinstance(x_tensor, mx.array) and x_tensor.ndim == 4 and x_tensor.shape[1] > 21:
        target_lat = x_tensor[:, :21, :, :]     # [16, 21, 32, 64]
        ref_lat = x_tensor[:, 21:, :, :]        # [16, K, 32, 64]

        if hasattr(model, "_patchify"):
            p_target = model._patchify(target_lat)  # [10752, 1536]
            p_ref = model._patchify(ref_lat)        # [1024, 1536]
            
            # Fixed: Use positional argument for axis (axis 0)
            x_patched = mx.concatenate([p_target, p_ref], 0)[None]  # [1, 11776, 1536]
        else:
            x_patched = x_tensor[None]
    else:
        if hasattr(model, "_patchify") and x_tensor.ndim == 4:
            x_patched = model._patchify(x_tensor)[None]
        else:
            x_patched = inp if inp.ndim == 5 else inp[None]

    # Temporarily override _patchify on model to pass pre-patchified tokens
    orig_patchify = getattr(model, "_patchify", None)
    model._patchify = lambda x: x.squeeze(0) if (isinstance(x, mx.array) and x.ndim == 3) else x

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
            model._patchify = orig_patchify

    return out
