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


def forward(model, latent_cfhw, t, context, rope_cos_sin, seq_len, cross_kv_caches=None):
    """One DiT forward over an assembled [C, F+K, H, W] latent (list-as-batch).

    latent_cfhw: list of [C, F+K, H, W] (one per batch element), OR a single [C,F+K,H,W].
    Returns list of [C, F+K, H, W] predicted velocities.
    """
    x_list = latent_cfhw if isinstance(latent_cfhw, list) else [latent_cfhw]
    ctx = context if isinstance(context, list) else [context]
    return model(x_list, t, ctx, seq_len, cross_kv_caches=cross_kv_caches, rope_cos_sin=rope_cos_sin)
