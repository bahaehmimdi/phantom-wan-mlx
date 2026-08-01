"""Sampling utilities for Phantom-Wan S2V (MLX)."""
from __future__ import annotations

import math
import mlx.core as mx


def get_schedule(steps: int, shift: float = 5.0) -> list[float]:
    """Flow matching timestep schedule with exponential shift."""
    timesteps = [1.0 - i / steps for i in range(steps)]
    shifted_timesteps = []
    for t in timesteps:
        if t == 0:
            shifted_timesteps.append(0.0)
        else:
            s_t = (shift * t) / (1.0 + (shift - 1.0) * t)
            shifted_timesteps.append(s_t)
    return shifted_timesteps


def sample_s2v(model, ref_lat, ctx, ctx_null, cfg, f_latent: int, h_lat: int, w_lat: int,
               steps: int = 50, shift: float = 5.0, guide_img: float = 5.0, guide_text: float = 7.5,
               seed: int = 0, verbose: bool = True, teacache_thresh: float = 0.0):
    """Sample S2V latent using Flow Matching with dual CFG and TeaCache support."""
    mx.random.seed(seed)
    
    # Latent noise initialization
    z = mx.random.normal((cfg.in_dim, f_latent, h_lat, w_lat), dtype=mx.bfloat16)
    
    timesteps = get_schedule(steps, shift=shift)
    
    # TeaCache tracking state
    accumulated_l1 = 0.0
    prev_input = None
    cached_residual = None
    skipped_steps = 0

    if verbose and teacache_thresh > 0.0:
        print(f"[TeaCache MLX] Enabled with threshold: {teacache_thresh}")

    for i in range(len(timesteps) - 1):
        t_curr = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_next - t_curr

        should_calc = True

        # Check relative L1 threshold against previous step input
        if teacache_thresh > 0.0 and prev_input is not None:
            l1_diff = mx.mean(mx.abs(z - prev_input)) / (mx.mean(mx.abs(prev_input)) + 1e-6)
            mx.eval(l1_diff)
            accumulated_l1 += l1_diff.item()

            if accumulated_l1 < teacache_thresh and cached_residual is not None:
                should_calc = False

        if should_calc:
            accumulated_l1 = 0.0

            # Forward pass: unconditional, text-only, and joint text+image guidance
            # 1. Full context (Text + Reference Image)
            v_cond = model(z, t=t_curr, context=ctx, ref_latents=ref_lat)
            
            # 2. Text-guided CFG (Null references)
            if guide_img != 1.0:
                v_text = model(z, t=t_curr, context=ctx, ref_latents=None)
            else:
                v_text = v_cond
                
            # 3. Unconditional CFG (Null context + Null references)
            if guide_text != 1.0:
                v_uncond = model(z, t=t_curr, context=ctx_null, ref_latents=None)
            else:
                v_uncond = v_text

            # Compute dual-guided velocity output
            v_pred = v_uncond + guide_text * (v_text - v_uncond) + guide_img * (v_cond - v_text)
            mx.eval(v_pred)

            # Store velocity output for residual prediction
            cached_residual = v_pred
        else:
            skipped_steps += 1
            v_pred = cached_residual

        prev_input = z

        # Euler step update
        z = z + v_pred * dt

    if verbose and teacache_thresh > 0.0:
        total_steps = len(timesteps) - 1
        speedup = total_steps / max(1, total_steps - skipped_steps)
        print(f"[TeaCache MLX] Skipped {skipped_steps}/{total_steps} steps (~{speedup:.2f}x speedup)")

    return z
