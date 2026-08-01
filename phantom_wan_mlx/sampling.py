"""Sampling utilities for Phantom-Wan S2V (MLX)."""
from __future__ import annotations

import math
import mlx.core as mx


def sample_s2v(model, ref_lat, ctx, ctx_null, cfg, f_latent: int, h_lat: int, w_lat: int,
               steps: int = 50, shift: float = 5.0, guide_img: float = 5.0, guide_text: float = 7.5,
               seed: int = 0, verbose: bool = True, teacache_thresh: float = 0.0):
    """Sample S2V latent using Flow Matching with dual CFG and TeaCache support."""
    mx.random.seed(seed)
    
    # Latent noise initialization: [1, in_dim, f_latent, h_lat, w_lat] or [in_dim, f_latent, h_lat, w_lat]
    # Match the shape expected by model forward
    z = mx.random.normal((1, cfg.in_dim, f_latent, h_lat, w_lat), dtype=mx.bfloat16)
    
    # Generate schedule
    timesteps = [1.0 - i / steps for i in range(steps)]
    shifted_timesteps = [(shift * t) / (1.0 + (shift - 1.0) * t) if t > 0 else 0.0 for t in timesteps]
    
    # TeaCache tracking state
    accumulated_l1 = 0.0
    prev_input = None
    cached_residual = None
    skipped_steps = 0

    if verbose and teacache_thresh > 0.0:
        print(f"[TeaCache MLX] Enabled with threshold: {teacache_thresh}")

    for i in range(len(shifted_timesteps) - 1):
        t_curr = shifted_timesteps[i]
        t_next = shifted_timesteps[i + 1]
        dt = t_next - t_curr

        # Convert float timestep to array for MLX model call
        t_arr = mx.array([t_curr * 1000.0], dtype=mx.float32)

        should_calc = True

        # Check TeaCache L1 threshold
        if teacache_thresh > 0.0 and prev_input is not None:
            l1_diff = mx.mean(mx.abs(z - prev_input)) / (mx.mean(mx.abs(prev_input)) + 1e-6)
            mx.eval(l1_diff)
            accumulated_l1 += l1_diff.item()

            if accumulated_l1 < teacache_thresh and cached_residual is not None:
                should_calc = False

        if should_calc:
            accumulated_l1 = 0.0

            # --- Forward Pass Calls ---
            # 1. Joint Subject + Text Pass (pass ref_lat as secondary positional or ref arg)
            if ref_lat is not None:
                v_cond = model(z, t_arr, ctx, ref_lat)
            else:
                v_cond = model(z, t_arr, ctx)
            
            # 2. Text-only Guidance Pass
            if guide_img != 1.0 and ref_lat is not None:
                v_text = model(z, t_arr, ctx)
            else:
                v_text = v_cond
                
            # 3. Unconditional Guidance Pass
            if guide_text != 1.0:
                v_uncond = model(z, t_arr, ctx_null)
            else:
                v_uncond = v_text

            # Compute dual CFG guidance prediction
            v_pred = v_uncond + guide_text * (v_text - v_uncond) + guide_img * (v_cond - v_text)
            mx.eval(v_pred)

            cached_residual = v_pred
        else:
            skipped_steps += 1
            v_pred = cached_residual

        prev_input = z

        # Step update
        z = z + v_pred * dt

    if verbose and teacache_thresh > 0.0:
        total_steps = len(shifted_timesteps) - 1
        speedup = total_steps / max(1, total_steps - skipped_steps)
        print(f"[TeaCache MLX] Skipped {skipped_steps}/{total_steps} steps (~{speedup:.2f}x speedup)")

    # Squeeze batch dim if needed by VAE decode
    if z.ndim == 5 and z.shape[0] == 1:
        z = z[0]

    return z
