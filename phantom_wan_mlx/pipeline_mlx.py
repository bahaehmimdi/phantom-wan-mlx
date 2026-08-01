"""Phantom-Wan S2V inference entry (MLX) with Maximum TeaCache Debug Logging.

    from phantom_wan_mlx import pipeline_mlx as P
    P.s2v("two friends walking", ["a.png", "b.png"], "out.mp4")

reference_images: list of paths (multi-subject <=4, each a distinct subject). See G1.
"""
from __future__ import annotations

from pathlib import Path
import types

import mlx.core as mx
import numpy as np
from PIL import Image

from .config import PhantomWanConfig
from .model import dit as DIT
from .model.reference import encode_references
from .sampling import sample_s2v
from .utils import weights as W
print('V2')
ROOT = Path(__file__).resolve().parents[1]
NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _format_tensor_info(obj):
    """Helper to inspect types, shapes, and structural signatures of objects."""
    if isinstance(obj, mx.array):
        return f"mx.array(shape={obj.shape}, dtype={obj.dtype})"
    elif isinstance(obj, (list, tuple)):
        items_str = ", ".join([_format_tensor_info(x) for x in obj])
        container_type = "tuple" if isinstance(obj, tuple) else "list"
        return f"{container_type}[len={len(obj)}: ({items_str})]"
    elif isinstance(obj, dict):
        keys_str = ", ".join([f"{k}: {_format_tensor_info(v)}" for k, v in obj.items()])
        return f"dict{{{keys_str}}}"
    else:
        return f"{type(obj).__name__}(val={str(obj)[:30]})"


class TeaCacheContext:
    """Dynamic feature caching context (TeaCache) with verbose telemetry logging."""
    def __init__(self, threshold: float = 0.15, ret_steps: int = 5):
        self.threshold = threshold
        self.ret_steps = ret_steps
        self.coefficients = [-23.9411516, 27.3094821, -0.493836371, 0.0425192456]
        self.poly_rescale = np.poly1d(self.coefficients)
        
        self.cnt = 0
        self.num_steps = 0
        self.accumulated_dist_even = 0.0
        self.accumulated_dist_odd = 0.0
        self.prev_inp_even = None
        self.prev_inp_odd = None
        self.prev_out_even = None
        self.prev_out_odd = None
        self.skipped_steps = 0
        self.total_evals = 0

    def reset(self, steps: int):
        self.cnt = 0
        self.num_steps = steps * 2  # cond + uncond calls
        self.accumulated_dist_even = 0.0
        self.accumulated_dist_odd = 0.0
        self.prev_inp_even = None
        self.prev_inp_odd = None
        self.prev_out_even = None
        self.prev_out_odd = None
        self.skipped_steps = 0
        self.total_evals = 0

    def should_compute(self, inp_tensor, is_even: bool) -> tuple[bool, str]:
        self.total_evals += 1
        stream_name = "COND (even)" if is_even else "UNCOND (odd)"
        
        # Always compute during retention window (beginning and end of denoising)
        if self.cnt < self.ret_steps or self.cnt >= (self.num_steps - self.ret_steps):
            if is_even:
                self.accumulated_dist_even = 0.0
                self.prev_inp_even = inp_tensor
            else:
                self.accumulated_dist_odd = 0.0
                self.prev_inp_odd = inp_tensor
            return True, f"RETENTION WINDOW (Step {self.cnt}/{self.num_steps}, {stream_name})"

        prev_inp = self.prev_inp_even if is_even else self.prev_inp_odd
        if prev_inp is None:
            if is_even:
                self.prev_inp_even = inp_tensor
            else:
                self.prev_inp_odd = inp_tensor
            return True, f"INITIAL STEP ({stream_name})"

        # Calculate relative distance based on latent feature difference
        diff = mx.mean(mx.abs(inp_tensor - prev_inp))
        norm = mx.mean(mx.abs(prev_inp)) + 1e-8
        rel_l1 = (diff / norm).item()
        scaled_dist = float(self.poly_rescale(rel_l1))

        if is_even:
            self.accumulated_dist_even += scaled_dist
            curr_accum = self.accumulated_dist_even
            if curr_accum < self.threshold:
                self.skipped_steps += 1
                return False, f"CACHE HIT ({stream_name}) -> accum_dist={curr_accum:.4f} < thresh={self.threshold}"
            self.accumulated_dist_even = 0.0
            self.prev_inp_even = inp_tensor
            return True, f"CACHE MISS ({stream_name}) -> accum_dist={curr_accum:.4f} >= thresh={self.threshold}"
        else:
            self.accumulated_dist_odd += scaled_dist
            curr_accum = self.accumulated_dist_odd
            if curr_accum < self.threshold:
                self.skipped_steps += 1
                return False, f"CACHE HIT ({stream_name}) -> accum_dist={curr_accum:.4f} < thresh={self.threshold}"
            self.accumulated_dist_odd = 0.0
            self.prev_inp_odd = inp_tensor
            return True, f"CACHE MISS ({stream_name}) -> accum_dist={curr_accum:.4f} >= thresh={self.threshold}"


def apply_teacache_to_dit(teacache_thresh: float, steps: int):
    """Hooks TeaCache cleanly around DIT.forward with deep signature telemetry."""
    tc = TeaCacheContext(threshold=teacache_thresh)
    tc.reset(steps)
    
    orig_dit_forward = DIT.forward

    def teacache_dit_forward(model, inp, t, context, rope_cos_sin, seq_len, cross_kv_caches=None):
        step_idx = tc.cnt
        is_even = (step_idx % 2 == 0)
        
        print(f"\n[TeaCache Log #{step_idx:03d}] --- ENTERING DIT.forward ---", flush=True)
        print(f"  > Input 'inp' Signature    : {_format_tensor_info(inp)}", flush=True)
        print(f"  > Timestep 't' Signature   : {_format_tensor_info(t)}", flush=True)
        print(f"  > Context Signature        : {_format_tensor_info(context)}", flush=True)
        print(f"  > rope_cos_sin Signature   : {_format_tensor_info(rope_cos_sin)}", flush=True)
        print(f"  > seq_len                  : {seq_len}", flush=True)
        
        # Extract main target latent array (first element in inp list if container)
        target_lat = inp[0] if isinstance(inp, (list, tuple)) else inp
        print(f"  > Isolated Target Latent   : {_format_tensor_info(target_lat)}", flush=True)
        
        compute, reason = tc.should_compute(target_lat, is_even)
        tc.cnt += 1

        print(f"  > Decision                 : {reason}", flush=True)

        if not compute:
            cached_out = tc.prev_out_even if is_even else tc.prev_out_odd
            print(f"  > REUSING CACHED OUTPUT    : {_format_tensor_info(cached_out)}", flush=True)
            return cached_out

        print(f"  > EXECUTING DiT FORWARD PASS...", flush=True)
        out = orig_dit_forward(
            model, inp, t, context, rope_cos_sin, seq_len, cross_kv_caches=cross_kv_caches
        )
        
        print(f"  > OUTPUT PRODUCED          : {_format_tensor_info(out)}", flush=True)

        if is_even:
            tc.prev_out_even = out
        else:
            tc.prev_out_odd = out

        return out

    DIT.forward = teacache_dit_forward
    return tc, orig_dit_forward


def encode_prompt(prompt: str, phantom_pth=None):
    """Offline umT5 encode → [L, 4096] context."""
    _, cfg = W.load_phantom_dit(phantom_pth or ROOT / "weights/phantom/Phantom-Wan-1.3B.pth")
    t5, tok = W.load_umt5(cfg)
    return W.encode_text(t5, tok, prompt, cfg.text_len)


def _save_video(frames_bchw, path, fps=16):
    import imageio
    v = ((np.array(frames_bchw[0]).transpose(1, 2, 3, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(path, list(v), fps=fps, quality=8)
    return path


def s2v(prompt: str, reference_images: list, output_path: str,
        size=(832, 480), frame_num: int = 81, steps: int = 50, shift: float = 5.0,
        guide_img: float = 5.0, guide_text: float = 7.5, seed: int = 0,
        phantom_pth=None, vae_pth=None, lossless_decode: bool = True,
        precomputed_context=None, precomputed_context_null=None, verbose: bool = True,
        teacache_thresh: float = 0.0):
    """Generate a subject-consistent video from a prompt + reference images."""
    w_px, h_px = size
    phantom_pth = phantom_pth or ROOT / "weights/phantom/Phantom-Wan-1.3B.pth"
    vae_pth = vae_pth or ROOT / "weights/wan-base/Wan2.1_VAE.pth"

    cfg_run = PhantomWanConfig.s2v_1_3b()
    model, cfg = W.load_phantom_dit(phantom_pth)               # cfg = mlx-video WanModelConfig

    tc_context = None
    orig_dit_forward = None
    if teacache_thresh > 0.0:
        print(f"\n" + "="*70, flush=True)
        print(f"[TEA-CACHE DIAGNOSTIC MODE INITIALIZED]", flush=True)
        print(f"  > Threshold : {teacache_thresh}", flush=True)
        print(f"  > Total Sampling Steps : {steps} (Total Model Calls = {steps * 2})", flush=True)
        print("="*70 + "\n", flush=True)
        tc_context, orig_dit_forward = apply_teacache_to_dit(teacache_thresh, steps)

    try:
        if precomputed_context is not None:
            ctx = precomputed_context
            ctx_null = precomputed_context_null if precomputed_context_null is not None else precomputed_context
        else:
            t5, tok = W.load_umt5(cfg)
            ctx = W.encode_text(t5, tok, prompt, cfg.text_len)
            ctx_null = W.encode_text(t5, tok, NEG_PROMPT, cfg.text_len)
            del t5

        enc = W.load_wan_vae(vae_pth, encoder=True)
        refs = [Image.open(p) for p in reference_images]
        ref_lat = encode_references(enc, refs, w_px, h_px)
        del enc

        f_latent = (frame_num - 1) // cfg_run.vae_stride[0] + 1     # temporal stride 4
        h_lat, w_lat = h_px // cfg_run.vae_stride[1], w_px // cfg_run.vae_stride[2]
        if verbose:
            print(f"f_latent={f_latent} (={frame_num} frames) grid {h_lat}x{w_lat}, K={ref_lat.shape[2]} refs", flush=True)

        x0 = sample_s2v(model, ref_lat, ctx, ctx_null, cfg, f_latent, h_lat, w_lat,
                        steps=steps, shift=shift, guide_img=guide_img, guide_text=guide_text,
                        seed=seed, verbose=verbose)

        if tc_context is not None and verbose:
            skipped = tc_context.skipped_steps
            total = tc_context.total_evals
            speedup = total / (total - skipped) if (total - skipped) > 0 else 1.0
            print("\n" + "="*70, flush=True)
            print("          TEA-CACHE FINAL PERFORMANCE SUMMARY          ", flush=True)
            print("="*70, flush=True)
            print(f"Total Model Calls (Cond + Uncond) : {total}", flush=True)
            print(f"Skipped Model Computations        : {skipped} ({skipped / total * 100:.1f}%)", flush=True)
            print(f"Effective Acceleration Factor     : {speedup:.2f}x Speedup", flush=True)
            print("="*70 + "\n", flush=True)

    finally:
        if orig_dit_forward is not None:
            DIT.forward = orig_dit_forward

    del model

    dec = W.load_wan_vae(vae_pth, encoder=False)
    if lossless_decode:
        from .streaming_decode import decode_streaming
        video = decode_streaming(dec, x0[None], chunk_lat=1)
    else:
        video = dec.decode(x0[None])
    mx.eval(video)
    return _save_video(video, output_path, fps=cfg_run.sample_fps)
