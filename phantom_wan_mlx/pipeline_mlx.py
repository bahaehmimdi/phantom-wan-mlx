"""Phantom-Wan S2V inference entry (MLX).

    from phantom_wan_mlx import pipeline_mlx as P
    P.s2v("two friends walking", ["a.png", "b.png"], "out.mp4")

reference_images: list of paths (multi-subject <=4, each a distinct subject). See G1.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .config import PhantomWanConfig
from .model.reference import encode_references
from .sampling import sample_s2v
from .utils import weights as W

ROOT = Path(__file__).resolve().parents[1]
NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def encode_prompt(prompt: str, phantom_pth=None):
    """Offline umT5 encode → [L, 4096] context. For the Swift escape hatch: encode prompts
    in Python, save the embeddings, pass them to a Swift consumer via precomputed_context
    (bypasses the ftfy cleaner + SentencePiece tokenizer + 11 GB umT5 at Swift runtime)."""
    _, cfg = W.load_phantom_dit(phantom_pth or ROOT / "weights/phantom/Phantom-Wan-1.3B.pth")
    t5, tok = W.load_umt5(cfg)
    return W.encode_text(t5, tok, prompt, cfg.text_len)


def _save_video(frames_bchw, path, fps=16):
    import imageio
    # frames_bchw: mx [1,3,T,H,W] in [-1,1] -> uint8 list
    v = ((np.array(frames_bchw[0]).transpose(1, 2, 3, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(path, list(v), fps=fps, quality=8)
    return path


def s2v(prompt: str, reference_images: list, output_path: str,
        size=(832, 480), frame_num: int = 81, steps: int = 50, shift: float = 5.0,
        guide_img: float = 5.0, guide_text: float = 7.5, seed: int = 0,
        phantom_pth=None, vae_pth=None, lossless_decode: bool = True,
        precomputed_context=None, precomputed_context_null=None, verbose: bool = True):
    """Generate a subject-consistent video from a prompt + reference images."""
    w_px, h_px = size
    phantom_pth = phantom_pth or ROOT / "weights/phantom/Phantom-Wan-1.3B.pth"
    vae_pth = vae_pth or ROOT / "weights/wan-base/Wan2.1_VAE.pth"

    cfg_run = PhantomWanConfig.s2v_1_3b()
    model, cfg = W.load_phantom_dit(phantom_pth)               # cfg = mlx-video WanModelConfig

    # text — escape hatch: precomputed [L,4096] umT5 context bypasses cleaning+tokenizer+umT5
    # entirely (Swift-port safety net: encode prompts in Python, ship the embeddings). See
    # encode_prompt() + docs/development/swift-port-concerns.md.
    if precomputed_context is not None:
        ctx = precomputed_context
        ctx_null = precomputed_context_null if precomputed_context_null is not None else precomputed_context
    else:
        t5, tok = W.load_umt5(cfg)
        ctx = W.encode_text(t5, tok, prompt, cfg.text_len)
        ctx_null = W.encode_text(t5, tok, NEG_PROMPT, cfg.text_len)
        del t5

    # reference latents (encoder VAE)
    enc = W.load_wan_vae(vae_pth, encoder=True)
    refs = [Image.open(p) for p in reference_images]
    ref_lat = encode_references(enc, refs, w_px, h_px)
    del enc

    # target latent grid
    f_latent = (frame_num - 1) // cfg_run.vae_stride[0] + 1     # temporal stride 4
    h_lat, w_lat = h_px // cfg_run.vae_stride[1], w_px // cfg_run.vae_stride[2]
    if verbose:
        print(f"f_latent={f_latent} (={frame_num} frames) grid {h_lat}x{w_lat}, K={ref_lat.shape[2]} refs", flush=True)

    x0 = sample_s2v(model, ref_lat, ctx, ctx_null, cfg, f_latent, h_lat, w_lat,
                    steps=steps, shift=shift, guide_img=guide_img, guide_text=guide_text,
                    seed=seed, verbose=verbose)
    del model

    # decode — streaming (lossless, flat memory) unblocks long video; whole-seq OOMs >~49 frames
    dec = W.load_wan_vae(vae_pth, encoder=False)
    if lossless_decode:
        from .streaming_decode import decode_streaming
        video = decode_streaming(dec, x0[None], chunk_lat=1)
    else:
        video = dec.decode(x0[None])
    mx.eval(video)
    return _save_video(video, output_path, fps=cfg_run.sample_fps)
