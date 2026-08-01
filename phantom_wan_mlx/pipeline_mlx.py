def s2v(prompt: str, reference_images: list, output_path: str,
        size=(832, 480), frame_num: int = 81, steps: int = 50, shift: float = 5.0,
        guide_img: float = 5.0, guide_text: float = 7.5, seed: int = 0,
        phantom_pth=None, vae_pth=None, lossless_decode: bool = True,
        precomputed_context=None, precomputed_context_null=None, verbose: bool = True,
        teacache_threshold: float = 0.0):
    """Generate a subject-consistent video from a prompt + reference images with TeaCache support."""
    w_px, h_px = size
    phantom_pth = phantom_pth or ROOT / "weights/phantom/Phantom-Wan-1.3B.pth"
    vae_pth = vae_pth or ROOT / "weights/wan-base/Wan2.1_VAE.pth"

    cfg_run = PhantomWanConfig.s2v_1_3b()
    model, cfg = W.load_phantom_dit(phantom_pth)

    teacache = None
    if teacache_threshold > 0.0:
        from .teacache import TeaCache
        model_name = "14B" if getattr(cfg, "num_layers", 30) == 40 else "1.3B"
        teacache = TeaCache(
            num_inference_steps=steps,
            model_name=model_name,
            threshold=teacache_threshold
        )
        if verbose:
            print(f"[TeaCache] Active for {model_name} with threshold={teacache_threshold}", flush=True)

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

    f_latent = (frame_num - 1) // cfg_run.vae_stride[0] + 1
    h_lat, w_lat = h_px // cfg_run.vae_stride[1], w_px // cfg_run.vae_stride[2]
    if verbose:
        print(f"f_latent={f_latent} (={frame_num} frames) grid {h_lat}x{w_lat}, K={ref_lat.shape[2]} refs", flush=True)

    x0 = sample_s2v(model, ref_lat, ctx, ctx_null, cfg, f_latent, h_lat, w_lat,
                    steps=steps, shift=shift, guide_img=guide_img, guide_text=guide_text,
                    seed=seed, verbose=verbose, teacache=teacache)
    del model

    dec = W.load_wan_vae(vae_pth, encoder=False)
    if lossless_decode:
        from .streaming_decode import decode_streaming
        video = decode_streaming(dec, x0[None], chunk_lat=1)
    else:
        video = dec.decode(x0[None])
    mx.eval(video)
    return _save_video(video, output_path, fps=cfg_run.sample_fps)
