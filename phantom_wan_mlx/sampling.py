"""Dual-scale chained CFG sampler for S2V (G1)."""
from __future__ import annotations

import mlx.core as mx

from .model import dit as DIT


def sample_s2v(
    model,
    ref_latents: mx.array,        # [1, 16, K, h, w] clean reference latents
    context: mx.array,            # text embedding [1, seq, 4096]
    context_null: mx.array,       # null/neg-prompt embedding [1, seq, 4096]
    cfg,                          # PhantomWanConfig / WanModelConfig (patch_size)
    f_latent: int,                # F target latent frames (21)
    h_latent: int,                # 32
    w_latent: int,                # 64
    steps: int = 50,
    shift: float = 5.0,
    guide_img: float = 5.0,
    guide_text: float = 7.5,
    seed: int = 0,
    verbose: bool = False,
):
    from mlx_video.models.wan_2.scheduler import FlowUniPCScheduler

    refs = ref_latents[0]                       # [16, K, h, w]
    k = refs.shape[1]
    refs_neg = mx.zeros_like(refs)
    patch = getattr(cfg, "patch_size", (1, 2, 2))

    rope, seq_len = DIT.prepare_grid(model, f_latent + k, h_latent, w_latent, patch)

    sched = FlowUniPCScheduler(num_train_timesteps=getattr(cfg, "num_train_timesteps", 1000))
    sched.set_timesteps(steps, shift=shift)

    mx.random.seed(seed)
    latent = mx.random.normal((16, f_latent + k, h_latent, w_latent))   # [16, 23, 32, 64]

    if context.ndim == 2:
        context = context[None]
    if context_null.ndim == 2:
        context_null = context_null[None]

    for i, t in enumerate(sched.timesteps):
        t_arr = mx.array([t])
        noisy_target = latent[:, :-k]                                    # [16, 21, 32, 64]
        
        # 5D Tensors [1, 16, 23, 32, 64]
        inp_refs = mx.concatenate([noisy_target, refs], axis=1)[None]   
        inp_zero = mx.concatenate([noisy_target, refs_neg], axis=1)[None]

        pos_it = DIT.forward(model, inp_refs, t_arr, context, rope, seq_len)[0]
        pos_i = DIT.forward(model, inp_refs, t_arr, context_null, rope, seq_len)[0]
        neg = DIT.forward(model, inp_zero, t_arr, context_null, rope, seq_len)[0]

        noise_pred = neg + guide_img * (pos_i - neg) + guide_text * (pos_it - pos_i)
        
        latent = sched.step(noise_pred[None], t, latent[None]).squeeze(0)
        mx.eval(latent)                                                 # Metal sync boundary
        
        if verbose:
            print(f"  step {i + 1}/{steps}", flush=True)

    return latent[:, :-k]                                               # strip reference frames -> [16, 21, 32, 64]
