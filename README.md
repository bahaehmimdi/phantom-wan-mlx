# phantom-wan-mlx

Apple MLX port of **[Phantom-video/Phantom](https://github.com/Phantom-video/Phantom)** (Phantom-Wan, ByteDance, ICCV 2025) — **multi-subject subject-to-video (S2V)**: compose up to 4 distinct characters into one shot from reference images. Apache-2.0.

> **Status: end-to-end working.** The 1.3B v1 path runs natively on Apple Silicon and a
> self-contained MLX artifact (bf16 + int4 DiT, Wan2.1 VAE, umT5-XXL) is built under
> [`dist/Phantom-Wan-1.3B-MLX/`](dist/Phantom-Wan-1.3B-MLX). The reference-injection
> mechanism (the only net-new surface) is specified in
> [`_research/G1_INJECTION.md`](_research/G1_INJECTION.md) and implemented in
> `phantom_wan_mlx/model/reference.py` + `phantom_wan_mlx/sampling.py`.

## Why this is cheap
Rides the **Wan2.1 substrate** published with mlx-video (Wan VAE 16-ch + umT5-XXL + 3D RoPE).
The DiT forward is **unchanged** stock Wan2.1 (`mlx_video.models.wan_2.WanModel`); Phantom's
contribution is purely input assembly + a custom CFG loop.

## The only new surface (G1)
1. **Reference encode** (`model/reference.py::encode_references`) — aspect-preserving LANCZOS
   resize + white-pad → [-1,1] → VAE-encode each ref to one latent frame.
2. **Temporal append** — concat K ref frames at the **tail** of the target latent; refs get
   ordinary trailing 3D-RoPE positions (no SA-3D). Re-clamp clean refs each step; strip the
   K-frame tail at the end (`model/dit.py`).
3. **Dual-scale chained CFG** (`sampling.py::sample_s2v`) — 3 DiT forwards/step:
   `neg(zero refs,∅) + w_img·(refs,∅ − neg) + w_text·(refs,text − refs,∅)`, defaults
   `guide_img=5.0`, `guide_text=7.5`. Scheduler: FlowUniPC, shift 5.0, 50 steps.
4. **Lossless streaming VAE decode** (`streaming_decode.py`) — temporal-chunked decode with
   flat peak memory so 81-frame output decodes (whole-sequence decode OOMs past ~49 frames).

## Usage

```python
from phantom_wan_mlx import pipeline_mlx as P
P.s2v(
    "two friends walking together in a park",
    ["subjectA.png", "subjectB.png"],
    "out.mp4",
    size=(832, 480), frame_num=81, steps=50,   teacache_threshold=0.15,
)
```

`s2v(prompt, reference_images, output_path, size=(832,480), frame_num=81, steps=50,
shift=5.0, guide_img=5.0, guide_text=7.5, seed=0, phantom_pth=None, vae_pth=None,
lossless_decode=True, precomputed_context=None, precomputed_context_null=None, verbose=True)`.
`precomputed_context` is the Swift escape hatch — encode umT5 prompts in Python via
`pipeline_mlx.encode_prompt(...)` and pass the embeddings, bypassing the 11 GB umT5 at runtime.

## Substrate / config (1.3B v1 oracle)
Wan2.1-T2V-1.3B: dim 1536 · ffn 8960 · 12 heads · 30 layers · patch (1,2,2) · 16-ch VAE
(stride 4/8/8) · umT5-xxl · text-only cross-attn (no CLIP). 14B variant config present but
not yet exported.

Pinned checkpoints (`config.py`, `_research/G3_CHECKPOINT.md`):
- Phantom: `bytedance-research/Phantom` @ `926cb19b…` → `Phantom-Wan-1.3B.pth`
- Substrate: `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec5126…` → `Wan2.1_VAE.pth`, umT5-XXL

## Layout
```
phantom_wan_mlx/
  config.py            # S2V 1.3B/14B config + pinned checkpoint IDs/revisions
  pipeline_mlx.py      # s2v(...) entry + encode_prompt() escape hatch
  sampling.py          # sample_s2v: dual-scale chained CFG loop
  model/reference.py   # ref encode + white-pad + temporal append
  model/dit.py         # stock Wan2.1 DiT grid/forward wrapper for F+K frames
  streaming_decode.py  # lossless temporal-chunked VAE decode (flat memory)
  utils/weights.py     # torch/diffusers -> MLX conversion + umT5 load
scripts/
  export_mlx.py        # build the self-contained dist/ artifact (bf16 + int4)
  gen_demo.py          # 2-subject e2e demo from .pth weights
  gen_int4_demo.py     # validate the published int4 artifact end-to-end
tests/smoke/           # substrate, dit_forward, reference, decode_stream
goldens/               # umt5 tokenizer golden
dist/Phantom-Wan-1.3B-MLX/   # published MLX artifact (model card + weights)
_research/G1_INJECTION.md     # the locked spec
refs/Phantom/                 # upstream clone (gitignored)
```

## Published artifact (`dist/Phantom-Wan-1.3B-MLX`)
Self-contained — both DiT precisions plus the shared Wan2.1 substrate:

| File | What |
|------|------|
| `transformer-bf16.safetensors` | DiT, bf16 (~2.84 GB) |
| `transformer-4bit.safetensors` | DiT, int4 group-size 64 (~0.98 GB), cosine 0.99633 vs bf16 |
| `vae-encoder.safetensors` / `vae-decoder.safetensors` | Wan2.1 16-ch VAE (bf16) |
| `t5_encoder.safetensors` | umT5-XXL text encoder (bf16) |
| `config.json` | architecture + sampling defaults |

Install (editable): `pip install -e .` (parity oracle: `pip install -e ".[parity]"`).

## License
Apache-2.0. Derived from Phantom-Wan (ByteDance), Wan2.1 (Wan-AI), mlx-video. See `NOTICE`.
