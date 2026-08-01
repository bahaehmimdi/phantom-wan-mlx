import time
import os
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm


class TeaCacheContext:
    """
    Context manager and controller for dynamic feature caching (TeaCache) in MLX.
    Tracks relative L1 distance between steps to skip redundant Transformer evaluations.
    """
    def __init__(self, threshold=0.15, ret_steps=5, coefficients=None):
        self.threshold = threshold
        self.ret_steps = ret_steps
        # Poly coefficients calibrated for Wan DiT dynamics
        self.coefficients = coefficients or [-23.94, 27.31, -0.49, 0.04]
        self.rescale_func = np.poly1d(self.coefficients)
        
        self.cnt = 0
        self.num_steps = 0
        self.accumulated_distance_even = 0.0
        self.accumulated_distance_odd = 0.0
        self.prev_emb_even = None
        self.prev_emb_odd = None
        self.prev_res_even = None
        self.prev_res_odd = None
        
        # Metrics tracking
        self.skipped_steps = 0
        self.total_evals = 0

    def reset(self, total_steps):
        self.cnt = 0
        self.num_steps = total_steps * 2  # cond + uncond per timestep
        self.accumulated_distance_even = 0.0
        self.accumulated_distance_odd = 0.0
        self.prev_emb_even = None
        self.prev_emb_odd = None
        self.prev_res_even = None
        self.prev_res_odd = None
        self.skipped_steps = 0
        self.total_evals = 0

    def should_compute(self, time_emb, is_even):
        """
        Determines whether to compute the DiT residual or reuse cached residual.
        """
        self.total_evals += 1
        
        # Always compute during retention steps (beginning/end of denoising)
        if self.cnt < self.ret_steps or self.cnt >= (self.num_steps - self.ret_steps):
            if is_even:
                self.accumulated_distance_even = 0.0
                self.prev_emb_even = time_emb
            else:
                self.accumulated_distance_odd = 0.0
                self.prev_emb_odd = time_emb
            return True

        prev_emb = self.prev_emb_even if is_even else self.prev_emb_odd
        
        if prev_emb is None:
            if is_even:
                self.prev_emb_even = time_emb
            else:
                self.prev_emb_odd = time_emb
            return True

        # Compute relative L1 distance in MLX
        diff = mx.mean(mx.abs(time_emb - prev_emb))
        norm = mx.mean(mx.abs(prev_emb)) + 1e-8
        rel_l1 = (diff / norm).item()
        
        scaled_dist = float(self.rescale_func(rel_l1))
        
        if is_even:
            self.accumulated_distance_even += scaled_dist
            if self.accumulated_distance_even < self.threshold:
                self.skipped_steps += 1
                return False
            self.accumulated_distance_even = 0.0
            self.prev_emb_even = time_emb
        else:
            self.accumulated_distance_odd += scaled_dist
            if self.accumulated_distance_odd < self.threshold:
                self.skipped_steps += 1
                return False
            self.accumulated_distance_odd = 0.0
            self.prev_emb_odd = time_emb
            
        return True


class DummyTransformerMLX(nn.Module):
    """
    Mock Transformer wrapper representing the loaded 4-bit safetensors weights.
    Applies the TeaCache logic inside the forward pass.
    """
    def __init__(self, weights_path):
        super().__init__()
        self.weights_path = weights_path
        self.teacache = None
        
    def __call__(self, x, timestep, context=None):
        # Time embedding computation
        time_emb = mx.sin(timestep * 0.01) + 0.1
        
        if self.teacache is not None:
            is_even = (self.teacache.cnt % 2 == 0)
            compute = self.teacache.should_compute(time_emb, is_even)
            self.teacache.cnt += 1
            
            if not compute:
                # Reuse residual from previous evaluation
                cached_res = self.teacache.prev_res_even if is_even else self.teacache.prev_res_odd
                return x + cached_res
            
            # Execute standard layer evaluations
            residual = x * 0.05 + 0.01  # Simulated block pass
            if is_even:
                self.teacache.prev_res_even = residual
            else:
                self.teacache.prev_res_odd = residual
                
            return x + residual
        else:
            return x * 1.05


class PipelineMLX:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        if self.model is None or getattr(self.model, "weights_path", None) != model_path:
            print(f"[MLX] Loading 4-bit Transformer from: {model_path}")
            self.model = DummyTransformerMLX(model_path)
            # Synchronize MLX compute stream
            mx.eval()

    def s2v(
        self,
        prompt: str,
        reference_images: list,
        output_path: str = "out.mp4",
        size: tuple = (512, 256),
        frame_num: int = 81,
        steps: int = 30,
        teacache_thresh: float = 0.15,
        phantom_pth: str = "",
    ):
        start_time = time.time()
        
        # 1. Load Model Weights
        self.load_model(phantom_pth)
        
        # 2. Setup TeaCache
        teacache = TeaCacheContext(threshold=teacache_thresh)
        teacache.reset(total_steps=steps)
        self.model.teacache = teacache
        
        print(f"[MLX] Prompt: '{prompt}'")
        print(f"[MLX] Processing {len(reference_images)} reference images at {size[0]}x{size[1]}")
        
        # 3. Latent Initialization (MLX arrays)
        # Latent dimensions: (B, C, F, H, W)
        lat_h, lat_w = size[1] // 8, size[0] // 8
        latents = mx.random.normal((1, 16, (frame_num - 1) // 4 + 1, lat_h, lat_w))
        
        # Timestep schedule
        timesteps = np.linspace(1000, 0, steps)
        
        # 4. Denoising Loop
        print(f"[MLX] Beginning sampling loop ({steps} steps) with TeaCache (threshold={teacache_thresh})...")
        for t in tqdm(timesteps, desc="Sampling Frames"):
            t_array = mx.array([t])
            
            # Conditional evaluation
            latents = self.model(latents, t_array, context=prompt)
            # Unconditional evaluation (CFG)
            latents = self.model(latents, t_array, context="")
            
            # Evaluate stream state periodically to free memory pressure
            mx.eval(latents)

        elapsed = time.time() - start_time
        
        # 5. Output Render Simulation
        print(f"[MLX] Decoding latents into video ({frame_num} frames)...")
        # Ensure target directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"OK")  # Mock output video file write

        # 6. Performance Summary
        skipped = teacache.skipped_steps
        total = teacache.total_evals
        speedup = total / (total - skipped) if (total - skipped) > 0 else 1.0
        
        print("\n" + "="*50)
        print("          TEA-CACHE PERFORMANCE SUMMARY          ")
        print("="*50)
        print(f"Total Model Calls (Cond + Uncond) : {total}")
        print(f"Skipped Model Computations        : {skipped} ({skipped / total * 100:.1f}%)")
        print(f"Effective Acceleration Factor     : {speedup:.2f}x Speedup")
        print(f"Total Execution Time              : {elapsed:.2f}s")
        print(f"Output Saved To                   : {output_path}")
        print("="*50 + "\n")


# Global Singleton Pipeline instance
pipeline_mlx = PipelineMLX()
