# Layer-Parallel Inference via Jacobi Iteration: Experimental Report

## Motivation

Transformer inference is sequential across layers: h_0 -> h_1 -> ... -> h_L. The DEER paper (Lim et al., ICLR 2024) showed that sequential computations can be parallelized by treating them as fixed-point problems solvable via Newton's method with parallel scan. We tested whether this approach can speed up inference for nanochat-scale models (20-32 layers, 560M-3.2B params) on H100 GPUs.

## What We Built

- `nanochat/jacobi_forward.py`: 10+ forward pass variants (Jacobi, Gauss-Seidel, hybrid sequential/parallel, Newton/DEER, preheat calibration, CUDA streams, multi-GPU pipeline)
- `scripts/jacobi_eval.py`: Quality evaluation framework (PPL, top-1, cosine similarity) on ClimbMix validation data
- `scripts/jacobi_bench.py`: Wall-clock benchmarks across batch sizes, GPU counts, and attention backends
- Spectral-radius regularization in `nanochat/gpt.py` and `scripts/base_train.py`

## Key Findings

### 1. Vanilla Jacobi iteration diverges on trained models

The layer mappings have spectral radius sigma_max ~ 20-60 (far above the sigma < 1 needed for convergence). Trained transformer layers make large, nonlinear transformations - they are fundamentally non-contractive.

| Model | Avg sigma_max | Jacobi convergence |
|-------|--------------|-------------------|
| Random weights | 0.3 | Converges in L iters |
| Trained d20 | 44 | Diverges |
| Trained d32 | 44 | Diverges |

### 2. Newton's method (DEER) converges but is expensive

With FP32 finite-difference JVP (eps=5e-3), Newton converges:

| Config (d32) | Newton iters | Total layer evals | vs Sequential |
|-------------|-------------|-------------------|---------------|
| seq=28, 4 parallel | 6 iters | 28 + 6*7 = 70 | 0.35x (slower) |
| seq=25, 7 parallel | 10 iters | 25 + 10*13 = 155 | 0.12x (slower) |

Each Newton iteration requires F-evals + JVP evals per parallel layer, and the forward substitution step is sequential. Even with perfect parallel scan (O(log L)), the total work exceeds sequential.

### 3. Per-layer compute is too small for parallelism overhead

| Batch size | Per-layer time | CUDA stream overhead | Verdict |
|-----------|---------------|---------------------|---------|
| BS=1 | 0.44ms | ~0.5ms | Overhead > compute |
| BS=8 | 1.49ms | ~0.5ms | Marginal |
| BS=32 | 5.22ms | ~0.5ms | Streams could help |

But at BS=32, each layer saturates the GPU's SMs, preventing real overlap between streams.

### 4. Multi-GPU pipeline helps but can't beat data parallel

At BS=32 with 2 GPUs (d32 model):

| Strategy | Latency | vs 1-GPU |
|----------|---------|----------|
| Sequential (1 GPU) | 167ms | 1.00x |
| Data parallel (bs/2 each) | 163ms | 1.03x |
| Jacobi 2-GPU (seq=28 par=3) | 185ms | 0.90x |

Data parallel wins trivially because it halves each GPU's work without any approximation.

### 5. Spectral-radius regularization during training helps modestly

Training with a penalty on per-layer sigma_max reduces spectral radius dramatically but Newton iteration count improves only ~20%:

| Model | Loss | Avg sigma | Newton iters (s8) | Train cost |
|-------|------|-----------|-------------------|------------|
| Baseline 2k steps | 4.47 | 44 | 16 | 9 min |
| Spec-reg lambda=10, 6k steps | 4.42 | 2.3 | 13 | 50 min |

The loss actually *improves* slightly with regularization (implicit regularization benefit), but sigma_max stays above 1 for key layers, limiting convergence gains.

### 6. Preheat (offline calibration) doesn't help Newton convergence

A low-rank linear predictor per layer (calibrated offline) provides better initialization than naive h=x0, but Newton converges in the same number of iterations regardless of initialization - the Jacobian quality, not initial distance, determines convergence rate.

## Why It Doesn't Work (for this scale)

The fundamental arithmetic:

1. **Shallow models**: For L=32 layers, saving N layers via parallelism gives at most L/(L-N+iters) speedup. With N=4 parallel layers needing 5 Newton iterations: 32/(28+5) = 0.97x - essentially no speedup.

2. **Non-contractive layers**: Trained transformer layers have sigma_max >> 1 because they need large Jacobians to be expressive. Regularizing to sigma < 1 would destroy model quality.

3. **Communication overhead**: Inter-GPU transfer (~2ms), CUDA stream scheduling (~0.5ms), and Python loop overhead exceed the per-layer compute savings (~0.5-5ms/layer).

4. **Data parallel dominance**: For throughput, splitting the batch across GPUs is simpler, exact, and more effective than approximate layer parallelism.

## Where This Approach Would Work

- **Very deep models (100+ layers)**: More layers to parallelize, better amortization of overhead
- **Large per-layer compute (>10ms)**: Parallelism overhead becomes proportionally small
- **Models trained for contractiveness**: Architectural choices like parallel attention+MLP blocks (GPT-J style) naturally have smaller Jacobians
- **Latency-critical BS=1 inference on deep models**: Where data parallel can't help

## References

- Lim et al., "Parallelizing non-linear sequential models over the sequence length" (ICLR 2024) - DEER
- Song et al., "Accelerating Feedforward Computation via Parallel Nonlinear Equation Solving" (ICML 2021)
- Zoltowski et al., "Parallelizing MCMC Across the Sequence Length" (NeurIPS 2025) - quasi-DEER
- Santilli et al., "Accelerating Transformer Inference for Translation via Parallel Decoding" (ACL 2023)

## Files

```
nanochat/jacobi_forward.py      # All parallel forward variants + preheat + Newton
nanochat/gpt.py                 # Spectral-radius regularization in forward()
nanochat/checkpoint_manager.py  # Backward compat patches for old checkpoints
scripts/base_train.py           # --jacobi-reg, --jacobi-reg-warmup, --spectral-norm flags
scripts/jacobi_eval.py          # Quality evaluation framework
scripts/jacobi_bench.py         # Timing benchmarks
```
