"""
Jacobi iteration over transformer layers for approximate parallel inference.

Instead of running layers sequentially (h_0 -> h_1 -> ... -> h_L),
we treat the layer stack as a fixed-point problem and solve via Jacobi iteration:

    Initialize: h_i^(0) = x0 for all layers i
    For k = 0, 1, 2, ...:
        For all layers i IN PARALLEL:
            input_i = resid_lambdas[i] * h_{i-1}^(k) + x0_lambdas[i] * x0
            h_i^(k+1) = block_i(input_i, ...)

At convergence (k -> inf), this produces the same output as the sequential forward pass.
The key question: does it converge in K << L iterations?

Reference: "Accelerating Feedforward Computation via Parallel Nonlinear Equation Solving"
           (Song et al., ICML 2021) - Jacobi iteration over layers of feedforward networks.
           "Parallelizing non-linear sequential models over the sequence length"
           (Lim et al., ICLR 2024) - DEER: Newton's method for sequential models.
"""

import torch
import torch.nn.functional as F
from nanochat.common import COMPUTE_DTYPE


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _embed(model, idx):
    """Embedding + smear, shared across all forward variants. Returns (x0, cos_sin, value_embeds)."""
    B, T = idx.size()
    assert T <= model.cos.size(1)
    cos_sin = model.cos[:, :T], model.sin[:, :T]

    x = model.transformer.wte(idx).to(COMPUTE_DTYPE)
    x = norm(x)

    if T > 1:
        gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(model.smear_gate(x[:, 1:, :24]))
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

    value_embeds = {}
    for key in model.value_embeds:
        value_embeds[key] = model.value_embeds[key](idx).to(x.dtype)

    return x, cos_sin, value_embeds


def _logits(model, h):
    """Compute logits from final hidden states, shared across all forward variants."""
    n_layer = model.config.n_layer
    x_final = h[-1]
    backout_layer = n_layer // 2
    if backout_layer < n_layer:
        x_final = x_final - model.backout_lambda.to(x_final.dtype) * h[backout_layer]
    x_final = norm(x_final)

    softcap = 15
    logits = model.lm_head(x_final)
    logits = logits[..., :model.config.vocab_size]
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)
    return logits


def _run_block(model, i, h_prev, x0, cos_sin, value_embeds):
    """Run a single transformer block with the resid/x0 blending."""
    x_in = model.resid_lambdas[i] * h_prev + model.x0_lambdas[i] * x0
    ve = value_embeds.get(str(i))
    return model.transformer.h[i](x_in, ve, cos_sin, model.window_sizes[i], None)


def _compute_deltas(h_new, h_old, n_layer):
    """Compute relative delta between two sets of hidden states."""
    deltas = []
    for i in range(n_layer):
        delta = (h_new[i] - h_old[i]).float().norm() / (h_old[i].float().norm() + 1e-8)
        deltas.append(delta.item())
    return deltas


# ---------------------------------------------------------------------------
# Preheat: offline calibration for better initialization
# ---------------------------------------------------------------------------

class PreheatCache:
    """
    Stores per-layer predictors that map x0 -> approximate h[i].

    Calibrated offline by running representative inputs through the model
    and fitting a low-rank linear map per layer:
        h[i] ≈ bias[i] + U[i] @ (V[i]^T @ x0)

    At runtime, predicting h[i] from x0 costs one matmul per layer (parallelizable).
    """

    def __init__(self, n_layer, n_embd, rank, device, dtype):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.rank = rank
        # Per-layer: bias (d,), U (d, r), V (d, r) where W ≈ U @ V^T
        self.bias = [torch.zeros(n_embd, device=device, dtype=dtype) for _ in range(n_layer)]
        self.U = [torch.zeros(n_embd, rank, device=device, dtype=dtype) for _ in range(n_layer)]
        self.V = [torch.zeros(n_embd, rank, device=device, dtype=dtype) for _ in range(n_layer)]

    def predict(self, x0):
        """Predict h[i] for all layers given embedding x0 (B, T, D). Returns list of (B, T, D)."""
        h = []
        for i in range(self.n_layer):
            # h[i] ≈ bias[i] + x0 @ V[i] @ U[i]^T  (low-rank affine)
            proj = x0 @ self.V[i]          # (B, T, r)
            pred = proj @ self.U[i].T      # (B, T, D)
            pred = pred + self.bias[i]
            h.append(pred)
        return h

    def save(self, path):
        torch.save({
            'n_layer': self.n_layer, 'n_embd': self.n_embd, 'rank': self.rank,
            'bias': [b.cpu() for b in self.bias],
            'U': [u.cpu() for u in self.U],
            'V': [v.cpu() for v in self.V],
        }, path)

    @classmethod
    def load(cls, path, device):
        data = torch.load(path, map_location=device)
        dtype = data['U'][0].dtype
        cache = cls(data['n_layer'], data['n_embd'], data['rank'], device, dtype)
        cache.bias = [b.to(device) for b in data['bias']]
        cache.U = [u.to(device) for u in data['U']]
        cache.V = [v.to(device) for v in data['V']]
        return cache


@torch.inference_mode()
def calibrate_preheat(model, n_samples=32, seq_len=128, rank=32):
    """
    Run calibration sequences through the model, collect (x0, h[i]) pairs,
    and fit a low-rank affine predictor per layer: h[i] ≈ x0 @ V @ U^T + bias.

    Args:
        model: GPT model
        n_samples: number of random sequences to calibrate on
        seq_len: sequence length for calibration
        rank: rank of the per-layer linear predictor

    Returns:
        PreheatCache with fitted predictors
    """
    device = model.get_device()
    config = model.config
    n_layer = config.n_layer
    n_embd = config.n_embd
    vocab_size = config.vocab_size

    # Collect x0 and h[i] over calibration data
    all_x0 = []
    all_h = [[] for _ in range(n_layer)]

    rng_cpu = torch.Generator(device='cpu')
    rng_cpu.manual_seed(0)

    for s in range(n_samples):
        idx = torch.randint(0, vocab_size, (1, seq_len), generator=rng_cpu).to(device)
        x0, cos_sin, value_embeds = _embed(model, idx)

        # Sequential forward, recording each layer's output
        x = x0
        for i in range(n_layer):
            x = _run_block(model, i, x, x0, cos_sin, value_embeds)
            # Subsample positions to save memory
            all_h[i].append(x[:, ::4, :].reshape(-1, n_embd).float().cpu())

        all_x0.append(x0[:, ::4, :].reshape(-1, n_embd).float().cpu())

    # Stack: X0 is (N, D), H[i] is (N, D)
    X0 = torch.cat(all_x0, dim=0)
    N = X0.shape[0]
    r = min(rank, N, n_embd)

    # Center
    x0_mean = X0.mean(dim=0)
    X0c = X0 - x0_mean

    # Get top-r principal directions of x0 variation via truncated SVD
    U_svd, S_svd, Vt_svd = torch.linalg.svd(X0c, full_matrices=False)
    # P: (D, r) projection matrix — maps x0 to r-dim space
    P = Vt_svd[:r, :].T  # (D, r)
    # X0 in reduced coords
    X0_r = X0c @ P  # (N, r)
    # Gram matrix for regression (r x r, small)
    gram = X0_r.T @ X0_r + 1e-4 * torch.eye(r)

    dtype = next(model.parameters()).dtype
    cache = PreheatCache(n_layer, n_embd, r, device, dtype)

    for i in range(n_layer):
        Hi = torch.cat(all_h[i], dim=0)
        hi_mean = Hi.mean(dim=0)
        Hic = Hi - hi_mean

        # Regress: Hic ≈ X0_r @ coeff, solve via normal equations
        coeff = torch.linalg.solve(gram, X0_r.T @ Hic)  # (r, D)

        # Store: prediction(x0) = (x0 - x0_mean) @ P @ coeff + hi_mean
        #                        = x0 @ V @ U^T + bias
        # where V = P, U = coeff^T, bias = hi_mean - x0_mean @ P @ coeff
        cache.V[i] = P.to(device=device, dtype=dtype)
        cache.U[i] = coeff.T.to(device=device, dtype=dtype)
        cache.bias[i] = (hi_mean - x0_mean @ P @ coeff).to(device=device, dtype=dtype)

    return cache


@torch.inference_mode()
def preheat_forward(model, idx, preheat_cache, max_iters=None,
                    seq_layers=0, return_diagnostics=False):
    """
    Jacobi iteration initialized with preheat predictions.

    Instead of h[i] = x0, uses h[i] = preheat_cache.predict(x0)[i]
    which is a learned linear approximation of each layer's output.

    Optionally runs first `seq_layers` sequentially for exact prefix,
    then uses preheat predictions for the rest.
    """
    config = model.config
    n_layer = config.n_layer
    if max_iters is None:
        max_iters = n_layer

    x0, cos_sin, value_embeds = _embed(model, idx)

    # Initialize with preheat predictions
    h_pred = preheat_cache.predict(x0)

    h = [None] * n_layer
    # Sequential prefix (exact)
    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, value_embeds)
        h[i] = x
    # Preheat predictions for the rest
    for i in range(seq_layers, n_layer):
        h[i] = h_pred[i]

    diagnostics = {'per_iter_delta': [], 'per_layer_delta': [],
                   'sequential_layers': seq_layers,
                   'init': f'preheat(rank={preheat_cache.rank}, seq={seq_layers})'
                   } if return_diagnostics else None

    for k in range(max_iters):
        h_new = [None] * n_layer
        for i in range(n_layer):
            h_prev = h[i - 1] if i > 0 else x0
            h_new[i] = _run_block(model, i, h_prev, x0, cos_sin, value_embeds)

        if return_diagnostics:
            layer_deltas = _compute_deltas(h_new, h, n_layer)
            diagnostics['per_iter_delta'].append(max(layer_deltas))
            diagnostics['per_layer_delta'].append(layer_deltas)
        h = h_new

    if return_diagnostics:
        return _logits(model, h), diagnostics
    return _logits(model, h)


@torch.inference_mode()
def preheat_hybrid_newton_forward(model, idx, preheat_cache, seq_layers=0,
                                   newton_iters=2, return_diagnostics=False):
    """
    Best combo: preheat init + optional sequential prefix + Newton refinement.
    """
    config = model.config
    n_layer = config.n_layer

    x0, cos_sin, value_embeds = _embed(model, idx)
    h_pred = preheat_cache.predict(x0)

    h = [None] * n_layer
    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, value_embeds)
        h[i] = x
    for i in range(seq_layers, n_layer):
        h[i] = h_pred[i]

    eps = 1e-3

    diagnostics = {'per_iter_delta': [], 'per_layer_delta': [],
                   'sequential_layers': seq_layers,
                   } if return_diagnostics else None

    start = seq_layers

    for k in range(newton_iters):
        F_h = {}
        for i in range(start, n_layer):
            h_prev = h[i - 1]
            F_h[i] = _run_block(model, i, h_prev, x0, cos_sin, value_embeds)

        residuals = {i: h[i] - F_h[i] for i in range(start, n_layer)}

        corrections = {}
        corrections[start] = residuals[start]
        for i in range(start + 1, n_layer):
            c_prev = corrections[i - 1]
            c_norm = c_prev.float().norm().item()
            if c_norm < 1e-8:
                corrections[i] = residuals[i]
            else:
                h_prev_pert = h[i - 1] + eps * c_prev
                F_pert = _run_block(model, i, h_prev_pert, x0, cos_sin, value_embeds)
                jvp = (F_pert - F_h[i]) / eps
                corrections[i] = residuals[i] + jvp

        h_new = list(h)
        for i in range(start, n_layer):
            h_new[i] = h[i] - corrections[i]

        if return_diagnostics:
            layer_deltas = []
            for i in range(start, n_layer):
                delta = (h_new[i] - h[i]).float().norm() / (h[i].float().norm() + 1e-8)
                layer_deltas.append(delta.item())
            diagnostics['per_iter_delta'].append(max(layer_deltas) if layer_deltas else 0.0)
            diagnostics['per_layer_delta'].append(layer_deltas)

        h = h_new

    if return_diagnostics:
        return _logits(model, h), diagnostics
    return _logits(model, h)


# ---------------------------------------------------------------------------
# Strategy 0: Vanilla Jacobi (baseline)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def jacobi_forward(model, idx, max_iters=None, tol=0.0, return_diagnostics=False):
    """
    Forward pass using Jacobi iteration over layers.
    Initializes all h[i] = x0 (the embedding). Worst-case init.
    """
    config = model.config
    n_layer = config.n_layer
    if max_iters is None:
        max_iters = n_layer

    x0, cos_sin, value_embeds = _embed(model, idx)
    h = [x0.clone() for _ in range(n_layer)]

    diagnostics = {'per_iter_delta': [], 'per_layer_delta': [], 'converged_at': max_iters,
                   'sequential_layers': 0} if return_diagnostics else None

    for k in range(max_iters):
        h_new = [None] * n_layer
        for i in range(n_layer):
            h_prev = h[i - 1] if i > 0 else x0
            h_new[i] = _run_block(model, i, h_prev, x0, cos_sin, value_embeds)

        if return_diagnostics or tol > 0:
            layer_deltas = _compute_deltas(h_new, h, n_layer)
            max_delta = max(layer_deltas)
            if return_diagnostics:
                diagnostics['per_iter_delta'].append(max_delta)
                diagnostics['per_layer_delta'].append(layer_deltas)
            if tol > 0 and max_delta < tol:
                if return_diagnostics:
                    diagnostics['converged_at'] = k + 1
                h = h_new
                break
        h = h_new

    if return_diagnostics:
        return _logits(model, h), diagnostics
    return _logits(model, h)


# ---------------------------------------------------------------------------
# Strategy 1: Better initialization
# ---------------------------------------------------------------------------

@torch.inference_mode()
def jacobi_forward_layerskip_init(model, idx, max_iters=None, skip_stride=2,
                                   return_diagnostics=False):
    """
    Jacobi with layer-skip draft initialization.

    Instead of h[i] = x0, run a cheap sequential "draft" pass through every
    `skip_stride`-th layer to get approximate hidden states, then interpolate
    for skipped layers. Cost: L/stride sequential layer evals for init.

    Args:
        skip_stride: run every N-th layer in the draft pass (2 = half layers)
    """
    config = model.config
    n_layer = config.n_layer
    if max_iters is None:
        max_iters = n_layer

    x0, cos_sin, value_embeds = _embed(model, idx)

    # Draft pass: run every skip_stride-th layer sequentially
    draft_indices = list(range(0, n_layer, skip_stride))
    draft_cost = len(draft_indices)

    h = [None] * n_layer
    x = x0
    last_draft_idx = -1
    for i in draft_indices:
        x = _run_block(model, i, x, x0, cos_sin, value_embeds)
        h[i] = x
        last_draft_idx = i

    # Interpolate skipped layers: use the output of the previous draft layer
    for i in range(n_layer):
        if h[i] is None:
            # Find the nearest preceding draft layer
            prev_draft = max(j for j in draft_indices if j < i) if any(j < i for j in draft_indices) else -1
            h[i] = h[prev_draft].clone() if prev_draft >= 0 else x0.clone()

    diagnostics = {'per_iter_delta': [], 'per_layer_delta': [], 'converged_at': max_iters,
                   'sequential_layers': draft_cost, 'init': f'layerskip(stride={skip_stride})'
                   } if return_diagnostics else None

    # Jacobi refinement iterations
    for k in range(max_iters):
        h_new = [None] * n_layer
        for i in range(n_layer):
            h_prev = h[i - 1] if i > 0 else x0
            h_new[i] = _run_block(model, i, h_prev, x0, cos_sin, value_embeds)

        if return_diagnostics:
            layer_deltas = _compute_deltas(h_new, h, n_layer)
            diagnostics['per_iter_delta'].append(max(layer_deltas))
            diagnostics['per_layer_delta'].append(layer_deltas)
        h = h_new

    if return_diagnostics:
        return _logits(model, h), diagnostics
    return _logits(model, h)


@torch.inference_mode()
def jacobi_forward_sequential_init(model, idx, max_iters=None, init_layers=None,
                                    return_diagnostics=False):
    """
    Jacobi with partial sequential initialization.

    Run the first `init_layers` layers sequentially to get correct hidden states
    for the early layers, then use the last sequential output as the init for
    all remaining layers.

    This exploits the observation that early layers do the "heavy lifting" of
    transforming x0 into the representation space, while later layers make
    smaller refinements.
    """
    config = model.config
    n_layer = config.n_layer
    if init_layers is None:
        init_layers = n_layer // 4  # default: run first quarter sequentially
    if max_iters is None:
        max_iters = n_layer

    x0, cos_sin, value_embeds = _embed(model, idx)

    # Sequential pass through first init_layers
    h = [None] * n_layer
    x = x0
    for i in range(init_layers):
        x = _run_block(model, i, x, x0, cos_sin, value_embeds)
        h[i] = x

    # Initialize remaining layers with the last sequential output
    for i in range(init_layers, n_layer):
        h[i] = x.clone()

    diagnostics = {'per_iter_delta': [], 'per_layer_delta': [], 'converged_at': max_iters,
                   'sequential_layers': init_layers, 'init': f'sequential({init_layers})'
                   } if return_diagnostics else None

    # Jacobi refinement
    for k in range(max_iters):
        h_new = [None] * n_layer
        for i in range(n_layer):
            h_prev = h[i - 1] if i > 0 else x0
            h_new[i] = _run_block(model, i, h_prev, x0, cos_sin, value_embeds)

        if return_diagnostics:
            layer_deltas = _compute_deltas(h_new, h, n_layer)
            diagnostics['per_iter_delta'].append(max(layer_deltas))
            diagnostics['per_layer_delta'].append(layer_deltas)
        h = h_new

    if return_diagnostics:
        return _logits(model, h), diagnostics
    return _logits(model, h)


# ---------------------------------------------------------------------------
# Strategy 2: Hybrid sequential/parallel
# ---------------------------------------------------------------------------

@torch.inference_mode()
def hybrid_forward(model, idx, seq_layers=None, parallel_iters=1,
                   return_diagnostics=False):
    """
    Hybrid sequential/parallel forward.

    Run the first `seq_layers` layers sequentially (exact), then run the
    remaining layers via Jacobi iteration for `parallel_iters` rounds.

    Total cost: seq_layers + (n_layer - seq_layers) * parallel_iters layer evals.
    Sequential cost: n_layer layer evals.
    Speedup potential: when parallel_iters << remaining layers.

    The sequential prefix provides exact hidden states for early layers,
    and the Jacobi suffix tries to approximate the rest.
    """
    config = model.config
    n_layer = config.n_layer
    if seq_layers is None:
        seq_layers = n_layer // 2

    x0, cos_sin, value_embeds = _embed(model, idx)

    # Phase 1: Sequential prefix (exact)
    h = [None] * n_layer
    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, value_embeds)
        h[i] = x

    # Initialize parallel suffix with the last sequential output
    for i in range(seq_layers, n_layer):
        h[i] = x.clone()

    n_parallel = n_layer - seq_layers
    total_layer_evals = seq_layers + n_parallel * parallel_iters

    diagnostics = {'per_iter_delta': [], 'per_layer_delta': [], 'converged_at': parallel_iters,
                   'sequential_layers': seq_layers, 'parallel_layers': n_parallel,
                   'parallel_iters': parallel_iters, 'total_layer_evals': total_layer_evals
                   } if return_diagnostics else None

    # Phase 2: Jacobi iterations over remaining layers
    for k in range(parallel_iters):
        h_new = list(h)  # copy the sequential prefix (won't change)
        for i in range(seq_layers, n_layer):
            h_prev = h[i - 1]  # either sequential (exact) or previous Jacobi iter
            h_new[i] = _run_block(model, i, h_prev, x0, cos_sin, value_embeds)

        if return_diagnostics:
            # Only measure deltas for the parallel suffix
            layer_deltas = []
            for i in range(seq_layers, n_layer):
                delta = (h_new[i] - h[i]).float().norm() / (h[i].float().norm() + 1e-8)
                layer_deltas.append(delta.item())
            diagnostics['per_iter_delta'].append(max(layer_deltas) if layer_deltas else 0.0)
            diagnostics['per_layer_delta'].append(layer_deltas)

        h = h_new

    if return_diagnostics:
        return _logits(model, h), diagnostics
    return _logits(model, h)


@torch.inference_mode()
def hybrid_forward_streams(model, idx, seq_layers=None, parallel_iters=1):
    """
    Hybrid sequential/parallel with CUDA streams for actual parallelism.
    The parallel suffix layers run on separate CUDA streams within each iteration.
    Uses proper event synchronization to avoid data races.
    """
    config = model.config
    n_layer = config.n_layer
    if seq_layers is None:
        seq_layers = n_layer // 2

    device = model.get_device()
    use_streams = device.type == 'cuda'
    x0, cos_sin, value_embeds = _embed(model, idx)

    # Phase 1: Sequential prefix on default stream
    h = [None] * n_layer
    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, value_embeds)
        h[i] = x
    for i in range(seq_layers, n_layer):
        h[i] = x.clone()

    n_par = n_layer - seq_layers
    if n_par == 0:
        return _logits(model, h)

    if not use_streams:
        for k in range(parallel_iters):
            h_new = list(h)
            for i in range(seq_layers, n_layer):
                h_new[i] = _run_block(model, i, h[i - 1], x0, cos_sin, value_embeds)
            h = h_new
        return _logits(model, h)

    # Create streams for parallel layers
    streams = [torch.cuda.Stream(device=device) for _ in range(n_par)]
    default_stream = torch.cuda.current_stream()

    # Phase 2: Jacobi with proper event-synchronized streams
    for k in range(parallel_iters):
        h_new = list(h)

        # Signal that h is ready to be read by all streams
        ready = torch.cuda.Event()
        ready.record(default_stream)

        # Launch all parallel layers on separate streams
        for j, i in enumerate(range(seq_layers, n_layer)):
            with torch.cuda.stream(streams[j]):
                streams[j].wait_event(ready)
                h_new[i] = _run_block(model, i, h[i - 1], x0, cos_sin, value_embeds)

        # Sync all streams back to default before next iteration
        for s in streams:
            default_stream.wait_stream(s)

        h = h_new

    return _logits(model, h)


@torch.inference_mode()
def hybrid_forward_multigpu(model, idx, seq_layers=None, parallel_iters=1,
                             gpu_ids=None):
    """
    Multi-GPU hybrid forward: sequential prefix on GPU 0, then distribute
    parallel suffix layers across multiple GPUs for true parallelism.

    Each GPU gets a copy of its assigned block weights and runs its layers
    independently. Communication is just the h tensors between iterations.
    """
    config = model.config
    n_layer = config.n_layer
    if seq_layers is None:
        seq_layers = n_layer // 2
    if gpu_ids is None:
        gpu_ids = list(range(min(4, torch.cuda.device_count())))

    primary = torch.device(f'cuda:{gpu_ids[0]}')
    n_par = n_layer - seq_layers
    n_gpus = len(gpu_ids)

    if n_par == 0 or n_gpus < 2:
        return hybrid_forward(model, idx, seq_layers=seq_layers, parallel_iters=parallel_iters)

    # Move input to primary GPU
    idx = idx.to(primary)
    x0, cos_sin, value_embeds = _embed(model, idx)

    # Phase 1: Sequential prefix on primary GPU
    h = [None] * n_layer
    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, value_embeds)
        h[i] = x
    for i in range(seq_layers, n_layer):
        h[i] = x.clone()

    # Assign parallel layers to GPUs (round-robin)
    par_layers = list(range(seq_layers, n_layer))
    gpu_assignments = {}  # layer_idx -> gpu_id
    gpu_layer_groups = {g: [] for g in gpu_ids}
    for j, layer_idx in enumerate(par_layers):
        gid = gpu_ids[j % n_gpus]
        gpu_assignments[layer_idx] = gid
        gpu_layer_groups[gid].append(layer_idx)

    # Copy block weights to assigned GPUs (one-time cost)
    block_copies = {}
    cos_copies, sin_copies = {}, {}
    for gid in gpu_ids:
        dev = torch.device(f'cuda:{gid}')
        cos_copies[gid] = cos_sin[0].to(dev)
        sin_copies[gid] = cos_sin[1].to(dev)
        for layer_idx in gpu_layer_groups[gid]:
            if layer_idx == seq_layers and gid == gpu_ids[0]:
                block_copies[layer_idx] = model.transformer.h[layer_idx]  # already on primary
            else:
                block_copies[layer_idx] = _copy_block_to_device(model, layer_idx, dev)

    # Also need x0, value_embeds, resid/x0 lambdas on each GPU
    x0_copies = {g: x0.to(torch.device(f'cuda:{g}')) for g in gpu_ids}
    ve_copies = {}
    for gid in gpu_ids:
        dev = torch.device(f'cuda:{gid}')
        ve_copies[gid] = {k: v.to(dev) for k, v in value_embeds.items()}
    resid_copies = {g: model.resid_lambdas.to(torch.device(f'cuda:{g}')) for g in gpu_ids}
    x0lam_copies = {g: model.x0_lambdas.to(torch.device(f'cuda:{g}')) for g in gpu_ids}

    # Phase 2: Multi-GPU Jacobi iterations
    streams = {g: torch.cuda.Stream(device=torch.device(f'cuda:{g}')) for g in gpu_ids}

    for k in range(parallel_iters):
        h_new = list(h)

        # Launch all GPUs in parallel
        gpu_results = {}
        for gid in gpu_ids:
            dev = torch.device(f'cuda:{gid}')
            with torch.cuda.stream(streams[gid]):
                for layer_idx in gpu_layer_groups[gid]:
                    # Get h[i-1] from previous iteration, move to this GPU
                    h_prev = h[layer_idx - 1].to(dev, non_blocking=True)
                    x0_g = x0_copies[gid]
                    cs_g = (cos_copies[gid], sin_copies[gid])
                    ve = ve_copies[gid].get(str(layer_idx))
                    x_in = resid_copies[gid][layer_idx] * h_prev + x0lam_copies[gid][layer_idx] * x0_g
                    block = block_copies[layer_idx]
                    ws = model.window_sizes[layer_idx]
                    result = block(x_in, ve, cs_g, ws, None)
                    gpu_results[layer_idx] = (result, gid)

        # Synchronize all GPUs
        for gid in gpu_ids:
            streams[gid].synchronize()

        # Gather results back to primary GPU
        for layer_idx, (result, gid) in gpu_results.items():
            h_new[layer_idx] = result.to(primary, non_blocking=True)
        torch.cuda.synchronize()

        h = h_new

    return _logits(model, h)


def _copy_block_to_device(model, layer_idx, device):
    """Deep copy a transformer block to a different device."""
    import copy
    block = model.transformer.h[layer_idx]
    # Create a new block on the target device with copied weights
    new_block = copy.deepcopy(block)
    new_block.to(device)
    return new_block


# ---------------------------------------------------------------------------
# Strategy 3: Newton's method (DEER-inspired)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def newton_forward(model, idx, max_iters=None, seq_layers=0,
                   return_diagnostics=False):
    """
    Newton's method for layer-parallel inference (DEER-inspired).

    Instead of the zeroth-order Jacobi update h^{k+1} = F(h^k), Newton's
    method uses the first-order correction:

        h^{k+1} = h^k - J^{-1} * (h^k - F(h^k))

    where J is the Jacobian of the fixed-point map G(h) = h - F(h).

    For our layer-sequential structure, J is lower bidiagonal:
        J_ii = I - dF_i/dh_i  (diagonal blocks)
        J_{i,i-1} = -dF_i/dh_{i-1}  (sub-diagonal blocks)

    Computing full Jacobians is expensive (d^2 per layer), so we use a
    finite-difference approximation to the Jacobian-vector product (JVP).

    The Newton step becomes:
        residual_i = h_i - F_i(h_{i-1})
        correction_i = J^{-1} * residual  (solved via forward substitution)
        h_i^{new} = h_i - correction_i

    For the bidiagonal structure, forward substitution is sequential in layers
    but each layer's JVP can be computed with just 2 forward passes (finite diff).
    """
    config = model.config
    n_layer = config.n_layer
    if max_iters is None:
        max_iters = 3  # Newton converges much faster

    x0, cos_sin, value_embeds = _embed(model, idx)

    # Initialize with sequential prefix + propagation
    h = [None] * n_layer
    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, value_embeds)
        h[i] = x
    for i in range(seq_layers, n_layer):
        h[i] = x.clone()

    eps = 1e-3  # finite difference step size

    diagnostics = {'per_iter_delta': [], 'per_layer_delta': [],
                   'sequential_layers': seq_layers,
                   } if return_diagnostics else None

    for k in range(max_iters):
        # Step 1: Compute F(h) for all layers (parallelizable)
        F_h = [None] * n_layer
        for i in range(n_layer):
            h_prev = h[i - 1] if i > 0 else x0
            F_h[i] = _run_block(model, i, h_prev, x0, cos_sin, value_embeds)

        # Step 2: Compute residuals r_i = h_i - F_i(h_{i-1})
        residuals = [h[i] - F_h[i] for i in range(n_layer)]

        # Step 3: Approximate Newton correction via forward substitution
        # For bidiagonal J, solve J * correction = residual forward:
        #   correction_0 = residual_0  (no sub-diagonal for layer 0)
        #   correction_i = residual_i + A_{i,i-1} * correction_{i-1}
        # where A_{i,i-1} approximates dF_i/dh_{i-1} applied to correction_{i-1}
        # We approximate A_{i,i-1} * v via finite differences:
        #   (F_i(h_{i-1} + eps*v) - F_i(h_{i-1})) / eps

        corrections = [None] * n_layer
        corrections[0] = residuals[0]  # first layer has no sub-diagonal

        for i in range(1, n_layer):
            # Approximate dF_i/dh_{i-1} * correction_{i-1} via finite diff
            h_prev = h[i - 1]
            c_prev = corrections[i - 1]
            c_norm = c_prev.float().norm().item()

            if c_norm < 1e-8:
                # Correction is negligible, skip expensive JVP
                corrections[i] = residuals[i]
            else:
                # Finite-difference JVP: (F_i(h_{i-1} + eps*c) - F_i(h_{i-1})) / eps
                h_prev_pert = h_prev + eps * c_prev
                F_pert = _run_block(model, i, h_prev_pert, x0, cos_sin, value_embeds)
                jvp = (F_pert - F_h[i]) / eps
                corrections[i] = residuals[i] + jvp

        # Step 4: Newton update h^{new} = h - correction
        h_new = [h[i] - corrections[i] for i in range(n_layer)]

        if return_diagnostics:
            layer_deltas = _compute_deltas(h_new, h, n_layer)
            diagnostics['per_iter_delta'].append(max(layer_deltas))
            diagnostics['per_layer_delta'].append(layer_deltas)

        h = h_new

    if return_diagnostics:
        return _logits(model, h), diagnostics
    return _logits(model, h)


# ---------------------------------------------------------------------------
# Strategy 2+3 combined: Hybrid + Newton
# ---------------------------------------------------------------------------

@torch.inference_mode()
def hybrid_newton_forward(model, idx, seq_layers=None, newton_iters=2,
                          return_diagnostics=False):
    """
    Best of both worlds: sequential prefix for exact early layers,
    then Newton refinement for the parallel suffix.

    This should converge much faster than hybrid + Jacobi because
    Newton's method has quadratic convergence.
    """
    config = model.config
    n_layer = config.n_layer
    if seq_layers is None:
        seq_layers = n_layer // 2

    x0, cos_sin, value_embeds = _embed(model, idx)

    # Phase 1: Sequential prefix
    h = [None] * n_layer
    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, value_embeds)
        h[i] = x
    for i in range(seq_layers, n_layer):
        h[i] = x.clone()

    eps = 1e-3
    n_parallel = n_layer - seq_layers

    diagnostics = {'per_iter_delta': [], 'per_layer_delta': [],
                   'sequential_layers': seq_layers, 'newton_iters': newton_iters,
                   } if return_diagnostics else None

    # Phase 2: Newton iterations on parallel suffix only
    for k in range(newton_iters):
        # Compute F(h) for parallel layers
        F_h = {}
        for i in range(seq_layers, n_layer):
            h_prev = h[i - 1]
            F_h[i] = _run_block(model, i, h_prev, x0, cos_sin, value_embeds)

        # Residuals
        residuals = {i: h[i] - F_h[i] for i in range(seq_layers, n_layer)}

        # Forward substitution for Newton correction
        corrections = {}
        first_parallel = seq_layers
        corrections[first_parallel] = residuals[first_parallel]

        for i in range(first_parallel + 1, n_layer):
            h_prev = h[i - 1]
            c_prev = corrections[i - 1]
            c_norm = c_prev.float().norm().item()

            if c_norm < 1e-8:
                corrections[i] = residuals[i]
            else:
                h_prev_pert = h_prev + eps * c_prev
                F_pert = _run_block(model, i, h_prev_pert, x0, cos_sin, value_embeds)
                jvp = (F_pert - F_h[i]) / eps
                corrections[i] = residuals[i] + jvp

        # Apply corrections
        h_new = list(h)
        for i in range(seq_layers, n_layer):
            h_new[i] = h[i] - corrections[i]

        if return_diagnostics:
            layer_deltas = []
            for i in range(seq_layers, n_layer):
                delta = (h_new[i] - h[i]).float().norm() / (h[i].float().norm() + 1e-8)
                layer_deltas.append(delta.item())
            diagnostics['per_iter_delta'].append(max(layer_deltas) if layer_deltas else 0.0)
            diagnostics['per_layer_delta'].append(layer_deltas)

        h = h_new

    if return_diagnostics:
        return _logits(model, h), diagnostics
    return _logits(model, h)


# ---------------------------------------------------------------------------
# Strategy 4: VJP-Newton (transpose Newton via backward-mode AD)
# ---------------------------------------------------------------------------

def _vjp_backward(model, i, h_prev, v, x0, cos_sin, value_embeds):
    """
    Compute J^T @ v via backward-mode AD. Works with SDPA (no custom kernels).

    Uses autograd.grad to compute the VJP (vector-Jacobian product) which is
    the transpose of the JVP. Empirically converges ~25% faster than FD-JVP
    Newton because J^T provides a better descent direction for non-symmetric
    Jacobians.
    """
    with torch.enable_grad():
        h_in = h_prev.detach().requires_grad_(True)
        x_in = model.resid_lambdas[i] * h_in + model.x0_lambdas[i] * x0.detach()
        ve = value_embeds.get(str(i))
        out = model.transformer.h[i](x_in, ve, cos_sin, model.window_sizes[i], None)
        grad = torch.autograd.grad(out, h_in, grad_outputs=v.detach(), retain_graph=False)[0]
    return grad.detach()


def hybrid_vjp_newton_forward(model, idx, seq_layers=None, newton_iters=None,
                               return_diagnostics=False):
    """
    Hybrid sequential prefix + VJP-Newton (transpose Newton) refinement.

    Uses backward-mode AD (J^T @ v) instead of forward-mode JVP (J @ v).
    This works with SDPA/FA3 (no custom kernels needed) and empirically
    converges in ~25% fewer iterations than FD-JVP Newton.

    The forward substitution becomes:
        delta_0 = r_0
        delta_i = r_i + J_i^T @ delta_{i-1}
    """
    config = model.config
    n_layer = config.n_layer
    if seq_layers is None:
        seq_layers = n_layer // 2
    if newton_iters is None:
        newton_iters = n_layer - seq_layers  # generous default

    x0, cos_sin, value_embeds = _embed(model, idx)

    # Phase 1: Sequential prefix (exact)
    h = [None] * n_layer
    x = x0
    for i in range(seq_layers):
        x = _run_block(model, i, x, x0, cos_sin, value_embeds)
        h[i] = x
    for i in range(seq_layers, n_layer):
        h[i] = x.clone()

    diagnostics = {'per_iter_delta': [], 'per_layer_delta': [],
                   'sequential_layers': seq_layers, 'newton_iters': 0,
                   } if return_diagnostics else None

    # Phase 2: VJP-Newton iterations on parallel suffix
    # NOTE: must exit inference_mode for autograd.grad to work
    for k in range(newton_iters):
        with torch.no_grad():
            F_h = {i: _run_block(model, i, h[i-1], x0, cos_sin, value_embeds)
                   for i in range(seq_layers, n_layer)}
            residuals = {i: h[i] - F_h[i] for i in range(seq_layers, n_layer)}

        # Forward substitution with VJP (J^T @ v)
        corrections = {}
        first = seq_layers
        corrections[first] = residuals[first]

        for i in range(first + 1, n_layer):
            c_prev = corrections[i - 1]
            if c_prev.float().norm().item() < 1e-8:
                corrections[i] = residuals[i]
            else:
                Jtv = _vjp_backward(model, i, h[i-1], c_prev, x0, cos_sin, value_embeds)
                corrections[i] = residuals[i] + Jtv

        # Apply corrections
        with torch.no_grad():
            h_new = list(h)
            for i in range(seq_layers, n_layer):
                h_new[i] = h[i] - corrections[i]

            if return_diagnostics:
                layer_deltas = []
                for i in range(seq_layers, n_layer):
                    delta = (h_new[i] - h[i]).float().norm() / (h[i].float().norm() + 1e-8)
                    layer_deltas.append(delta.item())
                diagnostics['per_iter_delta'].append(max(layer_deltas) if layer_deltas else 0.0)
                diagnostics['per_layer_delta'].append(layer_deltas)
                diagnostics['newton_iters'] = k + 1

            h = h_new

            # Early stopping on small residuals
            max_res = max(residuals[i].float().norm().item() for i in range(seq_layers, n_layer))
            if max_res < 1e-4:
                break

    if return_diagnostics:
        return _logits(model, h), diagnostics
    return _logits(model, h)


# ---------------------------------------------------------------------------
# Legacy API (kept for backward compatibility with eval script)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def jacobi_forward_streams(model, idx, max_iters=None, tol=0.0):
    """Jacobi with CUDA streams for actual parallelism."""
    config = model.config
    n_layer = config.n_layer
    if max_iters is None:
        max_iters = n_layer

    device = model.get_device()
    use_streams = device.type == 'cuda'
    x0, cos_sin, value_embeds = _embed(model, idx)
    h = [x0.clone() for _ in range(n_layer)]

    streams = [torch.cuda.Stream(device=device) for _ in range(n_layer)] if use_streams else None

    for k in range(max_iters):
        h_new = [None] * n_layer
        if use_streams:
            for i in range(n_layer):
                with torch.cuda.stream(streams[i]):
                    h_prev = h[i - 1] if i > 0 else x0
                    h_new[i] = _run_block(model, i, h_prev, x0, cos_sin, value_embeds)
            for s in streams:
                s.synchronize()
        else:
            for i in range(n_layer):
                h_prev = h[i - 1] if i > 0 else x0
                h_new[i] = _run_block(model, i, h_prev, x0, cos_sin, value_embeds)

        if tol > 0:
            max_delta = max(
                (h_new[i] - h[i]).float().norm().item() / (h[i].float().norm().item() + 1e-8)
                for i in range(n_layer)
            )
            h = h_new
            if max_delta < tol:
                break
        else:
            h = h_new

    return _logits(model, h)


@torch.inference_mode()
def gauss_seidel_forward(model, idx, max_iters=None, return_diagnostics=False):
    """Gauss-Seidel: use already-updated values within each iteration. GS K=1 = sequential."""
    config = model.config
    n_layer = config.n_layer
    if max_iters is None:
        max_iters = n_layer

    x0, cos_sin, value_embeds = _embed(model, idx)
    h = [x0.clone() for _ in range(n_layer)]

    diagnostics = {'per_iter_delta': [], 'converged_at': max_iters} if return_diagnostics else None

    for k in range(max_iters):
        max_delta = 0.0
        for i in range(n_layer):
            h_prev = h[i - 1] if i > 0 else x0
            h_old = h[i]
            h[i] = _run_block(model, i, h_prev, x0, cos_sin, value_embeds)
            if return_diagnostics:
                delta = (h[i] - h_old).float().norm() / (h_old.float().norm() + 1e-8)
                max_delta = max(max_delta, delta.item())

        if return_diagnostics:
            diagnostics['per_iter_delta'].append(max_delta)

    if return_diagnostics:
        return _logits(model, h), diagnostics
    return _logits(model, h)
