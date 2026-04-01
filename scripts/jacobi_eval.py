"""
Evaluate Jacobi iteration over transformer layers vs sequential forward pass.

Measures:
1. Convergence: how many iterations until hidden states stabilize
2. Quality: logit agreement, top-1 accuracy, cosine similarity vs sequential
3. Speed: wall-clock time comparison (sequential vs Jacobi with CUDA streams)

Usage:
    # Basic convergence analysis on a trained model
    python -m scripts.jacobi_eval --model-tag d12

    # With a specific checkpoint step
    python -m scripts.jacobi_eval --model-tag d12 --step 1000

    # Quick test with random weights (no checkpoint needed)
    python -m scripts.jacobi_eval --random --depth 12

    # Speed benchmark with CUDA streams
    python -m scripts.jacobi_eval --model-tag d12 --benchmark
"""

import argparse
import math
import time
import torch
import torch.nn.functional as F

from nanochat.common import compute_init, autodetect_device_type, COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig
from nanochat.jacobi_forward import (
    jacobi_forward, gauss_seidel_forward, jacobi_forward_streams,
    jacobi_forward_layerskip_init, jacobi_forward_sequential_init,
    hybrid_forward, hybrid_forward_streams, newton_forward, hybrid_newton_forward,
    calibrate_preheat, preheat_forward, preheat_hybrid_newton_forward,
)


def load_or_create_model(args, device):
    """Load a trained model or create one with random weights for testing."""
    if args.random:
        print(f"Creating random model with depth={args.depth}")
        config = GPTConfig(n_layer=args.depth)
        # Compute n_embd from depth like the training script does
        aspect_ratio = 64
        head_dim = 128
        raw_dim = args.depth * aspect_ratio
        n_head = max(1, round(raw_dim / head_dim))
        config.n_embd = n_head * head_dim
        config.n_head = n_head
        config.n_kv_head = n_head
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        model.init_weights()
        # The default init zeros out c_proj and mlp.c_proj, making blocks identity.
        # Randomize projections to simulate a trained model with non-trivial layer interactions.
        rng = torch.Generator(device='cpu')
        rng.manual_seed(123)
        n_embd = config.n_embd
        # Scale inversely with dim to produce non-trivial but stable activations
        s = 1.0 / (n_embd ** 0.5)
        for block in model.transformer.h:
            block.attn.c_proj.weight.data = torch.randn(n_embd, n_embd, generator=rng) * s
            block.mlp.c_proj.weight.data = torch.randn(n_embd, 4 * n_embd, generator=rng) * s
        model.eval()
        return model
    elif args.checkpoint_dir:
        from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys, find_last_step
        step = args.step if args.step else find_last_step(args.checkpoint_dir)
        model_data, _, meta_data = load_checkpoint(args.checkpoint_dir, step, device)
        if device.type in {"cpu", "mps"}:
            model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
        model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
        model_config_kwargs = meta_data["model_config"]
        _patch_missing_config_keys(model_config_kwargs)
        config = GPTConfig(**model_config_kwargs)
        _patch_missing_keys(model_data, config)
        # Move all tensors to target device
        model_data = {k: v.to(device) for k, v in model_data.items()}
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        model.init_weights()
        model.load_state_dict(model_data, strict=True, assign=True)
        model.eval()
        return model
    else:
        from nanochat.checkpoint_manager import load_model
        model, tokenizer, meta = load_model(
            args.source, device, phase="eval",
            model_tag=args.model_tag, step=args.step
        )
        return model


def create_test_input(model, device, seq_len=128, batch_size=1, n_batches=4):
    """
    Load real text from ClimbMix validation shard, tokenize, and create
    proper (input, target) batches for perplexity evaluation.

    Falls back to random tokens if dataset is unavailable.
    """
    import os

    base_dir = os.environ.get('NANOCHAT_BASE_DIR', os.path.expanduser('~/.cache/nanochat'))
    data_dir = os.path.join(base_dir, 'base_data_climbmix')

    if os.path.isdir(data_dir):
        try:
            import pyarrow.parquet as pq
            from nanochat.tokenizer import get_tokenizer

            parquet_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.parquet'))
            val_path = os.path.join(data_dir, parquet_files[-1])
            print(f"  Loading validation data from {val_path}")

            tokenizer = get_tokenizer()
            bos = tokenizer.get_bos_token_id()

            table = pq.read_table(val_path, columns=['text'])
            texts = table.column('text').to_pylist()

            # Tokenize enough text
            all_tokens = []
            needed = (seq_len + 1) * batch_size * n_batches * 2
            for text in texts:
                tokens = tokenizer.encode(text, prepend=bos)
                all_tokens.extend(tokens)
                if len(all_tokens) >= needed:
                    break

            # Pack into batches
            batches = []
            offset = 0
            for _ in range(n_batches):
                batch_seqs = []
                for _ in range(batch_size):
                    seq = all_tokens[offset:offset + seq_len + 1]
                    if len(seq) < seq_len + 1:
                        break
                    batch_seqs.append(seq)
                    offset += seq_len  # stride by seq_len (slight overlap of 1 for targets)
                if len(batch_seqs) < batch_size:
                    break
                tokens_tensor = torch.tensor(batch_seqs, dtype=torch.long, device=device)
                batches.append((tokens_tensor[:, :-1], tokens_tensor[:, 1:]))

            if len(batches) >= n_batches:
                print(f"  Created {len(batches)} batches from ClimbMix val split")
                return batches
            print(f"  Warning: only got {len(batches)} batches, falling back to random")
        except Exception as e:
            print(f"  Dataset loading failed ({e}), falling back to random tokens")

    # Fallback: random tokens
    print(f"  Using random tokens (dataset unavailable)")
    vocab_size = model.config.vocab_size
    rng = torch.Generator(device='cpu')
    rng.manual_seed(42)
    batches = []
    for _ in range(n_batches):
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=rng, device=device)
        batches.append((tokens[:, :-1], tokens[:, 1:]))
    return batches


def compute_loss(logits, targets):
    """Compute cross-entropy loss and perplexity."""
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction='mean'
    ).item()
    ppl = math.exp(min(loss, 20.0))  # cap to avoid overflow
    return loss, ppl


def compute_metrics(logits_ref, logits_test, targets):
    """Compute quality metrics: loss, perplexity, top-1 agreement, cosine similarity."""
    # Loss and perplexity from the test logits against ground truth targets
    loss, ppl = compute_loss(logits_test, targets)

    # Flatten to (B*T, vocab)
    ref = logits_ref.reshape(-1, logits_ref.size(-1))
    test = logits_test.reshape(-1, logits_test.size(-1))

    # Top-1 agreement vs sequential reference
    top1_ref = ref.argmax(dim=-1)
    top1_test = test.argmax(dim=-1)
    top1_agree = (top1_ref == top1_test).float().mean().item()

    # Cosine similarity (per position, averaged)
    cos_sim_raw = F.cosine_similarity(ref, test, dim=-1)
    cos_sim = cos_sim_raw[~cos_sim_raw.isnan()].mean().item() if not cos_sim_raw.isnan().all() else 0.0

    return {
        'loss': loss,
        'ppl': ppl,
        'top1_agree': top1_agree,
        'cos_sim': cos_sim,
    }


_REF_PPL = None  # set by run_convergence_analysis

def _print_header(title):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")

def _print_table_header():
    print(f"  {'Method':<36s} | {'PPL':>7} | {'ΔPPL':>7} | {'ΔPPL%':>6} | {'Top-1':>6} | {'Time':>8} | {'Speed':>6}")
    print(f"  {'-'*36}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}")

def _print_row(name, metrics, seq_layers=0, elapsed_ms=0.0):
    global _REF_PPL
    ppl = metrics['ppl']
    dppl = ppl - _REF_PPL if _REF_PPL else 0
    dppl_pct = 100 * dppl / _REF_PPL if _REF_PPL else 0
    speed = _REF_MS / elapsed_ms if elapsed_ms > 0 else 0
    ppl_s = f"{ppl:7.2f}" if ppl < 1e4 else f"{ppl:7.0f}"
    dppl_s = f"{dppl:+7.2f}" if abs(dppl) < 1e4 else f"{dppl:+7.0f}"
    print(f"  {name:<36s} | {ppl_s} | {dppl_s} | {dppl_pct:+5.1f}% | {metrics['top1_agree']:6.3f} | {elapsed_ms:7.1f}ms | {speed:5.2f}x")


def _timed(fn, device):
    """Run fn(), return (result, elapsed_ms)."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


def _eval_strategy(fn, model, eval_batches, device):
    """
    Evaluate a strategy across multiple batches, averaging metrics.
    fn(model, idx) -> logits
    Returns averaged metrics dict and mean time.
    """
    all_losses, all_ppls, all_top1s, all_cossims = [], [], [], []
    all_times = []

    for idx, targets in eval_batches:
        # Reference logits for this batch
        logits_ref = model.forward(idx)

        # Run the strategy
        (logits,), ms = _timed(lambda: (fn(model, idx),), device)
        m = compute_metrics(logits_ref, logits, targets)

        all_losses.append(m['loss'])
        all_ppls.append(m['ppl'])
        all_top1s.append(m['top1_agree'])
        all_cossims.append(m['cos_sim'])
        all_times.append(ms)

    return {
        'loss': sum(all_losses) / len(all_losses),
        'ppl': sum(all_ppls) / len(all_ppls),
        'top1_agree': sum(all_top1s) / len(all_top1s),
        'cos_sim': sum(all_cossims) / len(all_cossims),
    }, sum(all_times) / len(all_times)


@torch.inference_mode()
def run_convergence_analysis(model, eval_batches, device):
    """Run all strategies and compare quality vs sequential baseline."""
    n_layer = model.config.n_layer
    idx, targets = eval_batches[0]

    # Reference: sequential forward pass (averaged over all batches)
    global _REF_PPL, _REF_MS
    ref_metrics, ref_ms = _eval_strategy(lambda m, x: m.forward(x), model, eval_batches, device)
    _REF_PPL = ref_metrics['ppl']
    _REF_MS = ref_ms

    _print_header(f"Strategy Comparison: depth={n_layer}, seq_len={idx.size(1)}, batch={idx.size(0)}, "
                  f"n_batches={len(eval_batches)}")
    print(f"  Sequential baseline: {ref_ms:.1f}ms, loss={ref_metrics['loss']:.4f}, ppl={ref_metrics['ppl']:.2f}")
    print(f"  Averaged over {len(eval_batches)} batches")

    # Helper: evaluate a strategy fn(model, idx)->logits across all eval batches
    def ev(name, fn, seq_layers=0):
        m, ms = _eval_strategy(fn, model, eval_batches, device)
        _print_row(name, m, seq_layers=seq_layers, elapsed_ms=ms)

    # -----------------------------------------------------------------------
    # Full sweep: vary seq_layers from 0 to n_layer, for each method
    # This shows the real Pareto frontier of quality vs sequential cost
    # -----------------------------------------------------------------------
    seq_points = sorted(set([0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
                             n_layer // 4, n_layer // 2, 3 * n_layer // 4,
                             n_layer]))
    seq_points = [s for s in seq_points if 0 <= s <= n_layer]

    # --- Hybrid Jacobi (no streams): sweep seq ---
    _print_header("Hybrid Jacobi (sequential execution of parallel layers)")
    _print_table_header()
    for seq in seq_points:
        n_par = n_layer - seq
        if n_par == 0:
            continue
        for p in [1, 2, 3, 5, n_par]:
            if p > n_par:
                continue
            ev(f"Hybrid(seq={seq}, par={p})",
               lambda m, x, s=seq, p=p: hybrid_forward(m, x, seq_layers=s, parallel_iters=p),
               seq_layers=seq)

    # --- Hybrid Jacobi WITH CUDA streams ---
    if device.type == 'cuda':
        _print_header("Hybrid Jacobi + CUDA Streams (parallel layers overlap)")
        _print_table_header()
        for seq in seq_points:
            n_par = n_layer - seq
            if n_par == 0:
                continue
            for p in [1, 2, 3, 5, n_par]:
                if p > n_par:
                    continue
                ev(f"Streams(seq={seq}, par={p})",
                   lambda m, x, s=seq, p=p: hybrid_forward_streams(m, x, seq_layers=s, parallel_iters=p),
                   seq_layers=seq)

    # -----------------------------------------------------------------------
    # Verification (on first batch only)
    # -----------------------------------------------------------------------
    idx, targets = eval_batches[0]
    logits_ref = model.forward(idx)
    logits_gs1 = gauss_seidel_forward(model, idx, max_iters=1)
    gs1_max_diff = (logits_gs1 - logits_ref).abs().max().item()
    logits_jL = jacobi_forward(model, idx, max_iters=n_layer)
    jL_max_diff = (logits_jL - logits_ref).abs().max().item()
    print(f"\n--- Verification ---")
    print(f"  GS K=1 vs Sequential max logit diff: {gs1_max_diff:.2e} (should be ~0)")
    print(f"  Jacobi K={n_layer} vs Sequential max logit diff: {jL_max_diff:.2e} (should be ~0)")


@torch.inference_mode()
def run_speed_benchmark(model, idx, device, n_warmup=5, n_runs=20):
    """Benchmark sequential vs Jacobi forward pass speed."""
    n_layer = model.config.n_layer

    print(f"\n{'='*70}")
    print(f"Speed Benchmark: depth={n_layer}, seq_len={idx.size(1)}, batch={idx.size(0)}")
    print(f"{'='*70}")

    def bench(fn, name):
        # Warmup
        for _ in range(n_warmup):
            fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
        print(f"  {name:40s}: {avg*1000:8.2f} ms +/- {std*1000:6.2f} ms")
        return avg

    # Sequential baseline
    t_seq = bench(lambda: model.forward(idx), "Sequential forward")

    # Jacobi with different iteration counts
    for K in [1, 2, 3, n_layer // 2, n_layer]:
        if K > n_layer:
            continue
        t_j = bench(lambda K=K: jacobi_forward(model, idx, max_iters=K), f"Jacobi K={K}")

    # Jacobi with CUDA streams (only on CUDA)
    if device.type == 'cuda':
        for K in [1, 2, 3, n_layer // 2, n_layer]:
            if K > n_layer:
                continue
            t_js = bench(lambda K=K: jacobi_forward_streams(model, idx, max_iters=K), f"Jacobi+Streams K={K}")

    # Gauss-Seidel for comparison
    for K in [1, 2, 3]:
        t_gs = bench(lambda K=K: gauss_seidel_forward(model, idx, max_iters=K), f"Gauss-Seidel K={K}")


@torch.inference_mode()
def run_generation_comparison(model, idx, device):
    """Compare greedy generation quality between sequential and Jacobi."""
    n_layer = model.config.n_layer

    print(f"\n{'='*70}")
    print(f"Greedy Generation Comparison (first 64 tokens)")
    print(f"{'='*70}")

    # Get sequential logits for first position
    logits_ref = model.forward(idx)
    ref_tokens = logits_ref[:, -1, :].argmax(dim=-1)

    print(f"\n  Sequential top-1 token: {ref_tokens[0].item()}")

    for K in [1, 2, 3, n_layer // 2, n_layer]:
        if K > n_layer:
            continue
        logits_j = jacobi_forward(model, idx, max_iters=K)
        j_tokens = logits_j[:, -1, :].argmax(dim=-1)
        match = (ref_tokens == j_tokens).all().item()
        print(f"  Jacobi K={K:2d} top-1 token: {j_tokens[0].item()} {'OK' if match else 'MISMATCH'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Jacobi iteration over transformer layers')
    parser.add_argument('-i', '--source', type=str, default='base', help='Model source: base|sft|rl')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag (e.g. d12)')
    parser.add_argument('-s', '--step', type=int, default=None, help='Checkpoint step')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Direct path to checkpoint dir (overrides source/model-tag)')
    parser.add_argument('--random', action='store_true', help='Use random weights (no checkpoint needed)')
    parser.add_argument('--depth', type=int, default=12, help='Model depth (only with --random)')
    parser.add_argument('--seq-len', type=int, default=128, help='Sequence length for test input')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for test input')
    parser.add_argument('--benchmark', action='store_true', help='Run speed benchmark')
    parser.add_argument('--device-type', type=str, default='', help='Device: cuda|cpu|mps')
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, rank, local_rank, world_size, device = compute_init(device_type)

    model = load_or_create_model(args, device)

    print(f"Model: {model.config.n_layer} layers, {model.config.n_embd} dim, {model.config.n_head} heads")
    print(f"Device: {device}, dtype: {COMPUTE_DTYPE}")

    n_eval_batches = 8
    print(f"Loading eval data (batch_size={args.batch_size}, seq_len={args.seq_len})...")
    eval_batches = create_test_input(model, device, seq_len=args.seq_len,
                                      batch_size=args.batch_size, n_batches=n_eval_batches)
    idx, targets = eval_batches[0]

    # Run convergence analysis with all eval batches
    run_convergence_analysis(model, eval_batches, device)

    # Generation comparison
    run_generation_comparison(model, idx, device)

    # Speed benchmark (optional)
    if args.benchmark:
        run_speed_benchmark(model, idx, device)


if __name__ == "__main__":
    main()
