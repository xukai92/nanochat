"""
Advanced identity Newton experiments:
  Dir 1: Push more parallel layers (warm-starting from previous token)
  Dir 3: Adaptive K per token (K=1 for easy, K=2+ for hard)
  Dir 4: Speculative decoding with identity Newton as draft
"""
import os; os.environ['NANOCHAT_BASE_DIR'] = '/workspace/home/kai/src/research/nanochat/.nanochat_cache'
import faulthandler; faulthandler.enable()
import sys; sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

import torch, time, math
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys, find_last_step
from nanochat.jacobi_forward import _embed, _run_block, _logits
import nanochat.flash_attention as fa; fa._override_impl = 'sdpa'; fa.USE_FA3 = False

device = torch.device('cuda')


def load_mdl(path):
    step = find_last_step(path)
    md, _, meta = load_checkpoint(path, step, device)
    md = {k.removeprefix('_orig_mod.'): v for k, v in md.items()}
    cfg = meta['model_config']; _patch_missing_config_keys(cfg)
    config = GPTConfig(**cfg); _patch_missing_keys(md, config)
    md = {k: v.to(device) for k, v in md.items()}
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device); model.init_weights()
    model.load_state_dict(md, strict=True, assign=True); model.eval()
    return model


# Load eval data
from nanochat.tokenizer import get_tokenizer
import pyarrow.parquet as pq
tokenizer = get_tokenizer(); bos = tokenizer.get_bos_token_id()
table = pq.read_table('.nanochat_cache/base_data_climbmix/shard_06542.parquet', columns=['text'])
all_tokens = []
for t in table.column('text').to_pylist():
    all_tokens.extend(tokenizer.encode(t, prepend=bos))
    if len(all_tokens) > 50000:
        break

SL = 256


def identity_newton(model, idx, seq_layers, max_iters, h_init_override=None):
    """Identity Newton with optional warm-start from previous call's h."""
    L = model.config.n_layer
    with torch.no_grad():
        x0, cos_sin, ve = _embed(model, idx)
        h = [None] * L
        x = x0
        for i in range(seq_layers):
            x = _run_block(model, i, x, x0, cos_sin, ve)
            h[i] = x

        par = list(range(seq_layers, L))

        if h_init_override is not None:
            # Warm-start: use previous token's h as initial guess
            for i in par:
                h[i] = h_init_override[i]
        else:
            for i in par:
                h[i] = x.clone()

        for k in range(max_iters):
            F_h = {i: _run_block(model, i, h[i - 1], x0, cos_sin, ve) for i in par}
            res = {i: h[i] - F_h[i] for i in par}
            delta = {par[0]: res[par[0]]}
            for i in par[1:]:
                delta[i] = res[i] + delta[i - 1]
            for i in par:
                h[i] = h[i] - delta[i]

        return _logits(model, h), h  # return h for warm-starting next call


def adaptive_identity_newton(model, idx, seq_layers, max_K=4, threshold=0.1):
    """
    Dir 3: Adaptive K — run K=1 first, check residual norm,
    add more iterations only if residual exceeds threshold.
    """
    L = model.config.n_layer
    with torch.no_grad():
        x0, cos_sin, ve = _embed(model, idx)
        h = [None] * L
        x = x0
        for i in range(seq_layers):
            x = _run_block(model, i, x, x0, cos_sin, ve)
            h[i] = x
        par = list(range(seq_layers, L))
        for i in par:
            h[i] = x.clone()

        total_iters = 0
        for k in range(max_K):
            F_h = {i: _run_block(model, i, h[i - 1], x0, cos_sin, ve) for i in par}
            res = {i: h[i] - F_h[i] for i in par}

            # Check if we can stop early
            max_res = max(res[i].float().norm().item() / (h[i].float().norm().item() + 1e-8) for i in par)
            total_iters = k + 1

            delta = {par[0]: res[par[0]]}
            for i in par[1:]:
                delta[i] = res[i] + delta[i - 1]
            for i in par:
                h[i] = h[i] - delta[i]

            if max_res < threshold:
                break

        return _logits(model, h), total_iters


def speculative_identity_newton(model, idx, seq_layers, draft_K=1):
    """
    Dir 4: Speculative decoding with identity Newton as draft.

    1. Run identity Newton K=draft_K (fast, approximate)
    2. Run sequential forward (exact) to verify
    3. Compare: if draft matches, we save time in autoregressive generation
       because we can verify N tokens at once

    Returns: draft logits, exact logits, match rate
    """
    L = model.config.n_layer
    with torch.no_grad():
        x0, cos_sin, ve = _embed(model, idx)

        # Draft: identity Newton
        h_draft = [None] * L
        x = x0
        for i in range(seq_layers):
            x = _run_block(model, i, x, x0, cos_sin, ve)
            h_draft[i] = x
        par = list(range(seq_layers, L))
        for i in par:
            h_draft[i] = x.clone()

        for k in range(draft_K):
            F_h = {i: _run_block(model, i, h_draft[i - 1], x0, cos_sin, ve) for i in par}
            res = {i: h_draft[i] - F_h[i] for i in par}
            delta = {par[0]: res[par[0]]}
            for i in par[1:]:
                delta[i] = res[i] + delta[i - 1]
            for i in par:
                h_draft[i] = h_draft[i] - delta[i]

        draft_logits = _logits(model, h_draft)

        # Verify: sequential
        exact_logits = model.forward(idx)

        # Match rate
        draft_top1 = draft_logits.argmax(-1)
        exact_top1 = exact_logits.argmax(-1)
        match_rate = (draft_top1 == exact_top1).float().mean().item()

        return draft_logits, exact_logits, match_rate


# =====================================================================
# Benchmarks
# =====================================================================

model = load_mdl('.nanochat_cache/base_checkpoints/d20_base2k')
L = model.config.n_layer

batches = []
offset = 0
for _ in range(8):
    if offset + SL + 1 > len(all_tokens):
        break
    t = torch.tensor([all_tokens[offset:offset + SL + 1]], dtype=torch.long, device=device)
    batches.append((t[:, :-1], t[:, 1:]))
    offset += SL


def bench(fn, n_warm=5, n_run=20):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_run):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times) * 1000


seq_ms = bench(lambda: model.forward(batches[0][0]))
print(f'Model: {L} layers | Sequential: {seq_ms:.1f}ms')
print()

# =====================================================================
# Dir 1: Push more parallel layers with warm-starting
# =====================================================================
print('=' * 80)
print('  Dir 1: Warm-starting from previous token\'s hidden states')
print('=' * 80)
print()
print('Idea: in autoregressive generation, use the previous token\'s h as')
print('initial guess for the next token. The hidden states are similar between')
print('adjacent tokens, so this should reduce Newton iterations needed.')
print()

# Simulate sequential generation with warm-starting
with torch.no_grad():
    for seq in [17, 15, 13]:
        n_par = L - seq
        # Generate 8 tokens, measuring per-token identity Newton cost
        idx_gen = batches[0][0][:, :10]  # start with 10 tokens

        # Cold start (no warm-start)
        cold_times = []
        for tok in range(8):
            idx_t = idx_gen[:, :10 + tok]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            logits, h_out = identity_newton(model, idx_t, seq, max_iters=2)
            torch.cuda.synchronize()
            cold_times.append((time.perf_counter() - t0) * 1000)

        # Warm start (reuse previous h)
        warm_times = []
        h_prev = None
        for tok in range(8):
            idx_t = idx_gen[:, :10 + tok]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            logits, h_out = identity_newton(model, idx_t, seq, max_iters=2, h_init_override=h_prev)
            torch.cuda.synchronize()
            warm_times.append((time.perf_counter() - t0) * 1000)
            h_prev = h_out

        cold_avg = sum(cold_times) / len(cold_times)
        warm_avg = sum(warm_times) / len(warm_times)
        print(f'  seq={seq} par={n_par}: cold={cold_avg:.1f}ms warm={warm_avg:.1f}ms ({cold_avg / warm_avg:.2f}x)')

print()

# =====================================================================
# Dir 3: Adaptive K per token
# =====================================================================
print('=' * 80)
print('  Dir 3: Adaptive K (stop early when residual is small)')
print('=' * 80)
print()

with torch.no_grad():
    for seq in [17, 15, 13, 11]:
        n_par = L - seq
        total_iters = 0
        total_tokens = 0
        total_time = 0

        for idx, tgt in batches:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            logits, iters = adaptive_identity_newton(model, idx, seq, max_K=6, threshold=0.05)
            torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            total_iters += iters
            total_tokens += 1

        avg_K = total_iters / total_tokens
        avg_ms = total_time / total_tokens * 1000
        vs = seq_ms / avg_ms

        # Quality
        ref = model.forward(batches[0][0])
        logits_check, _ = adaptive_identity_newton(model, batches[0][0], seq, max_K=6, threshold=0.05)
        top1 = (logits_check.argmax(-1) == ref.argmax(-1)).float().mean().item()

        print(f'  seq={seq} par={n_par}: avg_K={avg_K:.1f} time={avg_ms:.1f}ms ({vs:.2f}x) top1={top1:.3f}')

print()

# =====================================================================
# Dir 4: Speculative decoding analysis
# =====================================================================
print('=' * 80)
print('  Dir 4: Speculative decoding (identity Newton as draft)')
print('=' * 80)
print()
print('If draft top-1 matches sequential, we accept the token without')
print('running the full sequential forward. Match rate determines speedup.')
print()

with torch.no_grad():
    for seq in [19, 17, 15, 13]:
        n_par = L - seq
        match_rates = []
        for idx, tgt in batches:
            _, _, mr = speculative_identity_newton(model, idx, seq, draft_K=1)
            match_rates.append(mr)
        avg_match = sum(match_rates) / len(match_rates)

        # Speculative speedup model:
        # If match_rate = p, we save (1-p) * verify_cost but pay draft_cost always
        draft_ms = bench(lambda: identity_newton(model, batches[0][0], seq, 1))

        # Amortized: we run draft always, verify only on mismatch
        # Expected time per token = draft_ms + (1-p) * seq_ms
        expected_ms = draft_ms + (1 - avg_match) * seq_ms
        vs = seq_ms / expected_ms

        print(f'  seq={seq} par={n_par} K=1: match={avg_match:.3f} draft={draft_ms:.1f}ms '
              f'expected={expected_ms:.1f}ms ({vs:.2f}x)')

        # With K=2
        match_rates_k2 = []
        for idx, tgt in batches:
            _, _, mr = speculative_identity_newton(model, idx, seq, draft_K=2)
            match_rates_k2.append(mr)
        avg_match_k2 = sum(match_rates_k2) / len(match_rates_k2)
        draft_ms_k2 = bench(lambda: identity_newton(model, batches[0][0], seq, 2))
        expected_k2 = draft_ms_k2 + (1 - avg_match_k2) * seq_ms
        vs_k2 = seq_ms / expected_k2

        print(f'  seq={seq} par={n_par} K=2: match={avg_match_k2:.3f} draft={draft_ms_k2:.1f}ms '
              f'expected={expected_k2:.1f}ms ({vs_k2:.2f}x)')
    print()
