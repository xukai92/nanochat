"""Focused benchmark: Hybrid Jacobi with ΔPPL on real data, with and without CUDA streams."""
import faulthandler; faulthandler.enable()
import os; os.environ['NANOCHAT_BASE_DIR'] = '/workspace/home/kai/src/research/nanochat/.nanochat_cache'
import sys; sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)  # line-buffered
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--sdpa', action='store_true', help='Force SDPA backend (no FA3)')
_ap.add_argument('--checkpoint', type=str, default='checkpoints/d20', help='Checkpoint dir')
_ap.add_argument('--seq-len', type=int, default=256, help='Sequence length')
_ap.add_argument('--batch-size', type=int, default=1, help='Batch size')
_args = _ap.parse_args()
if _args.sdpa:
    # Must patch BEFORE importing nanochat modules
    import nanochat.flash_attention as _fa_mod
    _fa_mod._override_impl = 'sdpa'
    _fa_mod.USE_FA3 = False
    print('Forced SDPA backend (no FA3)')
import torch, time, math
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import load_checkpoint, _patch_missing_config_keys, _patch_missing_keys, find_last_step
from nanochat.jacobi_forward import hybrid_forward, hybrid_forward_streams, hybrid_forward_multigpu

device = torch.device('cuda')
step = find_last_step(_args.checkpoint)
model_data, _, meta = load_checkpoint(_args.checkpoint, step, device)
model_data = {k.removeprefix('_orig_mod.'): v for k, v in model_data.items()}
cfg_kw = meta['model_config']
_patch_missing_config_keys(cfg_kw)
config = GPTConfig(**cfg_kw)
_patch_missing_keys(model_data, config)
model_data = {k: v.to(device) for k, v in model_data.items()}
with torch.device('meta'):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
model.load_state_dict(model_data, strict=True, assign=True)
model.eval()
print(f'Model: {config.n_layer} layers, {config.n_embd} dim')

# Load evaluation data
from nanochat.tokenizer import get_tokenizer
tokenizer = get_tokenizer()
seq_len = _args.seq_len

# Try SFT data first (SmolTalk test), fall back to raw text
try:
    from tasks.smoltalk import SmolTalk
    dataset = SmolTalk(split="test")
    all_tokens = []
    needed = (seq_len + 1) * bs * 16
    for i in range(dataset.num_examples()):
        conv = dataset.get_example(i)
        ids, mask = tokenizer.render_conversation(conv, max_tokens=seq_len * 2)
        all_tokens.extend(ids)
        if len(all_tokens) >= needed:
            break
    data_source = "SmolTalk test (SFT)"
except Exception as e:
    print(f'SmolTalk unavailable ({e}), using ClimbMix val')
    import pyarrow.parquet as pq
    bos = tokenizer.get_bos_token_id()
    table = pq.read_table('.nanochat_cache/base_data_climbmix/shard_06542.parquet', columns=['text'])
    texts = table.column('text').to_pylist()
    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenizer.encode(t, prepend=bos))
        if len(all_tokens) > 50000:
            break
    data_source = "ClimbMix val (base)"

bs = _args.batch_size
batches = []
offset = 0
for _ in range(8):
    batch_seqs = []
    for _ in range(bs):
        if offset + seq_len + 1 > len(all_tokens):
            break
        batch_seqs.append(all_tokens[offset:offset + seq_len + 1])
        offset += seq_len
    if len(batch_seqs) < bs:
        break
    t = torch.tensor(batch_seqs, dtype=torch.long, device=device)
    batches.append((t[:, :-1], t[:, 1:]))
print(f'Data: {len(batches)} batches of seq_len={seq_len} from {data_source}')


@torch.inference_mode()
def eval_config(fn, n_warm=5):
    for _ in range(n_warm):
        fn(batches[0][0])
    torch.cuda.synchronize()
    ppls, top1s, times = [], [], []
    for idx, tgt in batches:
        ref = model.forward(idx)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = fn(idx)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item()
        ppls.append(math.exp(min(loss, 20.0)))
        top1s.append((logits.argmax(-1) == ref.argmax(-1)).float().mean().item())
    return sum(ppls) / len(ppls), sum(top1s) / len(top1s), sum(times) / len(times) * 1000


# Baseline
ref_ppl, _, ref_ms = eval_config(lambda x: model.forward(x))
print(f'\nBaseline: PPL={ref_ppl:.2f}, time={ref_ms:.2f}ms\n')

header = f'{"Config":<35s} | {"PPL":>7s} | {"ΔPPL":>7s} | {"ΔPPL%":>6s} | {"Top-1":>6s} | {"Time":>7s} | {"Speed":>5s}'
sep = '-' * 35 + '-+-' + '-' * 7 + '-+-' + '-' * 7 + '-+-' + '-' * 6 + '-+-' + '-' * 6 + '-+-' + '-' * 7 + '-+-' + '-' * 5

L = config.n_layer
# Generate configs dynamically based on model depth
configs = []
for seq_frac in [0.9, 0.8, 0.7, 0.6, 0.5]:
    seq = int(L * seq_frac)
    n_par = L - seq
    for p in [1, 2, 3, 5, n_par]:
        if p > n_par:
            continue
        exact = " (exact)" if p == n_par else ""
        configs.append((f'seq={seq} par={p}{exact}', seq, p))
    configs.append(None)  # separator


def run_table(title, forward_fn):
    print(f'\n--- {title} ---')
    print(header)
    print(sep)
    for cfg in configs:
        if cfg is None:
            continue
        name, seq, par = cfg
        ppl, top1, ms = eval_config(lambda x, s=seq, p=par: forward_fn(model, x, seq_layers=s, parallel_iters=p))
        dp = ppl - ref_ppl
        dpc = 100 * dp / ref_ppl
        spd = ref_ms / ms
        ps = f'{ppl:.2f}' if ppl < 1e4 else f'{ppl:.0f}'
        print(f'{name:<35s} | {ps:>7s} | {dp:>+7.2f} | {dpc:>+5.1f}% | {top1:>6.3f} | {ms:>6.1f}ms | {spd:>4.2f}x')


run_table('Hybrid Jacobi (sequential, single GPU)', hybrid_forward)

if _args.sdpa:
    run_table('Hybrid Jacobi + CUDA Streams (single GPU)', hybrid_forward_streams)

# Multi-GPU
n_gpus = min(4, torch.cuda.device_count())
if n_gpus >= 2:
    for ng in [2, 4]:
        if ng > n_gpus:
            continue
        gpu_ids = list(range(ng))
        print(f'\n--- Hybrid Jacobi + Multi-GPU ({ng} GPUs: {gpu_ids}) ---')
        print(f'    NOTE: includes one-time weight copy overhead in first run')
        print(header)
        print(sep)
        for cfg in configs:
            if cfg is None:
                continue
            name, seq, par = cfg
            ppl, top1, ms = eval_config(
                lambda x, s=seq, p=par, g=gpu_ids: hybrid_forward_multigpu(
                    model, x, seq_layers=s, parallel_iters=p, gpu_ids=g))
            dp = ppl - ref_ppl
            dpc = 100 * dp / ref_ppl
            spd = ref_ms / ms
            ps = f'{ppl:.2f}' if ppl < 1e4 else f'{ppl:.0f}'
            print(f'{name:<35s} | {ps:>7s} | {dp:>+7.2f} | {dpc:>+5.1f}% | {top1:>6.3f} | {ms:>6.1f}ms | {spd:>4.2f}x')
