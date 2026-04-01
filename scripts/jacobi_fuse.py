"""
Fuse N parallel identity-Newton layers into one mega-block.

Instead of running 7 blocks sequentially on the same input,
concatenate their Q/K/V heads and MLP weights into a single
wider layer. One big matmul replaces 7 small sequential ones.

Results on d32 idn=0.5:
  7 blocks sequential:  7.29ms
  Fused mega-block:     3.71ms (0.51x = 1.97x faster)
  Projected total:      19.76ms (1.18x speedup vs 23.33ms sequential)
"""
import os; os.environ['NANOCHAT_BASE_DIR'] = '/workspace/home/kai/src/research/nanochat/.nanochat_cache'
import torch, time
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig, norm, apply_rotary_emb
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


def build_fused_weights(model, seq_layers):
    """Concatenate weights from parallel layers into one mega-block."""
    L = model.config.n_layer
    n_par = L - seq_layers
    D = model.config.n_embd

    W_q = torch.cat([model.transformer.h[i].attn.c_q.weight for i in range(seq_layers, L)], dim=0)
    W_k = torch.cat([model.transformer.h[i].attn.c_k.weight for i in range(seq_layers, L)], dim=0)
    W_v = torch.cat([model.transformer.h[i].attn.c_v.weight for i in range(seq_layers, L)], dim=0)
    W_o = torch.cat([model.transformer.h[i].attn.c_proj.weight for i in range(seq_layers, L)], dim=1)
    W_fc = torch.cat([model.transformer.h[i].mlp.c_fc.weight for i in range(seq_layers, L)], dim=0)
    W_proj = torch.cat([model.transformer.h[i].mlp.c_proj.weight for i in range(seq_layers, L)], dim=1)

    avg_resid = sum(model.resid_lambdas[i].item() for i in range(seq_layers, L)) / n_par
    avg_x0 = sum(model.x0_lambdas[i].item() for i in range(seq_layers, L)) / n_par

    return W_q, W_k, W_v, W_o, W_fc, W_proj, avg_resid, avg_x0


def fused_forward(h_prefix, x0, cos_sin, fused_weights, n_par, n_head, head_dim):
    """Run the fused mega-block."""
    W_q, W_k, W_v, W_o, W_fc, W_proj, avg_resid, avg_x0 = fused_weights
    B, T, D = h_prefix.shape
    total_heads = n_par * n_head

    x_in = avg_resid * h_prefix + avg_x0 * x0
    x_normed = norm(x_in)

    q = F.linear(x_normed, W_q.to(x_normed.dtype)).view(B, T, total_heads, head_dim)
    k = F.linear(x_normed, W_k.to(x_normed.dtype)).view(B, T, total_heads, head_dim)
    v = F.linear(x_normed, W_v.to(x_normed.dtype)).view(B, T, total_heads, head_dim)

    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    q, k = norm(q), norm(k)
    q = q * 1.2; k = k * 1.2

    q_t = q.transpose(1, 2); k_t = k.transpose(1, 2); v_t = v.transpose(1, 2)
    y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    y = y.transpose(1, 2).contiguous().view(B, T, -1)

    attn_out = F.linear(y, W_o.to(y.dtype))
    x_mid = x_in + attn_out

    x_mid_normed = norm(x_mid)
    h_fc = F.linear(x_mid_normed, W_fc.to(x_mid_normed.dtype))
    h_sq = F.relu(h_fc).square()
    mlp_out = F.linear(h_sq, W_proj.to(h_sq.dtype))

    return x_mid + mlp_out


def full_fused_forward(model, idx, seq_layers, fused_weights, n_par, n_head, head_dim):
    """Complete forward: sequential prefix + fused parallel suffix."""
    L = model.config.n_layer
    with torch.no_grad():
        x0, cos_sin, ve = _embed(model, idx)
        x = x0
        for i in range(seq_layers):
            x = _run_block(model, i, x, x0, cos_sin, ve)
        h_prefix = x

        fused_out = fused_forward(h_prefix, x0, cos_sin, fused_weights, n_par, n_head, head_dim)

        # Build h for _logits (needs h[-1] and h[L//2])
        h = [h_prefix] * L
        h[-1] = fused_out
        return _logits(model, h)


if __name__ == '__main__':
    model = load_mdl('.nanochat_cache/base_checkpoints/d32_idn05')
    L = model.config.n_layer
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head

    idx = torch.randint(0, 65536, (1, 256), device=device)

    def bench(fn, n_warm=10, n_run=50):
        for _ in range(n_warm): fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(n_run):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        return sum(times) / len(times) * 1000

    seq_ms = bench(lambda: model.forward(idx))
    print(f'Sequential: {seq_ms:.2f}ms')

    for seq in [29, 27, 25, 23]:
        n_par = L - seq
        fw = build_fused_weights(model, seq)
        fused_ms = bench(lambda s=seq, f=fw, n=n_par: full_fused_forward(model, idx, s, f, n, n_head, head_dim))
        vs = seq_ms / fused_ms

        with torch.no_grad():
            ref = model.forward(idx)
            fused_logits = full_fused_forward(model, idx, seq, fw, n_par, n_head, head_dim)
            cos_sim = F.cosine_similarity(ref.float().reshape(-1), fused_logits.float().reshape(-1), dim=0).item()
            top1 = (ref.argmax(-1) == fused_logits.argmax(-1)).float().mean().item()

        print(f'  seq={seq} par={n_par}: fused={fused_ms:.2f}ms ({vs:.2f}x) cos_sim={cos_sim:.4f} top1={top1:.3f}')
