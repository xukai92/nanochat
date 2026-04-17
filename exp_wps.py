"""Weight-pool sampling on a tiny GPT transformer.

Standalone experiment: trains a small GPT (4 layers, d=128, 4 heads) on
Shakespeare with two modes: baseline (standard nn.Linear) and ours
(per-group SIREN + pool generating all linear weights).

Tests whether the compression recipe from ResNet transfers to transformers
without architectural changes.
"""

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the SIREN + pool machinery from weight-pool-sampling
sys.path.insert(0, str(Path(__file__).parent.parent / "weight-pool-sampling" / "src"))
from wps.exp2_1_weight_field import SIRENLayer
from wps.exp2_3d_cnn_continuous import SirenWeightField


# ---- Tiny GPT ---------------------------------------------------------------


@dataclass
class TinyGPTConfig:
    vocab_size: int = 256  # byte-level
    seq_len: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, linear_fn):
        super().__init__()
        d = config.n_embd
        self.n_head = config.n_head
        self.head_dim = d // config.n_head
        self.c_qkv = linear_fn(f"layer{layer_idx}_attn_qkv", d, 3 * d)
        self.c_proj = linear_fn(f"layer{layer_idx}_attn_proj", d, d)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config, layer_idx, linear_fn):
        super().__init__()
        d = config.n_embd
        self.c_fc = linear_fn(f"layer{layer_idx}_mlp_fc", d, 4 * d)
        self.c_proj = linear_fn(f"layer{layer_idx}_mlp_proj", 4 * d, d)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config, layer_idx, linear_fn):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx, linear_fn)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config, layer_idx, linear_fn)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, config, kernel_source=None):
        super().__init__()
        self.config = config

        # Linear factory: standard or weight-field-backed
        if kernel_source is not None:
            self.kernel_source = kernel_source

            def linear_fn(name, d_in, d_out):
                shape = (d_out, d_in, 1, 1)
                kernel_source.register(name, shape)
                return WeightFieldLinear(name, d_in, d_out, kernel_source)
        else:
            self.kernel_source = None

            def linear_fn(name, d_in, d_out):
                return nn.Linear(d_in, d_out, bias=False)

        # Token + position embeddings (always standard, not generated)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.seq_len, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(config, i, linear_fn) for i in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # LM head (standard, not generated)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def generator_params(self):
        if self.kernel_source is not None:
            return sum(p.numel() for p in self.kernel_source.parameters())
        return sum(p.numel() for n, p in self.named_parameters()
                   if "wte" not in n and "wpe" not in n and "lm_head" not in n
                   and "ln" not in n)


# ---- Weight-field backed Linear layer ---------------------------------------


class WeightFieldLinear(nn.Module):
    """nn.Linear replacement that generates its weight from a kernel source."""

    def __init__(self, name: str, in_features: int, out_features: int,
                 kernel_source):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_source = kernel_source
        self._shape = (out_features, in_features, 1, 1)

    def forward(self, x):
        kernel = self.kernel_source.get(self.name, self._shape)
        weight = kernel.squeeze()  # (out, in)
        return F.linear(x, weight)


# ---- Kernel source (reuse from ResNet experiments) --------------------------


class TransformerPoolKernelSource(nn.Module):
    """Per-group SIREN + pool for transformer linear layers.

    Groups: one SIREN per transformer layer (4 linears per layer: qkv, proj,
    fc, proj_mlp). Each layer's SIREN generates weights for its 4 linears.
    Shared multiplicative pool + per-group additive pool.
    """

    def __init__(self, n_layer: int, pool_size: int = 2048,
                 wf_hidden: int = 32, wf_layers: int = 2,
                 omega_0: float = 30.0):
        super().__init__()
        self.n_layer = n_layer
        self.pool_size = pool_size
        # Per-layer SIREN (3-dim input: layer_pos, d_out_pos, d_in_pos)
        self.sirens = nn.ModuleDict()
        for i in range(n_layer):
            self.sirens[str(i)] = SirenWeightField(
                wf_hidden, wf_layers, omega_0,
                output_activation="none", coord_dim=3)
        # Shared pool + per-layer additive pool
        self.pool = nn.Parameter(torch.randn(pool_size) * 0.05)
        self.pool_adds = nn.ParameterDict({
            str(i): nn.Parameter(torch.zeros(pool_size))
            for i in range(n_layer)
        })
        self.layer_index: dict[str, tuple[int, int]] = {}
        self._layer_conv_count: dict[int, int] = {}

    def register(self, name: str, shape: tuple[int, ...]) -> None:
        # Parse layer index from name like "layer2_attn_qkv"
        parts = name.split("_")
        layer_idx = int(parts[0].replace("layer", ""))
        conv_idx = self._layer_conv_count.get(layer_idx, 0)
        self._layer_conv_count[layer_idx] = conv_idx + 1
        self.layer_index[name] = (layer_idx, conv_idx)

    def get(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        out_d, in_d, _, _ = shape
        layer_idx, conv_idx = self.layer_index[name]
        device = self.pool.device

        # 3D coordinates: (linear_position_in_layer, d_out, d_in)
        n_linears = self._layer_conv_count[layer_idx]
        l_norm = conv_idx / max(n_linears - 1, 1)

        do = torch.arange(out_d, dtype=torch.float, device=device) / max(out_d - 1, 1)
        di = torch.arange(in_d, dtype=torch.float, device=device) / max(in_d - 1, 1)
        grid = torch.stack(torch.meshgrid(do, di, indexing="ij"), dim=-1)  # (out, in, 2)
        l_coord = torch.full((*grid.shape[:-1], 1), l_norm, device=device)
        coords = torch.cat([l_coord, grid], dim=-1)  # (out, in, 3)

        raw = self.sirens[str(layer_idx)](coords).squeeze(-1)  # (out, in)

        # Hash pool + additive
        num_elem = out_d * in_d
        stride = max(1, self.pool_size // max(n_linears, 1))
        offset = conv_idx * stride
        indices = (torch.arange(num_elem, device=device) + offset) % self.pool_size
        pool_vals = self.pool[indices].view(out_d, in_d)
        add_vals = self.pool_adds[str(layer_idx)][indices].view(out_d, in_d)

        return (torch.sigmoid(raw) * pool_vals + add_vals).unsqueeze(-1).unsqueeze(-1)


# ---- Data: Shakespeare byte-level ------------------------------------------


def get_shakespeare_data(seq_len, device):
    """Download Shakespeare and return train/val tensors of byte tokens."""
    data_path = Path("data/shakespeare.txt")
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
    text = data_path.read_text()
    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long, device=device)
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, seq_len, batch_size, device, n_batches=20):
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, seq_len, batch_size, device)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / n_batches


# ---- Training ---------------------------------------------------------------


def train(config_name, use_wps, epochs=50, lr=3e-4, batch_size=64,
          pool_size=2048, wf_hidden=32, seed=0,
          pool_lr_mult=1.0, epochs_override=None):
    if epochs_override is not None:
        epochs = epochs_override
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    config = TinyGPTConfig()
    train_data, val_data = get_shakespeare_data(config.seq_len, device)

    if use_wps:
        ks = TransformerPoolKernelSource(
            n_layer=config.n_layer, pool_size=pool_size,
            wf_hidden=wf_hidden, wf_layers=2)
        model = TinyGPT(config, kernel_source=ks).to(device)
    else:
        model = TinyGPT(config).to(device)

    if pool_lr_mult != 1.0 and use_wps:
        pool_params = [p for n, p in model.named_parameters() if "pool" in n]
        other_params = [p for n, p in model.named_parameters() if "pool" not in n]
        optimizer = torch.optim.Adam([
            {"params": other_params, "lr": lr},
            {"params": pool_params, "lr": lr * pool_lr_mult},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = len(train_data) // (batch_size * config.seq_len)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    print(f"[{config_name}] total_params={model.num_params()} "
          f"generator_params={model.generator_params()}")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_steps = 0
        for _ in range(steps_per_epoch):
            x, y = get_batch(train_data, config.seq_len, batch_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_steps += 1

        train_loss = total_loss / max(n_steps, 1)
        val_loss = estimate_loss(model, val_data, config.seq_len, batch_size, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(f"  epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    final_val = estimate_loss(model, val_data, config.seq_len, batch_size, device, n_batches=50)

    history["best_val_loss"] = best_val
    history["final_val_loss"] = final_val
    history["num_params"] = model.num_params()
    history["generator_params"] = model.generator_params()

    print(f"  >> final_val_loss={final_val:.4f}  "
          f"total_params={model.num_params()}  gen_params={model.generator_params()}")
    return history


CONFIGS = {
    "baseline": {"use_wps": False},
    "wps_p2048": {"use_wps": True, "pool_size": 2048},
    "wps_p4096": {"use_wps": True, "pool_size": 4096},
    # Large pool (matching CNN finding: pool vocabulary is the lever)
    "wps_p16k": {"use_wps": True, "pool_size": 16384},
    "wps_p32k": {"use_wps": True, "pool_size": 32768},
    # Pool LR sweep: pool at 5x base LR
    "wps_p16k_poollr": {"use_wps": True, "pool_size": 16384, "pool_lr_mult": 5.0},
    # Longer training
    "wps_p16k_long": {"use_wps": True, "pool_size": 16384, "epochs_override": 200},
    # Combined: big pool + long training + pool LR
    "wps_p32k_long_plr": {"use_wps": True, "pool_size": 32768,
                           "epochs_override": 200, "pool_lr_mult": 5.0},
    # Fair baseline at 200 epochs
    "baseline_200ep": {"use_wps": False, "epochs_override": 200},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--out-dir", default="results/exp_wps")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CONFIGS[args.config]
    history = train(args.config, seed=args.seed, epochs=args.epochs, **cfg)

    out_path = out_dir / f"{args.config}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
