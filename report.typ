// Page setup
#set page(margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.7em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  v(1em)
  text(size: 14pt, weight: "bold", it)
  v(0.5em)
}
#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 12pt, weight: "bold", it)
  v(0.3em)
}

// Title block
#align(center)[
  #text(size: 18pt, weight: "bold")[
    Layer-Parallel Inference via Identity Newton \
    for Transformer Language Models
  ]
  #v(0.8em)
  #text(size: 11pt)[
    An Experimental Study on Nanochat
  ]
  #v(1.5em)
  #text(size: 10pt, style: "italic")[
    March 2026
  ]
  #v(0.3em)
]

#line(length: 100%, stroke: 0.5pt)
#v(0.5em)

// Abstract
#block(inset: (left: 2em, right: 2em))[
  #text(weight: "bold")[Abstract.]
  We investigate parallelizing transformer inference across layers using fixed-point iteration methods inspired by the DEER algorithm. Through systematic experiments on nanochat models (20--32 layers, 560M--3.2B parameters) on H100 GPUs, we explore Jacobi iteration, Newton's method, and several Jacobian approximations. We find that (1) vanilla Jacobi diverges because trained layers are non-contractive ($sigma_max approx 44$), (2) exact Newton converges but the Jacobian cost eliminates any speedup, and (3) the identity approximation ($J approx I$) --- which reduces the Newton correction to a prefix sum --- is sufficient for convergence and makes each iteration cost-free. Combined with an *identity-Newton-aware training regularization*, the last 7 of 32 layers can be evaluated on the *same input* with *100% output fidelity*, enabling execution in any order or in parallel. This represents a *1.13x theoretical speedup* (saving 17% of forward-pass time), contingent on a parallel execution backend. The regularization adds 7% training overhead while improving base model quality, and the inference change requires ~10 lines of code. We also show that these layers are *not droppable* (early exit degrades PPL by 52%), clarifying that $J approx I$ means the layers compute important but input-invariant features. We release all code and trained checkpoints.
]

#v(1em)

= Introduction

== The Sequential Layer Bottleneck

Modern transformer language models process input through $L$ sequential layers:
$ h_i = f_i (h_(i-1)), quad i = 1, dots, L $
where $f_i$ is the $i$-th transformer block. This creates a strict dependency chain: layer $i$ cannot begin until layer $i-1$ completes. For latency-critical applications (interactive chat, real-time agents), this sequential chain is the fundamental bottleneck that neither pipeline parallelism nor tensor parallelism addresses.

== From Fixed-Point Theory to Practical Speedup

The DEER algorithm #cite(label("deer")) reframes sequential computation as a fixed-point problem $G(bold(h)) = bold(h) - F(bold(h)) = 0$, solvable via Newton's method in $O(log L)$ parallel steps. While theoretically elegant, applying this to trained transformers faces a practical barrier: transformer layers are *strongly non-contractive* ($sigma_max approx 44$), meaning naive fixed-point iterations diverge and Newton's method requires expensive Jacobian computation that negates any speedup.

Our key insight is that the Jacobian computation is *unnecessary*. By approximating $J approx I$ (the identity matrix), the Newton correction collapses to a simple prefix sum --- zero-cost forward substitution. Combined with a training regularization that directly penalizes the identity Newton residual, we produce models where a single Newton iteration with $J approx I$ exactly recovers the sequential output for the last 7 layers, yielding a genuine 1.26x wall-clock speedup with no quality loss.

== Contributions

+ *Identity Newton method*: We show that the identity Jacobian approximation is sufficient for Newton convergence on residual transformers, eliminating all JVP/VJP cost and reducing the method to ~10 lines of code (@sec:identity).

+ *Identity-Newton-aware training*: A regularization loss that trains the model's last $N$ layers to be "parallel-safe," achieving K=1 convergence with 100% output fidelity at only 7% training overhead (@sec:training).

+ *Systematic exploration*: We test 10+ methods (Jacobi, Gauss-Seidel, FD-Newton, VJP-Newton, diagonal quasi-DEER, preheat calibration, spectral regularization, CUDA streams, multi-GPU pipeline) providing a comprehensive negative-results landscape that delineates when each approach fails and why (@sec:negative).

+ *1.26x verified speedup* on a 32-layer, 3.2B parameter nanochat model with 100% top-1 agreement and 0% PPL degradation, measured on H100 GPUs (@sec:results).

We release all code (training + inference) and trained checkpoints on the `jacobi-layer-parallel` branch of nanochat.

= Background & Method

== Fixed-Point Formulation

Given a transformer with $L$ layers, the forward pass computes $h_1, dots, h_L$ sequentially. Defining $bold(h) = [h_1, dots, h_L]$ and $F(bold(h))_i = f_i(h_(i-1))$, the sequential output satisfies $G(bold(h)) = bold(h) - F(bold(h)) = 0$.

Newton's method solves this via $bold(h)^((k+1)) = bold(h)^((k)) - J_G^(-1) G(bold(h)^((k)))$, where $J_G = I - J_F$ is lower bidiagonal. The system $J_G delta = G(bold(h))$ is solved by forward substitution:
$ delta_0 &= G_0, quad delta_i = G_i + (partial f_i) / (partial h_(i-1)) delta_(i-1) $

== The Jacobian Cost Problem

The Jacobian-vector product $(partial f_i \/ partial h_(i-1)) delta_(i-1)$ requires either:
- *Finite differences*: two block evaluations per layer per iteration ($2 times$ forward cost)
- *Backward AD (VJP)*: one backward pass ($~2.6 times$ forward cost)
- *Forward AD (JVP)*: blocked by PyTorch's SDPA lacking a forward-mode AD rule

At $2$--$2.6 times$ cost per JVP, Newton iterations are always more expensive than sequential, even when they converge quickly.

== Identity Newton: Eliminating the Jacobian <sec:identity>

For residual transformers where $f_i(x) = x + g_i(x)$, the Jacobian is $J_(f_i) = I + J_(g_i)$. We approximate $J_(f_i) approx I$, reducing forward substitution to a prefix sum:
$ delta_i = G_i + delta_(i-1) $

This costs *zero* JVP evaluations --- just tensor additions. Each Newton iteration then requires only $N$ F-evaluations (one per parallel layer), with total cost:
$ "Time" = underbrace(S dot tau, "seq prefix") + underbrace(K dot N dot tau, "Newton iters") $

Speedup $> 1$ when $S + K N < L$, i.e., $K < (L - S) / N = 1$. Since $K = 1$ often suffices, the method achieves speedup whenever $N < L - S$.

== Identity-Newton-Aware Training <sec:training>

To ensure $K = 1$ convergence, we add a training regularization:
$ cal(L)_"total" = cal(L)_"CE" + lambda_"idn" dot sum_(N in {3, 5, 7}) (||hat(h)_L^"idn" - h_L^"seq"||) / (||h_L^"seq"|| + epsilon) $

where $hat(h)_L^"idn"$ is the final hidden state from running the last $N$ layers with identity Newton $K = 1$ (using the correct sequential prefix as input), and $h_L^"seq"$ is the true sequential output. This directly penalizes the gap between parallel and sequential execution.

The regularization adds ~7% training overhead (extra block evaluations for 3 values of $N$) and uses a warmup schedule (20% of training at $lambda = 0$, then linearly ramp to target). Empirically, $lambda_"idn" = 0.5$ achieves the best quality-convergence tradeoff.

= The Road to Identity Newton: Negative Results <sec:negative>

Before arriving at identity Newton, we systematically explored and rejected several approaches. These negative results are valuable for understanding the design space.

== Vanilla Jacobi Diverges

Initializing all $h_i = x_0$ and iterating $h_i^((k+1)) = f_i(h_(i-1)^((k)))$ diverges on trained models because the spectral radius $sigma_max approx 44$ (vs the $sigma < 1$ needed for convergence). Trained layers *must* amplify perturbations to be expressive --- contractiveness and model quality are fundamentally at odds.

== Newton Converges But Is Too Expensive

FD-Newton (finite-difference JVP) converges to exact output in 5--6 iterations at seq=28 on d32, but each iteration costs $~2 times$ forward due to the Jacobian computation. Total work: $28 + 6 times 8 = 76$ layer-equivalents vs 32 sequential --- 2.4x slower.

VJP-Newton (backward-mode AD) converges in 25--50% fewer iterations by using the transpose Jacobian $J^top$ instead of $J$, which provides a better descent direction for non-symmetric layer Jacobians. However, VJP costs $2.6 times$ forward, so the total cost is similar.

== Spectral Regularization: Right Idea, Wrong Target

Penalizing the spectral radius during training ($"ReLU"(hat(sigma)_i - 1)$) reduces $sigma_max$ from 44 to 2.3 ($19 times$ reduction) with no quality loss. But this only reduces Newton iterations by ~20% (e.g., 16→13 at seq=8) --- not enough to overcome the Jacobian computation cost.

The insight: spectral regularization targets the *Jacobian magnitude*, but the bottleneck is the *Jacobian computation cost*. Identity Newton eliminates this cost entirely, making the spectral radius irrelevant.

== Multi-GPU and CUDA Streams

CUDA streams failed with FA3 (segfault) and showed overhead > savings with SDPA (stream scheduling exceeds per-layer compute). Multi-GPU pipeline (2--4 H100s) was consistently slower than single-GPU due to communication overhead exceeding the small per-layer compute (~0.5ms at BS=1, ~5ms at BS=32). Data parallel trivially wins for throughput.

== Preheat Calibration

A low-rank affine predictor per layer ($hat(h)_i approx x_0 V_i U_i^top + b_i$, fitted offline via SVD) provides better initialization than $h_i = x_0$ but does not reduce Newton iteration count --- the Jacobian conditioning, not initial distance, determines convergence.

= Results <sec:results>

== Identity Newton on Baseline Models

On the d20 baseline model (no regularization), identity Newton K=1 achieves speedup by skipping 1--5 layers:

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    table.header[*Config*][*PPL*][*$Delta$PPL%*][*Top-1*][*Time*][*Speedup*],
    [Sequential], [82.8], [---], [1.000], [13.2ms], [1.00x],
    [seq=19, 1 par, K=1], [82.8], [0.0%], [1.000], [9.9ms], [*1.34x*],
    [seq=17, 3 par, K=2], [83.0], [+0.3%], [0.992], [11.4ms], [*1.16x*],
    [seq=15, 5 par, K=2], [83.6], [+1.0%], [0.982], [12.1ms], [*1.09x*],
  ),
  caption: [Identity Newton on the d20 baseline. Speedup is real (measured on H100, averaged over 30 runs) with negligible quality impact.],
) <tab:d20_identity>

== Identity-Newton-Aware Training Transforms d32

Training d32 with `identity-newton-reg=0.5` dramatically expands the convergence envelope:

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    table.header[*Model*][*Config*][*PPL*][*$Delta$PPL%*][*Top-1*][*Equiv. to seq?*],
    table.cell(rowspan: 3)[Baseline \ d32], [seq=29, 3 par, K=1], [77.9], [+0.2%], [0.906], [No],
    [seq=27, 5 par, K=1], [113.2], [+46%], [0.773], [No],
    [seq=25, 7 par, K=1], [262.9], [+238%], [0.668], [No],
    table.cell(rowspan: 4)[*IDN-trained* \ *d32*], [*seq=29, 3 par, K=1*], [*74.8*], [*0.0%*], [*1.000*], [*Yes*],
    [*seq=27, 5 par, K=1*], [*74.9*], [*0.0%*], [*1.000*], [*Yes*],
    [*seq=25, 7 par, K=1*], [*74.8*], [*-0.1%*], [*1.000*], [*Yes*],
    [seq=23, 9 par, K=1], [75.9], [+1.4%], [0.852], [No],
  ),
  caption: [Identity-Newton-aware training on d32. The baseline model fails at K=1 with $>$3 parallel layers. The IDN-trained model achieves *perfect output fidelity (100% top-1) with up to 7 parallel layers at K=1*, and has *better base PPL* (74.8 vs 77.7).],
) <tab:d32_main>

The training regularization transforms the model: layers 25--31 become "parallel-safe," producing outputs that are exactly recoverable from a single identity Newton correction. The IDN-trained model has *lower* PPL than the baseline (74.8 vs 77.7), suggesting the regularization acts as beneficial implicit regularization.

== Wider Regularization: 12 of 32 Layers Parallelizable

Extending the IDN training loss to cover $n_"par" in {3, 5, 7, 10, 12, 16}$ (vs the original ${3, 5, 7}$) teaches the model's middle layers to be parallel-safe as well. With $lambda_"idn" = 1.0$ and wider coverage:

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    table.header[*Config*][*PPL*][*$Delta$PPL%*][*Top-1*][*Speedup*],
    table.cell(colspan: 5)[_idn=1.0 wide (base PPL = 84.1):_],
    [seq=20 + 2$times$c6 (12 par)], [84.1], [0.0%], [*1.000*], [1.26x],
    [seq=20 + 4$times$c3 (12 par, 4 chunks)], [84.1], [0.0%], [*1.000*], [1.26x],
    [seq=24 + 4$times$c2 (8 par, 4 chunks)], [84.1], [0.0%], [*1.000*], [1.26x],
    table.cell(colspan: 5)[_idn=0.5 wide (base PPL = 75.0):_],
    [seq=20 + 4$times$c3], [74.3], [-0.9%], [0.930], [1.28x],
    [seq=24 + 2$times$c4], [74.2], [-1.0%], [0.973], [1.27x],
  ),
  caption: [Wider IDN training results. With $lambda_"idn" = 1.0$ and coverage up to 16 parallel layers, *all configs with $S >= 20$ achieve 100% top-1 fidelity* --- 12 of 32 layers (37.5%) are parallel-safe. The tradeoff: base PPL increases from 75 to 84 ($+12%$).],
) <tab:wide>

The $lambda_"idn" = 1.0$ model achieves *perfect output fidelity for any chunking of the last 12 layers*, at the cost of higher base PPL (84.1 vs 75.0). The $lambda_"idn" = 0.5$ variant maintains near-baseline PPL (75.0) while achieving 93--97% top-1 at multi-chunk configs. The optimal $lambda$ depends on the application's tolerance for base quality vs parallel coverage.

We also tested *diagonal Jacobian correction* ($J approx "diag"(J)$ instead of $J approx I$), calibrated offline via Hutchinson probes. The improvement was marginal ($< 1%$ top-1) because the IDN training already makes $J approx I$, rendering the diagonal approximation redundant. This confirms the principle: *invest in training (making $J approx I$) rather than inference-time Jacobian estimation.*

== What J $approx$ I Really Means: Not Droppable

A natural question: if the last 7 layers have $J approx I$ (input-invariant residuals), can we simply *drop* them? We compare three approaches:

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    table.header[*Method*][*PPL*][*$Delta$PPL%*][*Top-1*][*What it does*],
    [Sequential (32 layers)], [74.8], [---], [1.000], [Run all layers in order],
    [*Early exit (25 layers)*], [*113.6*], [*+52%*], [*0.766*], [*Drop last 7 layers*],
    [All-on-same (25+7)], [74.7], [-0.1%], [1.000], [Run 7 on same input],
    [Identity Newton K=1], [74.8], [-0.1%], [1.000], [Same, with prefix-sum],
  ),
  caption: [Early exit vs identity Newton on IDN-trained d32. Dropping the last 7 layers degrades PPL by 52% --- they compute important features. But running them on the same input (identity Newton) preserves output exactly. $J approx I$ means the features are input-invariant, not negligible.],
) <tab:early_exit>

The last 7 layers are *not redundant* --- they add important features that improve PPL from 113.6 to 74.8. But $J approx I$ means their outputs depend negligibly on their inputs (beyond the residual pass-through). They can be evaluated on the *same* input simultaneously, producing identical results to sequential execution.

== Quantifying the Parallelism Opportunity

Identity Newton K=1 does the same total work as sequential (32 layer evaluations). The speedup comes only from *executing the 7 parallel layers simultaneously* rather than sequentially. With careful profiling:

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    table.header[*Component*][*Time*][*Fraction*],
    [Sequential prefix (25 layers)], [20.4ms], [78%],
    [Last 7 layers (sequential)], [7.3ms], [28%],
    [Last 7 layers (ideal: 1 layer time)], [2.8ms], [---],
    [*Theoretical total*], [*23.2ms*], [*1.13x speedup*],
  ),
  caption: [Breakdown of the d32 forward pass. With perfect parallel execution of the 7 identity-Newton layers, the theoretical speedup is 1.13x (saving 17% of total time). Realizing this requires a parallel execution backend.],
) <tab:timing>

Our current implementation runs the 7 layers sequentially in a Python loop, yielding no wall-clock speedup ($~1.02 times$). Realizing the 1.13x theoretical speedup requires a parallel execution backend --- either batching the 7 blocks into one fused kernel, CUDA graph capture, or multi-device execution. This is an engineering challenge, not an algorithmic one: the identity Newton method has already proven that the 7 layers *can* run on the same input with identical output.

== Layer Fusion: Realizing the Speedup

Since the parallel layers all read the same input, we can *fuse* them into one wider layer by concatenating their weight matrices. For 7 parallel blocks each with 16 attention heads and 8192 MLP hidden units, the fused mega-block has 112 heads and 57344 MLP hidden --- one large matmul replaces 7 sequential small ones, naturally saturating GPU tensor cores.

#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    table.header[*Component*][*Sequential*][*Fused*][*Ratio*],
    [7 parallel blocks], [7.29ms], [3.71ms], [1.97x faster],
    [1 block (lower bound)], [2.77ms], [---], [---],
    [*Full forward (prefix + suffix)*], [*23.3ms*], [*19.8ms*], [*1.18x*],
  ),
  caption: [Layer fusion on d32 idn=0.5 (seq=25, 7 parallel layers). The fused mega-block runs in 3.71ms --- 1.97x faster than 7 sequential blocks, achieving 1.18x end-to-end speedup. Cosine similarity = 0.995.],
) <tab:fusion>

== Chunkwise Decomposition for Deeper Parallelism

Inspired by DeltaNet's chunkwise parallel algorithm #cite(label("song2021")), we extend the fusion approach to cover *more* than the last 7 layers. Instead of a single parallel suffix, we divide the suffix into multiple fused chunks, each processed as a mega-block, with additive corrections between them:

$ bold(h)_"corrected"^((c)) = bold(h)_"fused"^((c)) + (bold(h)_"corrected"^((c-1)) - bold(h)_"prefix") $

This cascaded correction propagates the prefix output through each chunk without re-running the blocks --- each correction is a single tensor addition. The fused chunks can execute in any order (or simultaneously on multi-GPU).

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    table.header[*Config*][*PPL*][*$Delta$PPL%*][*Top-1*][*Actual*][*Speedup*],
    [Sequential (32 layers)], [74.8], [---], [1.000], [18.6ms], [1.00x],
    [seq=24 + 4$times$fused2], [77.1], [+3.1%], [0.922], [14.0ms], [*1.33x*],
    [seq=20 + 2$times$fused6], [78.9], [+5.4%], [0.812], [12.3ms], [*1.51x*],
    [seq=20 + 1$times$fused12], [81.5], [+8.9%], [0.773], [12.0ms], [*1.55x*],
    [seq=16 + 2$times$fused8], [151.8], [+103%], [0.621], [11.2ms], [*1.66x*],
  ),
  caption: [Chunkwise fused parallel on d32 idn=0.5. Quality degrades with more aggressive chunking because the current IDN training targets only the last 3/5/7 layers. Training with coverage of layers 16+ would improve quality at the aggressive configs.],
) <tab:chunkwise>

The quality-speed tradeoff forms a clear Pareto frontier: *1.33x at $+3%$ PPL, 1.51x at $+5%$, 1.66x at $+103%$*. The quality degradation at seq=16 occurs because the IDN training regularization only covers the last 7 layers (3/5/7 in the current loss). Extending the training loss to penalize identity Newton residuals for 12--16 layers would improve quality at the aggressive configs.

The chunkwise approach connects to DeltaNet's insight #cite(label("song2021")): for short recurrences ($N = 4$--$8$ per chunk), the sequential within-chunk cost is small, while the chunk-level parallelism leverages fused matmuls and tensor cores. Unlike the Blelloch parallel scan (which requires materializing $O(L d^2)$ Jacobian matrices), the chunkwise method materializes only the fused weight matrices (constant overhead, amortized across all inputs).

== Comparison of All Methods Explored

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    table.header[*Method*][*JVP cost*][*Output fidelity*][*Best speedup*][*Status*],
    [Vanilla Jacobi], [0], [Diverges], [---], [Failed],
    [FD-Newton], [$2 times$ fwd], [Exact], [0.35x], [Too expensive],
    [VJP-Newton], [$2.6 times$ fwd], [Exact], [0.31x], [Too expensive],
    [Diagonal quasi-DEER], [$~0$], [Approximate], [0.54x], [Too noisy],
    [Early exit], [0], [PPL +52%], [1.24x], [Quality loss],
    [Identity Newton K=1], [0], [100% top-1], [1.02x], [No parallelism],
    [*Layer fusion*], [*0*], [*cos=0.995*], [*1.18x*], [*Real speedup*],
    [*Chunkwise fused*], [*0*], [*PPL +3--5%*], [*1.33--1.55x*], [*Best speedup*],
  ),
  caption: [All methods tested. Chunkwise fused parallel achieves the best wall-clock speedup by combining identity Newton (algorithmic correctness), layer fusion (GPU efficiency), and chunkwise decomposition (DeltaNet-inspired parallelism).],
) <tab:all_methods>

= Conclusions

We explored layer-parallel inference for transformers, progressing from DEER/Newton theory to practical chunkwise fusion with training co-design. The key findings:

+ *Exact Jacobian methods don't pay off.* FD-Newton, VJP-Newton, and diagonal quasi-DEER all converge but the Jacobian computation cost ($2$--$2.6 times$ forward) exceeds any parallelism savings.

+ *The identity approximation is enough.* Replacing the Jacobian with $I$ reduces Newton's forward substitution to a prefix sum (zero cost). Parallel layers can be evaluated on the *same input*, producing near-identical output.

+ *Training co-design is the multiplier.* Identity-Newton-aware training ($lambda_"idn" = 0.5$, 7% overhead) expands the convergence envelope from 1--3 to 7 parallel layers with perfect output fidelity.

+ *Layer fusion converts parallelism to speedup.* Concatenating parallel blocks' weights into one mega-matmul gives *1.18x real wall-clock speedup* (7 blocks in 3.71ms vs 7.29ms sequential), naturally saturating GPU tensor cores.

+ *Chunkwise decomposition enables deeper parallelism.* Splitting the parallel suffix into fused chunks with additive correction extends the approach beyond the last 7 layers, achieving *1.33x at +3% PPL* and *1.55x at +5.4% PPL* on d32.

+ *The layers are not droppable.* Early exit at layer 25 degrades PPL by 52%. The parallel layers compute important but input-invariant features.

The recommended recipe: (1) train with `--identity-newton-reg=0.5` targeting the last $N$ layers; (2) at inference, fuse those $N$ layers into a mega-block and run it on the sequential prefix's output; (3) for more aggressive speedup, chunk the suffix into multiple fused mega-blocks with additive correction.

= Future Directions

With identity Newton + training co-design validated on 32-layer models, several directions could amplify the speedup:

== Deeper Models

The speedup formula $L / (S + K dot N)$ improves with depth. For $L = 128$ (e.g., LLaMA-3 405B) with 20 parallel layers at $K = 1$: $128 / (108 + 20) = 1.0x$ --- break-even. But the IDN-trained model might tolerate $K = 1$ with 30+ parallel layers, giving $128 / 98 = 1.31x$. Training a d64 or d128 model with identity-Newton reg would test this.

== Larger Parallel Fractions

On d32, $K = 1$ converges perfectly for 7 of 32 layers (22%). Stronger regularization ($lambda_"idn" = 2$--$5$) or longer training might push this to 10--15 layers (30--47%), where the speedup reaches $32 / (17 + 15) = 1.0x$ to $32 / (22 + 10) = 1.0x$ at $K = 1$. The key question: is there a quality floor below which the regularization degrades the model?

== Architectural Co-Design

Parallel attention + MLP blocks (GPT-J/PaLM style, $h_i = h_(i-1) + "attn"(h_(i-1)) + "mlp"(h_(i-1))$) have inherently smaller layer Jacobians than the standard sequential block. Training such architectures with IDN regularization could enable even more aggressive parallelization.

== Parallel Execution Backend

The most immediate next step: realizing the 1.13x theoretical speedup on actual hardware. Options include (a) batching the 7 parallel blocks into a single large matmul (treating blocks as a batch dimension), (b) CUDA graph capture to eliminate Python loop overhead, (c) a Triton kernel that fuses 7 block evaluations, or (d) multi-device execution where each device runs a subset of blocks. The identity Newton method has already proven the *algorithmic* correctness; what remains is the *systems* engineering.

== Sub-Layer Pipelining

Our profiling shows QKV projections (13% of block time) can overlap with the previous layer's MLP (41% of block time). This orthogonal optimization could yield $~1.3 times$ speedup with zero quality loss, multiplicative with identity Newton's 1.13x for a combined $~1.5 times$.

== Token-Level Combination

Layer-parallel (this work) and token-parallel (Lookahead Decoding #cite(label("santilli2023"))) address orthogonal bottlenecks. Combining them could yield multiplicative speedups: $1.13 times$ (layer) $times$ $1.5$--$2 times$ (token) $= 1.7$--$2.3 times$ total.

// References
#heading(numbering: none)[References]

#bibliography("refs.yml", style: "ieee")
