# Level 3 — Profiling a GPT-2 style Transformer

This level moves from toy ops (Level 1) and mixed precision (Level 2) to a real
decoder-only transformer trained on a Tesla T4. The goal is to identify where
time actually goes during one full training step and to back every claim with
numbers extracted from a Kineto trace.

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA Tesla T4 (Turing, 40 SMs, sm_75) |
| CUDA runtime | 12.0 |
| Framework | PyTorch + `torch.profiler` with CUDA activity |
| Model | GPT-2 style decoder, 6 layers, d_model=512, 8 heads (64/head), d_ff=2048, vocab=50257, max_seq=256 |
| Batch | batch=8, seq_len=128 |
| Steps profiled | 3 active steps after 1 wait + 1 warm-up |

The runnable notebook is `transformer_profiling.ipynb`. The raw trace it
captures is saved as `gpt2.pt.trace.json` and can be opened directly in
[perfetto.ui](https://ui.perfetto.dev) or `chrome://tracing`.

## Results

Total GPU kernel time across the 3 profiled steps: **266.65 ms**.

### GPU time by role

| Bucket | Time | Share |
|---|---:|---:|
| Matmul (cuBLAS sgemm, all shapes) | 142.65 ms | **53.50%** |
| Adam optimizer (`multi_tensor_apply`) | 53.64 ms | 20.12% |
| Elementwise (add, gelu, mul, etc.) | ~38 ms | ~14.2% |
| Memory-efficient attention (fwd + bwd CUTLASS FMHA) | 18.06 ms | **6.77%** |
| Reductions, layernorm backward, misc | ~14 ms | ~5.4% |

### Forward vs backward vs optimizer (CPU-side annotations)

| Step | forward | backward | optimizer | bwd/fwd |
|---|---:|---:|---:|---:|
| #2 | 14.28 ms | 16.09 ms | 3.49 ms | 1.13× |
| #3 | 10.40 ms | 15.04 ms | 3.64 ms | 1.45× |
| #4 |  9.91 ms | 37.61 ms | 3.05 ms | 3.80× |
| **avg** | **11.53 ms** | **22.92 ms** | **3.40 ms** | **1.99×** |

Step #4's 37.6 ms backward is a jitter outlier — either a cuBLAS heuristic
swap on a different tile shape or Python GC hitting the CPU scheduler. Keep
it in the table so the average reflects realistic variance rather than a
cherry-picked best case.

### Top individual kernels

| Share | Time | n | Kernel |
|---:|---:|---:|---|
| 20.31% | 54.17 ms | 95 | `volta_sgemm_128x64_tn` |
| 18.65% | 49.74 ms | 96 | `volta_sgemm_128x64_nt` |
| 14.53% | 38.74 ms | 72 | `volta_sgemm_128x64_nn` |
|  5.38% | 14.33 ms | 24 | `fmha_cutlassB_f32_aligned_64x64_k64` (attention backward) |
|  4.32% | 11.51 ms | 12 | `multi_tensor_apply_kernel` (Adam) |
|  1.40% |  3.72 ms | 18 | `fmha_cutlassF_f32_aligned_64x64_rf_sm75` (attention forward) |

## What to take away

- **Matmul dominates. Attention does not.** At seq_len=128 the memory-efficient
  attention kernels are under 7% of GPU time. The sgemm calls in the QKV
  projection, output projection, and MLP layers add up to 53.5%. The popular
  "attention is the bottleneck" framing only holds at longer contexts where
  the O(T²) term in attention catches up with the O(T·d²) term in the linear
  layers.
- **Backward is ~2× forward, for the right reason.** Every linear layer in
  backward launches two sgemms (grad w.r.t. activations + grad w.r.t.
  weights), compared to one in forward. The kernel table confirms this:
  `sgemm_*_tn` and `sgemm_*_nt` (backward) together roughly double
  `sgemm_*_nn` (forward).
- **Adam is surprisingly expensive.** ~20% of GPU time is in
  `multi_tensor_apply` kernels. On small models the optimizer can rival
  attention for GPU time; this is the usual motivation for fused / foreach
  optimizers.
- **Average, don't cherry-pick.** Step-to-step variance on an unloaded T4
  was large enough (1.13× to 3.80×) that a single step is not enough to draw
  conclusions from.

## Files

- `transformer_profiling.ipynb` — runnable Colab notebook (T4 runtime).
- `gpt2.pt.trace.json` — Kineto trace produced by the notebook above;
  open in perfetto.ui for per-kernel inspection.
- `README.md` — this file.

## Next step (Level 4 candidate)

Repeat the measurement at `seq_len=1024` and plot matmul% vs attention% as a
function of sequence length. The crossover point is the interesting number.
