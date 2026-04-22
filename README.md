# ML Accelerator Profiling Labs

Hands-on GPU profiling labs using PyTorch Profiler, mixed precision benchmarking,
and Perfetto trace analysis. Built to develop practical GPU performance engineering
skills on real training workloads.

**Author:** Vishwas Somashekara Reddy
**Stack:** PyTorch · PyTorch Profiler · Perfetto · Google Colab · NVIDIA CUDA

---

## Results summary

| Lab | Model | Hardware | Metric | Result |
|-----|-------|----------|--------|--------|
| Level 1 | FeedForward (FashionMNIST) | T4 (Colab) | Self CPU time total | 80.160 ms |
| Level 1 | FeedForward (FashionMNIST) | T4 (Colab) | Self CUDA time total | 287.420 µs |
| Level 2 | FeedForward (FashionMNIST) | T4 (Colab) | FP32 elapsed time | 1.743 sec |
| Level 2 | FeedForward (FashionMNIST) | T4 (Colab) | AMP elapsed time | 1.273 sec |
| Level 2 | FeedForward (FashionMNIST) | T4 (Colab) | Wallclock speedup (AMP vs FP32) | **1.37x** |
| Level 2 | FeedForward (FashionMNIST) | T4 (Colab) | CUDA time FP32 → AMP | 17.631 ms → 8.476 ms (**2.08x**) |
| Level 3 | GPT-2 style (6L, d=512, 8h) | T4 (Colab) | Matmul share of GPU time | **53.50%** |
| Level 3 | GPT-2 style (6L, d=512, 8h) | T4 (Colab) | Attention share of GPU time | **6.77%** |
| Level 3 | GPT-2 style (6L, d=512, 8h) | T4 (Colab) | Adam optimizer share of GPU time | 20.12% |
| Level 3 | GPT-2 style (6L, d=512, 8h) | T4 (Colab) | Backward / forward ratio (CPU-side) | **1.99x** |

> CUDA-only speedup in Level 2 (2.08x) is higher than wallclock speedup (1.37x) because CPU overhead and data loading are constant across both runs. The GPU kernel efficiency gain from AMP is the real story.

> In Level 3, matmul dominates GPU time at short sequence lengths — "attention is the bottleneck" is a long-context story. The O(T²) attention term only catches the O(T·d²) linear term at much longer sequences than we used here (seq_len=128).

---

## Lab progression

### Level 1 — GPU profiling with PyTorch Profiler

**Goal:** Understand GPU execution timeline and operator-level performance.

- Trains a feedforward network on FashionMNIST (batch size 256, 1 epoch)
- Captures CPU + CUDA execution traces using `torch.profiler`
- Exports `.pt.trace.json` for Perfetto visualization
- Identifies top operators by CUDA time: `aten::linear`, backward pass, optimizer step

**Key finding:** Backward pass consistently takes longer than forward pass. CPU→GPU data transfer latency is minimal relative to compute.

📁 [`Level1_gpu_profiling_pytorch/`](./Level1_gpu_profiling_pytorch/)

---

### Level 2 — Mixed precision profiling (FP32 vs AMP)

**Goal:** Quantify the performance impact of mixed precision training.

- Runs identical workload under FP32 baseline and PyTorch AMP (FP16)
- Captures separate trace files: `fp32.pt.trace.json`, `amp.pt.trace.json`
- Compares elapsed time, CUDA kernel time, and memory usage

**Key finding:** AMP cuts CUDA kernel time by 2.08x by running eligible ops in FP16. Wallclock speedup is 1.37x — lower than CUDA speedup because CPU and data loading overhead is unchanged.

📁 [`Level2_mixed_precision_profiling/`](./Level2_mixed_precision_profiling/)

---

### Level 3 — Transformer profiling (GPT-2 style on T4)

**Goal:** Identify where GPU time actually goes during one training step of a decoder-only transformer, and back every claim with numbers parsed directly from a Kineto trace.

- 6-layer GPT-2 style model (d_model=512, 8 heads × 64 dim, d_ff=2048, vocab=50257)
- Batch 8, seq_len 128, Tesla T4, PyTorch `torch.profiler` with CUDA activity
- Captures a Kineto trace across 3 active steps (266.65 ms of GPU kernel time)
- Buckets 1,405 GPU kernels by role and cross-checks against CPU-side annotations

**Key findings:**

- **Matmul dominates, attention does not.** cuBLAS sgemm kernels account for 53.50% of GPU time. Memory-efficient attention (CUTLASS FMHA, fwd + bwd) is only 6.77%.
- **Backward runs 1.99x forward, for the right reason.** Each linear layer launches two sgemms in backward (grad w.r.t. activations + grad w.r.t. weights) versus one in forward. The kernel table confirms it: `sgemm_*_tn` + `sgemm_*_nt` (backward) ≈ 2x `sgemm_*_nn` (forward).
- **Adam is not free.** ~20% of GPU time is spent in `multi_tensor_apply` kernels. Fused / foreach optimizers exist for a reason.

📁 [`Level3_transformer_profiling/`](./Level3_transformer_profiling/)

---

## How to run

All labs run on **Google Colab** (free T4 GPU). No local setup needed.

1. Open the `.ipynb` notebook in the relevant folder
2. Click **Runtime → Change runtime type → T4 GPU**
3. Run all cells
4. Download the `.pt.trace.json` output
5. Open [Perfetto UI](https://ui.perfetto.dev) and drag in the trace file to visualize the GPU timeline

---

## What's next

| Lab | Topic | Status |
|-----|-------|--------|
| Level 4 | Operator-level breakdown — % CUDA time by op type, scan across sequence lengths | Planned |
| Level 5 | Distributed training — DDP communication vs compute overlap | Planned |

---

## Tools used

| Tool | Purpose |
|------|---------|
| PyTorch Profiler | Capturing CPU + CUDA execution traces |
| Perfetto | Visualizing GPU execution timelines |
| PyTorch AMP | Mixed precision training (FP16/FP32) |
| Google Colab | Free T4 GPU environment |
| nvidia-smi | GPU hardware validation |
