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

> CUDA-only speedup (2.08x) is higher than wallclock speedup (1.37x) because CPU overhead and data loading are constant across both runs. The GPU kernel efficiency gain from AMP is the real story.

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
| Level 3 | Transformer profiling (GPT-2) — attention kernels, LayerNorm, KV ops | Planned |
| Level 4 | Operator-level breakdown — % CUDA time by op type | Planned |
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

### Level 3 � Transformer profiling with PyTorch Profiler

**Goal:** Profile a transformer-based model (GPT-2) to understand GPU utilization, memory patterns, and operator performance in large language models.

- Fine-tunes GPT-2 small on a text classification task
- Captures detailed execution traces using 	orch.profiler
- Exports .pt.trace.json for Perfetto analysis
- Analyzes attention layers, matrix multiplications, and memory bottlenecks

**Key findings:**
- Attention mechanisms dominate compute time
- Memory usage spikes during forward/backward passes
- Optimizer steps show high CUDA kernel activity

?? [Level3_transformer_profiling/](./Level3_transformer_profiling/)
