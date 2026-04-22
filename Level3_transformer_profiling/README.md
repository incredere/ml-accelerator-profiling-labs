# Level 3 — Transformer profiling with PyTorch Profiler

**Goal:** Profile GPT-2 small to understand which GPU operations dominate transformer training — attention, matrix multiply, or optimizer overhead.

- Trains GPT-2 small with `torch.profiler` active
- Exports `.pt.trace.json` for Perfetto visualization
- Measures CUDA kernel time split across matmul, attention, and optimizer ops

**Key findings:**
- Matrix multiply (sgemm) dominates GPU compute: **35.6%** of CUDA time across three kernel variants (tn / nt / nn)
- Attention (fmha) is only **3.6%** of CUDA compute — not the bottleneck in this workload
- Adam optimizer accounts for **10.1%** of CUDA time
- Backward pass takes **2.10×** longer than forward pass

📁 [Transformer profiling notebook](./transformer_profiling.ipynb)  
📁 [GPT-2 trace file](./gpt2.pt.trace.json)

## Prerequisites

- PyTorch with CUDA support
- Transformers library
- PyTorch Profiler
- Google Colab or local GPU environment

## Running the lab

1. Open `transformer_profiling.ipynb` in Colab
2. Follow the cells to train and profile GPT-2
3. Download the trace file for local analysis
4. Use Perfetto to visualize the timeline

## Analysis questions

- Which operators consume the most CUDA time?
- How does memory usage change during training?
- What are the bottlenecks in transformer layers?
