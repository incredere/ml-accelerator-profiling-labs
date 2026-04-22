# Level 3 — Transformer profiling with PyTorch Profiler

**Goal:** Profile a transformer-based model (GPT-2) to understand GPU utilization, memory patterns, and operator performance in large language models.

- Fine-tunes GPT-2 small on a text classification task
- Captures detailed execution traces using `torch.profiler`
- Exports .pt.trace.json for Perfetto analysis
- Analyzes attention layers, matrix multiplications, and memory bottlenecks

**Key findings:**
- Attention mechanisms dominate compute time
- Memory usage spikes during forward/backward passes
- Optimizer steps show high CUDA kernel activity

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
