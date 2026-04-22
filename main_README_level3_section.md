### Level 3 — Transformer profiling with PyTorch Profiler

**Goal:** Profile a transformer-based model (GPT-2) to understand GPU utilization, memory patterns, and operator performance in large language models.

- Fine-tunes GPT-2 small on a text classification task
- Captures detailed execution traces using 	orch.profiler
- Exports .pt.trace.json for Perfetto analysis
- Analyzes attention layers, matrix multiplications, and memory bottlenecks

**Key findings:**
- Attention mechanisms dominate compute time
- Memory usage spikes during forward/backward passes
- Optimizer steps show high CUDA kernel activity

📁 [Level3_transformer_profiling/](./Level3_transformer_profiling/)
