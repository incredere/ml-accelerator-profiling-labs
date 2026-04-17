# Mixed Precision GPU Profiling (FP32 vs FP16)

This lab demonstrates how mixed precision training improves GPU performance compared to FP32 precision.

The experiment compares execution behavior using PyTorch AMP and analyzes traces using PyTorch Profiler and Perfetto.

## Objective

Understand how reduced precision improves training throughput and GPU efficiency.

## Experiment Setup

Framework: PyTorch  
Profiler: PyTorch Profiler  
Trace viewer: Perfetto  
Platform: Google Colab GPU  

## Experiments

FP32 baseline training  
Mixed precision training using AMP  

## Key Learnings

Mixed precision reduces compute time and memory usage.

AMP automatically selects safe operations to run in FP16.

GPU kernels execute faster when precision is reduced.

## Files

mixed_precision_profiling.ipynb → experiment notebook  
fp32.pt.trace.json → FP32 execution trace  
amp.pt.trace.json → AMP execution trace  

## Outcome

Demonstrated understanding of GPU optimization techniques using mixed precision training.

## Author

Harish Nagunuri
ML Infrastructure | GPU Performance | Kubernetes
