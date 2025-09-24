# Inference Benchmark Report

Generated: Thu Sep 25 04:37:49 2025

## Overview

This report summarizes inference accuracy, latency and throughput across different backends (PyTorch CPU, PyTorch GPU, TensorRT FP32/FP16/INT8 where available).

## Results Summary

### pytorch_cpu

- Accuracy: **0.0560**

- Latency per image (ms): 14.45

- Throughput (img/s): 1081.40

### pytorch_gpu

- Accuracy: **0.0560**

- Latency per image (ms): 0.11

- Throughput (img/s): 141158.22

### trt_fp32

- Error: `'tensorrt_bindings.tensorrt.Builder' object has no attribute 'build_engine'`

### trt_fp16

- Error: `'tensorrt_bindings.tensorrt.Builder' object has no attribute 'build_engine'`

### trt_int8

- Error: `'tensorrt_bindings.tensorrt.Builder' object has no attribute 'build_engine'`

## Automated Observations

- GPU vs CPU speedup (latency): ~130.53x


## Plots and Confusion Matrices

- `cm_cpu.png`, `cm_gpu.png`, `cm_trt_fp32.png`, etc. contain confusion matrices.

- `mis_cpu.png`, `mis_gpu.png`, `mis_trt_fp32.png`, etc. contain example misclassified images.
