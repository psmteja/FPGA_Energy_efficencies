# ONNX Inference Profiling and FPGA Co-Design Analysis

## Overview

This repository provides a reproducible workflow for analyzing **ONNX inference performance** and guiding **FPGA hardware/software co-design**.  
It estimates:

- **Critical-path latency** across the model DAG,
- **Energy breakdown** between compute and data movement,
- **Candidate FPGA kernels** for offload, and
- **Amdahl’s Law-based end-to-end speedup**.

The workflow uses only a **forward pass** (no training data required).

---

## 1. Inputs

- **Model:** FP32 ONNX file (with weights).  
  Example: `resnet50-fp32.onnx` (opset 12).
- **Sample Input:** One or more random or real tensors matching model I/O.
- **Hardware:** CPU/GPU host (for profiling), optional FPGA estimator.

---

## 2. I/O Specification Example

