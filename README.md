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

```
Input:  (N, 3, 224, 224) float32, NCHW
        RGB image; resize shortest side→256, center-crop 224×224;
        normalize mean=255×[0.485,0.456,0.406], std=255×[0.229,0.224,0.225];
        transpose HWC→CHW.
Output: (N, 1000) float32 logits (ImageNet-1k)
```

---

## 3. Profiling Setup

```bash
pip install onnx onnxruntime numpy pillow matplotlib
```

### Example Script
`profiling_example.py` (excerpt):

```python
import onnxruntime as ort, numpy as np

sess_opts = ort.SessionOptions()
sess_opts.enable_profiling = True
sess = ort.InferenceSession("resnet50-fp32.onnx", sess_options=sess_opts)
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
sess.run(None, {sess.get_inputs()[0].name: x})
profile_file = sess.end_profiling()
print("Profile JSON:", profile_file)
```

The generated JSON includes per-operator timing (`start_time`, `end_time`).

---

## 4. Operator-Level Metrics

Use ONNX shape inference to calculate FLOPs and memory traffic.

```python
from onnx import shape_inference, load
import numpy as np

m = shape_inference.infer_shapes(load("resnet50-fp32.onnx"))
vi = {v.name: v for v in list(m.graph.input)+list(m.graph.value_info)+list(m.graph.output)}

def bytes_of(v):
    t = v.type.tensor_type
    if not t.HasField("shape"): return 0
    dims = [d.dim_value for d in t.shape.dim if d.dim_value]
    return np.prod(dims) * 4  # FP32

for node in m.graph.node:
    in_b = sum(bytes_of(vi.get(n, 0)) for n in node.input)
    out_b = sum(bytes_of(vi.get(n, 0)) for n in node.output)
    print(node.name, node.op_type, in_b, out_b)
```

This produces approximate **activation** and **weight byte counts** per operator.

---

## 5. Roofline Model

1. Measure your host’s **peak FLOPs** and **memory bandwidth** (e.g., STREAM).  
2. Compute **Arithmetic Intensity (AI) = FLOPs / Bytes**.  
3. Determine whether each operator is **compute-bound** or **memory-bound** by comparing:
   ```
   t_compute = FLOPs / PeakFLOPs
   t_mem     = Bytes / Bandwidth
   ```

---

## 6. Critical Path Extraction

Using operator-level latencies and the ONNX DAG, compute the longest dependency chain:

```python
# pseudo
for node in topo_sorted_nodes:
    start = max(finish[p] for p in parents[node]) if parents[node] else 0
    finish[node] = start + latency[node]
critical_time = max(finish.values())
```

The chain yielding `critical_time` represents the **critical path**.

---

## 7. Energy Breakdown

### (A) Measured Energy (CPU/GPU)
Use power sensors:
- CPU: `RAPL` counters (`/sys/class/powercap/intel-rapl`)
- GPU: `nvidia-smi --query-gpu=power.draw --format=csv`

Estimate per-op energy:
```
E_op = Power_avg × Latency_op
```

### (B) FPGA Estimate
Use vendor power estimators (Vitis AI, Intel Quartus) to project `E_compute` and `E_mem` for synthesized kernels.

---

## 8. Candidate FPGA Kernels

Select operators that:
- Contribute >10–20% of total latency or energy,
- Are **regular** (dense GEMMs, Conv2d, MLPs),
- Have high **arithmetic intensity**, and
- Can be **fused** to avoid DRAM transfers.

Example candidates:
| Kernel | Ops | % Latency | % Energy | Bound | Comment |
|---------|-----|------------|-----------|--------|----------|
| K1 | Conv2d | 38 | 42 | Compute | Fuse BN/ReLU |
| K2 | GEMM (MLP) | 25 | 19 | Compute | Dense & regular |
| K3 | Attention (QKᵀ+Softmax+V) | 14 | 22 | Memory | Streamable |

---

## 9. Amdahl’s Law Speedup

Estimate overall system improvement:

```
S_total = 1 / ((1 - f) + f / S_k)
```
Where:
- `f` = fraction of time spent in FPGA-accelerated ops
- `S_k` = FPGA kernel speedup factor (from vendor tool or estimate)

Report both **latency** and **energy** improvements.

---

## 10. Deliverables

- **CSV Tables:**
  - `ops_profile.csv` → per-op latency, FLOPs, bytes, energy
  - `critical_path.csv` → list of ops on critical path
- **Plots (optional):**
  - Roofline scatter (AI vs GFLOPs)
  - Energy breakdown bar chart
  - Critical-path Gantt chart

---

## 11. Summary Pipeline

```
ONNX Model
   ↓
ONNX Runtime Profiling (latency)
   ↓
Shape Inference (bytes/FLOPs)
   ↓
Roofline Classification (compute vs memory)
   ↓
Critical Path Analysis
   ↓
Energy Attribution
   ↓
FPGA Kernel Selection
   ↓
Amdahl’s Law Speedup Estimate
```

---

## References

- ONNX Runtime Profiling: https://onnxruntime.ai  
- ONNX Model Zoo (ResNet-50): https://github.com/onnx/models  
- AMD Vitis AI: https://www.xilinx.com/products/design-tools/vitis.html  
- Intel OpenVINO: https://www.intel.com/openvino  
- FINN, hls4ml, VTA, DNNWeaver — Open FPGA research toolchains.

---

**Author:** Sai Puppala 

---

## Steps for Execution

### 1) profile latency
python 01_profile_onnx.py --model resnet50-fp32.onnx --runs 5

# 2) shapes, bytes, FLOPs
python 02_shapes_flops_bytes.py --model resnet50-fp32.onnx

# 3) merge + roofline tags (set your host ceilings)
python 03_merge_profile_metrics.py --peak-flops 1.0e12 --mem-bw 5.0e10

# 4) critical path
python 04_critical_path.py --model resnet50-fp32.onnx

# 5) energy estimate (set your measured avg platform power)
python 05_energy_estimate.py --avg_power_w 65.0

# 6) plots
python 06_plots.py
