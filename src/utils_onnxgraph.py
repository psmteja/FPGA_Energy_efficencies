#!/usr/bin/env python3
"""
Utility helpers for ONNX graph analysis:
- load_onnx
- topo_sort_nodes
- build_name_maps
- get_parents_map
- infer_shapes_map
- estimate_flops(op_type, attrs, input_shapes, output_shapes)
"""
import onnx
from onnx import shape_inference
from typing import Dict, List, Tuple, Any
import numpy as np

def load_onnx(path: str) -> onnx.ModelProto:
    return onnx.load(path)

def build_name_maps(model: onnx.ModelProto):
    g = model.graph
    value_info = {v.name: v for v in list(g.input)+list(g.value_info)+list(g.output)}
    initializers = {init.name: init for init in g.initializer}
    nodes = list(g.node)
    return nodes, value_info, initializers

def infer_shapes_map(model: onnx.ModelProto):
    inferred = shape_inference.infer_shapes(model)
    return build_name_maps(inferred)

def _dims_of(value_info) -> List[int]:
    ttype = value_info.type.tensor_type
    if not ttype.HasField("shape"):
        return []
    dims = []
    for d in ttype.shape.dim:
        if d.dim_value:
            dims.append(int(d.dim_value))
        else:
            dims.append(0)  # dynamic; fill later or treat as zero
    return dims

def bytes_of(value_info) -> int:
    ttype = value_info.type.tensor_type
    if not ttype.HasField("shape"):
        return 0
    if ttype.elem_type != onnx.TensorProto.FLOAT:
        # Best-effort: assume 4 bytes if unknown type is present
        elem_bytes = 4
    else:
        elem_bytes = 4
    dims = _dims_of(value_info)
    if 0 in dims:  # dynamic => unknown
        return 0
    n = int(np.prod(dims)) if dims else 0
    return n * elem_bytes

def tensor_shape(value_info) -> List[int]:
    return _dims_of(value_info)

def get_parents_map(model: onnx.ModelProto) -> Dict[str, List[str]]:
    """Return a mapping node_name -> list of parent node_names (by data dependency)."""
    g = model.graph
    # Map output tensor -> producer node name
    producer = {}
    for n in g.node:
        for out in n.output:
            producer[out] = n.name or f"{n.op_type}__{id(n)}"
    parents = {}
    for n in g.node:
        name = n.name or f"{n.op_type}__{id(n)}"
        ps = []
        for inp in n.input:
            if inp in producer:
                ps.append(producer[inp])
        parents[name] = ps
    return parents

def topo_sort_nodes(model: onnx.ModelProto) -> List[str]:
    parents = get_parents_map(model)
    # Kahn's algorithm
    indeg = {n: 0 for n in parents.keys()}
    for n, ps in parents.items():
        for p in ps:
            indeg[n] += 1
    S = [n for n,d in indeg.items() if d == 0]
    order = []
    # Build child map for efficiency
    children = {n: [] for n in parents}
    for n, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(n)
    while S:
        v = S.pop()
        order.append(v)
        for w in children.get(v, []):
            indeg[w] -= 1
            if indeg[w] == 0:
                S.append(w)
    # Append any remaining nodes (in case of isolated components)
    for n in parents:
        if n not in order:
            order.append(n)
    return order

def _int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def estimate_flops(op_type: str, attrs: Dict[str, Any], input_shapes: List[List[int]], output_shapes: List[List[int]]) -> float:
    """
    Very rough FLOPs estimate for common ops (FP32 MAC = 2 FLOPs).
    Returns FLOPs per invocation (not per second).
    """
    op = op_type.lower()
    if op in ("gemm", "matmul"):
        # infer MxK x KxN
        if len(input_shapes) >= 2 and len(input_shapes[0]) >= 2 and len(input_shapes[1]) >= 2:
            M = input_shapes[0][-2]
            K = input_shapes[0][-1]
            N = input_shapes[1][-1]
            if 0 in (M, K, N): return 0.0
            macs = M * K * N
            return 2.0 * macs
        return 0.0
    if op == "conv":
        # assume NCHW
        # Inputs: X [N,Cin,H,W], W [Cout,Cin,kH,kW]
        if len(input_shapes) >= 2 and len(input_shapes[0]) == 4 and len(input_shapes[1]) == 4:
            N, Cin, H, W = input_shapes[0]
            Cout, CinW, kH, kW = input_shapes[1]
            if Cin == 0: Cin = CinW  # fallback
            if 0 in (N, Cin, H, W, Cout, kH, kW): return 0.0
            # Output shape can include stride/padding; if provided, use it
            if output_shapes and len(output_shapes[0]) == 4:
                _, _, Ho, Wo = output_shapes[0]
            else:
                Ho, Wo = H, W
            macs = N * Cout * Ho * Wo * Cin * kH * kW
            return 2.0 * macs
        return 0.0
    # Elementwise ops: approx 1 flop per element (very rough)
    if output_shapes and output_shapes[0]:
        numel = 1
        for d in output_shapes[0]:
            if d == 0: return 0.0
            numel *= d
        return float(numel)
    return 0.0
