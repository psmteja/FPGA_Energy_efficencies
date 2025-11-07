#!/usr/bin/env python3
"""
Infers tensor shapes and estimates per-op activation/weight bytes and FLOPs.
Usage:
  python 02_shapes_flops_bytes.py --model resnet50-fp32.onnx
"""
import argparse, pandas as pd, onnx
from utils_onnxgraph import load_onnx, infer_shapes_map, bytes_of, tensor_shape, estimate_flops

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    model = load_onnx(args.model)
    nodes, value_info, inits = infer_shapes_map(model)

    rows = []
    for n in nodes:
        name = n.name or f"{n.op_type}"
        in_shapes = [tensor_shape(value_info[i]) for i in n.input if i in value_info]
        out_shapes = [tensor_shape(value_info[o]) for o in n.output if o in value_info]
        in_bytes = sum(bytes_of(value_info[i]) for i in n.input if i in value_info)
        out_bytes = sum(bytes_of(value_info[o]) for o in n.output if o in value_info)
        # weight bytes (initializers consumed by this node)
        w_bytes = 0
        for i in n.input:
            if i in inits:
                init = inits[i]
                if init.data_type == onnx.TensorProto.FLOAT:
                    elbytes = 4
                else:
                    elbytes = 4
                count = 1
                for d in init.dims:
                    count *= int(d) if d else 0
                w_bytes += count * elbytes
        # FLOPs estimate
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        flops = estimate_flops(n.op_type, attrs, in_shapes, out_shapes)
        rows.append({
            "op_name": name,
            "op_type": n.op_type,
            "in_bytes": in_bytes,
            "out_bytes": out_bytes,
            "weight_bytes": w_bytes,
            "flops": flops
        })
    df = pd.DataFrame(rows)
    df.to_csv("ops_metrics.csv", index=False)
    print("Wrote ops_metrics.csv")

if __name__ == "__main__":
    main()
