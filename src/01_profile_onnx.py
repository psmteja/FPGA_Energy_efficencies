#!/usr/bin/env python3
"""
Runs ONNX Runtime with profiling enabled and emits profile.json and ops_profile.csv
Usage:
  python 01_profile_onnx.py --model resnet50-fp32.onnx --runs 5 --provider CPUExecutionProvider
"""
import argparse, json, time
import numpy as np
import onnxruntime as ort
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--input-name", default=None, help="Override input name (auto-detect if None)")
    ap.add_argument("--input-shape", default="1,3,224,224", help="Comma-separated shape for random input")
    args = ap.parse_args()

    sess_opts = ort.SessionOptions()
    sess_opts.enable_profiling = True
    sess = ort.InferenceSession(args.model, sess_options=sess_opts, providers=[args.provider])

    if args.input_name is None:
        input_name = sess.get_inputs()[0].name
    else:
        input_name = args.input_name

    shape = tuple(int(s) for s in args.input_shape.split(","))
    x = np.random.randn(*shape).astype(np.float32)

    # warmup
    sess.run(None, {input_name: x})
    # profiled runs
    for _ in range(args.runs):
        sess.run(None, {input_name: x})

    profile_file = sess.end_profiling()
    with open(profile_file, "r") as f:
        prof = json.load(f)

    rows = []
    for e in prof:
        if "args" in e and "op_name" in e["args"]:
            name = e["args"].get("op_name", "")
            cat = e.get("cat", "")
            dur = e.get("dur", 0.0) / 1e6  # ns -> ms
            ts  = e.get("ts", 0.0) / 1e6   # ns -> ms (relative)
            op_type = e["args"].get("op_type", "")
            rows.append({"op_name": name, "op_type": op_type, "category": cat, "start_ms": ts, "latency_ms": dur})
    df = pd.DataFrame(rows)
    df = df.groupby(["op_name","op_type"], as_index=False)["latency_ms"].median().sort_values("latency_ms", ascending=False)
    df.to_csv("ops_profile.csv", index=False)
    print("Wrote ops_profile.csv")
    print("Raw profile JSON:", profile_file)

if __name__ == "__main__":
    main()
