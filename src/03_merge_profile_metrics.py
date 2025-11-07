#!/usr/bin/env python3
"""
Merges ops_profile.csv and ops_metrics.csv, computes arithmetic intensity, t_compute, t_mem, and bound.
Usage:
  python 03_merge_profile_metrics.py --peak-flops 1.0e12 --mem-bw 5.0e10
"""
import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops_profile", default="ops_profile.csv")
    ap.add_argument("--ops_metrics", default="ops_metrics.csv")
    ap.add_argument("--peak-flops", type=float, required=True, help="Host peak FP32 FLOPs, e.g., 1.0e12")
    ap.add_argument("--mem-bw", type=float, required=True, help="Host sustained memory bandwidth in bytes/s, e.g., 5.0e10")
    args = ap.parse_args()

    p = pd.read_csv(args.ops_profile)
    m = pd.read_csv(args.ops_metrics)

    df = pd.merge(m, p, on=["op_name","op_type"], how="left")
    df["bytes_total"] = df[["in_bytes","out_bytes"]].fillna(0).sum(axis=1)
    # If you assume weights streamed from DRAM each run, include them:
    df["bytes_plus_w"] = df["bytes_total"] + df["weight_bytes"].fillna(0)

    # Arithmetic intensity with/without weights
    df["AI_now"] = df["flops"] / df["bytes_total"].replace(0, np.nan)
    df["AI_plus_w"] = df["flops"] / df["bytes_plus_w"].replace(0, np.nan)

    # Time estimates from ceilings
    df["t_compute_ms"] = (df["flops"] / args.peak_flops) * 1e3
    df["t_mem_ms"]     = (df["bytes_total"] / args.mem_bw) * 1e3

    # Bound classification (based on larger of the two)
    def bound_row(r):
        if np.isnan(r["t_compute_ms"]) or np.isnan(r["t_mem_ms"]):
            return "unknown"
        return "compute" if r["t_compute_ms"] >= r["t_mem_ms"] else "memory"
    df["bound"] = df.apply(bound_row, axis=1)

    df.to_csv("ops_analysis.csv", index=False)
    print("Wrote ops_analysis.csv")

if __name__ == "__main__":
    main()
