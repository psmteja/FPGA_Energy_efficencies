#!/usr/bin/env python3
"""
Estimates per-op energy from latency and power, and splits energy into compute vs memory using roofline proportions.
Usage:
  python 05_energy_estimate.py --ops_analysis ops_analysis.csv --avg_power_w 65.0
Optional:
  --compute_bias 0.7    # If op is compute-bound, fraction to attribute to compute (else memory-bound uses 1-compute_bias)
"""
import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops_analysis", default="ops_analysis.csv")
    ap.add_argument("--avg_power_w", type=float, required=True)
    ap.add_argument("--compute_bias", type=float, default=0.7)
    args = ap.parse_args()

    df = pd.read_csv(args.ops_analysis)

    # If you have measured per-op latency, prefer that; else fall back to max(t_compute, t_mem)
    lat = df["latency_ms"].fillna(np.maximum(df["t_compute_ms"].fillna(0), df["t_mem_ms"].fillna(0)))
    df["latency_ms_eff"] = lat

    E_total = (args.avg_power_w * (lat / 1000.0))  # Joules
    frac_compute = np.where(df["bound"] == "compute", args.compute_bias,
                     np.where(df["bound"] == "memory", 1.0 - args.compute_bias, 0.5))
    df["E_compute_J"] = E_total * frac_compute
    df["E_memory_J"]  = E_total * (1.0 - frac_compute)
    df["E_total_J"]   = E_total

    df.to_csv("ops_energy.csv", index=False)
    summary = df["E_total_J"].sum()
    with open("energy_total_J.txt","w") as f:
        f.write(f"{summary:.6f}\n")
    print("Wrote ops_energy.csv, total energy=%.6f J" % summary)

if __name__ == "__main__":
    main()
