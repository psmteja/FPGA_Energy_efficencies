#!/usr/bin/env python3
"""
Generates three plots:
- roofline_scatter.png : Arithmetic Intensity vs. GFLOPs/s achieved (or proxies), sized by latency
- energy_breakdown.png : Stacked bar by op (E_compute vs E_memory)
- critical_path_gantt.png : Gantt-style plot of ops on critical path (index only)
Rules: use matplotlib, one chart per figure, no seaborn, no explicit colors.
Usage:
  python 06_plots.py
"""
import argparse, pandas as pd, numpy as np
import matplotlib.pyplot as plt

def plot_roofline(df):
    plt.figure()
    # Approx achieved GFLOPs/s proxy: flops / latency
    flops = df["flops"].fillna(0).values
    lat_ms = df["latency_ms"].fillna(1e-3).values
    gflops_s = (flops / (lat_ms/1000.0)) / 1e9
    ai = df["AI_now"].replace([np.inf, -np.inf], np.nan).fillna(0).values
    sizes = 20 + 200 * (lat_ms / (lat_ms.max() if lat_ms.max()>0 else 1.0))
    plt.scatter(ai, gflops_s, s=sizes)
    plt.xlabel("Arithmetic Intensity (FLOPs / Byte)")
    plt.ylabel("Approx. Achieved GFLOPs/s")
    plt.title("Roofline Scatter (proxy)")
    plt.grid(True)
    plt.savefig("roofline_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_energy(df):
    plt.figure()
    ops = df["op_name"].astype(str).values
    E_comp = df["E_compute_J"].fillna(0).values
    E_mem  = df["E_memory_J"].fillna(0).values
    idx = np.arange(len(ops))
    plt.bar(idx, E_comp)
    plt.bar(idx, E_mem, bottom=E_comp)
    plt.xticks(idx[::max(1,len(idx)//10)], [o[:12] for o in ops[::max(1,len(idx)//10)]] , rotation=45, ha="right")
    plt.ylabel("Energy (J)")
    plt.title("Energy Breakdown per Operator")
    plt.tight_layout()
    plt.savefig("energy_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_gantt(cp_ops, lat_map):
    plt.figure()
    y = np.arange(len(cp_ops))
    starts = []
    lasts = []
    t = 0.0
    for op in cp_ops:
        d = float(lat_map.get(op, 0.0))
        starts.append(t)
        lasts.append(d)
        t += d
    plt.barh(y, lasts, left=starts)
    plt.yticks(y, [o[:30] for o in cp_ops])
    plt.xlabel("Time (ms)")
    plt.title("Critical Path (Gantt Approx)")
    plt.tight_layout()
    plt.savefig("critical_path_gantt.png", dpi=150, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops_analysis", default="ops_analysis.csv")
    ap.add_argument("--ops_energy", default="ops_energy.csv")
    ap.add_argument("--ops_profile", default="ops_profile.csv")
    ap.add_argument("--critical_path", default="critical_path.csv")
    args = ap.parse_args()

    df_a = pd.read_csv(args.ops_analysis)
    df_e = pd.read_csv(args.ops_energy) if Path(args.ops_energy).exists() else None
    df_p = pd.read_csv(args.ops_profile) if Path(args.ops_profile).exists() else None
    df_c = pd.read_csv(args.critical_path) if Path(args.critical_path).exists() else None

    if df_a is not None:
        plot_roofline(df_a)

    if df_e is not None:
        plot_energy(df_e)

    if df_c is not None and df_p is not None:
        lat_map = {r["op_name"]: r["latency_ms"] for _,r in df_p.iterrows()}
        cp_ops = list(df_c["op_name"].values)
        plot_gantt(cp_ops, lat_map)

    print("Wrote roofline_scatter.png, energy_breakdown.png, critical_path_gantt.png (where applicable)")

if __name__ == "__main__":
    from pathlib import Path
    main()
