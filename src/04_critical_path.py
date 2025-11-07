#!/usr/bin/env python3
"""
Computes critical-path time using ONNX DAG and measured per-op latency from ops_profile.csv.
Outputs critical_path.csv with the chain and total time.
Usage:
  python 04_critical_path.py --model resnet50-fp32.onnx --ops_profile ops_profile.csv
"""
import argparse, pandas as pd
from utils_onnxgraph import load_onnx, get_parents_map, topo_sort_nodes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ops_profile", default="ops_profile.csv")
    args = ap.parse_args()

    model = load_onnx(args.model)
    parents = get_parents_map(model)
    order = topo_sort_nodes(model)

    prof = pd.read_csv(args.ops_profile)
    # build latency map; fallback to small epsilon if missing
    lat_ms = {row["op_name"]: float(row["latency_ms"]) for _,row in prof.iterrows()}

    earliest_finish = {}
    choice_parent = {}
    for n in order:
        preds = parents.get(n, [])
        estart = max((earliest_finish.get(p, 0.0) for p in preds), default=0.0)
        efinish = estart + float(lat_ms.get(n, 0.0))
        earliest_finish[n] = efinish
        # choose the parent that gave the max
        if preds:
            pbest = max(preds, key=lambda p: earliest_finish.get(p, 0.0))
        else:
            pbest = None
        choice_parent[n] = pbest

    # find sink with maximum finish time
    end_node = max(earliest_finish, key=lambda k: earliest_finish[k])
    total_time = earliest_finish[end_node]

    # backtrack
    chain = []
    cur = end_node
    while cur is not None:
        chain.append(cur)
        cur = choice_parent[cur]
    chain = list(reversed(chain))

    pd.DataFrame({"op_name": chain}).to_csv("critical_path.csv", index=False)
    with open("critical_path_time_ms.txt","w") as f:
        f.write(f"{total_time:.6f}\n")
    print("Wrote critical_path.csv and critical_path_time_ms.txt (%.3f ms)" % total_time)

if __name__ == "__main__":
    main()
