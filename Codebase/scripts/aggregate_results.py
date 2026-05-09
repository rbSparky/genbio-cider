#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = Path(args.runs)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    allcsv = list(runs.glob("*/summary.csv"))
    df = pd.concat([pd.read_csv(p) for p in allcsv], ignore_index=True)
    agg = df.groupby("method", as_index=False).agg({
        "top1_success":"mean","top10_hits":"mean","best_percentile":"mean","normalized_regret":"mean","unique_loci":"mean","invalid_action_rate":"mean","cost_usd":"mean"
    })
    df.to_csv(out / "all_results.csv", index=False)
    agg.to_csv(out / "table_main.csv", index=False)

if __name__ == "__main__":
    main()
