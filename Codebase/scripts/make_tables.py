#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ORDER = [
"random","plm_greedy","plm_diverse","ucb","ei","thompson","prior_first_al","folde","cider_fixed","cider_qwen35_4b","cider_qwen3_8b","cider_gpt_oss_20b"
]

def f(x):
    return f"{x:.3f}"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--tables", required=True); ap.add_argument("--tex", default="paper_table_snippets.tex")
    args=ap.parse_args()
    t = pd.read_csv(Path(args.tables)/"table_main.csv")
    t["order"] = t.method.map({m:i for i,m in enumerate(ORDER)})
    t = t.sort_values("order")
    lines = []
    for _, r in t.iterrows():
        lines.append(f"{r.method} & {f(r.top1_success)} & {f(r.top10_hits)} & {f(r.best_percentile)} & {f(r.normalized_regret)} & {f(r.unique_loci)} & {f(r.invalid_action_rate)} \\")
    Path(args.tex).write_text("\n".join(lines)+"\n")

if __name__=='__main__':
    main()
