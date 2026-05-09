#!/usr/bin/env python3
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

main = list(csv.DictReader(open("results/runs/hard_20assay/summary.csv")))
abl = list(csv.DictReader(open("results/runs/ablations_hard/summary.csv")))
llm = list(csv.DictReader(open("results/runs/llm_comparison_hard/summary.csv")))


def vals(rows, method, key):
    return np.array([float(r[key]) for r in rows if r["method"] == method], dtype=float)


def mean_std(rows, method, key):
    x = vals(rows, method, key)
    if len(x) == 0:
        return np.nan, np.nan
    return float(x.mean()), float(x.std())


# Figure 2: single-column budget curve (no std bars)
budgets = [16, 24, 32, 48]
methods = [
    ("random", "Random"),
    ("plm_greedy", "PLM greedy"),
    ("folde_lite", "FolDE"),
    ("cider_fixed", "CIDER no-agent"),
    ("cider_gpt_oss_20b", "CIDER-Agent"),
]
x = np.array(budgets, dtype=float)
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for m, label in methods:
    ym = []
    for b in budgets:
        rows = list(csv.DictReader(open(f"results/runs/budget_scaling_hard/budget_{b}/summary.csv")))
        sub = np.array([float(r["top1_success"]) for r in rows if r["method"] == m], dtype=float)
        ym.append(float(sub.mean()))
    ax.plot(x, ym, marker="o", linewidth=1.8, label=label)
ax.set_xlabel("Budget (queries)")
ax.set_ylabel("Top-1% success")
ax.set_xticks(budgets)
ax.set_ylim(0.2, 1.02)
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=6, ncol=1, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig2_discovery_curves.png", dpi=240)
fig.savefig(OUT / "fig2_discovery_curves.pdf")
plt.close(fig)


# Figure 3: assay-wise ranked deltas (clearer than sparse histogram)
assays = sorted(set(r["DMS_id"] for r in main))
deltas = []
for a in assays:
    c = np.mean([float(r["top1_success"]) for r in main if r["DMS_id"] == a and r["method"] == "cider_gpt_oss_20b"])
    f = np.mean([float(r["top1_success"]) for r in main if r["DMS_id"] == a and r["method"] == "folde_lite"])
    deltas.append((a, c - f))
vals_delta = np.array([d[1] for d in deltas], dtype=float)
wins = int(np.sum(vals_delta > 0))
ties = int(np.sum(vals_delta == 0))
loss = int(np.sum(vals_delta < 0))
order = np.argsort(vals_delta)
vals_sorted = vals_delta[order]
xidx = np.arange(len(vals_sorted))
fig, ax = plt.subplots(figsize=(4.2, 2.9))
ax.axhline(0.0, color="black", linewidth=1.0)
for i, v in enumerate(vals_sorted):
    color = "#2ca02c" if v > 0 else ("#d62728" if v < 0 else "#7f7f7f")
    ax.vlines(i, 0, v, color=color, linewidth=2.0, alpha=0.9)
    ax.plot(i, v, "o", color=color, markersize=4)
ax.set_xlabel("Assays (sorted by delta)")
ax.set_ylabel("Top-1% success delta")
ax.set_xticks([])
ax.set_ylim(-1.1, 1.1)
ax.text(
    0.02,
    0.96,
    f"mean={np.mean(vals_delta):.3f}, wins/ties/loss={wins}/{ties}/{loss}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=7,
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
)
fig.tight_layout()
fig.savefig(OUT / "fig3_paired_wins_v2.png", dpi=240)
fig.savefig(OUT / "fig3_paired_wins_v2.pdf")
plt.close(fig)


# Figure 5: quality-diversity frontier (no std bars)
meths = ["random", "plm_greedy", "plm_diverse", "folde_lite", "cider_fixed", "cider_gpt_oss_20b"]
labels = {
    "random": "Random",
    "plm_greedy": "PLM greedy",
    "plm_diverse": "PLM + div",
    "folde_lite": "FolDE",
    "cider_fixed": "CIDER no-agent",
    "cider_gpt_oss_20b": "CIDER-Agent",
}
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for m in meths:
    xm, _ = mean_std(main, m, "unique_loci")
    ym, _ = mean_std(main, m, "top10_hits")
    if m == "cider_gpt_oss_20b":
        ax.plot(xm, ym, "o", ms=7, color="#1f77b4", label=labels[m], zorder=5)
    else:
        ax.plot(xm, ym, "o", ms=5, alpha=0.8, label=labels[m])
    ax.text(xm + 0.2, ym + 0.05, labels[m], fontsize=6)
ax.set_xlabel("Unique loci")
ax.set_ylabel("Top-10 hits")
ax.grid(alpha=0.2)
ax.margins(x=0.15, y=0.15)
fig.tight_layout()
fig.savefig(OUT / "fig5_diversity_frontier.png", dpi=240)
fig.savefig(OUT / "fig5_diversity_frontier.pdf")
plt.close(fig)


# Figure 7: reliability + efficiency (real means)
labels7 = ["Top-1", "Unique loci/40", "Cost USD"]
vals7 = {
    "LLM-only (Grok)": [
        mean_std(llm, "llm_grok41fast_only", "top1_success")[0],
        mean_std(llm, "llm_grok41fast_only", "unique_loci")[0] / 40.0,
        mean_std(llm, "llm_grok41fast_only", "cost_usd")[0],
    ],
    "Local LM direct": [
        mean_std(abl, "local_lm_direct", "top1_success")[0],
        mean_std(abl, "local_lm_direct", "unique_loci")[0] / 40.0,
        mean_std(abl, "local_lm_direct", "cost_usd")[0],
    ],
    "CIDER fixed": [
        mean_std(main, "cider_fixed", "top1_success")[0],
        mean_std(main, "cider_fixed", "unique_loci")[0] / 40.0,
        mean_std(main, "cider_fixed", "cost_usd")[0],
    ],
    "CIDER-Agent": [
        mean_std(main, "cider_gpt_oss_20b", "top1_success")[0],
        mean_std(main, "cider_gpt_oss_20b", "unique_loci")[0] / 40.0,
        mean_std(main, "cider_gpt_oss_20b", "cost_usd")[0],
    ],
}
x7 = np.arange(len(labels7))
w = 0.2
fig, ax = plt.subplots(figsize=(5.2, 2.8))
for i, (k, v) in enumerate(vals7.items()):
    ax.bar(x7 + (i - 1.5) * w, v, w, label=k)
ax.set_xticks(x7, labels7, rotation=12, ha="right")
ax.set_ylim(0.0, 1.05)
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig7_agent_reliability.png", dpi=240)
fig.savefig(OUT / "fig7_agent_reliability.pdf")
plt.close(fig)

print("real figures updated")
