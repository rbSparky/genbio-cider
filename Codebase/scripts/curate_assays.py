#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from src.cider.safety import is_safe_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--min_candidates", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_repo_files("OATML-Markslab/ProteinGym_v1", repo_type="dataset") if f.startswith("DMS_substitutions/") and f.endswith(".parquet")]
    dfs = []
    for f in files:
        p = hf_hub_download("OATML-Markslab/ProteinGym_v1", repo_type="dataset", filename=f)
        dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, axis=0, ignore_index=True)
    required = ["DMS_id", "mutant", "mutated_sequence", "DMS_score"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing columns: {miss}")

    df = df.dropna(subset=["DMS_score", "DMS_id", "mutant"]).copy()
    manifests = []
    for dms_id, g in df.groupby("DMS_id"):
        g = g.groupby("mutant", as_index=False).agg({"mutated_sequence":"first","DMS_score":"mean"})
        text = " ".join([str(dms_id)])
        safe = is_safe_text(text)
        q99_count = int((g.DMS_score >= g.DMS_score.quantile(0.99)).sum()) if len(g) else 0
        use = int(safe and len(g) >= args.min_candidates and q99_count >= 10)
        g = g.reset_index(drop=True)
        g["candidate_id"] = [f"{dms_id}_cand_{i:05d}" for i in range(len(g))]
        g.to_parquet(out / f"{dms_id}.parquet", index=False)
        manifests.append({"DMS_id": dms_id, "n": len(g), "q99_count": q99_count, "safe": int(safe), "use": use})
    m = pd.DataFrame(manifests).sort_values("n", ascending=False)
    m.to_csv(args.manifest, index=False)
    print(f"wrote {args.manifest} with {len(m)} assays, usable={(m.use==1).sum()}")

if __name__ == "__main__":
    main()
