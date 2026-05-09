#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from src.cider.runner import run_campaign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    assay_ids = None
    if cfg.get("assay_ids_csv"):
        import pandas as pd
        assay_ids = pd.read_csv(cfg["assay_ids_csv"])["DMS_id"].astype(str).tolist()
    run_campaign(
        manifest_csv=Path(cfg["manifest_csv"]),
        processed_dir=Path(cfg["processed_dir"]),
        out_dir=Path(args.out),
        seeds=cfg["seeds"],
        methods=cfg["methods"],
        assay_ids=assay_ids,
        rounds=int(cfg.get("rounds", 3)),
        batch=int(cfg.get("batch", 16)),
        obs_noise_std=float(cfg.get("obs_noise_std", 0.0)),
        obs_noise_hetero=float(cfg.get("obs_noise_hetero", 0.0)),
    )

if __name__ == "__main__":
    main()
