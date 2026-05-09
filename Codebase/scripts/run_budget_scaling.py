#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import pandas as pd
from src.cider.runner import run_campaign

BUDGET_SPECS = {
    16: (1, 16),
    24: (3, 8),
    32: (2, 16),
    48: (3, 16),
    64: (4, 16),
    96: (6, 16),
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    assay_ids=None
    if cfg.get('assay_ids_csv'):
        assay_ids=pd.read_csv(cfg['assay_ids_csv'])['DMS_id'].astype(str).tolist()
    out=Path(args.out)
    for b,(r,bs) in BUDGET_SPECS.items():
        o=out/f'budget_{b}'
        run_campaign(
            manifest_csv=Path(cfg['manifest_csv']),
            processed_dir=Path(cfg['processed_dir']),
            out_dir=o,
            seeds=cfg['seeds'],
            methods=cfg['methods'],
            assay_ids=assay_ids,
            rounds=r,
            batch=bs,
        )

if __name__=='__main__':
    main()
