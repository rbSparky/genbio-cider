# CIDER-Bench (Curated Benchmark Access)

This folder provides direct access to the benchmark used in the paper.

## What is included
- `metadata/`:
  - curated assay manifests and safety-filter metadata used to define benchmark splits.
- `configs/`:
  - benchmark run configs for main, hard split, LLM comparison, ablations, and budget scaling.

## Canonical benchmark definition
The benchmark is defined by:
1. curated assay manifests in `metadata/`
2. protocol settings in `configs/*.yaml`
3. campaign objective and metrics in `main.tex` (Method/Benchmark sections)

## Quick start
From `final_codebase/`:

```bash
python3 scripts/curate_assays.py --out data/processed --manifest data/metadata/assay_manifest.csv
python3 scripts/run_campaigns.py --config configs/hard_20assay.yaml --out results/runs/hard_20assay
```

If you already have curated processed data, use the provided manifests/configs directly.
