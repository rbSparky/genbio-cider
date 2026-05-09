
## Included

- `benchmark/` easy-access benchmark package (manifests + configs + usage notes).
- `main.tex`, bibliography/style files needed to compile the paper.
- `figures/` used by the paper.
- `src/cider/` core implementation.
- Core scripts only:
  - `curate_assays.py`
  - `run_campaigns.py`
  - `run_llm_comparison.py`
  - `run_budget_scaling.py`
  - `aggregate_results.py`
  - `make_tables.py`
  - `make_figures_real.py`
  - `claim_gate.py`
- `configs/*.yaml` run configurations.
- `results/tables/` and `results/figures/` aggregated artifacts only.
- `data/metadata/` curation metadata only.


## Compile paper
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
