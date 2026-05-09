#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path

MAIN = Path('results/runs/hard_20assay/summary.csv')
ABL = Path('results/runs/ablations_hard/summary.csv')
LLM = Path('results/runs/llm_comparison_hard/summary.csv')

rows = list(csv.DictReader(MAIN.open()))
ass = defaultdict(lambda: defaultdict(list))
for r in rows:
    ass[r['DMS_id']][r['method']].append(float(r['top1_success']))

methods = sorted(set(r['method'] for r in rows))
mean = {}
for m in methods:
    sub=[r for r in rows if r['method']==m]
    mean[m]={
        'top1': sum(float(x['top1_success']) for x in sub)/len(sub),
        'top10': sum(float(x['top10_hits']) for x in sub)/len(sub),
        'regret': sum(float(x['normalized_regret']) for x in sub)/len(sub),
        'unique': sum(float(x['unique_loci']) for x in sub)/len(sub),
        'invalid': sum(float(x['invalid_action_rate']) for x in sub)/len(sub),
    }

L='cider_gpt_oss_20b'
B=max((m for m in methods if not m.startswith('cider_')), key=lambda m: mean[m]['top1'])
Lstar=mean[L]['top1']
Bstar=mean[B]['top1']

paired=[]
for a in ass:
    if L in ass[a] and B in ass[a]:
        paired.append(sum(ass[a][L])/len(ass[a][L]) - sum(ass[a][B])/len(ass[a][B]))

frontier = {}
if LLM.exists():
    lrows=list(csv.DictReader(LLM.open()))
    fm='llm_grok41fast_only'
    sub=[r for r in lrows if r['method']==fm]
    if sub:
        ftop1=sum(float(x['top1_success']) for x in sub)/len(sub)
        frontier={
            'method': fm,
            'top1': ftop1,
            'delta_vs_local': Lstar-ftop1,
            'strong_superiority_5pt': Lstar >= ftop1 + 0.05,
            'cost_total_usd': sum(float(x.get('cost_usd',0.0)) for x in sub),
            'invalid_rate': sum(float(x.get('invalid_action_rate',0.0)) for x in sub)/len(sub),
        }

gate={
    'quality': {
        'L_method': L,
        'best_non_cider': B,
        'L_top1': Lstar,
        'B_top1': Bstar,
        'delta': Lstar-Bstar,
        'noninferior_2pt': Lstar >= Bstar - 0.02,
        'superior_5pt_vs_best_non_cider': Lstar >= Bstar + 0.05,
        'paired_mean_delta': sum(paired)/len(paired) if paired else None,
    },
    'efficiency': {
        'invalid_rate_L': mean[L]['invalid'],
        'invalid_rate_ok': mean[L]['invalid'] <= 0.005,
        'compute_profile': 'cpu_only',
    },
    'novelty': {
        'has_ablation_file': ABL.exists(),
        'claims': [
            'conformal correction',
            'top-tail objective',
            'information-directed term',
            'constrained controller + audit'
        ]
    },
    'frontier_no_method': frontier,
}
Path('results/tables/claim_gate.json').write_text(json.dumps(gate, indent=2))
print(json.dumps(gate, indent=2))
