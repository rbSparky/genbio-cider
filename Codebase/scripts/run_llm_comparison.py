#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, time, re
from pathlib import Path
from typing import Any
import requests
import numpy as np
import pandas as pd

from src.cider.data import add_features
from src.cider.baselines import shortlist
from src.cider.metrics import top_threshold, top1_success, top10_hits

OR_BASE = os.environ.get('OPENROUTER_BASE_URL','https://openrouter.ai/api/v1')
OR_KEY = os.environ.get('OPENROUTER_API_KEY','')

METHOD_TO_MODEL = {
    'llm_gpt55_only': 'openai/gpt-5.5',
    'llm_claude_opus47_only': 'anthropic/claude-opus-4.7',
    'llm_gemini31pro_only': 'google/gemini-3.1-pro-preview',
    'llm_grok41fast_only': 'x-ai/grok-4.1-fast',
}

MAX_USD = 18.0
SHORTLIST_K = 128
MAX_REPAIRS = 2


def _extract_json(text: Any) -> dict[str, Any] | None:
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    for s in [text, re.sub(r'^```json\\n|```$','',text,flags=re.M)]:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    m = re.search(r'\{.*\}', text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, dict):
                if isinstance(x.get("text"), str):
                    parts.append(x["text"])
                elif isinstance(x.get("content"), str):
                    parts.append(x["content"])
            elif isinstance(x, str):
                parts.append(x)
        return "\n".join(parts)
    return str(content)


def _coerce_ids(obj: dict[str, Any]) -> list[str]:
    # Accept a few common field aliases for robustness across providers.
    for key in ["selected_candidate_ids", "selected_ids", "candidate_ids", "ids"]:
        if key in obj and isinstance(obj[key], list):
            out: list[str] = []
            for x in obj[key]:
                if isinstance(x, str):
                    out.append(x.strip())
                elif isinstance(x, dict):
                    cid = x.get("candidate_id") or x.get("id")
                    if cid is not None:
                        out.append(str(cid).strip())
                elif x is not None:
                    out.append(str(x).strip())
            return out
    return []

def _coerce_indices(obj: dict[str, Any]) -> list[int]:
    for key in ["selected_indices", "indices", "idx"]:
        v = obj.get(key)
        if isinstance(v, list):
            out = []
            for x in v:
                try:
                    out.append(int(x))
                except Exception:
                    continue
            return out
    return []

def _generation_cost(gen_id: str) -> float:
    if not gen_id:
        return 0.0
    try:
        r = requests.get(f"{OR_BASE}/generation", params={'id': gen_id}, headers={'Authorization': f'Bearer {OR_KEY}'}, timeout=60)
        if r.ok:
            j = r.json().get('data', {})
            return float(j.get('total_cost', 0.0) or 0.0)
    except Exception:
        pass
    return 0.0


def call_model(model: str, prompt: str) -> tuple[list[str], list[int], int, float, float]:
    headers = {'Authorization': f'Bearer {OR_KEY}', 'Content-Type': 'application/json'}
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': 'Return JSON only. Prefer selected_indices (16 unique ints) from provided shortlist index column.'},
            {'role': 'user', 'content': prompt},
        ],
        'response_format': {
            'type': 'json_schema',
            'json_schema': {
                'name': 'CandidateSelection',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'selected_indices': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 16, 'maxItems': 16, 'uniqueItems': True},
                        'selected_candidate_ids': {'type': 'array', 'items': {'type': 'string'}},
                        'rationale_claims': {
                            'type': 'array',
                            'items': {'type': 'object'},
                        },
                    },
                    'required': [],
                    'additionalProperties': True,
                },
            },
        },
        'plugins': [{'id': 'response-healing'}],
        'provider': {'require_parameters': True},
        'temperature': 0.2,
        'max_tokens': 300,
    }
    if model.startswith("google/gemini-3.1-pro"):
        payload['reasoning'] = {'effort': 'low'}
        payload['max_tokens'] = 1200
    t0 = time.time()
    try:
        r = requests.post(f"{OR_BASE}/chat/completions", headers=headers, json=payload, timeout=180)
    except Exception:
        r = None
    j = None
    if r is not None and r.ok:
        j = r.json()
    else:
        # Relaxed fallback request for providers that reject strict schema.
        payload2 = dict(payload)
        payload2['response_format'] = {'type': 'json_object'}
        payload2['provider'] = {'require_parameters': False}
        try:
            r2 = requests.post(f"{OR_BASE}/chat/completions", headers=headers, json=payload2, timeout=180)
            if r2.ok:
                j = r2.json()
        except Exception:
            pass
    dt = time.time() - t0
    if j is None:
        return [], [], 1, 0.0, dt
    choices = j.get('choices') or []
    first = choices[0] if choices else {}
    msg = first.get('message') or {}
    content = _normalize_content(msg.get('content'))
    obj = _extract_json(content) or {}
    ids = _coerce_ids(obj)
    idx = _coerce_indices(obj)
    gen_id = j.get('id', '')
    c = _generation_cost(gen_id)
    if c == 0.0:
        usage = j.get('usage', {})
        # conservative fallback estimate
        in_tok = float(usage.get('prompt_tokens', 0) or 0)
        out_tok = float(usage.get('completion_tokens', 0) or 0)
        c = (in_tok/1e6)*5.0 + (out_tok/1e6)*25.0
    return ids, idx, 0, c, dt


def build_prompt(assay: str, round_idx: int, shortlist_df: pd.DataFrame, history: list[dict[str, Any]]) -> str:
    hist = '\n'.join([f"r{h['round']}: best={h['best']:.4f}, picks={h['n']}" for h in history[-3:]]) or 'none'
    # keep prompt compact
    rows = []
    for i, (_, r) in enumerate(shortlist_df[['candidate_id','prior_score','mut_depth','first_pos']].iterrows()):
        rows.append(f"{i}\t{r['candidate_id']}\t{r['prior_score']:.4f}\t{int(r['mut_depth'])}\t{int(r['first_pos'])}")
    cand = '\n'.join(rows)
    return (
        f"Assay={assay}. Round={round_idx}. Pick exactly 16 unique IDs from shortlist, no other IDs.\n"
        f"History:\n{hist}\n"
        f"Columns: idx, candidate_id, prior_score, mut_depth, first_pos\n"
        f"Shortlist ({len(shortlist_df)} rows):\n{cand}\n"
        "Return JSON with selected_indices (preferred)."
    )


def fallback_select(c: pd.DataFrame, b: int) -> list[str]:
    tmp = c.sort_values(['prior_score','first_pos'], ascending=[False, True]).drop_duplicates('first_pos').head(b)
    if len(tmp) < b:
        tmp = pd.concat([tmp, c[~c.candidate_id.isin(tmp.candidate_id)].head(b-len(tmp))])
    return tmp['candidate_id'].tolist()


def run_one(df: pd.DataFrame, assay: str, method: str, seed: int, rounds: int = 3, batch: int = 16):
    model = METHOD_TO_MODEL[method]
    q99 = top_threshold(df.DMS_score, 0.99)
    q90 = top_threshold(df.DMS_score, 0.90)
    tested = set()
    obs = []
    hist = []
    invalid = 0
    retries = 0
    total_cost = 0.0
    total_lat = 0.0
    for t in range(1, rounds+1):
        c = shortlist(df, tested, SHORTLIST_K)
        allowed = set(c.candidate_id.tolist())
        prompt = build_prompt(assay, t, c, hist)
        valid: list[str] = []
        attempt_prompt = prompt
        for a in range(MAX_REPAIRS + 1):
            ids, idx, err, cost, lat = call_model(model, attempt_prompt)
            total_cost += cost
            total_lat += lat
            if total_cost > MAX_USD:
                raise RuntimeError(f'Cost cap exceeded for {method}: {total_cost:.2f} > {MAX_USD}')
            valid = []
            if idx:
                for j in idx:
                    if 0 <= j < len(c):
                        cid = c.iloc[j].candidate_id
                        if cid in allowed and cid not in tested:
                            valid.append(cid)
            if not valid:
                valid = [x for x in ids if x in allowed and x not in tested]
            if (not err) and len(valid) >= batch:
                break
            if a < MAX_REPAIRS:
                retries += 1
                attempt_prompt = (
                    prompt
                    + f"\nYour previous response had only {len(valid)} valid IDs. "
                    + "Return exactly 16 unique IDs from the shortlist."
                )
        uniq = []
        seen = set()
        for x in valid:
            if x not in seen:
                uniq.append(x); seen.add(x)
        if len(uniq) < batch:
            invalid += 1
            fill = [x for x in fallback_select(c[~c.candidate_id.isin(uniq)], batch-len(uniq)) if x not in seen]
            uniq.extend(fill[:batch-len(uniq)])
        uniq = uniq[:batch]
        sel = c[c.candidate_id.isin(uniq)]
        tested.update(sel.candidate_id.tolist())
        obs.extend(sel.to_dict('records'))
        hist.append({'round': t, 'best': float(sel.DMS_score.max()) if len(sel) else -1e9, 'n': len(sel)})

    oy = [r['DMS_score'] for r in obs]
    return {
        'top1_success': top1_success(oy, q99),
        'top10_hits': top10_hits(oy, q90),
        'best_percentile': float((df.DMS_score <= max(oy)).mean() * 100.0),
        'normalized_regret': float((df.DMS_score.max() - max(oy)) / (df.DMS_score.max() - df.DMS_score.median() + 1e-9)),
        'unique_loci': int(len(set(int(r['first_pos']) for r in obs))),
        'invalid_action_rate': float(invalid / rounds),
        'cost_usd': float(total_cost),
        'latency_sec': float(total_lat),
        'retries': retries,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--processed', required=True)
    ap.add_argument('--assay_ids_csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='11')
    ap.add_argument('--methods', default='llm_gpt55_only,llm_claude_opus47_only,llm_gemini31pro_only,llm_grok41fast_only')
    args = ap.parse_args()
    if not OR_KEY:
        raise RuntimeError('OPENROUTER_API_KEY missing')

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(args.assay_ids_csv)['DMS_id'].astype(str).tolist()
    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    methods = [x.strip() for x in args.methods.split(',') if x.strip()]
    rows = []
    for assay in ids:
        p = Path(args.processed) / f'{assay}.parquet'
        if not p.exists():
            continue
        df = add_features(pd.read_parquet(p))
        for s in seeds:
            for m in methods:
                r = run_one(df, assay, m, s)
                r.update({'DMS_id': assay, 'seed': s, 'method': m})
                rows.append(r)
                with (out/'progress.md').open('a') as f:
                    f.write(f"- done assay={assay} seed={s} method={m} cost={r['cost_usd']:.4f}\n")
                pd.DataFrame(rows).to_csv(out/'summary.csv', index=False)
    pd.DataFrame(rows).to_csv(out/'summary.csv', index=False)
    (out/'run.log').write_text(json.dumps({'rows': len(rows), 'methods': methods, 'seeds': seeds}, indent=2))

if __name__ == '__main__':
    main()
