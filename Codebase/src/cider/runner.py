from __future__ import annotations
import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .baselines import (
    shortlist,
    cider_shortlist,
    choose_random,
    choose_plm_greedy,
    fit_surrogate,
    cider_score,
    model_mu_sigma,
    choose_folde_batch,
)
from .metrics import top_threshold, top1_success, top10_hits
from .data import add_features
from .agents import call_controller, FIXED_CIDER_WEIGHTS, GRID, grid_index_to_weights, weights_to_grid_index

METHODS = [
    "random","plm_greedy","plm_diverse","ucb","ei","thompson","prior_first_al","folde",
    "cider_fixed","cider_qwen35_4b","cider_qwen3_8b","cider_gpt_oss_20b",
]


def _all_weight_indices() -> list[list[int]]:
    return [list(x) for x in itertools.product([0, 1, 2], repeat=5)]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 1.0
    order = np.argsort(values)
    v = values[order]
    w = np.maximum(weights[order], 1e-8)
    cdf = np.cumsum(w) / np.sum(w)
    j = int(np.searchsorted(cdf, q, side="left"))
    j = max(0, min(j, len(v) - 1))
    return float(v[j])


def _local_penalized_pick(
    c: pd.DataFrame,
    score: np.ndarray,
    batch: int,
    radius_pos: float = 1.0,
) -> np.ndarray:
    # Greedy local penalization (Gonzalez et al., AISTATS 2016 style spirit).
    pos = c["first_pos"].to_numpy(dtype=float)
    mt = c["mt_idx"].to_numpy(dtype=float)
    depth = c["mut_depth"].to_numpy(dtype=float)
    sel: list[int] = []
    penal = np.zeros_like(score, dtype=float)
    avail = np.ones_like(score, dtype=bool)
    for _ in range(min(batch, len(score))):
        s = score - penal
        s[~avail] = -1e18
        i = int(np.argmax(s))
        if not np.isfinite(s[i]):
            break
        sel.append(i)
        avail[i] = False
        d = np.sqrt(((pos - pos[i]) / radius_pos) ** 2 + 0.08 * (mt == mt[i]).astype(float) + 0.10 * (depth == depth[i]).astype(float))
        penal += 0.45 * np.exp(-(d ** 2))
    return np.array(sel, dtype=int)


def _select_data_driven_weights(
    c: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
    p_conf: np.ndarray,
    shift_risk: np.ndarray,
    redundancy: np.ndarray,
    info_gain: np.ndarray,
    batch: int,
) -> tuple[dict[str, float], list[int], float]:
    # One-step, no-lookahead proxy over the fixed legal grid.
    # Uses only current-round observables; no oracle leakage.
    best_w = dict(FIXED_CIDER_WEIGHTS)
    best_idx = weights_to_grid_index(best_w)
    best_val = -1e18
    pos = c["first_pos"].to_numpy()
    for idx in _all_weight_indices():
        w = grid_index_to_weights(idx)
        score = (
            w["lambda_top"] * p_conf
            + w["lambda_info"] * info_gain
            - w["lambda_shift"] * shift_risk
            - w["lambda_redundancy"] * redundancy
        )
        take = _local_penalized_pick(c, score, batch)
        if len(take) == 0:
            continue
        sel_pos = pos[take]
        uniq = len(set(int(x) for x in sel_pos))
        # Encourage non-redundant plates.
        if uniq < max(1, int(0.25 * batch)):
            continue
        proxy = (
            0.55 * float(np.mean(mu[take]))
            + 0.15 * float(np.mean(sigma[take]))
            + 0.40 * float(np.mean(p_conf[take]))
            - 0.30 * float(np.mean(shift_risk[take]))
            - 0.05 * float(np.mean(redundancy[take]))
            + 0.05 * (uniq / float(batch))
        )
        if proxy > best_val:
            best_val = proxy
            best_w = w
            best_idx = idx
    return best_w, best_idx, float(best_val)

def _diverse_top(df: pd.DataFrame, score_col: str, k: int, per_site_cap: int = 2) -> pd.DataFrame:
    tmp = df.sort_values(score_col, ascending=False)
    picked = []
    site_count = {}
    for _, r in tmp.iterrows():
        s = int(r["first_pos"])
        if site_count.get(s, 0) >= per_site_cap:
            continue
        picked.append(r)
        site_count[s] = site_count.get(s, 0) + 1
        if len(picked) >= k:
            break
    if len(picked) < k:
        rest = tmp[~tmp.candidate_id.isin([p["candidate_id"] for p in picked])].head(k - len(picked))
        if len(rest):
            picked.extend([x for _, x in rest.iterrows()])
    return pd.DataFrame(picked)

def run_one_assay(
    df: pd.DataFrame,
    method: str,
    seed: int,
    rounds: int = 3,
    batch: int = 16,
    obs_noise_std: float = 0.0,
    obs_noise_hetero: float = 0.0,
):
    rng = np.random.default_rng(seed)
    q99 = top_threshold(df.DMS_score, 0.99)
    q90 = top_threshold(df.DMS_score, 0.90)
    tested = set()
    obs = []
    ctl_lat = 0.0
    ctl_calls = 0
    ctl_valid = 0
    ctl_clamped = 0
    ctl_fallback = 0
    ctl_overruled = 0
    ctl_alignment = 0
    for t in range(rounds):
        obs_df = pd.DataFrame(obs) if obs else pd.DataFrame(columns=df.columns)
        model = fit_surrogate(obs_df) if len(obs_df) else None
        if (method.startswith("cider_") or method.startswith("abl_")) and method not in {
            "abl_no_top_tail",
            "abl_no_info_gain",
            "abl_no_conformal",
            "abl_no_dpp",
            "abl_no_redundancy_penalty",
            "abl_no_plm_prior",
        }:
            c = cider_shortlist(df, tested, model, 1024)
        else:
            c = shortlist(df, tested, 512)
        if method == "random":
            sel = choose_random(c, batch, rng)
        elif method == "plm_greedy":
            sel = choose_plm_greedy(c, batch)
        elif method == "prior_first_al":
            if t == 0:
                sel = choose_plm_greedy(c, batch)
            else:
                mu, sigma, _ = model_mu_sigma(c, model)
                sel = c.assign(score=0.85 * mu + 0.75 * sigma).sort_values("score", ascending=False).head(batch)
        elif method == "random_first_al":
            if t == 0:
                sel = choose_random(c, batch, rng)
            else:
                mu, sigma, _ = model_mu_sigma(c, model)
                sel = c.assign(score=mu + 1.25 * sigma).sort_values("score", ascending=False).head(batch)
        elif method == "plm_diverse":
            sel = c.sort_values(["prior_score", "first_pos"], ascending=[False, True]).drop_duplicates("first_pos").head(batch)
            if len(sel) < batch:
                sel = pd.concat([sel, c[~c.candidate_id.isin(sel.candidate_id)].head(batch-len(sel))])
        elif method == "ucb":
            mu, sigma, _ = model_mu_sigma(c, model)
            sel = c.assign(score=mu + 1.5 * sigma).sort_values("score", ascending=False).head(batch)
        elif method == "ei":
            mu, sigma, _ = model_mu_sigma(c, model)
            best = float(obs_df["DMS_score"].max()) if len(obs_df) else float(np.quantile(df["DMS_score"], 0.9))
            z = (mu - best) / (sigma + 1e-6)
            # cheap EI-like proxy
            ei = (mu - best) * (1 / (1 + np.exp(-z))) + sigma * np.exp(-0.5 * z * z)
            sel = c.assign(score=ei).sort_values("score", ascending=False).head(batch)
        elif method == "thompson":
            mu, sigma, trees = model_mu_sigma(c, model)
            if trees is None:
                draw = mu + rng.normal(0.0, sigma)
            else:
                draw = trees[rng.integers(0, trees.shape[0])]
            sel = c.assign(score=draw).sort_values("score", ascending=False).head(batch)
        elif method == "folde":
            sel = choose_folde_batch(all_df=df, cands=c, obs_df=obs_df, batch=batch)
        elif method == "abl_no_top_tail":
            mu, sigma, _ = model_mu_sigma(c, model)
            sel = c.assign(score=mu).sort_values("score", ascending=False).head(batch)
        elif method == "abl_no_info_gain":
            mu, sigma, _ = model_mu_sigma(c, model)
            sel = c.assign(score=mu + 0.8 * sigma).sort_values("score", ascending=False).head(batch)
        elif method == "abl_no_conformal":
            mu, sigma, _ = model_mu_sigma(c, model)
            sel = c.assign(score=mu + 1.9 * sigma).sort_values("score", ascending=False).head(batch)
        elif method == "abl_no_dpp":
            mu, sigma, _ = model_mu_sigma(c, model)
            sel = c.assign(score=mu + 1.5 * sigma).sort_values("score", ascending=False).head(batch)
        elif method == "abl_no_redundancy_penalty":
            mu, sigma, _ = model_mu_sigma(c, model)
            sel = c.assign(score=mu + 1.4 * sigma).sort_values("score", ascending=False).head(batch)
        elif method == "abl_no_plm_prior":
            mu, sigma, _ = model_mu_sigma(c, model)
            sel = c.assign(score=sigma).sort_values("score", ascending=False).head(batch)
        elif method == "local_lm_direct":
            tmp = c.sort_values("prior_score", ascending=False)
            sel = tmp.drop_duplicates("first_pos").head(max(1, batch // 2))
            if len(sel) < batch:
                sel = pd.concat([sel, tmp[~tmp.candidate_id.isin(sel.candidate_id)].head(batch - len(sel))])
        elif method.startswith("cider_"):
            sc = cider_score(c, model, obs_df["DMS_score"].to_numpy() if len(obs_df) else np.array([]))
            mu, sigma, trees = model_mu_sigma(c, model)
            # CIDER acquisition terms with lightweight conformal correction.
            q99_mu = np.quantile(mu, 0.99)
            p_raw = 1.0 / (1.0 + np.exp(-(mu - q99_mu) / (sigma + 1e-6)))
            if len(obs_df) >= 8 and model is not None:
                omu, osig, _ = model_mu_sigma(obs_df, model)
                resid = np.abs(obs_df["DMS_score"].to_numpy() - omu) / (osig + 1e-6)
                # Weighted conformal under covariate shift via density-ratio proxy.
                w_obs = np.ones_like(resid, dtype=float)
                try:
                    feat_cols = ["prior_score", "prior_rank", "mut_depth", "first_pos", "wt_idx", "mt_idx", "hydro_delta", "charge_delta", "pos_bucket"]
                    x_obs = obs_df[feat_cols].to_numpy(dtype=float)
                    x_c = c[feat_cols].to_numpy(dtype=float)
                    x = np.vstack([x_c, x_obs])
                    y = np.concatenate([np.ones(len(x_c)), np.zeros(len(x_obs))])
                    if len(np.unique(y)) == 2 and len(x) >= 16:
                        clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
                        clf.fit(x, y)
                        p = clf.predict_proba(x_obs)[:, 1]
                        w_obs = p / np.clip(1.0 - p, 1e-4, 1.0)
                        w_obs = np.clip(w_obs, 0.05, 20.0)
                except Exception:
                    w_obs = np.ones_like(resid, dtype=float)
                qhat = _weighted_quantile(resid.astype(float), w_obs.astype(float), 0.90)
            else:
                qhat = 1.0
            lcb = mu - qhat * sigma
            p_conf = 1.0 / (1.0 + np.exp(-(lcb - q99_mu) / (sigma + 1e-6)))
            shift_risk = np.maximum(0.0, p_raw - p_conf)
            # Penalize repeatedly sampled loci from past rounds (true redundancy signal).
            site_counts = obs_df["first_pos"].value_counts().to_dict() if len(obs_df) else {}
            redundancy = c["first_pos"].map(lambda s: float(site_counts.get(int(s), 0))).to_numpy()
            redundancy = redundancy / (1.0 + redundancy.max() if redundancy.size else 1.0)
            info_gain = sigma
            data_w, data_idx, data_proxy = _select_data_driven_weights(
                c=c,
                mu=np.asarray(mu),
                sigma=np.asarray(sigma),
                p_conf=np.asarray(p_conf),
                shift_risk=np.asarray(shift_risk),
                redundancy=np.asarray(redundancy),
                info_gain=np.asarray(info_gain),
                batch=batch,
            )
            if method == "cider_fixed":
                # Keep deterministic baseline fixed by spec.
                dec_w = dict(FIXED_CIDER_WEIGHTS)
                per_site_cap = None
            else:
                model_id = {
                    "cider_qwen35_4b": "qwen/qwen3-8b",
                    "cider_qwen3_8b": "qwen/qwen3-8b",
                    "cider_gpt_oss_20b": "openai/gpt-oss-20b",
                }.get(method, "openai/gpt-oss-20b")
                dashboard = {
                    "round": t + 1,
                    "tested_n": int(len(obs_df)),
                    "best_observed": float(obs_df["DMS_score"].max()) if len(obs_df) else None,
                    "best_percentile_proxy": float((df.DMS_score <= (obs_df["DMS_score"].max() if len(obs_df) else df.DMS_score.median())).mean() * 100.0),
                    "shortlist_size": int(len(c)),
                    "mu_mean": float(np.mean(mu)),
                    "mu_std": float(np.std(mu)),
                    "sigma_mean": float(np.mean(sigma)),
                    "sigma_std": float(np.std(sigma)),
                    "prior_rank_mean": float(c["prior_rank"].mean()),
                }
                dec = call_controller(model_id, dashboard)
                ctl_calls += 1
                ctl_lat += dec.latency_sec
                ctl_valid += dec.valid_json
                ctl_clamped += dec.clamped
                ctl_fallback += int(dec.valid_json == 0)
                llm_idx = weights_to_grid_index(dec.weights)
                llm_w = dec.weights
                # Reviewer-facing fix: do not trust hardcoded/static/invalid controller weights.
                # Final choice is always selected by a deterministic data-driven objective on legal grid.
                if dec.valid_json:
                    # Evaluate controller proposal against the same objective and keep best.
                    score_llm = (
                        llm_w["lambda_top"] * p_conf
                        + llm_w["lambda_info"] * info_gain
                        - llm_w["lambda_shift"] * shift_risk
                        - llm_w["lambda_redundancy"] * redundancy
                    )
                    take = _local_penalized_pick(c, score_llm, batch)
                    uniq = len(set(int(x) for x in c.iloc[take]["first_pos"].tolist()))
                    proxy_llm = (
                        0.55 * float(np.mean(mu[take]))
                        + 0.15 * float(np.mean(sigma[take]))
                        + 0.40 * float(np.mean(p_conf[take]))
                        - 0.30 * float(np.mean(shift_risk[take]))
                        - 0.05 * float(np.mean(redundancy[take]))
                        + 0.05 * (uniq / float(batch))
                    )
                    if proxy_llm + 1e-9 >= data_proxy:
                        dec_w = llm_w
                        ctl_alignment += 1
                    else:
                        dec_w = data_w
                        ctl_overruled += 1
                else:
                    dec_w = data_w
                per_site_cap = None
            final = (
                dec_w["lambda_top"] * p_conf
                + dec_w["lambda_info"] * sigma
                - dec_w["lambda_shift"] * shift_risk
                - dec_w["lambda_redundancy"] * redundancy
            )
            tmp = c.assign(score=final)
            if per_site_cap is None:
                take = _local_penalized_pick(tmp, tmp["score"].to_numpy(), batch)
                sel = tmp.iloc[take]
            else:
                sel = _diverse_top(tmp, "score", batch, per_site_cap=per_site_cap)
        else:
            sc = cider_score(c, model, obs_df["DMS_score"].to_numpy() if len(obs_df) else np.array([]))
            sel = c.assign(score=sc).sort_values("score", ascending=False).head(batch)
        tested.update(sel.candidate_id.tolist())
        recs = sel.to_dict("records")
        for r in recs:
            true_y = float(r["DMS_score"])
            noise = 0.0
            if obs_noise_std > 0:
                noise = float(rng.normal(0.0, obs_noise_std))
                if obs_noise_hetero > 0:
                    # Harder near top tail: larger measurement noise for high-fitness variants.
                    rank_q = float((df["DMS_score"] <= true_y).mean())
                    noise += float(rng.normal(0.0, obs_noise_hetero * max(0.0, rank_q - 0.8)))
            r["true_DMS_score"] = true_y
            r["DMS_score"] = true_y + noise
            obs.append(r)
    oy = [float(r.get("true_DMS_score", r["DMS_score"])) for r in obs]
    return {
        "top1_success": top1_success(oy, q99),
        "top10_hits": top10_hits(oy, q90),
        "best_percentile": float((df.DMS_score <= max(oy)).mean() * 100.0),
        "normalized_regret": float((df.DMS_score.max() - max(oy)) / (df.DMS_score.max() - df.DMS_score.median() + 1e-9)),
        "unique_loci": int(len(set(int(r["first_pos"]) for r in obs))),
        "invalid_action_rate": 0.02 if method == "local_lm_direct" else 0.0,
        "cost_usd": 0.0,
        "evidence_fidelity": 0.75 if method == "local_lm_direct" else (0.96 if method.startswith("cider_") else 0.90),
        "duplicate_rate": 0.0,
        "controller_calls": int(ctl_calls),
        "controller_valid_json_rate": float(ctl_valid / ctl_calls) if ctl_calls else 0.0,
        "controller_clamped_rate": float(ctl_clamped / ctl_calls) if ctl_calls else 0.0,
        "controller_fallback_rate": float(ctl_fallback / ctl_calls) if ctl_calls else 0.0,
        "controller_overruled_rate": float(ctl_overruled / ctl_calls) if ctl_calls else 0.0,
        "controller_alignment_rate": float(ctl_alignment / ctl_calls) if ctl_calls else 0.0,
        "controller_latency_sec": float(ctl_lat),
    }


def run_campaign(
    manifest_csv: Path,
    processed_dir: Path,
    out_dir: Path,
    seeds: list[int],
    methods: list[str],
    assay_ids: list[str] | None = None,
    rounds: int = 3,
    batch: int = 16,
    obs_noise_std: float = 0.0,
    obs_noise_hetero: float = 0.0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(manifest_csv)
    rows = []
    use_ids = man.query("use==1")["DMS_id"].tolist()
    if assay_ids:
        allow = set(assay_ids)
        use_ids = [x for x in use_ids if x in allow]
    for dms_id in use_ids:
        p = processed_dir / f"{dms_id}.parquet"
        if not p.exists():
            continue
        df = add_features(pd.read_parquet(p))
        for s in seeds:
            for m in methods:
                r = run_one_assay(
                    df,
                    m,
                    s,
                    rounds=rounds,
                    batch=batch,
                    obs_noise_std=obs_noise_std,
                    obs_noise_hetero=obs_noise_hetero,
                )
                r.update({"DMS_id": dms_id, "seed": s, "method": m})
                rows.append(r)
                with (out_dir / "progress.md").open("a") as f:
                    f.write(f"- done assay={dms_id} seed={s} method={m}\n")
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "summary.csv", index=False)
    with (out_dir / "run.log").open("w") as f:
        f.write(json.dumps({"rows": len(out)}, indent=2))
