from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

FEAT_COLS = ["prior_score", "prior_rank", "mut_depth", "first_pos", "wt_idx", "mt_idx", "hydro_delta", "charge_delta", "pos_bucket"]


def shortlist(df: pd.DataFrame, tested: set[str], k: int = 512):
    c = df[~df.candidate_id.isin(tested)].copy()
    if len(c) <= k:
        return c
    c = c.sort_values("prior_score", ascending=False)
    n1 = int(k * 0.4)
    n2 = int(k * 0.2)
    top = c.head(n1)
    rnd = c.iloc[n1:].sample(n=min(n2, max(0, len(c)-n1)), random_state=0)
    rem = c.drop(top.index).drop(rnd.index, errors="ignore")
    mid = rem.head(k - len(top) - len(rnd))
    return pd.concat([top, mid, rnd], axis=0).head(k)


def cider_shortlist(df: pd.DataFrame, tested: set[str], model, k: int = 1024):
    c = df[~df.candidate_id.isin(tested)].copy()
    if len(c) <= k:
        return c
    c = c.sort_values("prior_score", ascending=False)
    n_top = int(k * 0.40)
    n_div = int(k * 0.20)
    n_unc = int(k * 0.25)
    n_rnd = max(1, k - n_top - n_div - n_unc)

    top = c.head(n_top)
    # site-diverse candidates from high prior region
    div = c.drop(top.index).drop_duplicates("first_pos").head(n_div)

    rem = c.drop(top.index).drop(div.index, errors="ignore")
    if model is not None and len(rem) > 0:
        pool = rem if len(rem) <= 8000 else rem.sample(8000, random_state=17)
        mu, sigma, _ = model_mu_sigma(pool, model)
        unc = pool.assign(_u=sigma).sort_values("_u", ascending=False).head(n_unc).drop(columns="_u")
    else:
        unc = rem.head(n_unc)
    rem2 = rem.drop(unc.index, errors="ignore")
    rnd = rem2.sample(n=min(n_rnd, len(rem2)), random_state=23) if len(rem2) else rem2
    out = pd.concat([top, div, unc, rnd], axis=0).drop_duplicates("candidate_id").head(k)
    return out


def choose_random(cands: pd.DataFrame, b: int, rng: np.random.Generator):
    idx = rng.choice(len(cands), size=min(b, len(cands)), replace=False)
    return cands.iloc[idx]


def choose_plm_greedy(cands: pd.DataFrame, b: int):
    return cands.sort_values("prior_score", ascending=False).head(b)


def fit_surrogate(obs: pd.DataFrame):
    if len(obs) < 8:
        return None
    X = obs[FEAT_COLS].to_numpy()
    y = obs["DMS_score"].to_numpy()
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0)
    m.fit(X, y)
    return m


def cider_score(cands: pd.DataFrame, model, obs_scores):
    X = cands[FEAT_COLS].to_numpy()
    if model is None:
        mu = cands["prior_score"].to_numpy()
        sigma = np.full(len(cands), np.std(mu) + 1e-3)
    else:
        trees = np.vstack([t.predict(X) for t in model.estimators_])
        mu = trees.mean(0)
        sigma = trees.std(0) + 1e-3
    q99 = np.quantile(mu, 0.99)
    p_top = 1 / (1 + np.exp(-(mu - q99) / (sigma + 1e-6)))
    info = sigma
    shift = np.maximum(0, p_top - (p_top - 0.1 * sigma))
    red = 1 / (1 + np.abs(cands["first_pos"].to_numpy()))
    return 0.55*p_top + 0.25*info - 0.10*shift - 0.10*red


def model_mu_sigma(cands: pd.DataFrame, model):
    X = cands[FEAT_COLS].to_numpy()
    if model is None:
        mu = cands["prior_score"].to_numpy()
        sigma = np.full(len(cands), np.std(mu) + 1e-3)
        trees = None
    else:
        trees = np.vstack([t.predict(X) for t in model.estimators_])
        mu = trees.mean(0)
        sigma = trees.std(0) + 1e-3
    return mu, sigma, trees


def fit_folde_mlp_ensemble(
    all_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    ensemble_size: int = 5,
    hidden_dims: tuple[int, int] = (100, 50),
    pretrain: bool = True,
    pretrain_n: int = 1024,
    obs_oversample: int = 24,
):
    if len(obs_df) == 0:
        return None
    X_obs = obs_df[FEAT_COLS].to_numpy(dtype=float)
    y_obs = obs_df["DMS_score"].to_numpy(dtype=float)
    if pretrain:
        ref = all_df.sample(n=min(pretrain_n, len(all_df)), random_state=13)
        X_pre = ref[FEAT_COLS].to_numpy(dtype=float)
        y_pre = ref["prior_score"].to_numpy(dtype=float)
        y_pre = (y_pre - y_pre.mean()) / (y_pre.std() + 1e-8)
        X_train = np.vstack([X_pre, np.repeat(X_obs, obs_oversample, axis=0)])
        y_train = np.concatenate([y_pre, np.repeat(y_obs, obs_oversample)])
    else:
        X_train = X_obs
        y_train = y_obs
    models = []
    for i in range(ensemble_size):
        m = MLPRegressor(
            hidden_layer_sizes=hidden_dims,
            activation="relu",
            solver="adam",
            alpha=1e-5,
            learning_rate_init=3e-4,
            max_iter=250,
            random_state=101 + i,
        )
        m.fit(X_train, y_train)
        models.append(m)
    return models


def folde_predict_ensemble(cands: pd.DataFrame, models) -> np.ndarray:
    X = cands[FEAT_COLS].to_numpy(dtype=float)
    pred = np.stack([m.predict(X) for m in models], axis=1)  # (N, E)
    return pred


def _constant_liar_sample(ensemble_preds: np.ndarray, q_slate_size: int, lie_mult: float = 6.0, ucb_beta: float = 0.0) -> list[int]:
    # Adapted from FolDE constant-liar logic: use ensemble covariance + repeated GP-style liar updates.
    if ensemble_preds.ndim != 2:
        raise ValueError("ensemble_preds must be 2D (N, E)")
    n, e = ensemble_preds.shape
    if n < q_slate_size:
        q_slate_size = n
    if e < 3:
        # fallback to mean-ranking when ensemble is too small
        return np.argsort(ensemble_preds.mean(axis=1))[::-1][:q_slate_size].tolist()
    pred = ensemble_preds.T  # (E, N)
    prior_mean = pred.mean(axis=0)
    devs = pred - prior_mean[None, :]
    cov = (devs.T @ devs) / float(e)
    lie_var = (lie_mult * np.median(pred.std(axis=0))) ** 2
    cov = cov + lie_var * np.eye(n)
    L = float(prior_mean.min())
    selected: list[int] = []
    for _ in range(q_slate_size):
        var = np.clip(np.diag(cov), 1e-10, None)
        sigma = np.sqrt(var)
        ucb = prior_mean + ucb_beta * sigma
        if selected:
            ucb[np.array(selected, dtype=int)] = -np.inf
        idx = int(np.argmax(ucb))
        selected.append(idx)
        k_i = cov[:, idx].copy()
        v_i = max(var[idx], 1e-10)
        delta = (L - prior_mean[idx]) / v_i
        prior_mean = prior_mean + k_i * delta
        cov = cov - np.outer(k_i, k_i) / v_i
        cov = 0.5 * (cov + cov.T)
    return selected


def choose_folde_batch(
    all_df: pd.DataFrame,
    cands: pd.DataFrame,
    obs_df: pd.DataFrame,
    batch: int,
):
    # Round 1 zero-shot: naturalness-only (here prior_score is our naturalness proxy).
    if len(obs_df) == 0:
        return cands.sort_values("prior_score", ascending=False).head(batch)
    models = fit_folde_mlp_ensemble(all_df=all_df, obs_df=obs_df, pretrain=True)
    if models is None:
        return cands.sort_values("prior_score", ascending=False).head(batch)
    preds = folde_predict_ensemble(cands, models)
    chosen_idx = _constant_liar_sample(preds, q_slate_size=batch, lie_mult=6.0, ucb_beta=0.0)
    return cands.iloc[chosen_idx]
