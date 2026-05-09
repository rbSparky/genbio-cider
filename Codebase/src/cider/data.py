from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

MUT_RE = re.compile(r"([A-Z])(\d+)([A-Z])")
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
HYDRO = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3
}
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5}

@dataclass
class Assay:
    dms_id: str
    df: pd.DataFrame


def parse_mutant(mut: str):
    toks = MUT_RE.findall(str(mut))
    if not toks:
        return [], 0
    pos = [int(t[1]) for t in toks]
    return toks, len(toks)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    parsed = x["mutant"].map(parse_mutant)
    x["mut_depth"] = parsed.map(lambda t: t[1])
    x["mut_positions"] = parsed.map(lambda t: ",".join(str(z[1]) for z in t[0]))
    x["first_pos"] = parsed.map(lambda t: int(t[0][0][1]) if t[0] else -1)
    x["wt_aa"] = parsed.map(lambda t: t[0][0][0] if t[0] else "X")
    x["mt_aa"] = parsed.map(lambda t: t[0][0][2] if t[0] else "X")
    x["wt_idx"] = x["wt_aa"].map(lambda a: AA_IDX.get(a, -1))
    x["mt_idx"] = x["mt_aa"].map(lambda a: AA_IDX.get(a, -1))
    x["aa_same"] = (x["wt_aa"] == x["mt_aa"]).astype(float)
    x["hydro_delta"] = x["mt_aa"].map(lambda a: HYDRO.get(a, 0.0)) - x["wt_aa"].map(lambda a: HYDRO.get(a, 0.0))
    x["charge_delta"] = x["mt_aa"].map(lambda a: CHARGE.get(a, 0.0)) - x["wt_aa"].map(lambda a: CHARGE.get(a, 0.0))
    # simple prior proxy from mutation depth and position frequency (no score leakage)
    pos_freq = x["first_pos"].value_counts().to_dict()
    x["prior_score"] = (
        -0.20 * x["mut_depth"]
        + 0.20 * x["aa_same"]
        - 0.07 * x["hydro_delta"].abs()
        - 0.12 * x["charge_delta"].abs()
        + x["first_pos"].map(lambda p: 1.0 / (1 + pos_freq.get(p, 0)))
    )
    x["prior_score"] = x["prior_score"].astype(float)
    x["prior_rank"] = x["prior_score"].rank(method="average", pct=True)
    x["pos_bucket"] = np.where(x["first_pos"] < 0, -1, (x["first_pos"] // 20).astype(int))
    return x


def load_assays(processed_dir: Path, manifest: pd.DataFrame) -> list[Assay]:
    out = []
    for _, r in manifest.iterrows():
        if int(r.get("use", 0)) != 1:
            continue
        p = processed_dir / f"{r['DMS_id']}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        out.append(Assay(r["DMS_id"], add_features(df)))
    return out
