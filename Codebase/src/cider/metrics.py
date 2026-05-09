from __future__ import annotations
import numpy as np

def top_threshold(y, q):
    return float(np.quantile(np.asarray(y), q))

def top1_success(observed, q99):
    return int(np.max(observed) >= q99) if len(observed) else 0

def top10_hits(observed, q90):
    return int(np.sum(np.asarray(observed) >= q90)) if len(observed) else 0
