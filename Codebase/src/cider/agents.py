from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

OR_BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")

ALLOWED_MODES = [
    "exploit_confident_top_tail",
    "information_gain",
    "diversify_promising_sites",
    "recover_from_prior_failure",
    "calibration_conservative",
]
GRID = {
    "lambda_top": [0.30, 0.45, 0.60],
    "lambda_info": [0.10, 0.25, 0.40],
    "lambda_shift": [0.05, 0.15, 0.30],
    "lambda_redundancy": [0.05, 0.15, 0.30],
    "beta_dpp": [0.05, 0.10, 0.20],
}


@dataclass
class ControllerDecision:
    mode: str
    weights: dict[str, float]
    model: str
    latency_sec: float
    valid_json: int
    clamped: int
    error: str


FIXED_CIDER_WEIGHTS = {
    "lambda_top": 0.45,
    "lambda_info": 0.25,
    "lambda_shift": 0.15,
    "lambda_redundancy": 0.15,
    "beta_dpp": 0.10,
}


def grid_index_to_weights(w_idx: list[int]) -> dict[str, float]:
    keys = ["lambda_top", "lambda_info", "lambda_shift", "lambda_redundancy", "beta_dpp"]
    if len(w_idx) != 5:
        raise ValueError("w_idx must have length 5")
    out: dict[str, float] = {}
    for j, k in enumerate(keys):
        i = int(w_idx[j])
        if i < 0 or i >= len(GRID[k]):
            raise ValueError(f"invalid index {i} for {k}")
        out[k] = GRID[k][i]
    return out


def weights_to_grid_index(weights: dict[str, float]) -> list[int]:
    keys = ["lambda_top", "lambda_info", "lambda_shift", "lambda_redundancy", "beta_dpp"]
    idx: list[int] = []
    for k in keys:
        v = float(weights[k])
        xs = GRID[k]
        idx.append(int(min(range(len(xs)), key=lambda i: abs(xs[i] - v))))
    return idx


def _extract_json(text: Any) -> dict[str, Any] | None:
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    s = text.strip()
    for cand in [s, re.sub(r"^```json\\n|```$", "", s, flags=re.M)]:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    m = re.search(r"\{.*\}", s, flags=re.S)
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


def _nearest(key: str, val: float) -> float:
    xs = GRID[key]
    return min(xs, key=lambda x: abs(x - val))


def _parse_indices(obj: dict[str, Any]) -> tuple[str, dict[str, float], int] | None:
    mode = str(obj.get("mode", ""))
    if mode not in ALLOWED_MODES:
        return None
    w = obj.get("weights")
    if isinstance(w, dict):
        out = {}
        clamped = 0
        for k in GRID:
            try:
                v = float(w[k])
            except Exception:
                return None
            nv = _nearest(k, v)
            clamped += int(abs(nv - v) > 1e-9)
            out[k] = nv
        return mode, out, int(clamped > 0)
    # compact index format: w_idx = [i,j,k,l,m]
    wi = obj.get("w_idx")
    if isinstance(wi, list) and len(wi) == 5:
        try:
            ii = [int(x) for x in wi]
        except Exception:
            return None
        for x in ii:
            if x < 0 or x > 2:
                return None
        keys = ["lambda_top", "lambda_info", "lambda_shift", "lambda_redundancy", "beta_dpp"]
        out = {k: GRID[k][ii[j]] for j, k in enumerate(keys)}
        return mode, out, 0
    return None


def _parse_indices_from_text(s: str) -> tuple[str, dict[str, float], int] | None:
    m_mode = re.search(r'"mode"\s*:\s*"([^"]+)"', s)
    mode = m_mode.group(1).strip() if m_mode else "information_gain"
    if mode not in ALLOWED_MODES:
        mode = "information_gain"
    m_idx = re.search(r'"w_idx"\s*:\s*\[([^\]]+)\]', s)
    ii = None
    if m_idx:
        raw = [x.strip() for x in m_idx.group(1).split(",") if x.strip()]
        if len(raw) == 5:
            try:
                ii = [int(x) for x in raw]
            except Exception:
                ii = None
    if ii is None:
        # fallback: any bracketed list of exactly five ints in the output text
        m_any = re.search(r"\[\s*([0-9]\s*,\s*){4}[0-9]\s*\]", s)
        if not m_any:
            return None
        raw2 = m_any.group(0).strip("[]")
        try:
            ii = [int(x.strip()) for x in raw2.split(",")]
        except Exception:
            return None
    if len(ii) != 5:
        return None
    if any((x < 0 or x > 2) for x in ii):
        return None
    keys = ["lambda_top", "lambda_info", "lambda_shift", "lambda_redundancy", "beta_dpp"]
    out = {k: GRID[k][ii[j]] for j, k in enumerate(keys)}
    return mode, out, 0


def call_controller(model_id: str, dashboard: dict[str, Any], max_retries: int = 6) -> ControllerDecision:
    if not OR_KEY:
        return ControllerDecision(
            mode="information_gain",
            weights=dict(FIXED_CIDER_WEIGHTS),
            model=model_id,
            latency_sec=0.0,
            valid_json=0,
            clamped=0,
            error="controller_key_missing",
        )
    system = (
        "You are a constrained acquisition controller. Return JSON only with keys mode and w_idx. "
        "Allowed mode: exploit_confident_top_tail, information_gain, diversify_promising_sites, "
        "recover_from_prior_failure, calibration_conservative. "
        "Set w_idx as 5 integers in [0,1,2] mapping to weights grid in order: "
        "[lambda_top,lambda_info,lambda_shift,lambda_redundancy,beta_dpp]."
    )
    user = json.dumps({"dashboard": dashboard}, separators=(",", ":"))
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "CiderController",
                "schema": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string"},
                        "w_idx": {"type": "array", "items": {"type": "integer"}, "minItems": 5, "maxItems": 5},
                    },
                    "required": ["mode", "w_idx"],
                },
            },
        },
        "plugins": [{"id": "response-healing"}],
        "provider": {"require_parameters": True},
        "temperature": 0.1,
        "max_tokens": 220,
    }
    headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
    t0 = time.time()
    last_err = "invalid_controller_output"
    for attempt in range(max_retries):
        payload_try = dict(payload)
        if attempt > 0:
            payload_try["messages"] = payload["messages"] + [
                {
                    "role": "user",
                    "content": f"Previous output invalid. Retry {attempt}. Return ONLY JSON with mode and w_idx length 5.",
                }
            ]
            payload_try["response_format"] = {"type": "json_object"}
            payload_try["provider"] = {"require_parameters": False}
        try:
            r = requests.post(f"{OR_BASE}/chat/completions", headers=headers, json=payload_try, timeout=120)
            if r.ok:
                j = r.json()
                content = _normalize_content(((j.get("choices") or [{}])[0].get("message") or {}).get("content"))
                obj = _extract_json(content)
                if isinstance(obj, dict):
                    parsed = _parse_indices(obj)
                    if parsed is not None:
                        mode, weights, clamped = parsed
                        return ControllerDecision(
                            mode=mode,
                            weights=weights,
                            model=model_id,
                            latency_sec=time.time() - t0,
                            valid_json=1,
                            clamped=clamped,
                            error="",
                        )
                parsed2 = _parse_indices_from_text(content)
                if parsed2 is not None:
                    mode, weights, clamped = parsed2
                    return ControllerDecision(
                        mode=mode,
                        weights=weights,
                        model=model_id,
                        latency_sec=time.time() - t0,
                        valid_json=1,
                        clamped=clamped,
                        error="",
                    )
                last_err = "parse_or_schema_invalid"
            else:
                last_err = f"http_{r.status_code}"
        except Exception as e:
            last_err = f"request_error:{type(e).__name__}"
    return ControllerDecision(
        mode="information_gain",
        weights=dict(FIXED_CIDER_WEIGHTS),
        model=model_id,
        latency_sec=time.time() - t0,
        valid_json=0,
        clamped=0,
        error=f"controller_failed:{last_err}",
    )
