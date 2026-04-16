"""Fuse cheap scores with optional agent outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import Settings
from app.utils import clip01


def compute_final_scores(
    tx: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Add final_fraud_score, final_economic_priority, final_rank_score columns."""
    out = tx.copy()
    if "arbitration_fraud_prob" not in out.columns:
        out["arbitration_fraud_prob"] = out["base_risk_score"]
    fraud_col = out["arbitration_fraud_prob"].astype(float)

    out["final_fraud_score"] = clip01(
        0.65 * fraud_col.fillna(out["base_risk_score"]) + 0.35 * out["base_risk_score"].astype(float)
    )
    econ = out["economic_risk_score"].astype(float).fillna(0.0)
    out["final_economic_priority"] = clip01(econ)
    cluster = out.get("cluster_risk", pd.Series(0.0, index=out.index)).astype(float).fillna(0.0)

    wf = settings.fusion_weight_fraud
    we = settings.fusion_weight_economic
    wc = settings.fusion_weight_cluster
    s = wf + we + wc
    wf, we, wc = wf / s, we / s, wc / s

    out["final_rank_score"] = clip01(
        wf * out["final_fraud_score"] + we * out["final_economic_priority"] + wc * cluster
    ).fillna(out["base_risk_score"].astype(float))
    return out


def apply_threshold(
    tx: pd.DataFrame,
    mode: str,
    threshold: float | None,
    top_k: int | None,
    percentile: float | None,
) -> pd.Series:
    """Return boolean mask for flagged fraud."""
    scores = tx["final_rank_score"].astype(float)
    if mode == "top_k" and top_k is not None and top_k > 0:
        k = min(top_k, len(scores))
        cutoff = scores.nlargest(k).min()
        return scores >= cutoff
    if mode == "percentile" and percentile is not None:
        q = float(np.clip(percentile, 0.0, 1.0))
        cutoff = scores.quantile(q)
        mask = scores >= cutoff
        if not mask.any() and len(scores) > 0:
            return scores >= scores.max()
        return mask
    thr = float(threshold if threshold is not None else 0.55)
    mask = scores >= thr
    if not mask.any() and len(scores) > 0:
        return scores >= scores.nlargest(1).min()
    return mask


def adaptive_threshold(
    tx: pd.DataFrame,
    target_min_flags: int = 1,
    max_flags_ratio: float = 0.35,
) -> pd.Series:
    """Pick a percentile-based cutoff with guards for empty or full outputs."""
    n = len(tx)
    if n == 0:
        return pd.Series([], dtype=bool)
    scores = tx["final_rank_score"].astype(float)
    ratio = min(max_flags_ratio, max(target_min_flags / n, 0.02))
    q = 1.0 - ratio
    cutoff = scores.quantile(q)
    mask = scores >= cutoff
    if mask.sum() < target_min_flags:
        mask = scores >= scores.nlargest(target_min_flags).min()
    if mask.mean() > max_flags_ratio:
        tighter = scores >= scores.quantile(1.0 - max_flags_ratio)
        if tighter.any():
            mask = tighter
    if not mask.any():
        mask = scores >= scores.nlargest(target_min_flags).min()
    return mask
