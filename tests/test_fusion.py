import pandas as pd

from app.config import load_settings
from app.fusion import apply_threshold, compute_final_scores


def test_fusion_and_threshold():
    settings = load_settings()
    df = pd.DataFrame(
        {
            "base_risk_score": [0.2, 0.8, 0.5],
            "economic_risk_score": [0.1, 0.9, 0.4],
            "cluster_risk": [0.1, 0.2, 0.3],
            "arbitration_fraud_prob": [0.2, 0.9, 0.55],
        }
    )
    out = compute_final_scores(df, settings)
    assert "final_rank_score" in out.columns
    mask = apply_threshold(out, "top_k", None, 1, None)
    assert mask.sum() == 1
