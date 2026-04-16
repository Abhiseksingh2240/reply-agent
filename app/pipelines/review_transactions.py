"""Selective LLM review, fusion, and prediction export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.agents import arbitration_agent, audio_signal_agent
from app.config import Settings, is_llm_configured, load_settings
from app.feature_store import cache_paths, load_normalized
from app.fusion import adaptive_threshold, apply_threshold, compute_final_scores
from app.io_loader import load_dataset
from app.models import RunSummary
from app.observability.tracing import LangfuseRun, new_session_id
from app.output_writer import mirror_default_outputs, write_langfuse_session, write_predictions, write_run_summary
from app.llm.audio_transcription import load_transcription_cache
from app.llm.openrouter_client import OpenRouterClient
from app.pipelines.llm_parallel import (
    run_specialists_parallel,
    run_specialists_with_transcription_parallel,
)
from app.scoring import score_transactions
from app.utils import stable_json_hash


Mode = Literal["fixed", "top_k", "percentile", "adaptive"]


def _load_scored_tables(data_dir: Path, settings: Settings, force_rescore: bool) -> tuple[pd.DataFrame, Path]:
    bundle = load_dataset(data_dir)
    paths = cache_paths(settings.data_cache_dir, bundle.data_dir)
    scored_path = paths["dir"] / "scored_transactions.parquet"
    if force_rescore or not scored_path.exists():
        from app.pipelines.build_features import run_build_features

        run_build_features(data_dir, settings=settings, force=force_rescore)
    scored = pd.read_parquet(scored_path)
    return scored, paths["dir"]


def _nearby_texts(
    mails: pd.DataFrame,
    sms: pd.DataFrame,
    ts: pd.Timestamp,
    hours: float = 72.0,
) -> dict[str, Any]:
    if pd.isna(ts):
        return {"mails": [], "sms": []}
    lo = ts - pd.Timedelta(hours=hours)
    hi = ts + pd.Timedelta(hours=hours)
    mslice = mails[(mails["mail_timestamp"] >= lo) & (mails["mail_timestamp"] <= hi)] if not mails.empty else mails
    sslice = sms[(sms["sms_timestamp"] >= lo) & (sms["sms_timestamp"] <= hi)] if not sms.empty else sms
    return {
        "mails": mslice.head(6)["subject"].fillna("").tolist() if "subject" in mslice.columns else [],
        "sms": sslice.head(8)["sms_text"].fillna("").tolist() if "sms_text" in sslice.columns else [],
    }


def _audio_paths_for_transcription(audio: pd.DataFrame, biotag: str | None, settings: Settings) -> list[Path]:
    if audio.empty or not settings.enable_audio:
        return []
    picked: list[Path] = []
    for _, r in audio.iterrows():
        name = str(r.get("name") or "")
        p = Path(str(r.get("path") or ""))
        if not p.is_file():
            continue
        if biotag and biotag in name.replace("_", "-"):
            picked.append(p)
    if not picked:
        for _, r in audio.head(5).iterrows():
            p = Path(str(r.get("path") or ""))
            if p.is_file():
                picked.append(p)
    cap = max(1, int(settings.audio_transcription_max_files))
    return picked[:cap]


def _audio_payload(
    audio: pd.DataFrame,
    biotag: str | None,
    settings: Settings,
    transcripts_by_path: dict[str, str] | None = None,
) -> dict[str, Any]:
    if audio.empty or not settings.enable_audio:
        return {"files": [], "note": "audio_disabled"}
    transcripts_by_path = transcripts_by_path or {}
    rows: list[dict[str, Any]] = []
    for _, r in audio.iterrows():
        name = str(r.get("name") or "")
        pstr = str(r.get("path") or "")
        if biotag and biotag in name.replace("_", "-"):
            rows.append(
                {
                    "name": name,
                    "size_bytes": int(r.get("size_bytes") or 0),
                    "path": pstr,
                    "transcript_excerpt": (transcripts_by_path.get(pstr) or "")[:2000],
                }
            )
    if not rows:
        for _, r in audio.head(5).iterrows():
            pstr = str(r.get("path") or "")
            rows.append(
                {
                    "name": str(r.get("name")),
                    "size_bytes": int(r.get("size_bytes") or 0),
                    "path": pstr,
                    "transcript_excerpt": (transcripts_by_path.get(pstr) or "")[:2000],
                }
            )
    return {"files": rows[:8]}


def _should_transcribe(row: pd.Series, settings: Settings) -> bool:
    if not (settings.enable_audio and settings.enable_audio_transcription):
        return False
    pr = str(row.get("llm_review_priority", "low"))
    br = float(row.get("base_risk_score", 0.0) or 0.0)
    er = float(row.get("economic_risk_score", 0.0) or 0.0)
    return pr == "high" or br >= 0.48 or er >= 0.45


def _serialize_row(row: pd.Series) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            d[str(k)] = v.isoformat()
        elif isinstance(v, (float, int, str, bool)) or v is None:
            d[str(k)] = v
        else:
            d[str(k)] = str(v)
    return d


def _heuristic_arbitration(row: pd.Series) -> dict[str, float]:
    p = float(
        0.45 * row["base_risk_score"]
        + 0.25 * row["phishing_score"]
        + 0.18 * row["txn_anomaly_score"]
        + 0.12 * row["geo_score"]
    )
    p = max(p, 0.08 * row["economic_risk_score"])
    return {"final_fraud_probability": float(np.clip(p, 0, 1)), "final_economic_priority": float(row["economic_risk_score"])}


def _load_llm_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_llm_cache(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _select_candidates(tx: pd.DataFrame, review_budget: int) -> pd.DataFrame:
    mask = tx["llm_review_priority"].isin(["high", "medium"])
    cand = tx[mask].copy()
    if cand.empty:
        cand = tx.nlargest(min(5, len(tx)), "base_risk_score")
    cand = cand.sort_values(["base_risk_score", "economic_risk_score"], ascending=False)
    return cand.head(max(1, review_budget))


def run_review(
    data_dir: Path,
    output: Path,
    settings: Settings | None = None,
    review_budget: int = 40,
    threshold: float | None = None,
    top_k: int | None = None,
    percentile: float | None = None,
    mode: Mode = "adaptive",
    force_rescore: bool = False,
    session_id: str | None = None,
) -> RunSummary:
    settings = settings or load_settings()
    session_id = session_id or new_session_id(settings)
    lf = LangfuseRun(settings, session_id, dataset=str(data_dir))
    lf.start_trace()

    scored, cache_dir = _load_scored_tables(data_dir, settings, force_rescore=force_rescore)
    tables = load_normalized(cache_dir)

    cache_file = cache_dir / "llm_decisions.json"
    llm_cache = _load_llm_cache(cache_file)
    tx_cache_path = cache_dir / "audio_transcripts.json"
    tx_disk_cache = load_transcription_cache(tx_cache_path)

    client = OpenRouterClient(settings, session_id=session_id) if is_llm_configured(settings) else None

    candidates = _select_candidates(scored, review_budget=review_budget)
    reviewed_ids: set[str] = set()
    arbitration_probs: dict[str, float] = {}

    audio_reviewed = 0
    audio_transcription_calls = 0

    for _, row in candidates.iterrows():
        tid = str(row["transaction_id"])
        sig = stable_json_hash(
            {
                "tid": tid,
                "base": float(row["base_risk_score"]),
                "flags": row.get("reason_flags", ""),
            }
        )
        cache_key = f"{tid}:{sig}"
        if cache_key in llm_cache:
            arbitration_probs[tid] = float(llm_cache[cache_key]["final_fraud_probability"])
            reviewed_ids.add(tid)
            continue

        payload_common = {
            "transaction": _serialize_row(row.drop(labels=["phishing_flags"], errors="ignore")),
            "nearby_comms": _nearby_texts(tables["mails"], tables["sms"], row["timestamp"]),
            "cheap_scores": {
                "base_risk_score": float(row["base_risk_score"]),
                "economic_risk_score": float(row["economic_risk_score"]),
                "txn_anomaly_score": float(row["txn_anomaly_score"]),
                "geo_score": float(row["geo_score"]),
                "phishing_score": float(row["phishing_score"]),
                "flags": row.get("reason_flags", ""),
            },
        }

        if client is None:
            arb = _heuristic_arbitration(row)
            llm_cache[cache_key] = arb
            arbitration_probs[tid] = arb["final_fraud_probability"]
            reviewed_ids.add(tid)
            continue

        with lf.span("llm_review", {"transaction_id": tid}):
            ap_paths = _audio_paths_for_transcription(
                tables["audio"],
                str(row.get("sender_biotag")) if pd.notna(row.get("sender_biotag")) else None,
                settings,
            )
            transcript_map: dict[str, str] = {}
            tx_api = 0
            if ap_paths and _should_transcribe(row, settings):
                inv, soc, mob, eco, transcript_map, tx_api = run_specialists_with_transcription_parallel(
                    client,
                    payload_common,
                    row,
                    settings,
                    ap_paths,
                    tx_disk_cache,
                    tx_cache_path,
                )
                audio_transcription_calls += tx_api
                lf.log_event(
                    "audio_transcription",
                    {"transaction_id": tid, "api_calls": tx_api, "paths": [str(p) for p in ap_paths]},
                )
            else:
                inv, soc, mob, eco = run_specialists_parallel(
                    client,
                    payload_common,
                    row,
                    max_workers=settings.llm_parallel_workers,
                )

            audio_out = {"scam_signal_score": 0.0, "confidence": 0.0, "reasons": []}
            if settings.enable_audio:
                ap = _audio_payload(
                    tables["audio"],
                    str(row.get("sender_biotag")) if pd.notna(row.get("sender_biotag")) else None,
                    settings,
                    transcripts_by_path=transcript_map,
                )
                if ap.get("files"):
                    audio_out = audio_signal_agent.run_agent(client, ap).model_dump()
                    audio_reviewed += 1

            arb_payload = {
                "cheap": payload_common["cheap_scores"],
                "agents": {
                    "transaction_investigator": inv,
                    "social_engineering": soc,
                    "mobility": mob,
                    "economic": eco,
                    "audio": audio_out,
                },
            }
            arb_obj = arbitration_agent.run_agent(client, settings, arb_payload)
            arb = arb_obj.model_dump()
            lf.log_llm(
                "arbitration",
                settings.openrouter_model_strong,
                str(arb_payload)[:4000],
                arb,
                usage=None,
            )
            final_prob = float(arb_obj.final_fraud_probability)
        llm_cache[cache_key] = {"final_fraud_probability": final_prob}
        arbitration_probs[tid] = final_prob
        reviewed_ids.add(tid)

    _save_llm_cache(cache_file, llm_cache)

    scored = scored.copy()
    probs = scored["transaction_id"].astype(str).map(arbitration_probs)
    scored["arbitration_fraud_prob"] = pd.to_numeric(probs, errors="coerce").fillna(scored["base_risk_score"])

    scored = compute_final_scores(scored, settings)

    if mode == "top_k":
        mask = apply_threshold(scored, "top_k", threshold, top_k, None)
    elif mode == "percentile":
        mask = apply_threshold(scored, "percentile", threshold, None, percentile or 0.92)
    elif mode == "fixed":
        mask = apply_threshold(scored, "fixed", threshold or 0.55, None, None)
    else:
        mask = adaptive_threshold(scored)

    flagged = scored[mask].sort_values("final_rank_score", ascending=False)
    ids = flagged["transaction_id"].astype(str).tolist()

    write_predictions(ids, output)
    out_dir = output.parent
    write_langfuse_session(session_id, out_dir)
    mirror_default_outputs(output, out_dir)

    summary = RunSummary(
        dataset_path=str(data_dir.resolve()),
        total_transactions=int(len(scored)),
        flagged_count=len(ids),
        llm_reviewed=len(reviewed_ids),
        audio_reviewed=audio_reviewed,
        audio_transcription_calls=audio_transcription_calls,
        langfuse_session_id=session_id,
        score_stats={
            "final_rank_score_mean": float(scored["final_rank_score"].mean()),
            "final_rank_score_p95": float(scored["final_rank_score"].quantile(0.95)),
        },
    )
    write_run_summary(summary, out_dir)
    lf.flush()
    return summary
