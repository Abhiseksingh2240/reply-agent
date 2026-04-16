"""Parallel OpenRouter calls for specialist agents and optional transcription."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from app.agents import (
    economic_impact_agent,
    mobility_presence_agent,
    social_engineering_agent,
    transaction_investigator,
)
from app.config import Settings
from app.llm.audio_transcription import (
    save_transcription_cache,
    transcribe_audio_file,
    transcription_cache_key,
)
from app.llm.openrouter_client import OpenRouterClient


def run_specialists_parallel(
    client: OpenRouterClient,
    payload_common: dict[str, Any],
    row: pd.Series,
    max_workers: int = 4,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run the four non-arbitration agents concurrently."""

    def _inv() -> dict[str, Any]:
        return transaction_investigator.run_agent(client, payload_common).model_dump()

    def _soc() -> dict[str, Any]:
        return social_engineering_agent.run_agent(
            client,
            {
                "nearby": payload_common["nearby_comms"],
                "phishing_score": float(row["phishing_score"]),
                "user_vulnerability": float(row["user_vulnerability"]),
            },
        ).model_dump()

    def _mob() -> dict[str, Any]:
        return mobility_presence_agent.run_agent(
            client,
            {
                "transaction_time": str(row["timestamp"]),
                "location_field": str(row.get("location", "")),
                "geo_scores": {
                    "geo_score": float(row["geo_score"]),
                    "geo_distance_anomaly": float(row.get("geo_distance_anomaly", 0.0)),
                },
            },
        ).model_dump()

    def _eco() -> dict[str, Any]:
        return economic_impact_agent.run_agent(
            client,
            {
                "amount": float(row["amount"]),
                "salary": float(row.get("salary", 0.0)),
                "balance_after": float(row.get("balance_after", 0.0)),
            },
        ).model_dump()

    workers = max(1, min(int(max_workers), 4))
    names = ("inv", "soc", "mob", "eco")
    funcs = (_inv, _soc, _mob, _eco)
    defaults: dict[str, dict[str, Any]] = {
        "inv": {"suspicious": False, "confidence": 0.0, "reasons": []},
        "soc": {"manipulation_likelihood": 0.0, "confidence": 0.0, "reasons": []},
        "mob": {"presence_plausible": True, "confidence": 0.0, "reasons": []},
        "eco": {
            "economic_severity": 0.0,
            "cash_out_sequence": False,
            "confidence": 0.0,
            "reasons": [],
        },
    }
    results = {k: dict(v) for k, v in defaults.items()}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(fn): name for fn, name in zip(funcs, names)}
        for fut in as_completed(list(fut_map.keys())):
            name = fut_map[fut]
            try:
                results[name] = fut.result()
            except Exception:
                results[name] = dict(defaults[name])
    return results["inv"], results["soc"], results["mob"], results["eco"]


def run_specialists_with_transcription_parallel(
    client: OpenRouterClient,
    payload_common: dict[str, Any],
    row: pd.Series,
    settings: Settings,
    ap_paths: list[Path],
    tx_disk_cache: dict[str, str],
    tx_cache_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str], int]:
    """Run four specialists and pending audio transcriptions in one thread pool."""
    transcript_map: dict[str, str] = {}
    pending_tr: list[tuple[str, Path]] = []
    for p in ap_paths:
        ck = transcription_cache_key(p)
        ps = str(p)
        if ck in tx_disk_cache:
            transcript_map[ps] = tx_disk_cache[ck]
        else:
            pending_tr.append((ck, p))

    pool = min(12, max(4, int(settings.llm_parallel_workers) + max(1, len(pending_tr))))

    defaults: dict[str, dict[str, Any]] = {
        "inv": {"suspicious": False, "confidence": 0.0, "reasons": []},
        "soc": {"manipulation_likelihood": 0.0, "confidence": 0.0, "reasons": []},
        "mob": {"presence_plausible": True, "confidence": 0.0, "reasons": []},
        "eco": {
            "economic_severity": 0.0,
            "cash_out_sequence": False,
            "confidence": 0.0,
            "reasons": [],
        },
    }
    results: dict[str, dict[str, Any]] = {k: dict(v) for k, v in defaults.items()}

    def _inv() -> dict[str, Any]:
        return transaction_investigator.run_agent(client, payload_common).model_dump()

    def _soc() -> dict[str, Any]:
        return social_engineering_agent.run_agent(
            client,
            {
                "nearby": payload_common["nearby_comms"],
                "phishing_score": float(row["phishing_score"]),
                "user_vulnerability": float(row["user_vulnerability"]),
            },
        ).model_dump()

    def _mob() -> dict[str, Any]:
        return mobility_presence_agent.run_agent(
            client,
            {
                "transaction_time": str(row["timestamp"]),
                "location_field": str(row.get("location", "")),
                "geo_scores": {
                    "geo_score": float(row["geo_score"]),
                    "geo_distance_anomaly": float(row.get("geo_distance_anomaly", 0.0)),
                },
            },
        ).model_dump()

    def _eco() -> dict[str, Any]:
        return economic_impact_agent.run_agent(
            client,
            {
                "amount": float(row["amount"]),
                "salary": float(row.get("salary", 0.0)),
                "balance_after": float(row.get("balance_after", 0.0)),
            },
        ).model_dump()

    meta: dict[Any, tuple[Any, ...]] = {}
    with ThreadPoolExecutor(max_workers=pool) as ex:
        meta[ex.submit(_inv)] = ("inv",)
        meta[ex.submit(_soc)] = ("soc",)
        meta[ex.submit(_mob)] = ("mob",)
        meta[ex.submit(_eco)] = ("eco",)
        for ck, p in pending_tr:
            meta[ex.submit(transcribe_audio_file, client, p, settings)] = ("tr", ck, str(p))

        for fut in as_completed(list(meta.keys())):
            tag = meta[fut][0]
            try:
                if tag == "tr":
                    _, ck, pstr = meta[fut]
                    txt = fut.result()
                    tx_disk_cache[ck] = txt
                    transcript_map[pstr] = txt
                else:
                    results[tag] = fut.result()
            except Exception:
                if tag == "tr":
                    _, ck, pstr = meta[fut]
                    tx_disk_cache[ck] = ""
                    transcript_map[pstr] = ""
                else:
                    results[tag] = dict(defaults[tag])

    if pending_tr:
        save_transcription_cache(tx_cache_path, tx_disk_cache)

    return results["inv"], results["soc"], results["mob"], results["eco"], transcript_map, len(pending_tr)
