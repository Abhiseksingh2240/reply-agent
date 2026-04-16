"""LLM agent: economic impact."""

from __future__ import annotations

from typing import Any

from app.llm.openrouter_client import OpenRouterClient
from app.llm import prompts
from app.models import EconomicImpactOut


def run_agent(client: OpenRouterClient, payload: dict[str, Any]) -> EconomicImpactOut:
    msgs = prompts.economic_prompt(payload)
    data = client.chat_json(msgs)
    return EconomicImpactOut.model_validate(
        {
            "economic_severity": float(data.get("economic_severity", 0.0)),
            "cash_out_sequence": bool(data.get("cash_out_sequence", False)),
            "confidence": float(data.get("confidence", 0.0)),
            "reasons": list(data.get("reasons") or []),
        }
    )
