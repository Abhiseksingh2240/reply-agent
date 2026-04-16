"""Short, cost-aware prompts."""

from __future__ import annotations


def transaction_investigator_prompt(payload: dict) -> list[dict[str, str]]:
    system = (
        "You are a bank fraud analyst. Judge if the described outbound transaction is suspicious. "
        "Output strict JSON with keys: suspicious (boolean), confidence (0-1 number), reasons (string array, max 5 short items)."
    )
    user = {"role": "user", "content": str(payload)[:8000]}
    return [{"role": "system", "content": system}, user]


def social_engineering_prompt(payload: dict) -> list[dict[str, str]]:
    system = (
        "You analyze SMS/email context for social engineering. "
        "Output JSON keys: manipulation_likelihood (0-1), confidence (0-1), reasons (string array, max 5)."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": str(payload)[:8000]}]


def mobility_prompt(payload: dict) -> list[dict[str, str]]:
    system = (
        "You assess geographic plausibility of a transaction versus mobility logs. "
        "Output JSON keys: presence_plausible (boolean), confidence (0-1), reasons (string array, max 5)."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": str(payload)[:8000]}]


def economic_prompt(payload: dict) -> list[dict[str, str]]:
    system = (
        "You assess economic severity and possible cash-out sequences. "
        "Output JSON keys: economic_severity (0-1), cash_out_sequence (boolean), confidence (0-1), reasons (string array, max 5)."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": str(payload)[:6000]}]


def audio_prompt(payload: dict) -> list[dict[str, str]]:
    system = (
        "You evaluate audio-related fraud signals from metadata or transcript snippets. "
        "Output JSON keys: scam_signal_score (0-1), confidence (0-1), reasons (string array, max 5)."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": str(payload)[:6000]}]


def arbitration_prompt(payload: dict) -> list[dict[str, str]]:
    system = (
        "You fuse deterministic risk scores with specialist agent JSON opinions. "
        "Output JSON keys: final_fraud_probability (0-1), final_economic_priority (0-1), "
        "recommend_fraud (boolean), explanation (single short sentence)."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": str(payload)[:9000]}]
