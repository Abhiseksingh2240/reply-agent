"""Pydantic models for structured agent outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentVerdict(BaseModel):
    verdict: Literal["suspicious", "not_suspicious", "plausible", "improbable", "manipulated", "benign"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class TransactionInvestigatorOut(BaseModel):
    suspicious: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class SocialEngineeringOut(BaseModel):
    manipulation_likelihood: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class MobilityPresenceOut(BaseModel):
    presence_plausible: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class EconomicImpactOut(BaseModel):
    economic_severity: float = Field(ge=0.0, le=1.0)
    cash_out_sequence: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class AudioSignalOut(BaseModel):
    scam_signal_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class ArbitrationOut(BaseModel):
    final_fraud_probability: float = Field(ge=0.0, le=1.0)
    final_economic_priority: float = Field(ge=0.0, le=1.0)
    recommend_fraud: bool
    explanation: str = ""


class RunSummary(BaseModel):
    dataset_path: str
    total_transactions: int
    flagged_count: int
    llm_reviewed: int
    audio_reviewed: int
    audio_transcription_calls: int = 0
    langfuse_session_id: str
    score_stats: dict[str, Any] = Field(default_factory=dict)
