# Fraud Agent (Reply AI Agent Challenge 2026)

Python 3.11+ hybrid fraud pipeline: **deterministic screening on every transaction**, **selective multi-agent LLM review** (OpenRouter), and **Langfuse** tracing with a single session per run.

## Architecture

1. **Ingestion** (`app/io_loader.py`, `app/normalizer.py`, `app/feature_store.py`): load `transactions.csv`, `users.json`, `locations.json`, `mails.json`, `sms.json`, optional `audio/` listing; normalize to UTC timestamps, snake_case columns, cached Parquet under `data_cache/`.
2. **Cheap scoring** (`app/scoring.py`): transaction anomaly, geo-temporal consistency, phishing exposure, user vulnerability; outputs `base_risk_score`, `economic_risk_score`, `reason_flags`, `llm_review_priority`.
3. **Selective agents** (`app/agents/` + `app/llm/`): investigator, social engineering, mobility, economic impact, optional audio, then arbitration (stronger model). Calls are **budgeted** and **cached** per transaction hash in `data_cache/.../llm_decisions.json`.
4. **Fusion & thresholding** (`app/fusion.py`): configurable weights for fraud vs economic vs cluster signal; `adaptive` (default), `fixed`, `top_k`, or `percentile` modes with guards against empty outputs.
5. **Outputs** (`app/output_writer.py`): UTF-8 prediction file, `langfuse_session_id.txt`, `run_summary.json`, and `fraudulent_transactions.txt` mirroring the primary `--output` path.

## Setup

```bash
cd fraud_agent
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
copy .env.example .env   # then fill keys
```

### Environment

See `.env.example`. Required for full agent behavior:

- `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL_CHEAP`, `OPENROUTER_MODEL_STRONG`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` (challenge Langfuse URL)

If OpenRouter keys are missing, the pipeline still runs using **heuristic fusion** (no network calls) so CI and dry runs stay reproducible.

## Commands

From the `fraud_agent` directory:

```bash
python -m app.main build-features --data-dir "path/to/dataset"
python -m app.main review --data-dir "path/to/dataset" --output ./outputs/preds.txt
python -m app.main run --data-dir "path/to/dataset" --output ./outputs/preds.txt
```

Useful flags on `review` / `run`:

- `--review-budget N` — cap LLM-reviewed transactions.
- `--mode adaptive|fixed|top_k|percentile` with `--threshold`, `--top-k`, or `--percentile`.
- `--enable-audio` — attach linked audio metadata to the audio agent payload.
- `--enable-transcription` — enable high-risk audio transcription with cached transcript excerpts.
- `--llm-workers N` — override `LLM_PARALLEL_WORKERS` for this run.
- `--force` / `--force-rescore` — rebuild normalized caches.

## Outputs and submission

After `run` or `review`:

- Predictions: `--output` path (UTF-8, one `transaction_id` per line).
- Copy for naming convenience: `outputs/fraudulent_transactions.txt`.
- Langfuse session id: `outputs/langfuse_session_id.txt`.
- Run metadata: `outputs/run_summary.json`.

## Tests

```bash
set PYTHONPATH=%CD%
python -m pytest
```
