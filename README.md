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

**Security:** never commit real keys. If keys were pasted into a chat or ticket, rotate them in the provider dashboards.

## Commands

From the `fraud_agent` directory (so `app` is importable):

```bash
python -m app.main build-features --data-dir "path/to/dataset"
python -m app.main review --data-dir "path/to/dataset" --output ./outputs/preds.txt
python -m app.main run --data-dir "path/to/dataset" --output ./outputs/preds.txt
```

Useful flags on `review` / `run`:

- `--review-budget N` — cap LLM-reviewed transactions (each shortlisted row runs the specialist stack + arbitration).
- `--mode adaptive|fixed|top_k|percentile` with `--threshold`, `--top-k`, or `--percentile`.
- `--enable-audio` — attach linked audio file metadata (paths, sizes, optional transcript excerpts) to the audio agent payload.
- `--enable-transcription` — set `ENABLE_AUDIO_TRANSCRIPTION` for this run. When enabled with `--enable-audio`, high-risk / high-value shortlisted rows trigger **OpenRouter `input_audio` transcription** (base64, format from extension). Results are cached in `data_cache/.../audio_transcripts.json` by file fingerprint. The four specialist agents and pending transcriptions run **in one thread pool** to overlap network latency.
- `--llm-workers N` — override `LLM_PARALLEL_WORKERS` for this run (when `N > 0`).
- `--force` / `--force-rescore` — rebuild normalized caches.

Environment knobs: `OPENROUTER_MODEL_TRANSCRIPTION` (must support audio input on OpenRouter), `AUDIO_TRANSCRIPTION_MAX_BYTES`, `AUDIO_TRANSCRIPTION_MAX_FILES`, `LLM_PARALLEL_WORKERS`.

## Outputs and submission

After `run` or `review`:

- Predictions: `--output` path (UTF-8, one `transaction_id` per line).
- Copy for naming convenience: `outputs/fraudulent_transactions.txt`.
- Langfuse session id: `outputs/langfuse_session_id.txt` (also printed to stdout).
- Run metadata: `outputs/run_summary.json` (includes `audio_transcription_calls`: OpenRouter transcription requests excluding cache hits).

Evaluation zip should include **complete source**, `requirements.txt`, `.env.example`, and this README with the same commands.

## Tests

```bash
set PYTHONPATH=%CD%
python -m pytest
```

## Generalization notes

Rules avoid dataset-specific hardcoding (no fixed IBANs, cities, or challenge IDs). Phishing, mobility, and monetary signals are expressed as **relative** behaviors (salary ratios, rolling activity, text heuristics, time windows).

Transcription runs only when audio is enabled, transcription is enabled, and the row meets **risk gates** (`llm_review_priority == high`, or `base_risk_score` / `economic_risk_score` above configurable thresholds in code) so API spend stays focused on ambiguous or high-impact cases.
