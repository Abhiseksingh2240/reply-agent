"""CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print

from app.config import load_settings
from app.pipelines.build_features import run_build_features
from app.pipelines.review_transactions import run_review

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("build-features")
def build_features(
    data_dir: Path = typer.Option(..., "--data-dir", help="Dataset directory containing CSV/JSON"),
    force: bool = typer.Option(False, "--force", help="Rebuild cached normalized tables"),
) -> None:
    """Normalize inputs and compute deterministic scores."""
    settings = load_settings()
    path = run_build_features(data_dir, settings=settings, force=force)
    print(f"[green]Wrote scored features to[/green] {path}")


@app.command("review")
def review(
    data_dir: Path = typer.Option(..., "--data-dir"),
    output: Path = typer.Option(Path("outputs/preds.txt"), "--output"),
    review_budget: int = typer.Option(40, "--review-budget"),
    threshold: Optional[float] = typer.Option(None, "--threshold"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    percentile: Optional[float] = typer.Option(None, "--percentile"),
    mode: str = typer.Option("adaptive", "--mode", help="adaptive|fixed|top_k|percentile"),
    force_rescore: bool = typer.Option(False, "--force-rescore"),
    enable_audio: bool = typer.Option(False, "--enable-audio"),
    enable_transcription: bool = typer.Option(False, "--enable-transcription"),
    llm_workers: int = typer.Option(0, "--llm-workers", help="Override LLM_PARALLEL_WORKERS when >0"),
) -> None:
    """Run selective LLM review and write predictions."""
    settings = load_settings()
    settings.enable_audio = enable_audio or settings.enable_audio
    settings.enable_audio_transcription = enable_transcription
    if llm_workers > 0:
        settings.llm_parallel_workers = llm_workers
    summary = run_review(
        data_dir=data_dir,
        output=output,
        settings=settings,
        review_budget=review_budget,
        threshold=threshold,
        top_k=top_k,
        percentile=percentile,
        mode=mode,  # type: ignore[arg-type]
        force_rescore=force_rescore,
    )
    print(f"[bold]Langfuse session ID:[/bold] {summary.langfuse_session_id}")
    print(f"[bold]Flagged[/bold] {summary.flagged_count} / {summary.total_transactions}")
    if summary.audio_transcription_calls:
        print(f"[bold]Audio transcription API calls:[/bold] {summary.audio_transcription_calls}")


@app.command("run")
def run_all(
    data_dir: Path = typer.Option(..., "--data-dir"),
    output: Path = typer.Option(Path("outputs/preds.txt"), "--output"),
    review_budget: int = typer.Option(40, "--review-budget"),
    threshold: Optional[float] = typer.Option(None, "--threshold"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    percentile: Optional[float] = typer.Option(None, "--percentile"),
    mode: str = typer.Option("adaptive", "--mode"),
    force: bool = typer.Option(False, "--force", help="Rebuild caches"),
    enable_audio: bool = typer.Option(False, "--enable-audio"),
    enable_transcription: bool = typer.Option(False, "--enable-transcription"),
    llm_workers: int = typer.Option(0, "--llm-workers", help="Override LLM_PARALLEL_WORKERS when >0"),
) -> None:
    """End-to-end: build features then review."""
    settings = load_settings()
    settings.enable_audio = enable_audio or settings.enable_audio
    settings.enable_audio_transcription = enable_transcription
    if llm_workers > 0:
        settings.llm_parallel_workers = llm_workers
    run_build_features(data_dir, settings=settings, force=force)
    summary = run_review(
        data_dir=data_dir,
        output=output,
        settings=settings,
        review_budget=review_budget,
        threshold=threshold,
        top_k=top_k,
        percentile=percentile,
        mode=mode,  # type: ignore[arg-type]
        force_rescore=force,
    )
    print(f"[bold]Langfuse session ID:[/bold] {summary.langfuse_session_id}")
    print(f"[bold]Flagged[/bold] {summary.flagged_count} / {summary.total_transactions}")
    if summary.audio_transcription_calls:
        print(f"[bold]Audio transcription API calls:[/bold] {summary.audio_transcription_calls}")


if __name__ == "__main__":
    app()
