"""CLI entrypoint for the CVPR 2026 DataCV Challenge."""

from pathlib import Path
from typing import Annotated, Literal

import typer

from src.config import Config, DatasetMode
from src.log import setup_logging
from src.submission import create_submission_zip
from src.task1 import run_task1_sync
from src.task2 import run_task2_sync

app = typer.Typer(
    name="datacv",
    help="CVPR 2026 DataCV Challenge — VLM visual illusion understanding",
    add_completion=False,
)


def _resolve_exp_name(exp_name: str, output_dir: Path) -> str:
    """If exp_name is empty, auto-generate run_001, run_002, etc."""
    if exp_name:
        return exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("run_"))
    if not existing:
        return "run_001"
    last_num = int(existing[-1].split("_")[1])
    return f"run_{last_num + 1:03d}"


def _parse_samples(samples_str: str) -> list[int]:
    """Parse comma-separated sample indices like '0,5,11,25'."""
    if not samples_str.strip():
        return []
    return [int(s.strip()) for s in samples_str.split(",")]


def _resolve_sample_selection(
    *,
    explicit_samples: str,
    limit: int,
) -> tuple[list[int], int, str]:
    """Resolve CLI selection options into a concrete sample list."""
    sample_indices = _parse_samples(explicit_samples)
    return sample_indices, limit, explicit_samples or "all"


@app.command()
def task1(
    model: Annotated[str, typer.Option(help="pydantic-ai model string")] = "google-gla:gemini-3-flash-preview",
    mode: Annotated[DatasetMode, typer.Option(help="Dataset split")] = DatasetMode.VAL,
    exp_name: Annotated[str, typer.Option(help="Experiment name (auto-increments if empty)")] = "",
    output_dir: Annotated[Path, typer.Option(help="Base output directory")] = Path("outputs"),
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.0,
    top_p: Annotated[float, typer.Option(help="Nucleus sampling top-p")] = 1.0,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    max_tokens: Annotated[int, typer.Option(help="Max output tokens")] = 4096,
    google_thinking_level: Annotated[
        Literal["minimal", "low", "medium", "high"] | None,
        typer.Option(help="Google Gemini thinking level: minimal|low|medium|high"),
    ] = None,
    limit: Annotated[int, typer.Option(help="Limit number of samples (0=all)")] = 0,
    samples: Annotated[str, typer.Option(help="Comma-separated sample indices (e.g. '0,5,11')")] = "",
    max_tool_rounds: Annotated[int, typer.Option(help="Max request rounds with tools")] = 10,
    max_concurrency: Annotated[int, typer.Option(help="Max concurrent sample requests")] = 15,
    allow_mixed_model_repair: Annotated[
        bool, typer.Option(help="Allow reusing snapshot records when prior run used a different model")
    ] = False,
    num_votes: Annotated[int, typer.Option(help="Majority voting: number of runs per sample")] = 1,
) -> None:
    """Run Task I: Binary VQA on visual illusion images (agentic mode)."""
    resolved_name = _resolve_exp_name(exp_name, output_dir)
    try:
        resolved_samples, resolved_limit, selection_desc = _resolve_sample_selection(
            explicit_samples=samples,
            limit=limit,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None

    config = Config(
        model_name=model,
        task1_mode=mode,
        exp_name=resolved_name,
        output_dir=output_dir,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
        google_thinking_level=google_thinking_level,
        limit=resolved_limit,
        samples=resolved_samples,
        max_tool_rounds=max_tool_rounds,
        max_concurrency=max_concurrency,
        allow_mixed_model_repair=allow_mixed_model_repair,
        num_votes=num_votes,
    )
    setup_logging(config.exp_dir)

    vote_str = f" [VOTE x{num_votes}]" if num_votes > 1 else ""
    typer.echo(
        f"Task I [{mode.value.upper()}]{vote_str} — model={model}, exp={resolved_name}, "
        f"limit={resolved_limit or 'all'}, samples={selection_desc}"
    )
    predictions = run_task1_sync(config)
    typer.echo(f"Done. {len(predictions)} predictions saved to {config.exp_dir}")


@app.command()
def task2(
    model: Annotated[str, typer.Option(help="pydantic-ai model string")] = "google-gla:gemini-3-flash-preview",
    mode: Annotated[DatasetMode, typer.Option(help="Dataset split")] = DatasetMode.VAL,
    exp_name: Annotated[str, typer.Option(help="Experiment name (auto-increments if empty)")] = "",
    output_dir: Annotated[Path, typer.Option(help="Base output directory")] = Path("outputs"),
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.0,
    top_p: Annotated[float, typer.Option(help="Nucleus sampling top-p")] = 1.0,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    max_tokens: Annotated[int, typer.Option(help="Max output tokens")] = 32000,
    google_thinking_level: Annotated[
        Literal["minimal", "low", "medium", "high"] | None,
        typer.Option(help="Google Gemini thinking level: minimal|low|medium|high"),
    ] = None,
    limit: Annotated[int, typer.Option(help="Limit number of samples (0=all)")] = 0,
    samples: Annotated[str, typer.Option(help="Comma-separated sample indices (e.g. '0,18,59')")] = "",
    max_tool_rounds: Annotated[int, typer.Option(help="Max request rounds with tools")] = 10,
    max_concurrency: Annotated[int, typer.Option(help="Max concurrent sample requests")] = 15,
    allow_mixed_model_repair: Annotated[
        bool, typer.Option(help="Allow reusing snapshot records when prior run used a different model")
    ] = False,
    num_votes: Annotated[int, typer.Option(help="Majority voting: number of runs per sample")] = 1,
) -> None:
    """Run Task II: MCQ on visual illusion and anomaly images (agentic mode)."""
    resolved_name = _resolve_exp_name(exp_name, output_dir)
    try:
        resolved_samples, resolved_limit, selection_desc = _resolve_sample_selection(
            explicit_samples=samples,
            limit=limit,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None

    config = Config(
        model_name=model,
        task2_mode=mode,
        exp_name=resolved_name,
        output_dir=output_dir,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
        google_thinking_level=google_thinking_level,
        limit=resolved_limit,
        samples=resolved_samples,
        max_tool_rounds=max_tool_rounds,
        max_concurrency=max_concurrency,
        allow_mixed_model_repair=allow_mixed_model_repair,
        num_votes=num_votes,
    )
    setup_logging(config.exp_dir)

    vote_str = f" [VOTE x{num_votes}]" if num_votes > 1 else ""
    typer.echo(
        f"Task II [{mode.value.upper()}]{vote_str} — model={model}, exp={resolved_name}, "
        f"limit={resolved_limit or 'all'}, samples={selection_desc}"
    )
    predictions = run_task2_sync(config)
    typer.echo(f"Done. {len(predictions)} predictions saved to {config.exp_dir}")


@app.command()
def submit(
    task: Annotated[int, typer.Argument(help="Task number (1 or 2)")],
    exp_name: Annotated[str, typer.Option(help="Experiment name")] = "",
    output_dir: Annotated[Path, typer.Option(help="Base output directory")] = Path("outputs"),
) -> None:
    """Package predictions + model.json into result.zip for submission."""
    if task not in (1, 2):
        typer.echo("Error: task must be 1 or 2", err=True)
        raise typer.Exit(code=1)

    exp_dir = output_dir / exp_name
    if not exp_dir.exists():
        typer.echo(f"Error: experiment directory not found: {exp_dir}", err=True)
        raise typer.Exit(code=1)

    predictions_name = "predictions.txt" if task == 1 else "result.txt"
    predictions_path = exp_dir / predictions_name
    model_info_path = exp_dir / "model.json"

    if not predictions_path.exists():
        typer.echo(f"Error: {predictions_path} not found", err=True)
        raise typer.Exit(code=1)
    if not model_info_path.exists():
        typer.echo(f"Error: {model_info_path} not found", err=True)
        raise typer.Exit(code=1)

    zip_path = exp_dir / "result.zip"
    create_submission_zip(predictions_path, model_info_path, zip_path)
    typer.echo(f"Submission zip created: {zip_path}")


if __name__ == "__main__":
    app()
