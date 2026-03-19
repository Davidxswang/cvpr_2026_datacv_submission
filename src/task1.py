"""Task I runner: Binary VQA on visual illusions (agentic mode with drawing tools)."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from PIL import Image
from pydantic_ai import Agent, BinaryContent, ModelResponse
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from src.agents import TASK1_SYSTEM_PROMPT, create_task1_agent, create_task1_rescue_agent
from src.models import ModelInfo, Prediction, Task1Output, Task1Sample
from src.pricing import compute_cost
from src.snapshot import (
    ExperimentSnapshot,
    SampleRecord,
    SampleUsage,
    is_sample_failed,
    load_experiment_snapshot,
    save_experiment_snapshot,
    serialize_messages,
)
from src.submission import save_model_info, save_predictions
from src.tools import ToolCallRecord, ToolDeps, create_tool_deps
from src.visualize import generate_report

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

_DEFAULT_FALLBACK_ANSWER = 1
_HTTP_MAX_RETRIES_429 = 60
_HTTP_MAX_RETRIES_503 = 6
_HTTP_MAX_BACKOFF_SECONDS = 30
_RESCUE_REQUEST_LIMIT = 2


def _image_to_png_content(image_path: Path) -> BinaryContent:
    """Load any image format and normalize to PNG BinaryContent for model input."""
    with Image.open(image_path) as img:
        has_alpha = img.mode in {"RGBA", "LA", "P"}
        normalized = img.convert("RGBA" if has_alpha else "RGB")
        with BytesIO() as buf:
            normalized.save(buf, format="PNG")
            png_bytes = buf.getvalue()
    return BinaryContent(data=png_bytes, media_type="image/png")


def _samples_from_csv(config: Config) -> list[Task1Sample]:
    """Load Task I evaluation samples from the selected CSV manifest."""
    df = pd.read_csv(config.task1_csv)
    samples: list[Task1Sample] = []
    for row_idx, row in df.iterrows():
        sample_index = int(str(row["index"])) if "index" in row.index else int(str(row_idx))
        ground_truth: int | None = int(row["answer"]) if "answer" in row.index else None
        samples.append(
            Task1Sample(
                index=sample_index,
                image_path=Path(Path(str(row["image_path"])).name),
                prompt=str(row["prompt"]),
                label=ground_truth,
            )
        )
    return samples


def _filter_samples(samples: list[Task1Sample], config: Config) -> list[Task1Sample]:
    """Apply --samples and --limit filters."""
    if config.samples:
        index_set = set(config.samples)
        samples = [s for s in samples if s.index in index_set]
        logger.info("Filtered to %d specific sample(s): %s", len(samples), sorted(index_set))
    elif config.limit > 0:
        samples = samples[: config.limit]
        logger.warning("Limiting to %d samples", config.limit)
    return samples


def _usage_from_result(result: object, model_name: str) -> SampleUsage:
    """Convert pydantic-ai usage object into our serializable usage model."""
    usage = result.usage()  # type: ignore[attr-defined]
    sample_usage = SampleUsage(
        requests=usage.requests,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        total_tokens=usage.total_tokens,
    )
    sample_usage.cost_usd = compute_cost(sample_usage, model_name)
    return sample_usage


def _load_existing_records(config: Config) -> dict[int, SampleRecord]:
    """Load existing successful/failed records for idempotent repair runs."""
    snapshot_path = config.exp_dir / "snapshot.json"
    if not snapshot_path.exists():
        return {}

    try:
        snapshot = load_experiment_snapshot(config.exp_dir)
    except Exception as exc:
        logger.warning("Unable to load existing snapshot at %s: %s", snapshot_path, exc)
        return {}

    if snapshot.task != "task1":
        logger.warning(
            "Existing snapshot task mismatch (found=%s, expected=task1); ignoring old snapshot.",
            snapshot.task,
        )
        return {}

    if snapshot.model_name != config.model_name and not config.allow_mixed_model_repair:
        logger.warning(
            "Existing snapshot model mismatch (found=%s, current=%s); ignoring old snapshot "
            "(set --allow-mixed-model-repair to reuse successful prior records).",
            snapshot.model_name,
            config.model_name,
        )
        return {}

    if snapshot.model_name != config.model_name and config.allow_mixed_model_repair:
        logger.warning(
            "Mixed-model repair enabled: reusing successful records from %s and repairing failed records with %s.",
            snapshot.model_name,
            config.model_name,
        )

    for record in snapshot.samples:
        if record.inference_model is None:
            record.inference_model = snapshot.model_name

    return {record.index: record for record in snapshot.samples}


def _build_task1_augmented_prompt(prompt: str, deps: ToolDeps) -> str:
    return (
        f"Image dimensions: {deps.original_width}x{deps.original_height} pixels (width x height).\n"
        f'The original image has resource ID: "original".\n\n'
        f"{prompt}"
    )


def _build_task1_rescue_prompt(prompt: str, error: Exception) -> str:
    """Build a final-attempt prompt after round/output-validation failures."""
    return (
        "Rescue pass: a prior tool-calling attempt failed due to a runtime limit or output validation issue.\n"
        f"Failure signal: {type(error).__name__}: {error}\n"
        "Now produce the final answer immediately with NO tool calls.\n\n"
        f"{prompt}"
    )


async def _process_sample_once(
    sample: Task1Sample,
    *,
    config: Config,
    agent: Agent[ToolDeps, Task1Output],
    rescue_agent: Agent[ToolDeps, Task1Output],
    usage_limits: UsageLimits | None,
) -> tuple[Prediction, SampleRecord]:
    """Run a single evaluation sample once (no voting)."""
    index = sample.index
    image_path = sample.image_path
    prompt = sample.prompt
    full_image_path = config.task1_image_dir / image_path
    exp_dir = config.exp_dir

    answer = _DEFAULT_FALLBACK_ANSWER
    reasoning = ""
    raw_response = ""
    sample_usage = SampleUsage()
    sample_tool_calls: list[ToolCallRecord] | None = None
    sample_conversation: list[dict[str, object]] | None = None
    status = "success"
    failure_reason: str | None = None
    rescue_used = False

    last_error: Exception | None = None

    image_content = _image_to_png_content(full_image_path)
    retry_429 = 0
    retry_503 = 0

    while True:
        try:
            deps = create_tool_deps(full_image_path, exp_dir=exp_dir, sample_index=index)
            sample_tool_calls = deps.tool_calls
            result = await agent.run(
                [image_content, _build_task1_augmented_prompt(prompt, deps)],
                deps=deps,
                usage_limits=usage_limits,
            )

            output: Task1Output = result.output
            answer = output.answer
            reasoning = output.reasoning
            all_msgs = result.all_messages()
            model_responses = [m for m in all_msgs if isinstance(m, ModelResponse)]
            raw_response = str(model_responses[-1]) if model_responses else ""
            sample_conversation = serialize_messages(all_msgs)
            sample_usage = _usage_from_result(result, config.model_name)
            last_error = None
            break

        except ModelHTTPError as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            if status_code == 429 and retry_429 < _HTTP_MAX_RETRIES_429:
                retry_429 += 1
                wait_seconds = min(2 ** min(retry_429, 5), _HTTP_MAX_BACKOFF_SECONDS)
                logger.warning(
                    "Sample %d got 429 (%d/%d) — retrying in %ds",
                    index,
                    retry_429,
                    _HTTP_MAX_RETRIES_429,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)
                continue
            if status_code == 503 and retry_503 < _HTTP_MAX_RETRIES_503:
                retry_503 += 1
                wait_seconds = min(2 ** min(retry_503, 5), _HTTP_MAX_BACKOFF_SECONDS)
                logger.warning(
                    "Sample %d got 503 (%d/%d) — retrying in %ds",
                    index,
                    retry_503,
                    _HTTP_MAX_RETRIES_503,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)
                continue
            logger.warning("Sample %d (%s) primary pass failed with %s", index, image_path, exc)
            break

        except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
            last_error = exc
            logger.warning("Sample %d (%s) primary pass hit limit/validation issue: %s", index, image_path, exc)
            break

        except Exception as exc:
            last_error = exc
            logger.exception("Sample %d (%s) primary pass failed unexpectedly", index, image_path)
            break

    if last_error is not None and isinstance(last_error, (UnexpectedModelBehavior, UsageLimitExceeded)):
        try:
            rescue_deps = create_tool_deps(full_image_path, exp_dir=exp_dir, sample_index=index)
            rescue_result = await rescue_agent.run(
                [image_content, _build_task1_rescue_prompt(prompt, last_error)],
                deps=rescue_deps,
                usage_limits=UsageLimits(request_limit=_RESCUE_REQUEST_LIMIT),
            )
            rescue_output: Task1Output = rescue_result.output
            answer = rescue_output.answer
            reasoning = f"RESCUE PASS after {type(last_error).__name__}: {last_error}\n\n{rescue_output.reasoning}"
            rescue_msgs = rescue_result.all_messages()
            model_responses = [m for m in rescue_msgs if isinstance(m, ModelResponse)]
            raw_response = str(model_responses[-1]) if model_responses else ""
            sample_conversation = serialize_messages(rescue_msgs)
            sample_usage = _usage_from_result(rescue_result, config.model_name)
            status = "success"
            failure_reason = None
            rescue_used = True
            last_error = None
        except Exception as rescue_exc:
            last_error = rescue_exc
            logger.warning("Sample %d (%s) rescue pass failed: %s", index, image_path, rescue_exc)

    if last_error is not None:
        status = "failed"
        failure_reason = f"{type(last_error).__name__}: {last_error}"
        answer = _DEFAULT_FALLBACK_ANSWER
        reasoning = f"FALLBACK: {failure_reason}"

    ground_truth = sample.label
    is_correct: bool | None = None
    if ground_truth is not None:
        is_correct = answer == ground_truth

    record = SampleRecord(
        index=index,
        image_path=str(image_path),
        user_prompt=prompt,
        model_raw_response=raw_response,
        model_reasoning=reasoning,
        model_answer=answer,
        inference_model=config.model_name,
        ground_truth=ground_truth,
        is_correct=is_correct,
        status=status,
        failure_reason=failure_reason,
        rescue_used=rescue_used,
        usage=sample_usage,
        tool_calls=sample_tool_calls,
        conversation=sample_conversation,
        timestamp=SampleRecord.now_timestamp(),
    )
    prediction = Prediction(index=index, answer=answer)
    return prediction, record


async def _process_sample(
    sample: Task1Sample,
    *,
    config: Config,
    agent: Agent[ToolDeps, Task1Output],
    rescue_agent: Agent[ToolDeps, Task1Output],
    usage_limits: UsageLimits | None,
    semaphore: asyncio.Semaphore,
) -> tuple[Prediction, SampleRecord]:
    """Process a sample with concurrency control and optional majority voting."""
    async with semaphore:
        if config.num_votes <= 1:
            pred, record = await _process_sample_once(
                sample,
                config=config,
                agent=agent,
                rescue_agent=rescue_agent,
                usage_limits=usage_limits,
            )
            logger.info(
                "Sample %d → %d (correct=%s, status=%s)",
                sample.index,
                pred.answer,
                record.is_correct,
                record.status,
            )
            return pred, record

        # Majority voting: run N times and pick the most common answer.
        votes: list[tuple[Prediction, SampleRecord]] = []
        for vote_idx in range(config.num_votes):
            pred, record = await _process_sample_once(
                sample,
                config=config,
                agent=agent,
                rescue_agent=rescue_agent,
                usage_limits=usage_limits,
            )
            votes.append((pred, record))
            logger.info(
                "Sample %d vote %d/%d → %d (status=%s)",
                sample.index,
                vote_idx + 1,
                config.num_votes,
                pred.answer,
                record.status,
            )

        answer_counts: Counter[int] = Counter(p.answer for p, _ in votes)
        majority_answer = answer_counts.most_common(1)[0][0]

        matching_records = [r for p, r in votes if p.answer == majority_answer]
        assert matching_records
        best_record = next((r for r in matching_records if not is_sample_failed(r)), matching_records[0])

        vote_answers = [p.answer for p, _ in votes]
        best_record.model_reasoning = (
            f"MAJORITY VOTE ({dict(answer_counts)}): {majority_answer}\n"
            f"Individual votes: {vote_answers}\n\n"
            f"{best_record.model_reasoning}"
        )
        best_record.model_answer = majority_answer
        if best_record.ground_truth is not None:
            best_record.is_correct = majority_answer == best_record.ground_truth

        best_record.usage = SampleUsage(
            requests=sum(r.usage.requests for _, r in votes),
            input_tokens=sum(r.usage.input_tokens for _, r in votes),
            output_tokens=sum(r.usage.output_tokens for _, r in votes),
            cache_read_tokens=sum(r.usage.cache_read_tokens for _, r in votes),
            cache_write_tokens=sum(r.usage.cache_write_tokens for _, r in votes),
            total_tokens=sum(r.usage.total_tokens for _, r in votes),
            cost_usd=sum(r.usage.cost_usd for _, r in votes),
        )

        if all(is_sample_failed(r) for r in matching_records):
            best_record.status = "failed"
            if best_record.failure_reason is None:
                best_record.failure_reason = "Majority answer came from failed attempts"
        else:
            best_record.status = "success"
            best_record.failure_reason = None

        final_pred = Prediction(index=sample.index, answer=majority_answer)
        logger.info(
            "Sample %d → %d (majority of %s, correct=%s, status=%s)",
            sample.index,
            majority_answer,
            vote_answers,
            best_record.is_correct,
            best_record.status,
        )
        return final_pred, best_record


async def run_task1(config: Config) -> list[Prediction]:
    """Run Task I inference on all samples using agentic drawing tools.

    Returns:
        List of Prediction(index, answer) objects.
    """
    exp_dir = config.exp_dir

    agent = create_task1_agent(config)
    rescue_agent = create_task1_rescue_agent(config)

    usage_limits: UsageLimits = UsageLimits(request_limit=config.max_tool_rounds)
    logger.info("Using agentic mode with drawing tools (max %d rounds)", config.max_tool_rounds)
    logger.info("Rescue pass enabled for round/output-validation failures")
    if config.num_votes > 1:
        logger.info("Majority voting enabled: %d votes per sample", config.num_votes)

    eval_samples = _samples_from_csv(config)
    eval_samples = _filter_samples(eval_samples, config)

    existing_records = _load_existing_records(config)
    samples_to_run: list[Task1Sample] = []
    reused_records: dict[int, SampleRecord] = {}
    reused_indices: list[int] = []
    for sample in eval_samples:
        existing = existing_records.get(sample.index)
        if existing is None:
            samples_to_run.append(sample)
            continue
        if is_sample_failed(existing):
            samples_to_run.append(sample)
            logger.info("Sample %d marked failed in existing snapshot; scheduling repair.", sample.index)
            continue
        reused_records[sample.index] = existing
        reused_indices.append(sample.index)

    logger.info(
        "Task1 idempotent run: total=%d, reuse_success=%d, rerun=%d",
        len(eval_samples),
        len(reused_records),
        len(samples_to_run),
    )
    if reused_indices:
        preview = reused_indices[:20]
        suffix = "" if len(reused_indices) <= 20 else " ..."
        logger.info(
            "Reused successful samples from snapshot (count=%d, first %d): %s%s",
            len(reused_indices),
            len(preview),
            preview,
            suffix,
        )

    new_records: dict[int, SampleRecord] = {}
    if samples_to_run:
        semaphore = asyncio.Semaphore(config.max_concurrency)
        logger.info("Processing %d samples with max_concurrency=%d", len(samples_to_run), config.max_concurrency)

        tasks = [
            _process_sample(
                sample,
                config=config,
                agent=agent,
                rescue_agent=rescue_agent,
                usage_limits=usage_limits,
                semaphore=semaphore,
            )
            for sample in samples_to_run
        ]
        results = await asyncio.gather(*tasks)
        for _, record in results:
            new_records[record.index] = record

    sample_records: list[SampleRecord] = []
    for sample in eval_samples:
        if sample.index in new_records:
            sample_records.append(new_records[sample.index])
            continue
        existing = reused_records.get(sample.index)
        if existing is None:
            raise RuntimeError(f"Internal merge error: missing record for sample {sample.index}")
        sample_records.append(existing)

    selected_indices = {sample.index for sample in eval_samples}
    preserved_count = 0
    for existing in existing_records.values():
        if existing.index in selected_indices:
            continue
        sample_records.append(existing)
        preserved_count += 1
    if preserved_count:
        logger.info("Preserved %d untouched sample record(s) from existing snapshot", preserved_count)

    sample_records.sort(key=lambda s: s.index)
    predictions = [Prediction(index=record.index, answer=record.model_answer) for record in sample_records]

    labeled = [s for s in sample_records if s.is_correct is not None]
    if labeled:
        correct = sum(1 for s in labeled if s.is_correct)
        total = len(labeled)
        acc = correct / total
        logger.warning("Accuracy: %d/%d = %.1f%%", correct, total, acc * 100)

        gt1 = [s for s in labeled if s.ground_truth == 1]
        gt0 = [s for s in labeled if s.ground_truth == 0]
        if gt1:
            acc1 = sum(1 for s in gt1 if s.is_correct) / len(gt1)
            logger.warning(
                "  Original-img ACC (GT=1): %d/%d = %.1f%%",
                sum(1 for s in gt1 if s.is_correct),
                len(gt1),
                acc1 * 100,
            )
        if gt0:
            acc0 = sum(1 for s in gt0 if s.is_correct) / len(gt0)
            logger.warning(
                "  Perturbed-img ACC (GT=0): %d/%d = %.1f%%",
                sum(1 for s in gt0 if s.is_correct),
                len(gt0),
                acc0 * 100,
            )

    failed_count = sum(1 for s in sample_records if is_sample_failed(s))
    if failed_count:
        logger.warning("Task1 completed with %d fallback sample(s) still failing after repair", failed_count)

    total_usage = SampleUsage(
        requests=sum(s.usage.requests for s in sample_records),
        input_tokens=sum(s.usage.input_tokens for s in sample_records),
        output_tokens=sum(s.usage.output_tokens for s in sample_records),
        cache_read_tokens=sum(s.usage.cache_read_tokens for s in sample_records),
        cache_write_tokens=sum(s.usage.cache_write_tokens for s in sample_records),
        total_tokens=sum(s.usage.total_tokens for s in sample_records),
        cost_usd=sum(s.usage.cost_usd for s in sample_records),
    )

    save_experiment_snapshot(
        ExperimentSnapshot(
            exp_name=config.exp_name,
            task="task1",
            model_name=config.model_name,
            system_prompt=TASK1_SYSTEM_PROMPT,
            samples=sample_records,
            total_usage=total_usage,
            created_at=ExperimentSnapshot.now_timestamp(),
        ),
        exp_dir,
    )

    save_predictions(predictions, exp_dir / "predictions.txt")
    model_usage_counts: dict[str, int] = {}
    for record in sample_records:
        model_key = record.inference_model or config.model_name
        model_usage_counts[model_key] = model_usage_counts.get(model_key, 0) + 1
    models_used = sorted(model_usage_counts)

    model_parameters: dict[str, float | int | str | bool | list[str] | dict[str, int]] = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": config.seed,
        "max_tokens": config.max_tokens,
        "num_votes": config.num_votes,
    }
    if config.google_thinking_level is not None:
        model_parameters["google_thinking_level"] = config.google_thinking_level
    model_parameters["models_used"] = models_used
    model_parameters["sample_count_by_model"] = model_usage_counts
    model_parameters["mixed_model_repair"] = len(models_used) > 1
    model_parameters["repair_model"] = config.model_name
    model_name_for_submission = models_used[0] if len(models_used) == 1 else config.model_name

    save_model_info(
        ModelInfo(
            model=model_name_for_submission,
            parameters=model_parameters,
        ),
        exp_dir / "model.json",
    )

    generate_report(
        exp_dir=exp_dir,
        output_path=exp_dir / "report.html",
        resolve_images_from=config.task1_image_dir.resolve(),
    )

    return predictions


def run_task1_sync(config: Config) -> list[Prediction]:
    """Synchronous wrapper for run_task1."""
    return asyncio.run(run_task1(config))
