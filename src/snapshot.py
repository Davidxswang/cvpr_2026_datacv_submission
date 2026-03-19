"""Conversation snapshot system for debugging prompt engineering.

One snapshot.json per experiment containing all sample records.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from src.tools import ToolCallRecord

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic_ai.messages import ModelMessage

logger = logging.getLogger(__name__)


def _serialize_part(part: object) -> dict[str, Any]:
    """Serialize a single message part to a JSON-friendly dict.

    Binary content (images) is replaced with a placeholder string to keep
    the snapshot readable.
    """
    if not dataclasses.is_dataclass(part):
        return {"kind": type(part).__name__, "repr": repr(part)}

    result: dict[str, Any] = {}
    for f in dataclasses.fields(part):
        val = getattr(part, f.name)
        if isinstance(val, (str, int, float, bool, type(None))):
            result[f.name] = val
        elif isinstance(val, bytes):
            result[f.name] = f"<binary {len(val)} bytes>"
        elif isinstance(val, (list, tuple)):
            serialized: list[Any] = []
            for item in val:
                if isinstance(item, (str, int, float, bool, type(None))):
                    serialized.append(item)
                elif isinstance(item, bytes):
                    serialized.append(f"<binary {len(item)} bytes>")
                elif dataclasses.is_dataclass(item):
                    serialized.append(_serialize_part(item))
                else:
                    serialized.append(str(item))
            result[f.name] = serialized
        elif hasattr(val, "isoformat"):
            result[f.name] = val.isoformat()
        else:
            result[f.name] = str(val)
    return result


def serialize_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Serialize pydantic-ai ModelMessage list to JSON-serializable dicts.

    Preserves the full conversation structure: system prompts, user prompts,
    model thinking, tool calls, tool returns, and text responses. Binary
    content (images) is replaced with size placeholders.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {"kind": getattr(msg, "kind", type(msg).__name__)}
        if hasattr(msg, "parts"):
            entry["parts"] = [_serialize_part(p) for p in msg.parts]
        if hasattr(msg, "model_name") and msg.model_name:
            entry["model_name"] = msg.model_name
        result.append(entry)
    return result


class FewShotEntry(BaseModel):
    """A single few-shot example in a snapshot."""

    image_path: str
    prompt: str
    label: int
    similarity: float | None = None  # cosine similarity to test image (when using "similar" strategy)


class SampleUsage(BaseModel):
    """Token usage for a single sample inference."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class SampleRecord(BaseModel):
    """Conversation record for one sample within an experiment snapshot."""

    index: int
    image_path: str  # relative path to the query image
    user_prompt: str  # the text prompt sent with the image
    model_raw_response: str  # full raw text from the model
    model_reasoning: str  # extracted reasoning/thinking
    model_answer: int  # parsed final answer
    inference_model: str | None = None  # model used to produce this sample answer
    ground_truth: int | None = None  # if available
    is_correct: bool | None = None
    status: str = "success"  # "success" or "failed" for explicit repair detection
    failure_reason: str | None = None  # populated when status == "failed"
    rescue_used: bool = False  # true if rescue pass produced the final answer
    usage: SampleUsage = SampleUsage()
    few_shot_examples: list[FewShotEntry] | None = None  # per-sample few-shot (when using "similar" strategy)
    tool_calls: list[ToolCallRecord] | None = None  # tool call log (agentic mode only)
    conversation: list[dict[str, Any]] | None = None  # full message chain (agentic mode)
    timestamp: str

    @staticmethod
    def now_timestamp() -> str:
        return datetime.now(tz=UTC).isoformat()


def is_sample_failed(record: SampleRecord) -> bool:
    """Return True if a sample should be treated as failed for repair reruns.

    Supports both new explicit status fields and legacy snapshots that only
    tagged fallbacks in the reasoning text.
    """
    if record.status.lower() == "failed":
        return True
    return record.model_reasoning.strip().startswith("FALLBACK:")


class ExperimentSnapshot(BaseModel):
    """Complete experiment snapshot — all samples in one file."""

    exp_name: str
    task: str  # "task1" or "task2"
    model_name: str
    system_prompt: str
    few_shot_examples: list[FewShotEntry] = []  # Legacy; kept for backward compat with old snapshots
    samples: list[SampleRecord]
    total_usage: SampleUsage = SampleUsage()  # aggregated across all samples
    created_at: str

    @staticmethod
    def now_timestamp() -> str:
        return datetime.now(tz=UTC).isoformat()


# Resolve forward references (e.g., ToolCallRecord in SampleRecord.tool_calls)
# so this module works when imported standalone.
SampleRecord.model_rebuild()
ExperimentSnapshot.model_rebuild()


def save_experiment_snapshot(snapshot: ExperimentSnapshot, exp_dir: Path) -> Path:
    """Write the experiment snapshot to exp_dir/snapshot.json.

    Returns:
        Path to the written snapshot file.
    """
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / "snapshot.json"
    path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved experiment snapshot (%d samples) → %s", len(snapshot.samples), path)
    return path


def load_experiment_snapshot(exp_dir: Path) -> ExperimentSnapshot:
    """Load an experiment snapshot from exp_dir/snapshot.json."""
    path = exp_dir / "snapshot.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    snapshot = ExperimentSnapshot.model_validate(data)
    logger.info("Loaded experiment snapshot (%d samples) from %s", len(snapshot.samples), path)
    return snapshot
