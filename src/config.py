"""Global configuration for the DataCV challenge."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DatasetMode(StrEnum):
    """Supported evaluation split."""

    VAL = "val"
    TEST = "test"


class Config(BaseModel):
    """Flat global configuration. All paths, hyperparameters, and constants."""

    # Model selection (pydantic-ai format: "provider:model")
    model_name: str = "google-gla:gemini-3-flash-preview"

    # Generation parameters
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    max_tokens: int = 4096
    google_thinking_level: Literal["minimal", "low", "medium", "high"] | None = None

    # Task I inputs
    task1_mode: DatasetMode = DatasetMode.VAL
    task1_val_csv: Path = Path("external/VI-Probe-Competition/val.csv")
    task1_test_csv: Path = Path("data/task_1_test_set/test.csv")
    task1_val_image_dir: Path = Path("external/VI-Probe-Competition/val")
    task1_test_image_dir: Path = Path("data/task_1_test_set/test")

    # Task II inputs
    task2_mode: DatasetMode = DatasetMode.VAL
    task2_val_json: Path = Path("external/DataCV-2026-Challenge-CVPR-Task-II-Competition/val/val.json")
    task2_test_json: Path = Path("data/task_2_test_set/test.json")
    task2_val_image_dir: Path = Path("external/DataCV-2026-Challenge-CVPR-Task-II-Competition/val/images")
    task2_test_image_dir: Path = Path("data/task_2_test_set/images")

    # Experiment
    exp_name: str = ""
    output_dir: Path = Path("outputs")
    limit: int = 0  # 0 = process all samples
    samples: list[int] = Field(default_factory=list)  # Specific sample indices to run (empty = all)

    # Tool-calling (agentic mode)
    max_tool_rounds: int = 10  # UsageLimits(request_limit=...)

    # Majority voting
    num_votes: int = 1  # >1 enables majority voting over N runs per sample

    # Concurrency
    max_concurrency: int = 15
    allow_mixed_model_repair: bool = False

    # Reproducibility
    seeds: list[int] = Field(default=[42, 123, 456])

    @property
    def exp_dir(self) -> Path:
        """Resolved experiment output directory."""
        return self.output_dir / self.exp_name

    @property
    def task1_csv(self) -> Path:
        """Resolved Task I CSV manifest for the selected split."""
        return self.task1_val_csv if self.task1_mode == DatasetMode.VAL else self.task1_test_csv

    @property
    def task1_image_dir(self) -> Path:
        """Resolved Task I image directory for the selected split."""
        return self.task1_val_image_dir if self.task1_mode == DatasetMode.VAL else self.task1_test_image_dir

    @property
    def task2_json(self) -> Path:
        """Resolved Task II JSON manifest for the selected split."""
        return self.task2_val_json if self.task2_mode == DatasetMode.VAL else self.task2_test_json

    @property
    def task2_image_dir(self) -> Path:
        """Resolved Task II image directory for the selected split."""
        return self.task2_val_image_dir if self.task2_mode == DatasetMode.VAL else self.task2_test_image_dir
