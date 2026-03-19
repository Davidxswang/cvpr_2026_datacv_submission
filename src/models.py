"""Pydantic IO boundary models for both competition tasks."""

from pathlib import Path

from pydantic import BaseModel, Field

# --- pydantic-ai structured output types ---


class Task1Output(BaseModel):
    """Structured output from the VLM for Task I (Binary VQA)."""

    reasoning: str = Field(description="Step-by-step visual reasoning about the image")
    answer: int = Field(description="1 if yes (same), 0 if no (different)", ge=0, le=1)


class Task2Output(BaseModel):
    """Structured output from the VLM for Task II (MCQ)."""

    reasoning: str = Field(description="Step-by-step reasoning about the question and options")
    answer: str = Field(description="The selected option letter: A, B, C, or D", pattern=r"^[ABCD]$")


# --- Data loading models ---


class Task1Sample(BaseModel):
    """A single row from the Task I CSV manifest."""

    index: int
    image_path: Path
    prompt: str
    label: int | None = None  # None for val/test, 0 or 1 for few-shot


class Task2Sample(BaseModel):
    """A single entry from the Task II JSON manifest."""

    index: int
    image_name: str
    question: str
    option: str


# --- Output models ---


class Prediction(BaseModel):
    """A single prediction line: '{index} {answer}'."""

    index: int
    answer: int


class ModelInfo(BaseModel):
    """model.json schema matching competition format."""

    model: str
    parameters: dict[str, float | int | str | bool | list[str] | dict[str, int]]
