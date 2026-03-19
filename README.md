# DataCV 2026 Paper Reproduction Repository

This repository contains the cleaned inference code for our CVPR 2026 DataCV paper submission.
The system is an inference-only agentic VLM pipeline with immutable image resources and tool-assisted visual verification.
It supports both challenge tracks:

- Task 1: binary VQA on classic visual illusions
- Task 2: multiple-choice reasoning on real-world visual illusions and anomalies

The main reproduction configuration uses:

- model: `google-gla:gemini-3-flash-preview`
- temperature: `0.0`
- `num_votes=1`
- `max_tool_rounds=10`

Public validation and test labels are withheld, so this repo reproduces the prediction pipeline and output artifacts rather than local accuracy numbers.

## Clone

Clone with submodules:

```bash
git clone --recurse-submodules <YOUR_PUBLIC_REPO_URL>
cd cvpr_2026_datacv
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Environment Setup

This project uses `uv` and requires Python 3.13.

```bash
uv python install 3.13
uv sync
export GEMINI_API_KEY=YOUR_KEY
```

## Repository Layout

Validation data comes from the two public challenge repositories included as submodules:

- Task 1 validation: `external/VI-Probe-Competition/val.csv` and `external/VI-Probe-Competition/val/`
- Task 2 validation: `external/DataCV-2026-Challenge-CVPR-Task-II-Competition/val/val.json` and `external/DataCV-2026-Challenge-CVPR-Task-II-Competition/val/images/`

Task 1 validation files are packaged inside the Task 1 submodule as `Task1-Validation.zip`, so extract them once after cloning:

```bash
unzip -n external/VI-Probe-Competition/Task1-Validation.zip -d external/VI-Probe-Competition/
```

If you also want to reproduce the hidden-phase test predictions, place the official test files under a local `data/` directory with this layout:

```text
data/
├── task_1_test_set/
│   ├── test.csv
│   └── test/
│       ├── 0.png
│       └── ...
└── task_2_test_set/
    ├── test.json
    └── images/
        ├── 0.jpg
        └── ...
```

`data/` is local input data and is not tracked by git.

## Reproduce Validation Predictions

Task 1 validation:

```bash
uv run python main.py task1 \
  --model google-gla:gemini-3-flash-preview \
  --mode val \
  --exp-name task1_val_flash \
  --temperature 0.0 \
  --top-p 1.0 \
  --seed 42 \
  --max-tokens 4096 \
  --max-tool-rounds 10 \
  --max-concurrency 3 \
  --num-votes 1
```

Task 2 validation:

```bash
uv run python main.py task2 \
  --model google-gla:gemini-3-flash-preview \
  --mode val \
  --exp-name task2_val_flash \
  --temperature 0.0 \
  --top-p 1.0 \
  --seed 42 \
  --max-tokens 32000 \
  --max-tool-rounds 10 \
  --max-concurrency 3 \
  --num-votes 1
```

## Reproduce Test Predictions

Task 1 test:

```bash
uv run python main.py task1 \
  --model google-gla:gemini-3-flash-preview \
  --mode test \
  --exp-name task1_test_flash \
  --temperature 0.0 \
  --top-p 1.0 \
  --seed 42 \
  --max-tokens 4096 \
  --max-tool-rounds 10 \
  --max-concurrency 3 \
  --num-votes 1
```

Task 2 test:

```bash
uv run python main.py task2 \
  --model google-gla:gemini-3-flash-preview \
  --mode test \
  --exp-name task2_test_flash \
  --temperature 0.0 \
  --top-p 1.0 \
  --seed 42 \
  --max-tokens 32000 \
  --max-tool-rounds 10 \
  --max-concurrency 3 \
  --num-votes 1
```

## Optional Failed-Sample Repair

If a run is interrupted by provider-side failures (like error_code=429 all the time) and you want to repair only failed samples in the same experiment directory, rerun with `--allow-mixed-model-repair`.
One recovery configuration used during archived runs was:

In our archived submitted test outputs, Flash-Lite repair affected 340/630 Task 1 samples and 1/755 Task 2 samples.
If a reproduction requires significantly more repaired samples than this, worse results should be expected.
If a reproduction requires significantly fewer repaired samples than this, improved results should be expected.

Task 1 repair:

```bash
uv run python main.py task1 \
  --model google-gla:gemini-3.1-flash-lite-preview \
  --mode test \
  --exp-name task1_test_flash \
  --google-thinking-level high \
  --allow-mixed-model-repair \
  --max-tokens 4096 \
  --max-tool-rounds 10 \
  --max-concurrency 3 \
  --num-votes 1
```

Task 2 repair:

```bash
uv run python main.py task2 \
  --model google-gla:gemini-3.1-flash-lite-preview \
  --mode test \
  --exp-name task2_test_flash \
  --google-thinking-level high \
  --allow-mixed-model-repair \
  --max-tokens 32000 \
  --max-tool-rounds 10 \
  --max-concurrency 3 \
  --num-votes 1
```

## Package Submission Files

Task 1:

```bash
uv run python main.py submit 1 --exp-name task1_test_flash
```

Task 2:

```bash
uv run python main.py submit 2 --exp-name task2_test_flash
```

## Outputs

Each run writes an experiment directory under `outputs/<exp_name>/` with:

- Task 1: `predictions.txt`, `model.json`, `snapshot.json`, `report.html`
- Task 2: `result.txt`, `model.json`, `snapshot.json`, `report.html`
- optional tool images in `tool_images/`

The `submit` command additionally creates `result.zip`.

## Notes

- Validation and test labels are not public, so local runs reproduce predictions, logs, and submission artifacts, not official scores.
- Gemini API outputs may vary slightly across reruns even with the same settings.
- Validation reproduction works immediately after cloning the repo with submodules.
- Test reproduction additionally requires the official test files placed under `data/` as shown above.
