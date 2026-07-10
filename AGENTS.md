# RF-DETR Experiments - Agent Instructions

This repository is **not** the upstream RF-DETR project and is **not** aimed at contributing back to it. It is a research fork whose goal is to build a **new detection model that uses RF-DETR as its foundation** — swapping components (backbone, matcher, loss, optimizer, training recipe), measuring what each change buys, and keeping what wins.

There is no PR review process, no maintainer sign-off, no CI matrix to satisfy, and no user-facing documentation to maintain. The deliverables are working code and **written-up, reproducible experiments**.

**Where things live:**

- **Experiment write-ups:** [experiment_notes/](experiment_notes/) — start with its [README.md](experiment_notes/README.md) for the thread connecting all experiments and key results
- **Experiment scripts:** [scripts/](scripts/) — profiling, A/B runners, training entry points
- **Source (the fork):** [src/rfdetr/](src/rfdetr/)
- **Model configs:** [configs/](configs/)
- **Training outputs:** [runs/](runs/) (gitignored artifacts)

## Agent Responsibilities

As an AI agent working in this repo, you are responsible for:

1. **Treating experiments as first-class output**

    - Every substantive experiment (profiling, A/B, ablation) gets a write-up in `experiment_notes/`, written course-note style: build up the concept, then the implementation, then the *measured* result, ending with a **Reproduce** section
    - Update `experiment_notes/README.md` (the index and "key results at a glance") when adding or materially updating a note
    - Report numbers honestly — negative and null results are results; record them (see the Muon note for the pattern)
    - Convergence-affecting changes need an A/B against the current baseline, not vibes

2. **Keeping the fork runnable**

    - The existing test suite is a regression safety net, not a TDD mandate — run the relevant tests after touching shared code paths (`src/rfdetr/`), and add tests where a component's correctness is subtle (e.g. the matcher's SciPy-equivalence tests)
    - Experiment scripts in `scripts/` can be looser, but must actually run from the repo root after `uv sync --all-groups`

3. **Preserving comparability**

    - Don't silently change defaults that prior experiments depended on — if a baseline shifts, say so in the relevant note
    - New training features should be opt-in (config flag or callback) so old configs keep meaning what they meant

4. **Writing minimal, focused code**

    - Follow existing patterns in the codebase; avoid over-engineering and unnecessary abstractions
    - Prefer surgical changes to RF-DETR internals over parallel re-implementations

## Build & Development Environment

```bash
# Install uv (if not already installed)
pip install uv

# Full development environment (always use this)
uv sync --all-groups
```

**Prerequisites:** Python >=3.10

**Dependency information:** see `pyproject.toml`.

- **Core:** PyTorch, torchvision, transformers, supervision, pydantic
- **Optional extras:** `[train]`, `[lora]`, `[onnx]`, `[loggers]` (tensorboard, wandb, mlflow, clearml)
- **Version constraints:** PyTorch >=2.2.0,\<3.0.0; Transformers >=5.0.0,\<6.0.0

## Testing

```bash
# CPU tests
uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu" --ignore=tests/try_instantiate_all_models.py --timeout=240 --durations=50

# GPU tests (requires GPU)
uv run --no-sync pytest tests/ -m gpu -n 2 --reruns 1 --only-rerun "OutOfMemoryError" --timeout=600 --durations=20

# Lint/format (run before committing)
pre-commit run --all-files
```

**Testing principles for a research fork:**

- Tests exist to catch regressions in shared machinery (datasets, matcher, losses, model wiring), not to gate every experiment
- When a change is *correctness-critical* (e.g. an exact solver replacing SciPy), write equivalence tests against the reference implementation
- Mark GPU/heavy tests with `@pytest.mark.gpu`; use `@pytest.mark.parametrize` with `pytest.param(..., id="name")`
- It's fine for `tests/` to lag behind experimental features — it is not fine for a merged-to-branch change to break existing tests silently

## Running Experiments

- **Training entry point:** `scripts/train_rfdetr_dataset.py` (YOLO/COCO datasets under `datasets/`)
- **Profiling:** `scripts/profile_training.py`, `scripts/profile_inference.py`
- **A/B runners:** e.g. `scripts/dense_o2o_ab.sh`, `scripts/matcher_map_ab.py` — follow their pattern (fixed seeds, size-matched arms, same schedule) for new A/Bs
- **Long runs:** use `scripts/run_detached.sh`; outputs land in `runs/`
- All commands in write-ups assume execution from the **repo root**

## Code Quality

```bash
pre-commit run --all-files
```

Configuration: `.pre-commit-config.yaml` (ruff, mdformat, prettier, codespell, license headers) and `pyproject.toml` (`[tool.ruff]`).

**License header** — this fork derives from Apache-2.0 RF-DETR; keep the header on Python files under `src/` (the pre-commit hook enforces it):

```python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
```

## Architecture & Conventions

### Key Patterns

**Model Architecture:**

- RFDETR wrappers: `self.model` is the model context returned by `get_model()`
- Underlying PyTorch module: `self.model.model`
- Segmentation models return `pred_masks` as `torch.Tensor` or dict with keys `['spatial_features', 'query_features', 'bias']`

**Imports:**

```python
# Prefer direct project imports. Standard aliases such as `numpy as np`,
# `torch.nn.functional as F`, and lazy module aliases are allowed when conventional.
from rfdetr.util.misc import get_rank, get_world_size, is_main_process, save_on_master
from rfdetr.util.logger import get_logger

# Logger usage
logger = get_logger()  # Default name: "rf-detr", reads LOG_LEVEL env var

# TQDM (environment compatibility)
from tqdm.auto import tqdm  # NOT: from tqdm import tqdm
```

**Logging:**

- Use `logger.debug()` for detailed tensor/shape information (not `logger.info()`)
- Use `logger.info()` for high-level progress/status

**Checkpoint Handling:**

- Always check file existence before operations — training runs get interrupted

**Subprocess Usage:**

```python
import subprocess

result = subprocess.run(
    ["command", "arg1", "arg2"],
    check=True,  # Raise CalledProcessError on failure
    text=True,  # Return stdout/stderr as strings
    capture_output=True,
)
```

### Type Hints & Docstrings

- Type hints on all function parameters and return types in `src/rfdetr/`
- Google-style docstrings for public functions and classes; don't duplicate types in docstrings
- Target Python version: 3.10+
- Throwaway experiment scripts can be pragmatic, but anything imported by `src/rfdetr/` follows the full standard

## Common Workflow

1. **Setup:** `uv sync --all-groups`
2. **Before changes:** run the relevant tests to establish a baseline; skim related notes in `experiment_notes/` so you don't redo or contradict prior work
3. **Development:** minimal, focused changes following existing patterns; new training behavior behind an opt-in flag/callback
4. **Validation:** correctness tests for subtle components; an A/B or profile for anything claiming a speed/quality win
5. **Quality checks:** `pre-commit run --all-files`
6. **Write-up:** add/update the `experiment_notes/` document with measured results and a Reproduce section; update the notes index
7. **Commit** to the working branch — no PR ceremony required

## Security Considerations

- Validate inputs, especially file paths, URLs, and dataset-provided data
- Never commit API keys, tokens, or credentials (wandb/mlflow keys live in the environment)

---

**Note:** This file is for AI coding agents. Upstream RF-DETR docs (https://rfdetr.roboflow.com, https://github.com/roboflow/rf-detr) remain useful as reference for the base architecture, but their contribution process does not apply here.
