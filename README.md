# J-Reflect
Neurotech@Berkeley: FA26 Software Project

An executable research pipeline for answer-only GSM8K self-correction with Qwen2.5-7B-Instruct and J++ workspace readouts. Three LoRA arms share the same first answers, example manifest, targets, and seeds: own readout, no readout, and a readout from another problem.

The [build plan](docs/initial-experiment-plan.md) and [run guide](docs/run-guide.md) describe the implemented GSM8K protocol and staged execution. See [implementation verification](docs/implementation-status.md) for completed checks and the remaining GPU work.

The [revised proposal](Initial%20Experiment%20Readout-Trained%20Self-Correction%20on%20GSM8K%20Revised.md) instead specifies MATH, Qwen3.5-9B, and a released lens. That revision is not implemented by this pipeline or its notebook launcher.

For Jupyter, Colab, or a cloud A100 runtime, open the [notebook launcher](output/jupyter-notebook/run-jreflect.ipynb). It walks through environment setup, GPU checks, each experiment stage, and optional checkpoint backup/restore. Notebook helper functions were checked locally; the full cloud GPU sequence has not been executed here.

## Install and check locally

Python 3.12 and `uv` are recommended. Run commands from this repository's root.

```sh
UV_CACHE_DIR=.cache/uv uv sync --locked --extra dev --extra analysis --extra model
.venv/bin/jreflect doctor
.venv/bin/pytest -q
.venv/bin/jreflect demo --output artifacts/demo
```

The demo needs no model or dataset downloads. It checks splits, resumable generation caches, balanced controls, statistics, and report creation using explicitly synthetic data. Its output is **not an experiment result**.

For a real neural-network smoke check, using a tiny randomly initialized Qwen2 model:

```sh
git clone https://github.com/safety-research/jpp_lens.git vendor/jpp_lens
git -C vendor/jpp_lens checkout d7efc0de372e20589e899188c0cfe8434acc63ec
.venv/bin/jreflect smoke --output artifacts/smoke --upstream vendor/jpp_lens
.venv/bin/pytest -q
```

The pinned upstream checkout enables the additional Qwen2 LRP and Jacobian-fitting tests. It is ignored by Git; no upstream source is copied into this package. The smoke command performs generation, activation replay, LRP forward-equivalence checking, and a small LoRA fit on CPU. Tiny random weights do not measure GSM8K performance.

## Start the real experiment

```sh
.venv/bin/jreflect prepare --config configs/initial.toml --run artifacts/initial
.venv/bin/jreflect generate --run artifacts/initial --split dev_tune --limit 10
```

`prepare` resolves immutable model/dataset revisions and downloads **only GSM8K train**. The generation command needs the configured CUDA device and downloads the 7B model on first use. The default first check uses a **logit lens**, clearly labeled in its outputs. Fit J++ before proceeding to the scientific comparison.

Full lens fitting, development gates, training, and frozen final evaluation are documented in the [run guide](docs/run-guide.md). The CLI deliberately stops when a stage is incomplete, a provenance hash changes, a gate fails, or a CUDA-dependent command runs without CUDA. Full 7B fitting/training has not been validated by the local CPU tests.

## Outputs and reproducibility

Configuration is in `configs/initial.toml`; resolved settings live in each run's `run.json`. `uv.lock` pins Python packages. Caches retain prompt/generated token IDs, readout scores, source IDs, grades, software fingerprints, and checksums. Repeating a generation command skips completed items; interrupted LoRA training resumes at the last saved optimizer step.

`report` writes Markdown, JSON, CSV, and a PNG chart. It reports W→R/R→W, accuracy changes, format failures, exact McNemar tests with Holm correction, and paired item bootstrap intervals. Seeds remain clustered by item. Test access requires frozen adapters and protocol metadata.
