# J-Reflect Recovery

**New experiment: can decoded internal information help a frozen model correct its own mistakes?**

This project implements the small J++ pilot discussed in the research conversation. It is independent of [the earlier GSM8K adapter-training experiment](../README.md). It does not train a model or lens, import that project, or reuse its results.

## What is implemented

1. Generate program-verified arithmetic and variable-tracking problems, including a known intermediate value that is absent from the question and differs from the final answer.
2. Run **Qwen/Qwen3.5-9B** with answer-only, greedy decoding and thinking disabled. Capture the last prompt token's post-block activations **during the actual first generation prefill**, before the answer is emitted.
3. Decode those activations using the released `koayon/jpp-lenses` checkpoint, with a raw logit-lens baseline. Record top-k token IDs, text, and scores at fixed layers.
4. Fork the same saved first response into four independent second attempts: retry, error-only, unrelated J++ readout, and authentic J++ readout.
5. On initially wrong cases with a detected, unexpressed intermediate token, compare authentic feedback with target removal, a wrong replacement, and a matched-count sham removal.
6. Report detection rates, correction rates, harm to initially correct answers, paired intervals, McNemar tests, Holm-adjusted p-values, latent-information strata, and actual token/call/time costs.

**Scope:** the ablations manipulate feedback text. Native activation interventions, ordinary-critique and original J-lens correction arms, natural benchmark generalization, and cross-model introspection tests are future extensions. A positive result would demonstrate useful decoded information, not privileged introspection.

## Folder layout

```text
J-Reflect-Recovery/
  configs/pilot.toml          # model, lens, layers, data sizes, budgets
  src/jrecovery/              # standalone runner and analysis
  tests/                     # protocol and tiny neural integration checks
  docs/protocol.md            # hypotheses, controls, gates, interpretation
  uv.lock                    # resolved Python dependencies
  runs/                      # independent caches, activations, reports; ignored by Git
```

## Install and verify

Run from this directory with Python 3.12. The project has its own environment; the earlier experiment's environment is not required.

```sh
UV_CACHE_DIR=.cache/uv uv sync --locked --extra dev --extra model
.venv/bin/jrecovery doctor
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```

The model tests construct tiny random Qwen2 and Qwen3.5 networks, including the Qwen3.5 multimodal wrapper and hybrid text decoder. They check real generation-prefill capture and lens transport without downloading weights. They do **not** validate the scientific performance of the 9B model or released lens.

For an offline end-to-end check:

```sh
.venv/bin/jrecovery demo --run runs/demo
```

Alternatively, the demo needs only Python 3.12's standard library:

```sh
PYTHONPATH=src python3 -m jrecovery demo --run runs/demo-stdlib
```

This generates 30 development and 30 test problems and exercises all stages using **explicitly simulated responses and readouts**. All manifests, trial records, and reports carry `DEMO_NOT_RESEARCH`; the report cannot be interpreted as model accuracy. Choose a new directory to repeat the full demo. Individual stages can be resumed using the commands below.

## Run the real pilot

The default run requires a CUDA host capable of loading Qwen3.5-9B in bfloat16, plus inference overhead, CPU RAM for the released lens, and disk space for the model and lens. Quantization and multi-device sharding are not implemented. Use `doctor` to check CUDA before proceeding. No paid compute is provisioned by this code.

### 1. Prepare and inspect development

```sh
.venv/bin/jrecovery prepare --config configs/pilot.toml --run runs/pilot
.venv/bin/jrecovery collect --run runs/pilot --split dev --limit 10
```

`prepare` resolves immutable model and lens Hub revisions and generates both splits; it does not download model weights. `collect` downloads weights and the lens on first use. It does not silently substitute a logit lens if J++ loading fails. The readout checkpoint is validated against the model name, selected layers, hidden dimension, final-block target, and finite matrices; its content hash is saved.

Continue collection and inspect detection:

```sh
.venv/bin/jrecovery collect --run runs/pilot --split dev
.venv/bin/jrecovery report --run runs/pilot --split dev
```

The report is marked partial until all primary continuations are present. Readout diagnostics are available before correction. A missing readout does not establish that the model lacks the information; the prefill snapshot may precede the relevant computation.

### 2. Development correction and diagnostics

```sh
.venv/bin/jrecovery correct --run runs/pilot --split dev
.venv/bin/jrecovery ablate --run runs/pilot --split dev
.venv/bin/jrecovery report --run runs/pilot --split dev
```

`ablate` is optional and most informative after finding a feedback benefit. It runs only on the predefined detected-target subset and never modifies the main authentic condition. Its use of the answer key is explicitly labeled as an oracle-assisted diagnostic.

If the model makes too few genuine errors, change `data.tail_steps` or other prespecified development choices in a copied config and prepare a **new** run. Do not repurpose formatting failures as reasoning errors. The default dataset contains five to seven arithmetic operations; `tail_steps = 0` gives the simplest examples.

### 3. Freeze and evaluate held-out templates

```sh
.venv/bin/jrecovery freeze --run runs/pilot
.venv/bin/jrecovery collect --run runs/pilot --split test
.venv/bin/jrecovery correct --run runs/pilot --split test
.venv/bin/jrecovery ablate --run runs/pilot --split test
.venv/bin/jrecovery report --run runs/pilot --split test
```

The development gate requires at least 10 eligible wrong first attempts and a J++ hit rate greater than both the raw logit-lens and shuffled-readout hit rates. This is an exploratory feasibility gate, not a significance test. All development primary continuations must also be complete before freezing.

If the gate fails, a negative-result evaluation is still possible with an explicit recorded rationale:

```sh
.venv/bin/jrecovery freeze --run runs/pilot \
  --allow-weak-readout "Evaluate whether noisy readouts have any behavioral value; report as a negative-readout study."
```

The test sample size is fixed in the original config. Do not keep adding problems until a p-value crosses a threshold. Template splits test a limited change of operation pattern, not general mathematical reasoning.

## Resume and outputs

Repeat `collect`, `correct`, or `ablate` to skip completed records and continue. `--limit N` processes at most N **new** records; for correction and ablation a record is one condition for one problem. All conditions must be completed for final paired comparisons. Partial reports display their coverage.

```text
runs/pilot/
  run.json                   # config, resolved revisions, source/environment hashes
  backend.json               # execution model, lens hash, device, capture location
  data/{dev,test}.json        # generated problems and programmatic answer keys
  first/{split}/{id}.json     # one saved first attempt with raw readouts
  activations/{split}/        # CPU activation snapshots by default
  correction/{split}/        # immutable feedback plan and resumable continuations
  ablation/{split}/          # explicitly oracle-assisted diagnostic continuations
  frozen.json                # development gate and held-out protocol lock
  reports/{split}/
    report.md
    summary.json
    trials.csv
```

Raw JSON caches wrap data in `payload` with a checksum. Reports are ordinary JSON/CSV/Markdown. Source changes, package-version changes, model/lens changes, modified data, changed prompts, or corrupt caches cause a failure instead of silently mixing runs. Preparing on the same GPU environment used for inference avoids package-version mismatches. The released lens records a model identity but may not identify the exact model revision used for fitting; pinning the current checkpoint does not establish that missing historical provenance.

## Verification and research status

Local verification covers deterministic generators and grading, leakage controls, donor assignment, paired statistics, interrupted-run resumption, frozen evaluation, ablation matching, checkpoint validation, real tiny Qwen generation, hook cleanup, and direct readout equivalence. The included offline demo exercises the complete CLI pipeline.

The **full pretrained Qwen3.5-9B/released-J++ GPU run has not been executed locally**. Tiny random models and simulated outputs are engineering checks, not evidence for the hypothesis.

## Sources

- [Anthropic: Verbalizable Representations Form a Global Workspace](https://transformer-circuits.pub/2026/workspace/index.html)
- [J++ implementation and readout convention](https://github.com/safety-research/jpp_lens)
- [Released lenses and model-specific filenames](https://huggingface.co/koayon/jpp-lenses)
- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)

The native checkpoint reader implements the documented `norm(J @ h)` followed by the model head. It does not vendor or require the upstream lens-fitting code.
