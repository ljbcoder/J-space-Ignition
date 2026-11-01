# Implementation verification

Verified locally on November 1, 2026, with Python 3.12.11, PyTorch 2.15.0, Transformers 5.22.1, and the committed `uv.lock`.

- 39 tests pass, including exact numeric grading, disjoint splits, train-only streaming, checksum enforcement, interrupted generation, balancing, shuffled donor assignments, paired statistics, held-out correctness probes, and protocol freeze enforcement.
- A tiny random Qwen2 model exercises real generation and activation capture at both specified positions. Its local pretrained checkpoint and trained PEFT adapter reload successfully.
- The pinned upstream J++ implementation passes Qwen2 forward-equivalence and changed-gradient checks. A small router/Jacobian fit, checkpoint resume, expert weighting, held-out readout evaluation, and lens serialization round-trip pass on CPU.
- Tiny LoRA training preserves base weights, decreases target loss, and matches an uninterrupted run after optimizer checkpoint resume. The standalone smoke run completed four optimizer steps, with loss decreasing from 4.7859 to 4.3473. These are mechanics checks using random weights, not GSM8K results.
- The synthetic CLI demo generates Markdown/JSON/CSV/PNG reports and reuses completed cached items on rerun. Its report is explicitly labeled synthetic.
- Real GSM8K train preparation produced 6,973 training, 300 development-tuning, and 200 development-gate items. Model revision: `a09a35458c702b33eeacc393d103063234e8bc28`. Dataset revision: `740312add88f781978c0658806c59bc2815b9866`.

During verification, the initial eager Datasets loader materialized both source splits in its library cache despite requesting train. Preparation was changed to streaming and verified again, with a regression test. The original preparation directory is retained under `artifacts/initial-eager-preparation`; the current prepared run is `artifacts/initial`. No test predictions, performance analysis, or selection based on test outcomes was performed.

This machine reports no CUDA device. The real 7B generation, full lens fit, development go/no-go decision, three-arm training, and final test comparison remain unexecuted. Their memory use, runtime, lens quality, and scientific outcomes are not established by the CPU checks. The [run guide](run-guide.md) provides the commands for executing those stages on a CUDA host.
