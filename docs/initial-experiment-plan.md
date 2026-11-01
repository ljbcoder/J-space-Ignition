# Initial experiment build plan

Planning date: November 1, 2026. Implementation update: the staged CLI is built; see the [run guide](run-guide.md). Local checks cover tiny-model J++ fitting and LoRA training, and the real GSM8K training splits are prepared. Full-model GPU jobs and scientific evaluation have not been run. The repository inventory below records the starting state at planning time.

Build a reproducible Python experiment that tests whether J++ readouts improve answer-only GSM8K self-correction beyond both no-readout and shuffled-readout training. The first deliverable is a small, validated generation-and-readout pipeline. Full LoRA training follows only after the lens and development-set checks pass.

This implements [the initial experiment proposal](../Initial%20Experiment%20Readout-Trained%20Self-Correction%20on%20GSM8K.md). Choices below marked as proposed make underspecified parts of that proposal executable; they are not experimental findings.

## What exists and what needs checking

The repository currently contains the proposal, a short README, and an empty notebook directory. There is no application, dependency setup, training pipeline, or test suite to extend. Keep implementation in this repository, under `src/jreflect`; notebooks can later consume saved artifacts.

The local machine is Apple ARM64 with Python 3.14.7 and `uv` available. Plan CPU development and small tests locally, with full model runs on a separate CUDA machine. GPU access, device memory, storage, and runtime have not been verified.

Upstream inspected at `safety-research/jpp_lens` commit `d7efc0de372e20589e899188c0cfe8434acc63ec`:

- The LRP implementation explicitly handles `Qwen2MLP` and `Qwen2RMSNorm`. The generic Hugging Face adapter supports the usual decoder layout. This supports trying Qwen2.5 first, but does not establish end-to-end compatibility. [LRP source](https://github.com/safety-research/jpp_lens/blob/d7efc0de372e20589e899188c0cfe8434acc63ec/src/workspace_lens/lrp/lrp.py), [adapter](https://github.com/safety-research/jpp_lens/blob/d7efc0de372e20589e899188c0cfe8434acc63ec/src/jlens/hf.py)
- CLI defaults use Qwen3.6-27B layer indices, including indices outside a 28-layer model. Supply explicit layers and model identity everywhere. [Configuration](https://github.com/safety-research/jpp_lens/blob/d7efc0de372e20589e899188c0cfe8434acc63ec/src/workspace_lens/config.py), [Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- The bundled readout-evaluation correctness CSV only covers Qwen3.6-27B. Generate Qwen2.5's own correctness labels before fitting expert weights or scoring the lens. [Evaluation loader](https://github.com/safety-research/jpp_lens/blob/d7efc0de372e20589e899188c0cfe8434acc63ec/src/lens_evals/readout_evals/readout_eval_items.py)
- Upstream uses Python 3.12 and has tightly pinned dependencies with distinct CPU/CUDA extras. Resolve a compatible environment before installing training dependencies. Use a separate lens environment if the PEFT training stack conflicts. [Dependency specification](https://github.com/safety-research/jpp_lens/blob/d7efc0de372e20589e899188c0cfe8434acc63ec/pyproject.toml)
- No Qwen2.5-7B lens is listed in the released collection. The proposal's fallback, Qwen3.5-9B, has a released lens. [Lens collection](https://huggingface.co/koayon/jpp-lenses)

## Build sequence and completion checks

| Milestone | Implementation | Completion check |
| --- | --- | --- |
| 1. Reproducible foundation | Python package, configuration, artifact manifests, GSM8K split builder, numeric grading, prompt templates, CPU fixtures | Deterministic splits, no leakage, grading edge cases covered, synthetic pipeline reruns identically |
| 2. Model and lens compatibility | Qwen wrapper, explicit hooks and token positions, logit-lens baseline, tiny Qwen2 LRP tests, short GPU fit | Hooks preserve normal outputs; LRP changes backward behavior only; finite maps and valid readouts round-trip through disk |
| 3. Fit and validate J++ | Router fitting, expert Jacobian shards, expert weighting, separate held-out latent-task evaluation | Correct model-specific evaluation labels, disjoint weight-fit/evaluation items, quality and compute report |
| 4. GSM8K development gate | Cached base first attempts, J++ and logit readouts, content audit, correctness probes, accuracy/power estimate | Frozen probe evaluated on held-out dev items; explicit go/no-go report before full training |
| 5. Controlled LoRA runs | Balanced examples, fixed shuffled donors, target-only loss, three arms with matched seeds | All arms share example IDs, targets, initial weights, order, and training settings; tiny overfit and short dev pilot pass |
| 6. Frozen final evaluation | One shared test cache, all adapters, paired metrics/tests, uncertainty, report | All 1,319 test IDs accounted for, no retuning from test outcomes, reproducible results and limitations |

### 1. Foundation and data contracts

Use Python 3.12, `uv`, PyTorch, Transformers, Datasets, PEFT, NumPy, SciPy, scikit-learn, pytest, and Ruff. Prefer a small explicit training loop or Transformers Trainer with a verified target mask; avoid adding a framework solely for supervised fine-tuning. Pin resolved versions and the upstream commit after the compatibility check.

Proposed structure:

```text
configs/initial.yaml           # model, splits, readout, fitting, training, evaluation
src/jreflect/
  cli.py                      # resumable stage commands
  config.py                   # validated configuration and fingerprints
  artifacts.py                # manifests, checksums, atomic writes, resume checks
  data.py                     # source IDs, splits, gold answer extraction
  grading.py                  # numeric normalization and format validation
  prompts.py                  # first-attempt and three second-attempt templates
  model.py                    # loading, generation, activation capture
  lenses.py                   # upstream J++ integration and logit lens
  readouts.py                 # token filtering, scores, bands, text formatting
  sanity.py                   # content audit and correctness probes
  examples.py                 # balancing and shuffled donor assignments
  training.py                 # LoRA and target-only loss
  evaluation.py               # predictions, transitions, paired statistics
  reporting.py                # tables, plots, Markdown report
tests/                        # CPU fixtures; separately marked GPU checks
artifacts/<experiment>/<stage>/ # ignored generated data and checkpoints
```

Use the GSM8K `main` configuration, pin its dataset revision, and assign stable source IDs before shuffling. Hold out 500 of the 7,473 training items with split seed 42, leaving 6,973 for training. Reserve the 1,319-item test split for the frozen evaluation. Validate these counts against the pinned dataset. [Dataset](https://huggingface.co/datasets/openai/gsm8k)

Proposed dev subdivision: 300 items for formatting/hyperparameter development and 200 for a one-time readout gate, chosen deterministically before looking at outcomes. The final test loader is only available in the evaluation stage after a frozen protocol manifest exists.

Artifacts should preserve:

- Example ID, source split/revision, question, gold number, and separately stored reference solution.
- Exact first prompt token IDs, generated token IDs/text, parsed answer, format validity, correctness, and stop reason.
- Readout method, lens/model hashes, block index, token position, token IDs/text, ranks and scores; retain semantic-only and question-filtered rankings.
- Training instance ID, source ID, arm, donor source ID when shuffled, and target text.
- Run configuration, seeds, code/dependency versions, input checksums, progress, elapsed time, and peak GPU memory.

Prompt builders receive question, previous answer, and readout fields only. Gold answers, reference reasoning, correctness labels, and donor labels must never enter model inputs. Resume only when fingerprints match; reject incompatible caches instead of silently mixing them.

Parse the gold number after `####`. Grade generated answers using exact decimal/rational normalization rather than floating-point tolerance. Proposed primary format rule: a single numeric value after whitespace trimming; allow signs, valid thousands separators, and decimals. Extra prose, multiple answers, empty output, and truncation are format failures and count wrong. Log a secondary permissive score for diagnosis, without changing the primary rule after results are seen.

### 2. Generation, token positions, and compatibility

Use the frozen `Qwen/Qwen2.5-7B-Instruct` checkpoint in bf16 on GPU with its pinned tokenizer and chat template. Start with greedy generation, `do_sample=False`, and `max_new_tokens=32`; resolve length and EOS settings on development items, then freeze them. Record violations of answer-only formatting.

Start with individual examples for position correctness; introduce batching only after unpadded/padded outputs agree. Generate the first answer normally, then perform a deterministic replay with exact prompt IDs plus the first generated token. Capture only the requested residual vectors; no reference answer is replayed.

Define positions precisely:

- `last_prompt`: final token of the actual chat-formatted generation prefix, at index `prompt_length - 1`.
- `first_answer`: first generated token after it has been consumed by the model, at index `prompt_length`. The hidden state that predicts that token belongs to the preceding position.

Use post-block residuals to match upstream hooks. Proposed zero-based sampled blocks: `[3, 7, 11, 15, 19, 23]`, meaning every fourth completed block before the final target block 27. Assert that the fitted lens and activation capture use the same convention.

Before fitting the full model, construct a tiny random Qwen2 model locally. Verify hook placement, forward equivalence with LRP enabled/disabled, finite activation gradients, patch removal on error, lens serialization, and final-norm/unembedding behavior. Test target masking separately on the training model: the lens wrapper freezes parameters, so do not reuse that wrapped instance for LoRA training.

Then run a small GPU compatibility job: a few prompts, one source layer, and a short Jacobian shard. Record actual memory and time. Treat the proposal's 40–80 GB GPU and 1–2 GPU-day estimates as unverified planning assumptions, especially for nine adapter runs.

### 3. Lens fitting and validation

Reuse upstream fitting stages behind a thin adapter. Keep a pinned checkout for scripts and evaluation data, whose relative paths must resolve explicitly. Do not rewrite Jacobian fitting.

The recipe includes a router-fit corpus in addition to the 64 Jacobian sequences: use 1,000 WikiText-103 records for PCA/k-means routing, eight experts per layer, and 64 qualifying records truncated to 128 tokens for Jacobians. Skip the first 16 positions and the final position, use the `rlens` backward rules, and follow upstream's minimum-cluster fallback. Fit signed expert weights on a separate latent-task fitting subset; evaluate on its held-out complement. [Published fitting recipe](https://huggingface.co/koayon/jpp-lenses)

Fit the six required source layers initially, targeting block 27; expand only if a frozen dev format needs other layers. Generate the latent-task correctness filter using Qwen2.5 itself. Preserve task-level sample counts, correctness coverage, and the expert-weight split manifest.

Produce qualitative spider/legs and organ/chambers readouts plus held-out recall@1 and recall@10, with the logit lens evaluated on the same items and layers. The released 9B model's 55% recall@10 is context, not a directly comparable pass mark for different model/item coverage. Proposed operational gate: valid maps and positive held-out recall@10 improvement over the logit lens, followed by the GSM8K-specific gate below. Report uncertainty; an inconclusive result is not evidence that no workspace exists.

If compatibility or quality fails, write a diagnostic report. The proposal permits a Qwen3.5-9B fallback; treat it as a new experiment identity and regenerate all first attempts, splits/config fingerprints as needed, and controls. Never apply another model's lens to Qwen2.5.

### 4. Development readouts and go/no-go

Generate the 500 dev first attempts once. Compute J++ and logit-lens scores at identical blocks and positions. Save top 20 semantic tokens and a separate top 20 after question-word filtering. Apply filtering before top-k selection, with an explicit exclusion for special tokens as well as tokens without letters/digits.

Proposed correction-prompt format: middle blocks `[11, 15]`, late blocks `[19, 23]`, both positions, five unique decoded concepts per band. Start with reciprocal-rank aggregation across block/position rankings, avoiding direct comparison of differently scaled logits. Normalize whitespace/case for deduplication and whole-word prompt-copy removal. Retain numeric tokens, including digits occurring in the question; do not invent place values or join independent digits into numbers. Record this numeric exception explicitly.

Compute a diagnostic content-overlap measure using a frozen operation vocabulary and deterministic entity/word extraction from reference solutions, subtracting words present in the question. Audit a small sample by hand. Reference text is used only for this analysis, never to select readouts or construct correction prompts.

Train regularized linear correctness probes on sparse readout token scores using the 300-item development subset, with inner cross-validation for regularization. Fit feature selection and scaling on training folds only; use matching feature construction for J++ and logit lens. Freeze the probes, then score the untouched 200-item gate subset. Report AUROC, balanced accuracy, label counts, and paired bootstrap uncertainty. Both probes use the same first-token information, which can itself correlate with correctness.

Proposed conservative training gate: the held-out J++ probe's 95% AUROC interval is above 0.5 and the paired J++ minus logit-lens AUROC interval is above zero. If class counts are insufficient or intervals cross these thresholds, mark the result inconclusive/no-go for the expensive training stage. This is a project resource decision; it does not prove that readouts cannot help correction. A word-intermediate dataset would be a separately documented follow-up.

Estimate test error count and paired-test power from dev outcomes, without loading test questions or labels. The proposed 500 wrong test items is a heuristic; McNemar power depends on discordant second-attempt outcomes. Any harder benchmark extension must be declared before final evaluation and reported separately.

### 5. Examples and matched LoRA training

After the dev gate, generate and cache base first attempts/readouts for the 6,973 training items. Use all wrong examples and an equal number of right training instances. If right items are fewer, sample them with replacement using a fixed seed and record repeated source IDs. If either class is empty, stop with an explicit balancing error. Never balance or resample the evaluation sets.

Construct the three arms from one shared instance manifest:

| Arm | Correction input |
| --- | --- |
| Readout | Question, own first answer, own middle/late concept lines |
| No readout | Same question and first answer; omit the workspace introduction and both concept lines |
| Shuffled | Same question and first answer; use a different source item's already-formatted concept lines |

Assign shuffled donors within each split using a seeded derangement of unique source IDs before duplicating training instances. Never use the item's own readout or cross a train/dev/test boundary. Do not shuffle gold labels, previous answers, or targets. Preserve the same donor map across adapter seeds. The shuffled arm controls the marginal readout format and content distribution; log prompt token lengths because per-item lengths need not match.

Target the gold bare number plus the assistant termination token. Mask all prompt and padding tokens with `-100`; preserve the target/EOS under truncation and reject overlong prompts rather than silently cutting off labels. Begin without sequence packing.

Starting settings from the proposal: rank 16, alpha 32, dropout 0.05, learning rate `1e-4`, cosine schedule, two epochs. Explicit Qwen projection targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, verified against the loaded model. Proposed initial effective batch size 16, with microbatch/gradient accumulation set from measured memory. Resolve context limit and any warmup on dev and freeze them across arms.

Start with an 8–16-example overfit check and a short dev pilot for all three arms. Check loss masking, trainable parameter counts, base-weight immutability, answer formatting, change rate, and checkpoint reload. Then train one seed end-to-end; add matched seeds 43 and 44 to the primary seed 42 if the measured budget permits. Three seeds mean nine adapters, not three total runs. Finalize the seed set before opening the test set; one seed supports a pilot conclusion only.

### 6. Final evaluation and reporting

Freeze model/tokenizer/lens hashes, prompts, parser, readout format, hyperparameters, seed set, donor procedure, selected checkpoints, and statistical rules in `protocol.json`. Select checkpoints on dev only. Build one complete test first-attempt/readout cache with the frozen base model, then reuse it for every second-attempt arm and seed. Do not adapt prompts, training, or benchmark selection after seeing test outcomes.

Report first- and second-attempt accuracy, net accuracy change in percentage points, W→R conditional on initially wrong items, R→W conditional on initially right items, answer-change rate, and invalid-output rate. Use `null` with a count explanation when a conditional denominator is zero. Define changed answers using normalized numeric values when both parse; otherwise compare normalized raw strings.

For each prespecified seed, compare Readout with each control using exact paired McNemar tests and Holm correction across the two contrasts. With zero discordant pairs, return p=1. Use 10,000 paired item bootstrap draws for 95% CIs on metrics and arm differences. Reuse sampled item IDs across arms; seed-averaged summaries retain all seed predictions together within each sampled item. Do not treat repeated seeds on the same items as independent test examples.

Proposed interpretation of the proposal's across-seed success rule: positive gains over both controls and adjusted p<0.05 for both contrasts in every prespecified seed. Report all seeds, including failures; never choose a successful seed after the fact. Also report effect sizes when significance is absent.

Secondary checks: tuned first-attempt accuracy, removing readouts from the Readout adapter's correction prompt, and a small blinded error-type audit. Label these exploratory. Generating readouts from tuned weights requires a new lens-quality check and possibly refitting; it is outside the primary frozen-base pipeline.

Final artifacts: prediction rows, metrics JSON/CSV, paired contingency tables and adjusted p-values, bootstrap intervals, accuracy/transition plots, run manifests, compute usage, and a Markdown report stating what ran, what failed, and what remains uncertain. No outcome is assumed in advance.

## First implementation slice

Build milestones 1 and 2 first: package/configuration, deterministic splits, answer grading, three prompt templates, artifact manifests, tiny-model tests, and a command that processes 10 dev examples into first answers plus logit-lens readouts. Then connect a real J++ checkpoint and the fitting adapter. A logit lens or synthetic fixture must always be labeled as such; neither counts as a completed J++ experiment.

The first slice is complete when CPU checks pass and the same small GPU job can stop, resume, and reproduce its artifact IDs and outputs. Full fitting, the dev gate, training, and final evaluation remain separate measured stages.
