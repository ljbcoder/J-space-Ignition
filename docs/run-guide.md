# Running the initial experiment

The implementation runs stages explicitly so that a weak lens or a failed development gate stops the experiment before expensive training. CPU smoke checks validate mechanics; the 7B scientific experiment requires a CUDA host. No GPU rental or external job submission is automated.

## Environment

Install the locked environment and the exact upstream checkout using the README. The integration imports the pinned checkout directly and calls its public Python stages with explicit model/layer settings; it does not use upstream's Qwen3.6 defaults. A clean checkout at `d7efc0de372e20589e899188c0cfe8434acc63ec` is required. Verify `jreflect doctor` on the GPU host and run the small tests there before launching fitting. The committed lock is locally tested; CUDA memory, driver compatibility, throughput, and full-model lens quality must be measured on that host.

Use `.venv/bin/jreflect` below, or activate `.venv` and use `jreflect`. The CLI places Hugging Face and plotting caches in `.cache/` unless the corresponding environment variables are already set. Generated artifacts and model weights are ignored by Git.

Set the final seed list and hardware settings in `configs/initial.toml` before `prepare`. Defaults are three seeds (nine adapters), bf16, one CUDA device, effective batch 16, and two epochs. Use a separate experiment directory for changed configuration. Cache fingerprints intentionally refuse mixed software, data, or settings. Keep the code/dependencies unchanged during a run.

## Prepare and verify ten examples

```sh
.venv/bin/jreflect prepare --config configs/initial.toml --run artifacts/initial
.venv/bin/jreflect generate --run artifacts/initial --split dev_tune --limit 10
```

Preparation creates 6,973 train, 300 dev-tuning, and 200 dev-gate examples. The source revision, source IDs, and split checksums are persisted. Reference solutions live in a separate file and are not passed to prompt constructors. Test data is not loaded.

The first generation check captures logit-lens readouts at the last chat-prefix token and the consumed first generated token. `--limit` is a prefix of the full, stable split; rerun without the limit to extend the same cache. Only a complete split receives a completion marker, and downstream stages require it. Output rows contain `logit` explicitly; there is no stand-in J++ result.

Review a few rows under `artifacts/initial/base/logit/dev_tune/rows/`: exact token IDs, output length/format, layer indices, semantic rankings, question-filtered rankings, and middle/late concept lists. Strict primary grading counts explanations and truncated outputs as wrong. A permissive diagnostic grade is saved separately.

## Fit and validate J++

```sh
.venv/bin/jreflect jpp-fit --run artifacts/initial --stage router
.venv/bin/jreflect jpp-fit --run artifacts/initial --stage jacobians
.venv/bin/jreflect jpp-fit --run artifacts/initial --stage weights
.venv/bin/jreflect jpp-fit --run artifacts/initial --stage validate
```

`--stage all` executes the same sequence. `--upstream` can select a pinned checkout at another path.

The router uses 1,000 qualifying WikiText-103 records, saved with a dataset commit and content hash. Jacobians use the first 64 records, eight clusters, 128 tokens, skipping 16 leading positions and the last position. Six source blocks are fitted: zero-based `[3, 7, 11, 15, 19, 23]`, targeting final block 27. LRP mode is `rlens`. Jacobian checkpoints are saved every four prompts and can resume.

Expert weighting generates Qwen2.5's own latent-task correctness CSV, uses the upstream 20% weight-fit split, and keeps held-out latent items disjoint. `lens.pt` has a `.meta.json` sidecar binding it to the exact model revision and source commit. A checkpoint from another model cannot be substituted.

Validation writes `heldout_ranks.csv`, `heldout_scores.csv`, qualitative readouts, and `validation.json`. The operational lens gate requires held-out item-weighted macro recall@10 above the same-item logit lens. Missing task coverage makes this gate fail. The separate GSM8K gate below is still required. No 55% result is assumed for Qwen2.5.

Router fitting and expert-weight optimization restart their current stage if interrupted before saving; completed stages are reused. Jacobian fitting resumes its accumulated sufficient statistics. Plan GPU memory and time from observed measurements rather than the original rough budget.

## Measure the development signal

```sh
.venv/bin/jreflect generate --run artifacts/initial --split dev_tune --lens artifacts/initial/lens/lens.pt
.venv/bin/jreflect generate --run artifacts/initial --split dev_gate --lens artifacts/initial/lens/lens.pt
.venv/bin/jreflect gate --run artifacts/initial
```

Each J++ cache also includes the logit lens, measured on identical activations. Top-k filtering excludes special/non-semantic tokens and copied question words; numeric tokens are retained without assigning place value. Reciprocal-rank aggregation produces five concepts per band.

The probe fits feature extraction/scaling and regularization using only the 300 tuning items. The final AUROC comparison uses the 200 gate items. Both the J++ AUROC interval above 0.5 and the paired improvement interval above zero are required. Small class counts or overlapping intervals produce a no-go/inconclusive result. A reference-word overlap audit is a separate heuristic diagnostic, not the acceptance criterion or an automatic semantic validator.

`gate_started.json` records opening the gate before analysis. Rerunning the gate to tune against its outcomes is refused. Preserve failed/inconclusive runs; changes belong in a newly identified experiment and should be reported as follow-up work. A failed interrupted gate requires investigating the saved inputs rather than deleting its marker and silently retuning.

## Build matched examples and train

```sh
.venv/bin/jreflect generate --run artifacts/initial --split train --lens artifacts/initial/lens/lens.pt
.venv/bin/jreflect examples --run artifacts/initial
.venv/bin/jreflect train --run artifacts/initial --arm readout --seed 42 --max-steps 10
```

The short run checks training mechanics. Rerun without `--max-steps` to continue from the last saved optimizer step. A stage refuses to train unless the real development gate passed. Inspect output format and loss; conduct the local `smoke` first to verify the training implementation independently of model quality.

Train each configured seed and arm:

```sh
for seed in 42 43 44; do
  for arm in readout no_readout shuffled; do
    .venv/bin/jreflect train --run artifacts/initial --arm "$arm" --seed "$seed"
  done
done
```

All wrong training items are retained; right items are sampled to match, with replacement when necessary. Balancing happens once, before constructing arms. Shuffled donor assignment is a derangement of unique source IDs within a split, so oversampled instances cannot receive their own source's readout. No correctness labels enter prompts. Prompt/padding tokens are masked from loss, and overlong examples raise an error instead of truncating the targets.

Evaluate completed adapters on dev before freezing:

```sh
for seed in 42 43 44; do
  for arm in readout no_readout shuffled; do
    .venv/bin/jreflect predict --run artifacts/initial --split dev_tune --arm "$arm" --seed "$seed"
  done
done
.venv/bin/jreflect report --run artifacts/initial --split dev_tune
```

The current runner saves the final configured-epoch adapter; it does not choose a checkpoint by looking at test performance. Hyperparameter exploration should use separately identified runs and be recorded before final evaluation. Changing configured seeds after a run is prepared invalidates its provenance.

## Freeze, then evaluate test once

```sh
.venv/bin/jreflect freeze --run artifacts/initial
.venv/bin/jreflect prepare-test --run artifacts/initial
.venv/bin/jreflect generate --run artifacts/initial --split test --lens artifacts/initial/lens/lens.pt
for seed in 42 43 44; do
  for arm in readout no_readout shuffled; do
    .venv/bin/jreflect predict --run artifacts/initial --split test --arm "$arm" --seed "$seed"
  done
done
.venv/bin/jreflect report --run artifacts/initial --split test
```

Freeze requires every configured adapter to be complete and binds their file hashes, lens, example manifest, gate, code, environment, and statistical rules. The test cache must cover all 1,319 items; every arm receives the same first answers. Rerunning a completed command reuses stored predictions rather than generating another sample. There is no post-test training path within a frozen run.

Reports appear in `artifacts/initial/reports/test/`. JSON includes all metrics, conditional denominators, intervals, paired contingency counts, raw/adjusted p-values, and seed summaries. Fewer than three seeds is labeled a pilot. The success flag requires positive Readout improvements over both controls and Holm-adjusted p<0.05 in every configured seed, with at least three seeds. It is not forced true when the experiment runs successfully.

Optional exploratory predictions can be saved separately with `predict --arm readout --ablation readout_removed` or `--ablation tuned_first`, supplying the usual run/split/seed arguments. These caches do not enter primary comparisons. Error-type annotation and tuned-weight lens refitting remain research follow-ups; neither is inferred automatically.
