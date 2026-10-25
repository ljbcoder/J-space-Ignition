# Initial Experiment: Readout-Trained Self-Correction on MATH

Oct 25, 2026 · @arham

We fine-tune **Qwen3.5-9B** to make a second attempt on MATH competition problems given its first answer and a workspace readout from that attempt. The question is whether real readouts beat two controls trained the same way: no readout, and a readout shuffled from another problem. Nothing is written into the model's activations; the readout is shown as text and the model runs normally.

**Why this model.** Qwen3.5-9B already has released workspace lenses, so there is no lens to fit ([camilablank/workspace-lenses](https://huggingface.co/camilablank/workspace-lenses)). It fits on a single 24 GB GPU ([vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-9B)). The same release includes Qwen3.5-4B, a ready-made smaller scale point for later.

**Why MATH over GSM8K.** It's harder, so a 9B model makes plenty of errors, and its intermediates are often concepts rather than numbers: *prime*, *modulo*, *hypotenuse*, *permutation*. Those are single word tokens the lens can actually show, which GSM8K's multi-digit values are not.

## Setup

**Lens: use the best released one, fit nothing.**

| Lens for Qwen3.5-9B | Readout recall@10 | Where |
| --- | --- | --- |
| J++ Lens | 55.0% | Check [koayon/jpp-lenses](https://huggingface.co/koayon/jpp-lenses) for a 9B file |
| R-Lens | 37.5% | Released: `qwen3.5-9b/r-lens/lens.pt` in [workspace-lenses](https://huggingface.co/camilablank/workspace-lenses) |
| J-Lens | 34.2% | Released, same repo |

Recall figures are from the [J++ paper's Table 1](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace). Use J++ if a 9B file is published; otherwise use the R-Lens. Applying a lens is one line: softmax(W\_U · norm(J\_ℓ · h\_ℓ)) over the layer's activation.

**Validate in an afternoon.** Reproduce a readout from the papers ("the animal that spins webs" → *spider*, *legs* in middle layers) before running anything else.

**Turn off thinking mode.** Qwen3.5-9B reasons by default. Disable it (`enable_thinking: false`) so both attempts are genuinely answer-only.

**Compute**

- 9B in bf16 is about 18 GB of weights; one 24–80 GB GPU handles generation, readouts and LoRA.
- A readout is one matrix multiply per layer per position, so it adds almost nothing at inference.
- Total budget: roughly 1–2 GPU-days.

## Data

MATH has 7,500 train and 5,000 test problems drawn from competitions such as AMC 10, AMC 12 and AIME, each tagged with one of five difficulty levels and a subject ([dataset card](https://www.tensorflow.org/datasets/community_catalog/huggingface/competition_math)).

| Split | Source | Size | Use |
| --- | --- | --- | --- |
| Train | MATH train, minus dev | 7,000 | Generate training examples |
| Dev | Random 500 from MATH train, stratified by level | 500 | Fix readout format, prompts, hyperparameters |
| Test | Full MATH test | 5,000 | Final evaluation, touched once; also report the standard MATH-500 subset |

**Use the full test set, not just MATH-500.** The tests need 500+ wrong first answers. MATH-500 alone may not give that many, depending on the model's no-CoT accuracy.

**Both attempts are answer-only (no CoT).** The workspace carries the most when reasoning isn't written out, and an identical format across attempts and arms removes CoT as a confound.

**Grading.** MATH answers are LaTeX expressions (fractions, radicals, intervals), so string matching undercounts. Grade with a symbolic equivalence checker such as Hugging Face's Math-Verify, and spot-check 50 gradings by hand.

**What readouts will and won't show.** Concept words (*prime*, *factor*, *radius*, *probability*) are single tokens and readable. Numeric values are still split into digits by Qwen's tokenizer, so they won't appear as numbers.

**Contamination.** Assume Qwen3.5-9B has seen MATH: it was released in February 2026 and MATH has been public since 2021. This is standard for method papers comparing arms, because every arm uses the same model and items, so it doesn't bias the comparison. It does risk "fixes" that are really recall, so the experiment includes one clean check:

- **Memorization split.** For every test item, give the model the first half of the problem and check whether it completes the rest nearly verbatim. Tag items as likely-memorized or not.
- **Report the main result on both subsets.** If the readout effect holds on the not-memorized subset, contamination isn't driving it.
- State contamination as a limitation, and don't compare absolute accuracies with other papers.

## Step 1: readout sanity check

Before training anything, confirm readouts on MATH carry more than noise. This takes a day or two on the dev split.

1. Run the model greedy, answer-only, on the 500 dev items. Record correct/wrong and readouts (top 20) at every 4th layer, at the last prompt token and the first answer token.
2. **Content check.** From each MATH reference solution, extract the key concept words (theorem names, objects, operations), excluding words already in the problem. Score how many appear in the readout's top 10. Concepts the model needed but never stated are exactly what the workspace should hold.
3. **Error check.** Do readouts differ between wrong and correct items, in concepts or in meta-tokens such as *wrong* or *wait*? Train a linear probe on readout scores to predict correctness, and compare it with the same probe on logit-lens readouts.
4. **Baseline accuracy by level.** Record no-CoT accuracy per difficulty level. This tells you how many test errors you'll have and which levels have fixable rather than hopeless errors.

**Go/no-go:** proceed if the content check finds unstated solution concepts above a shuffled-solution baseline, and the readout probe beats chance and the logit lens. If not, there's nothing for training to use.

## Step 2: training data and readout format

**Generate data on-policy.** Run the base model, answer-only, on the full train split. Every item yields its first answer, a correct/wrong label and a readout. Using the model's own mistakes matters: SCoRe found that SFT on offline correction traces failed to teach self-correction ([Kumar et al., ICLR 2025](https://iclr.cc/virtual/2025/poster/30501)).

**Example construction**

- **Input:** problem + first answer + readout
- **Target:** the MATH gold answer, the final `\boxed{}` expression from the reference solution
- **Mix:** all wrong items, plus an equal number of right items whose target is the same answer. Without right items, the model learns to always change its answer.
- **No correctness signal in the prompt.** The model must decide whether to change its answer, which is the realistic setting.

**Prompt template** (identical across arms except the readout lines)

```
Problem: {problem}
Your previous answer: {first_answer}
While you answered, these concepts were most active in your
internal workspace (strongest first):
  middle layers: {top5_mid}
  late layers:   {top5_late}
Give only your final answer, in \boxed{}.
```

**Readout choices** (fix on dev, then freeze)

1. Positions: last prompt token and first answer token.
2. Layers: two bands, middle and late, top 5 each.
3. Filter non-semantic tokens and words copied verbatim from the problem.
4. Keep single digits; note they carry no place value.

## Step 3: three LoRA runs

All three runs use the same items, targets, hyperparameters and seed; only the readout lines differ.

| Run | Readout lines in the prompt | What it isolates |
| --- | --- | --- |
| **Readout** | The item's own readout | Main condition |
| **No readout** | Removed; prompt otherwise identical | Gain from self-correction training alone |
| **Shuffled** | Readout from a random other item | Gain from format and extra tokens |

**Starting hyperparameters** (adjust on dev)

- LoRA rank 16, alpha 32, dropout 0.05, on attention and MLP projections
- Learning rate 1e-4, cosine schedule, 2 epochs
- Loss on the target tokens only
- 3 seeds per run if compute allows; seed variance tells you how big a real effect must be

**Lens drift check.** The main evaluation takes first attempts and readouts from the frozen base model, so the lens stays valid. For the secondary check where the tuned model makes its own first attempt, rerun the Step 1 content check first; if readout quality drops, refit the lens on the tuned weights.

## Evaluation

**Protocol.** The frozen base model answers all 5,000 test items once, greedy, and its readouts are computed. Every tuned model then makes the second attempt from those same first answers. First attempts are identical across arms, so differences come only from the second attempt.

| Metric | Definition |
| --- | --- |
| **Net accuracy change** (headline) | Second-attempt accuracy minus first-attempt accuracy |
| W→R | Share of wrong first answers fixed |
| R→W | Share of right first answers broken |
| Change rate | Share of items where the answer changed at all |

**Tests.** Arms share items, so compare per-item outcomes with McNemar's test: Readout vs. No readout, and Readout vs. Shuffled. Bootstrap 95% CIs over items. With two primary contrasts, use Holm correction.

**Success criterion, set before running:** Readout beats both controls on net accuracy change with p < 0.05 after correction, consistently across seeds.

**Secondary checks**

- Results by difficulty level and subject: where do readouts help, and where are errors beyond repair?
- Tuned model's first attempt, no readout, vs. base model: did training improve reasoning itself?
- Second attempt with the readout removed, for the Readout model: did it learn recovery or only hint-following?
- MATH-500 subset numbers, for comparability with other work.

## Risks

| Risk | Sign | Response |
| --- | --- | --- |
| No J++ file for 9B | Not in the J++ release | Use the released R-Lens (37.5% vs. 55.0% recall@10); expect noisier readouts |
| Readouts carry little signal | Step 1 probe near chance | Move to a word-intermediate dataset where intermediates are known |
| Errors too hard to fix without CoT | Near-zero W→R in every arm at levels 4–5 | Restrict the primary analysis to levels 1–3 |
| SFT collapses to "repeat first answer" | Change rate near zero in all arms | Rebalance right/wrong items; consider an RL stage as in SCoRe |
| Grading errors | Hand spot-check disagrees with the checker | Fix the checker before any test-set run |
| Readout = No readout | Equal net gain | Itself a result: readouts add nothing beyond self-correction training at 9B |

## Timeline

1. **Week 1:** Load Qwen3.5-9B and the best released lens; validate one readout. Set up grading and the memorization probe. Run Step 1 on dev; go/no-go.
2. **Week 2:** Generate training data on-policy; fix readout format and hyperparameters on dev; train the three runs.
3. **Week 3:** Test-set evaluation, secondary checks, write-up.

## Open questions

- Is a J++ lens for Qwen3.5-9B in the koayon/jpp-lenses release, or only the larger models?
- What is Qwen3.5-9B's no-CoT MATH accuracy per level? It sets how many errors we have and which are fixable.

## After this experiment: synthetic data

If the MATH result is positive, the next experiment moves to a generated dataset. It answers what MATH can't:

- **Contamination-free by construction:** problems produced from templates with randomized values and answers computed in code.
- **Known intermediates:** each template has a chosen bridge concept, so you can check whether the readout contained it and whether that predicts a successful fix. That explains *why* readouts help, not just that they do.
- **Readable by the lens:** templates are built so the key intermediate is a single word token, avoiding the digit-tokenization problem.

Candidate templates: two-hop fact + arithmetic ("legs on the animal that spins webs, times 3"), modular arithmetic, counting and combinations, simple probability, basic geometry.

## Sources

- [J++ Lens (Ayonrinde & Lindsey, 2026)](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace) · [code](https://github.com/safety-research/jpp_lens) · [J++ lenses](https://huggingface.co/koayon/jpp-lenses)
- [Released J-/R-lenses incl. Qwen3.5-9B and 4B (camilablank/workspace-lenses)](https://huggingface.co/camilablank/workspace-lenses)
- [Verbalizable Representations Form a Global Workspace in Language Models (Gurnee et al., 2026)](https://huggingface.co/papers/2607.15495)
- [Qwen3.5-9B vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-9B)
- [MATH dataset card](https://www.tensorflow.org/datasets/community_catalog/huggingface/competition_math)
- [Training Language Models to Self-Correct via RL (SCoRe; Kumar et al., ICLR 2025)](https://iclr.cc/virtual/2025/poster/30501)
- [Self-Taught Self-Correction for Small Language Models](https://arxiv.org/pdf/2503.08681)
