# Initial Experiment: Readout-Trained Self-Correction on GSM8K

Oct 25, 2026 · @arham

We fine-tune **Qwen2.5-7B-Instruct** to make a second attempt on GSM8K problems given its first answer and a J++ readout of its own workspace from that attempt. The question is whether real readouts beat two controls trained the same way: no readout, and a readout shuffled from another problem. Nothing is written into the model's activations; the readout is shown as text and the model runs normally.

**Why 7B over 3B.** The workspace is a learned property, and it has only been shown at 9B and up ([Ayonrinde & Lindsey, 2026](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace)). A 3B model is less likely to have a readable one, and its no-CoT GSM8K errors are more likely to be hopeless rather than fixable. 7B still fits on one GPU and keeps lens fitting and LoRA runs cheap. If 7B works, 3B becomes a scale point later.

## Setup

**No released lens exists for a Qwen 7B model, so we fit our own.** The released J++ lenses cover 9B–284B models only. Fitting is the first real task and the first go/no-go.

**Fitting the J++ lens** (recipe from the J++ paper, scaled down):

- Code: the [jpp\_lens repo](https://github.com/safety-research/jpp_lens). First check it supports Qwen2.5's architecture; it was built for Qwen3.5/3.6, Gemma 4, Olmo 3 and DeepSeek.
- Fit data: 64 WikiText-103 sequences of 128 tokens, skipping the first 16 positions, as in the paper.
- 8 expert Jacobians per layer, LRP backward pass, non-semantic tokens filtered from readouts.
- Expert weights are learned on a small dev set of latent-extraction problems; use the repo's eval harness for these.
- Cost: the paper's 27B fit took about 9 H200-hours. Expect a few GPU-hours at 7B.

**Validate the lens before using it.** Reproduce the spider→legs readout and run the repo's readout evals. Target recall@10 near the paper's 9B figure (55.0%). Well below that, the readouts will mostly be noise.

**Compute**

- 7B in bf16 is about 15 GB of weights. One 40–80 GB GPU handles generation, readouts and LoRA.
- A readout is one matrix multiply per layer per position, so it adds almost nothing at inference.
- Total budget for this experiment: roughly 1–2 GPU-days.

## Data

| Split | Source | Size | Use |
| --- | --- | --- | --- |
| Train | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) train, minus dev | \~6,970 | Generate training examples |
| Dev | Random 500 from GSM8K train | 500 | Fix readout format, prompts, hyperparameters |
| Test | GSM8K test | 1,319 | Final evaluation only, touched once |

**Both attempts are answer-only (no CoT).** The workspace carries the most when reasoning isn't written out, and keeping the format identical across attempts and arms removes CoT as a confound. A CoT version is a follow-up, not part of this run.

**Caveat: multi-digit numbers won't appear in readouts.** The lens reads single tokens, and Qwen tokenizes digits one at a time; the J++ authors couldn't evaluate arithmetic on Qwen for this reason. GSM8K intermediates are mostly numbers, so readouts will show:

- quantity and operation words: *half*, *twice*, *per*, *total*, *remaining*, entity names
- single digits, without their place value
- meta-tokens such as *wrong* or *wait*

This weakens the signal compared with a word-intermediate dataset. It doesn't block the experiment: the training result is measured on accuracy, not on readout content. Step 1 checks whether enough signal survives.

## Step 1: readout sanity check

Before training anything, confirm readouts on GSM8K carry more than noise. This takes a day on the dev split.

1. Run the model greedy, answer-only, on the 500 dev items. Record correct/wrong and J++ readouts (top 20) at every 4th layer, at the last prompt token and the first answer token.
2. **Content check.** For each item, take the key words from the GSM8K reference solution (entities, operation words). Score how many appear in the readout's top 10, excluding words copied from the prompt.
3. **Error check.** Compare wrong vs. correct items: do readouts differ in operation words or meta-tokens? A linear probe on readout scores predicting correctness, compared with a probe on logit-lens readouts, gives one number.
4. **Baseline accuracy.** Record the no-CoT accuracy. You need roughly 500+ wrong test items for the tests later, which requires error rates above \~40%.

**Go/no-go:** proceed if readouts predict correctness above chance and beat the logit lens on that probe. If not, the readout has nothing for training to use, and the honest next move is a word-intermediate dataset rather than more GSM8K tuning.

## Step 2: training data and readout format

**Generate data on-policy.** Run the base model, answer-only, on the full train split. Every item yields its first answer, a correct/wrong label and a readout. Using the model's own mistakes matters: SCoRe found that SFT on offline correction traces failed to teach self-correction ([Kumar et al., ICLR 2025](https://iclr.cc/virtual/2025/poster/30501)).

**Example construction**

- **Input:** problem + first answer + readout
- **Target:** the GSM8K gold answer (the number after `####`), as a bare number
- **Mix:** all wrong items, plus an equal number of right items whose target is the same answer. Without right items, the model learns to always change its answer.
- **No correctness signal in the prompt.** The model must decide whether to change its answer, which is the realistic setting.

**Prompt template** (identical across arms except the readout lines)

```
Problem: {question}
Your previous answer: {first_answer}
While you answered, these concepts were most active in your
internal workspace (strongest first):
  middle layers: {top5_mid}
  late layers:   {top5_late}
Give your final answer as a number only.
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

**Protocol.** The frozen base model answers all 1,319 test items once, greedy, and its readouts are computed. Every tuned model then makes the second attempt from those same first answers. First attempts are identical across arms, so differences come only from the second attempt.

| Metric | Definition |
| --- | --- |
| **Net accuracy change** (headline) | Second-attempt accuracy minus first-attempt accuracy |
| W→R | Share of wrong first answers fixed |
| R→W | Share of right first answers broken |
| Change rate | Share of items where the answer changed at all |

**Tests.** Arms share items, so compare per-item outcomes with McNemar's test: Readout vs. No readout, and Readout vs. Shuffled. Bootstrap 95% CIs over items. With two primary contrasts, use Holm correction.

**Success criterion, set before running:** Readout beats both controls on net accuracy change with p < 0.05 after correction, consistently across seeds.

**Secondary checks**

- Tuned model's first attempt, no readout, vs. base model: did training improve reasoning itself?
- Second attempt with the readout removed, for the Readout model: did it learn recovery or only hint-following?
- Results split by first-attempt error type (arithmetic slip vs. wrong setup), using a small hand-labeled sample.

## Risks

| Risk | Sign | Response |
| --- | --- | --- |
| Lens won't fit or is poor at 7B | Repo lacks Qwen2.5 support, or recall@10 far below 55% | Switch to Qwen3.5-9B, which has released lenses |
| GSM8K readouts carry little signal | Step 1 probe near chance | Move to a word-intermediate dataset |
| SFT collapses to "repeat first answer" | Change rate near zero in all arms | Rebalance right/wrong items; consider an RL stage as in SCoRe |
| Readout = No readout | Equal net gain | Itself a result: readouts add nothing beyond self-correction training at 7B |
| Too few test errors for power | Base accuracy above \~60% | Add MATH or harder GSM8K variants to the test pool |

## Timeline

1. **Week 1:** Fit and validate the J++ lens on Qwen2.5-7B-Instruct.
2. **Week 2:** Step 1 sanity check on dev; go/no-go.
3. **Week 3:** Generate training data; fix readout format and hyperparameters on dev; train the three runs.
4. **Week 4:** Test-set evaluation, secondary checks, write-up.

## Open questions

- Does jpp\_lens support Qwen2.5's architecture out of the box?
- What is Qwen2.5-7B-Instruct's no-CoT GSM8K accuracy? It sets how many errors we have to work with.

## Sources

- [J++ Lens (Ayonrinde & Lindsey, 2026)](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace) · [code](https://github.com/safety-research/jpp_lens) · [released lenses](https://huggingface.co/koayon/jpp-lenses)
- [Verbalizable Representations Form a Global Workspace in Language Models (Gurnee et al., 2026)](https://huggingface.co/papers/2607.15495)
- [Training Language Models to Self-Correct via RL (SCoRe; Kumar et al., ICLR 2025)](https://iclr.cc/virtual/2025/poster/30501)
- [Self-Taught Self-Correction for Small Language Models](https://arxiv.org/pdf/2503.08681)
- [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k)
