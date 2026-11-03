# J-Reflect Recovery: run results for analysis

## What this experiment is

Question: can decoded internal information help a frozen language model correct its own mistakes? A frozen model (no training) answers program-generated arithmetic and variable-tracking problems answer-only, with thinking disabled. Before it answers, its residual-stream activations at the last prompt token are decoded with a released J++ lens (and a raw logit lens as baseline) into top-k tokens per layer. Each problem has a known hidden intermediate value. The model then gets a second attempt under four independent conditions: retry_only (neutral review), error_only (told it was wrong), shuffled_jpp (wrong + another problem's J++ readout), authentic_jpp (wrong + its own J++ readout). Primary contrasts: authentic minus shuffled, and authentic minus error_only, on initially wrong answers. Ablations edit the hidden value out of the feedback text. A development split gates a held-out test split.

Key definition: an attempt is 'latent eligible' only if the hidden intermediate is a single vocabulary token, absent from the question and the visible answer, and the answer parses. Detection = that exact token appears in the readout top-k at any layer.

## Run configuration

- Model: Qwen/Qwen3.5-9B @ c202236235762e1c871ad0ccb60c8ee5ba337b9a
- Lens: koayon/jpp-lenses/qwen3.5-9b/lens.pt @ 237dab6ea0e611d964bf407f13ba1655f1a6ae43; layers [4, 8, 12, 16, 20, 24, 28]; top-k 10
- Data: {'dev_count': 200, 'test_count': 1000, 'tail_steps': 3}; decoding greedy, thinking=False, max_new_tokens=64
- Mode: research; frozen: False

## Verification notes (checked after the run)

- Code and environment: source hash of the code that ran matches the repository; installed package versions match uv.lock; research mode on an NVIDIA L40S GPU; 24/24 unit tests pass and lint is clean.
- Grading: every gold answer was recomputed from its oracle steps and every first answer regraded independently: 0/200 correct (4/200 within 1% of gold). All 200 answers ended normally (EOS), none truncated; no thinking text.
- Readouts: all 200 attempts captured all 7 layers at the last prompt token during the first generation prefill.
- Correction prompts: 800 continuations (4 conditions x 200). Every authentic_jpp prompt contains that problem's own readout; every shuffled_jpp prompt uses a different problem's readout.
- Why 0 attempts are latent-eligible: the Qwen3.5-9B tokenizer (248,044 tokens) contains only the single digits 0-9 as numeric tokens, and its pre-tokenizer splits every number into individual digits. A multi-digit intermediate such as 84 can therefore never be a single token, so the detection measure and ablations cannot apply to any multi-digit target on this model. The freeze gate (>= 10 eligible errors) failed for this reason and the held-out test split was not run.
- Procedural deviations: the readout-inspection step before correction (collect --limit 10, then report) was skipped by the automation; the image installed with `uv sync --frozen` rather than `--locked` (identical versions). Neither affects the results.
- Cost: about $0.53 of GPU time on Modal.

## Notes from the researcher

(Add your own notes or questions here before sharing.)

## dev split

### Data facts (computed from saved records)
- First attempts saved: 200; grading: {'wrong': 200}
- Latent-eligible attempts: 0; ineligibility reasons: {'target_not_single_token': 200}

### Report

# J-Reflect recovery: dev

**Mode: research — complete**

First attempts: 200/200. Correct: 0; wrong: 200; format failures: 0.

## Readout detection

Eligible initially wrong attempts only. Recall is at the configured top-k, across the fixed layers.

| N | J++ hit rate | Logit hit rate | Shuffled hit rate |
| --- | --- | --- | --- |
| 0 | n/a | n/a | n/a |

## Correction and harm

| Condition | Wrong cases observed | Correction rate | Correct cases audited | Harm rate |
| --- | --- | --- | --- | --- |
| retry_only | 200/200 | 0.0% | 0 | n/a |
| error_only | 200/200 | 0.0% | 0 | n/a |
| shuffled_jpp | 200/200 | 0.0% | 0 | n/a |
| authentic_jpp | 200/200 | 0.0% | 0 | n/a |

## Paired contrasts

Differences and intervals are percentage points. Incomplete pairs are excluded and N is shown.

| Comparison | Pairs | Difference | Confidence interval | Holm p |
| --- | --- | --- | --- | --- |
| authentic_jpp_minus_shuffled_jpp | 200 | 0.0% | 0.0% to 0.0% | 1.0 |
| authentic_jpp_minus_error_only | 200 | 0.0% | 0.0% to 0.0% | 1.0 |

## Interpretation and scope

- Pilot estimates; no claim of introspection or native-feature causality.
- Detection is best rank over the fixed layer set; identical search exposure in both lens baselines.
- Shuffled source is chosen without labels. With partial first collection, the donor bank is provisional.
- Intervals bootstrap independent questions within templates; they do not establish generalization to new task families.
- Feedback has matched readout slots, not exactly matched serialized prompt-token lengths; actual costs are recorded.
- Correct first attempts receive neutral review instructions, never a false error signal.
- Oracle-guided ablations are diagnostic only. Textual intervention does not establish neural causality.

Saved continuation records: 800/800; ablations: 0/0.

Full detection comparisons, latent strata, ablations, and compute accounting are in `summary.json`.


### summary.json

```json
{
  "mode": "research",
  "split": "dev",
  "status": "complete",
  "first_attempts": 200,
  "first_attempts_expected": 200,
  "initial_correct": 0,
  "initial_wrong": 200,
  "first_format_failures": 0,
  "first_accuracy": 0.0,
  "corrections_observed": 800,
  "corrections_expected": 800,
  "ablations_observed": 0,
  "ablations_expected": 0,
  "detection": {
    "all_eligible": {
      "n": 0,
      "jpp_hit_rate": null,
      "logit_hit_rate": null,
      "shuffled_hit_rate": null,
      "jpp_mrr": null,
      "jpp_vs_logit": {
        "n": 0,
        "difference": null,
        "ci": [
          null,
          null
        ],
        "wins": 0,
        "losses": 0,
        "p": null
      },
      "jpp_vs_shuffled": {
        "n": 0,
        "difference": null,
        "ci": [
          null,
          null
        ],
        "wins": 0,
        "losses": 0,
        "p": null
      }
    },
    "initially_wrong": {
      "n": 0,
      "jpp_hit_rate": null,
      "logit_hit_rate": null,
      "shuffled_hit_rate": null,
      "jpp_mrr": null,
      "jpp_vs_logit": {
        "n": 0,
        "difference": null,
        "ci": [
          null,
          null
        ],
        "wins": 0,
        "losses": 0,
        "p": null
      },
      "jpp_vs_shuffled": {
        "n": 0,
        "difference": null,
        "ci": [
          null,
          null
        ],
        "wins": 0,
        "losses": 0,
        "p": null
      }
    },
    "initially_correct": {
      "n": 0,
      "jpp_hit_rate": null,
      "logit_hit_rate": null,
      "shuffled_hit_rate": null,
      "jpp_mrr": null,
      "jpp_vs_logit": {
        "n": 0,
        "difference": null,
        "ci": [
          null,
          null
        ],
        "wins": 0,
        "losses": 0,
        "p": null
      },
      "jpp_vs_shuffled": {
        "n": 0,
        "difference": null,
        "ci": [
          null,
          null
        ],
        "wins": 0,
        "losses": 0,
        "p": null
      }
    }
  },
  "conditions": {
    "retry_only": {
      "wrong_attempts_observed": 200,
      "wrong_attempts_expected": 200,
      "wrong_to_right": 0,
      "correction_rate": 0.0,
      "correct_attempts_observed": 0,
      "correct_to_wrong": 0,
      "harm_rate": null,
      "second_format_failures": 0,
      "answer_change_rate": 0.0,
      "prompt_tokens": 24515,
      "generated_tokens": 2128,
      "seconds": 97.42318386400001
    },
    "error_only": {
      "wrong_attempts_observed": 200,
      "wrong_attempts_expected": 200,
      "wrong_to_right": 0,
      "correction_rate": 0.0,
      "correct_attempts_observed": 0,
      "correct_to_wrong": 0,
      "harm_rate": null,
      "second_format_failures": 0,
      "answer_change_rate": 0.005,
      "prompt_tokens": 23915,
      "generated_tokens": 2129,
      "seconds": 96.9716282629995
    },
    "shuffled_jpp": {
      "wrong_attempts_observed": 200,
      "wrong_attempts_expected": 200,
      "wrong_to_right": 0,
      "correction_rate": 0.0,
      "correct_attempts_observed": 0,
      "correct_to_wrong": 0,
      "harm_rate": null,
      "second_format_failures": 0,
      "answer_change_rate": 0.0,
      "prompt_tokens": 85515,
      "generated_tokens": 2128,
      "seconds": 101.2528781830004
    },
    "authentic_jpp": {
      "wrong_attempts_observed": 200,
      "wrong_attempts_expected": 200,
      "wrong_to_right": 0,
      "correction_rate": 0.0,
      "correct_attempts_observed": 0,
      "correct_to_wrong": 0,
      "harm_rate": null,
      "second_format_failures": 0,
      "answer_change_rate": 0.0,
      "prompt_tokens": 85515,
      "generated_tokens": 2128,
      "seconds": 101.71020551800018
    }
  },
  "primary_comparisons": {
    "authentic_jpp_minus_shuffled_jpp": {
      "n": 200,
      "difference": 0.0,
      "ci": [
        0.0,
        0.0
      ],
      "wins": 0,
      "losses": 0,
      "p": 1.0,
      "holm_p": 1.0
    },
    "authentic_jpp_minus_error_only": {
      "n": 200,
      "difference": 0.0,
      "ci": [
        0.0,
        0.0
      ],
      "wins": 0,
      "losses": 0,
      "p": 1.0,
      "holm_p": 1.0
    }
  },
  "latent_strata": {
    "detected": {
      "n": 0,
      "difference": null,
      "ci": [
        null,
        null
      ],
      "wins": 0,
      "losses": 0,
      "p": null
    },
    "not_detected": {
      "n": 0,
      "difference": null,
      "ci": [
        null,
        null
      ],
      "wins": 0,
      "losses": 0,
      "p": null
    }
  },
  "ablations": {
    "authentic_minus_remove_target": {
      "n": 0,
      "difference": null,
      "ci": [
        null,
        null
      ],
      "wins": 0,
      "losses": 0,
      "p": null
    },
    "authentic_minus_replace_target": {
      "n": 0,
      "difference": null,
      "ci": [
        null,
        null
      ],
      "wins": 0,
      "losses": 0,
      "p": null
    },
    "authentic_minus_sham_remove": {
      "n": 0,
      "difference": null,
      "ci": [
        null,
        null
      ],
      "wins": 0,
      "losses": 0,
      "p": null
    }
  },
  "cost": {
    "generation_calls": 1000,
    "prompt_tokens": 236047,
    "generated_tokens": 10641,
    "generation_seconds": 494.0267820900003,
    "readout_seconds": 18.98189437399995
  },
  "limitations": [
    "Pilot estimates; no claim of introspection or native-feature causality.",
    "Detection is best rank over the fixed layer set; identical search exposure in both lens baselines.",
    "Shuffled source is chosen without labels. With partial first collection, the donor bank is provisional.",
    "Intervals bootstrap independent questions within templates; they do not establish generalization to new task families.",
    "Feedback has matched readout slots, not exactly matched serialized prompt-token lengths; actual costs are recorded.",
    "Correct first attempts receive neutral review instructions, never a false error signal.",
    "Oracle-guided ablations are diagnostic only. Textual intervention does not establish neural causality."
  ]
}
```

### Example transcripts (first 4 problems)

#### Example dev-00761e4435a4449b (add_multiply)
- Question: Start with 73. Add 11, then multiply the result by 4. Next multiply the result by 21. Next add 36. Next subtract 24. What is the final number?
- Gold answer: 7068; hidden intermediate (target): 84
- Oracle steps: 73 + 11 = 84; 84 * 4 = 336; 336 * 21 = 7056; 7056 + 36 = 7092; 7092 - 24 = 7068
- First answer: 'Answer: 13080' -> wrong
- Latent eligible: False ['target_not_single_token']
- J++ readout (top tokens per layer, last prompt position, before answering):
  - layer 4: '<u', '<br', '\\u', '<b', '<span', '\\n', '<strong', '\\s', '<a', '爲'
  - layer 8: '数学', 'Math', ' mathematical', ' Mathematical', ' Adobe', ' Euler', 'ithmetic', 'acers', 'geo', '.Euler'
  - layer 12: '’T', '’S', '1', '2', '7', ' Calculation', '’an', '9', '’B', '3'
  - layer 16: ' Calculator', ' Calculation', 'Bitcoin', ' Arithmetic', 'Calc', 'CT', 'RM', '推理', '的计算', '算力'
  - layer 20: '计算', ' calculation', ' Calculation', ' calculations', '的计算', ' Carn', 'Calc', '@g', ' calculate', ' Answer'
  - layer 24: '@c', '@g', '@m', '@d', '@s', '@end', '@a', '@include', '@if', '@Data'
  - layer 28: ' Answer', ' answer', 'Answer', ' ANSW', ' Answers', '_answer', ' answers', '.answer', '答案', ' answering'
- Raw logit-lens readout (same activations):
  - layer 4: '1', 'd', 'C', ' C', 'A', '正', '一', 'eng', 'x', 'ly'
  - layer 8: '卓', '技', '依法', '横', 'math', '晏', 'inking', '数学', 'ros', '沙'
  - layer 12: '昔', '牙', '完整的', '法', '蜂', 'ipot', '计算', '先行', '位', '影'
  - layer 16: '位数', '概', '理', ' ans', ' jih', '尾', '算', '莱特', "'ve", '的计算'
  - layer 20: 'TTTT', ' radios', 'OwnerId', '三八', 'noop', ' ans', '算', 'ans', 'arily', '接'
  - layer 24: ' ans', ' Answer', 'Answer', '晗', 'noop', 'ansible', 'ans', 'engan', 'uffle', '默'
  - layer 28: ' Answer', 'Answer', ' answer', '_answer', '答案', ' ANSW', '.answer', '-answer', 'answer', '答案是'
- Condition `error_only`
  - Feedback prompt:
    Your previous answer was incorrect. Solve the problem again.
    Return only Answer: <integer>.
  - Second answer: 'Answer: 13080' -> wrong
- Condition `authentic_jpp`
  - Feedback prompt:
    Your previous answer was incorrect. Solve the problem again.
    Automatically extracted diagnostic tokens follow. They may be noisy or irrelevant.
    [{"layer": 4, "tokens": ["<u", "<br", "\\u", "<b", "<span", "\\n", "<strong", "\\s", "<a", "爲"]}, {"layer": 8, "tokens": ["数学", "Math", " mathematical", " Mathematical", " Adobe", " Euler", "ithmetic", "acers", "geo", ".Euler"]}, {"layer": 12, "tokens": ["’T", "’S", "1", "2", "7", " Calculation", "’an", "9", "’B", "3"]}, {"layer": 16, "tokens": [" Calculator", " Calculation", "Bitcoin", " Arithmetic", "Calc", "CT", "RM", "推理", "的计算", "算力"]}, {"layer": 20, "tokens": ["计算", " calculation", " Calculation", " calculations", "的计算", " Carn", "Calc", "@g", " calculate", " Answer"]}, {"layer": 24, "tokens": ["@c", "@g", "@m", "@d", "@s", "@end", "@a", "@include", "@if", "@Data"]}, {"layer": 28, "tokens": [" Answer", " answer", "Answer", " ANSW", " Answers", "_answer", " answers", ".answer", "答案", " answering"]}]
    Return only Answer: <integer>.
  - Second answer: 'Answer: 13080' -> wrong
- Condition `shuffled_jpp` (donor dev-41d1f0e200fa3922)
  - Feedback prompt:
    Your previous answer was incorrect. Solve the problem again.
    Automatically extracted diagnostic tokens follow. They may be noisy or irrelevant.
    [{"layer": 4, "tokens": ["<u", "<br", "\\u", "<b", "<span", "\\n", "<strong", "\\s", "<a", "爲"]}, {"layer": 8, "tokens": ["数学", "Math", " mathematical", " Mathematical", "ithmetic", " Adobe", " Euler", "acers", "geo", "Calcul"]}, {"layer": 12, "tokens": ["’T", "’S", "1", "2", "7", " Calculation", "’an", "9", "’B", "3"]}, {"layer": 16, "tokens": [" Calculator", " Calculation", "Bitcoin", " Arithmetic", "Calc", "CT", "RM", "推理", "的计算", "算力"]}, {"layer": 20, "tokens": ["计算", " calculation", " Calculation", " calculations", "的计算", "Calc", " Carn", " Answer", " calculate", "@g"]}, {"layer": 24, "tokens": ["@c", "@g", "@m", "@s", "@d", "@end", "@a", "@include", "@if", "@Data"]}, {"layer": 28, "tokens": [" Answer", " answer", "Answer", " ANSW", " Answers", "_answer", " answers", ".answer", " answering", "答案"]}]
    Return only Answer: <integer>.
  - Second answer: 'Answer: 13080' -> wrong
- Condition `retry_only`
  - Feedback prompt:
    Review your previous answer. Keep it if correct; otherwise correct it.
    Return only Answer: <integer>.
  - Second answer: 'Answer: 13080' -> wrong

#### Example dev-00b2a3b0fd498833 (add_multiply)
- Question: Start with 35. Add 11, then multiply the result by 8. Next multiply the result by 18. Next add 18. Next subtract 15. What is the final number?
- Gold answer: 6627; hidden intermediate (target): 46
- Oracle steps: 35 + 11 = 46; 46 * 8 = 368; 368 * 18 = 6624; 6624 + 18 = 6642; 6642 - 15 = 6627
- First answer: 'Answer: 4686' -> wrong
- Latent eligible: False ['target_not_single_token']
- J++ readout (top tokens per layer, last prompt position, before answering):
  - layer 4: '<u', '<br', '\\u', '<b', '<span', '\\n', '<strong', '\\s', '<a', '爲'
  - layer 8: '数学', 'Math', ' mathematical', ' Mathematical', 'ithmetic', ' Adobe', ' Euler', 'acers', 'geo', 'Calcul'
  - layer 12: '’T', '’S', '1', ' Calculation', '2', '’an', '7', '’B', '9', '’n'
  - layer 16: ' Calculator', ' Calculation', 'Bitcoin', ' Arithmetic', 'Calc', 'CT', '的计算', '算力', '推理', 'RM'
  - layer 20: '计算', ' calculation', ' Calculation', ' calculations', '的计算', ' Carn', 'Calc', ' Answer', '@g', ' calculate'
  - layer 24: '@c', '@g', '@m', '@s', '@d', '@end', '@a', '@if', '@include', '@Data'
  - layer 28: ' Answer', ' answer', 'Answer', ' ANSW', ' Answers', '_answer', ' answers', '.answer', ' answering', '答案'
- Raw logit-lens readout (same activations):
  - layer 4: '1', 'd', 'C', ' C', 'A', '正', 'eng', '一', 'x', 'ly'
  - layer 8: '卓', '技', '横', '依法', 'math', 'inking', '晏', '数学', 'ros', ' math'
  - layer 12: '昔', '牙', '完整的', '法', '蜂', '先行', 'ipot', '位', '纠正', '计算'
  - layer 16: '位数', '概', '理', ' ans', '尾', ' jih', '莱特', '算', "'ve", '的计算'
  - layer 20: 'TTTT', '三八', ' radios', 'OwnerId', 'noop', '算', ' ans', '接', 'まず', '</think>'
  - layer 24: ' ans', ' Answer', 'Answer', '晗', 'ansible', 'noop', 'ans', 'mente', 'uffle', 'engan'
  - layer 28: ' Answer', 'Answer', ' answer', '_answer', '答案', ' ANSW', '-answer', '.answer', 'answer', '答案是'
- Condition `error_only`
  - Feedback prompt:
    Your previous answer was incorrect. Solve the problem again.
    Return only Answer: <integer>.
  - Second answer: 'Answer: 4686' -> wrong
- Condition `authentic_jpp`
  - Feedback prompt:
    Your previous answer was incorrect. Solve the problem again.
    Automatically extracted diagnostic tokens follow. They may be noisy or irrelevant.
    [{"layer": 4, "tokens": ["<u", "<br", "\\u", "<b", "<span", "\\n", "<strong", "\\s", "<a", "爲"]}, {"layer": 8, "tokens": ["数学", "Math", " mathematical", " Mathematical", "ithmetic", " Adobe", " Euler", "acers", "geo", "Calcul"]}, {"layer": 12, "tokens": ["’T", "’S", "1", " Calculation", "2", "’an", "7", "’B", "9", "’n"]}, {"layer": 16, "tokens": [" Calculator", " Calculation", "Bitcoin", " Arithmetic", "Calc", "CT", "的计算", "算力", "推理", "RM"]}, {"layer": 20, "tokens": ["计算", " calculation", " Calculation", " calculations", "的计算", " Carn", "Calc", " Answer", "@g", " calculate"]}, {"layer": 24, "tokens": ["@c", "@g", "@m", "@s", "@d", "@end", "@a", "@if", "@include", "@Data"]}, {"layer": 28, "tokens": [" Answer", " answer", "Answer", " ANSW", " Answers", "_answer", " answers", ".answer", " answering", "答案"]}]
    Return only Answer: <integer>.
  - Second answer: 'Answer: 4686' -> wrong
- Condition `shuffled_jpp` (donor dev-0666e1e41d1cd85f)
  - Feedback prompt:
    Your previous answer was incorrect. Solve the problem again.
    Automatically extracted diagnostic tokens follow. They may be noisy or irrelevant.
    [{"layer": 4, "tokens": ["<u", "<br", "\\u", "<b", "<span", "\\n", "<strong", "\\s", "<a", "爲"]}, {"layer": 8, "tokens": ["数学", "Math", " mathematical", " Mathematical", "ithmetic", " Adobe", " Euler", "acers", "geo", "Calcul"]}, {"layer": 12, "tokens": ["’T", "’S", "1", "7", "2", " Calculation", "9", "’an", "’B", "3"]}, {"layer": 16, "tokens": [" Calculator", " Calculation", "Bitcoin", " Arithmetic", "Calc", "CT", "推理", "的计算", "RM", " Bitcoin"]}, {"layer": 20, "tokens": ["计算", " calculation", " Calculation", " calculations", "的计算", " Carn", "Calc", "@g", " calculate", " Answer"]}, {"layer": 24, "tokens": ["@c", "@g", "@m", "@s", "@d", "@end", "@a", "@if", "@include", "@Data"]}, {"layer": 28, "tokens": [" Answer", " answer", "Answer", " ANSW", " Answers", "_answer", " answers", ".answer", "答案", " answering"]}]
    Return only Answer: <integer>.
  - Second answer: 'Answer: 4686' -> wrong
- Condition `retry_only`
  - Feedback prompt:
    Review your previous answer. Keep it if correct; otherwise correct it.
    Return only Answer: <integer>.
  - Second answer: 'Answer: 4686' -> wrong

#### Example dev-00ec5cc0d0a3d98a (track_add)
- Question: Execute these assignments in order: x = 41; x = x * 6; x = x + 25; y = x * 19; y = y * 31; y = y + 33; y = y - 28. What is y?
- Gold answer: 159624; hidden intermediate (target): 271
- Oracle steps: x = 41; x = 246; x = 271; y = 5149; y = 159619; y = 159652; y = 159624
- First answer: 'Answer: 146451' -> wrong
- Latent eligible: False ['target_not_single_token']
- J++ readout (top tokens per layer, last prompt position, before answering):
  - layer 4: '<u', '\\u', '<b', '<span', '<br', '<strong', '\\n', '\\s', '爲', '<a'
  - layer 8: '数学', 'Math', ' mathematical', 'comput', ' computational', 'Comput', ' Mathematical', '2', 'acers', ' computer'
  - layer 12: '1', '2', '’T', '’S', '7', '3', '9', '’s', 'ﬁ', '0'
  - layer 16: ' Calculator', ' Calculation', 'IBM', ' Arithmetic', '计算', '的计算', ' Logic', 'Calc', 'Logic', 'Bitcoin'
  - layer 20: '计算', ' Calculation', ' calculation', '的计算', 'Lambda', ' Aspen', ' Answer', ' calculations', ' Ragnar', ' Lambda'
  - layer 24: '@c', '@g', '@m', '@d', '@s', '@end', '@a', '@if', '@include', '@section'
  - layer 28: ' Answer', ' answer', 'Answer', ' ANSW', ' Answers', '_answer', ' answers', '.answer', '-answer', ' answering'
- Raw logit-lens readout (same activations):
  - layer 4: '1', 'd', ' C', 'C', 'A', '一', 'ID', 'x', 'eng', '正'
  - layer 8: '横', '卓', '技', '数学', 'math', '沙', 'ilag', 'ros', '粉', '晏'
  - layer 12: '昔', 'ix', '佐', '彬', '完整的', '牙', '韦', '理性', '数学', '彬彬'
  - layer 16: '概', '莱特', '位数', '的计算', ' calculation', 'angen', '稿', ' vertically', ' answer', ' Bere'
  - layer 20: 'TTTT', 'まず', '三八', 'uffle', 'OwnerId', '尻', 'istr', ' ans', '算', 'unnel'
  - layer 24: ' Answer', ' ans', 'Answer', ' answer', ' Ans', 'engan', 'ansible', ' ANSW', 'ans', 'noop'
  - layer 28: ' Answer', 'Answer', ' answer', '_answer', '-answer', ' ANSW', '答案', 'answer', '.answer', ' Answers'
- Condition `error_only`
  - Feedback prompt:
    Your previous answer was incorrect. Solve the problem again.
    Return only Answer: <integer>.
  - Second answer: 'Answer: 146451' -> wrong
- Condition `shuffled_jpp` (donor dev-d94a087275e2cd56)
  - Feedback prompt:
    Your previous answer was incorrect. Solve the problem again.
    Automatically extracted diagnostic tokens follow. They may be noisy or irrelevant.
    [{"layer": 4, "tokens": ["<u", "\\u", "<b", "<span", "<br", "<strong", "\\n", "\\s", "爲", "<a"]}, {"layer": 8, "tokens": ["数学", "Math", "comput", "acers", " computational", " mathematical", "2", "Comput", "ithmetic", " Mathematical"]}, {"layer": 12, "tokens": ["1", "’T", "2", "’S", "7", "3", "’s", "ﬁ", "9", "’re"]}, {"layer": 16, "tokens": [" Calculator", "IBM", " Calculation", " Arithmetic", "计算", "Bitcoin", " Logic", "的计算", "Calc", "Logic"]}, {"layer": 20, "tokens": ["计算", "Lambda", " calculation", " Answer", " Calculation", "的计算", " Lambda", " Aspen", "@g", " calculations"]}, {"layer": 24, "tokens": ["@c", "@g", "@m", "@d", "@s", "@a", "@end", "@if", "@include", "@section"]}, {"layer": 28, "tokens": [" Answer", " answer", "Answer", " ANSW", " Answers", "_answer", " answers", ".answer", "-answer", " answering"]}]
    Return only Answer: <integer>.
  - Second answer: 'Answer: 146451' -> wrong
- Condition `retry_only`
  - Feedback prompt:
    Review your previous answer. Keep it if correct; otherwise correct it.
    Return only Answer: <integer>.
  - Second answer: 'Answer: 146451' -> wrong
- Condition `authentic_jpp`
  - Feedback prompt:
    Your previous answer was incorrect. Solve the problem again.
    Automatically extracted diagnostic tokens follow. They may be noisy or irrelevant.
    [{"layer": 4, "tokens": ["<u", "\\u", "<b", "<span", "<br", "<strong", "\\n", "\\s", "爲", "<a"]}, {"layer": 8, "tokens": ["数学", "Math", " mathematical", "comput", " computational", "Comput", " Mathematical", "2", "acers", " computer"]}, {"layer": 12, "tokens": ["1", "2", "’T", "’S", "7", "3", "9", "’s", "ﬁ", "0"]}, {"layer": 16, "tokens": [" Calculator", " Calculation", "IBM", " Arithmetic", "计算", "的计算", " Logic", "Calc", "Logic", "Bitcoin"]}, {"layer": 20, "tokens": ["计算", " Calculation", " calculation", "的计算", "Lambda", " Aspen", " Answer", " calculations", " Ragnar", " Lambda"]}, {"layer": 24, "tokens": ["@c", "@g", "@m", "@d", "@s", "@end", "@a", "@if", "@include", "@section"]}, {"layer": 28, "tokens": [" Answer", " answer", "Answer", " ANSW", " Answers", "_answer", " answers", ".answer", "-answer", " answering"]}]
    Return only Answer: <integer>.
  - Second answer: 'Answer: 146451' -> wrong

#### Example dev-027ef29e93c0cdd1 (multiply_subtract)
- Question: Multiply 59 by 3. Then subtract 32. Next multiply the result by 19. Next add 14. Next subtract 14. What is the final number?
- Gold answer: 2755; hidden intermediate (target): 177
- Oracle steps: 59 * 3 = 177; 177 - 32 = 145; 145 * 19 = 2755; 2755 + 14 = 2769; 2769 - 14 = 2755
- First answer: 'Answer: 3482' -> wrong
- Latent eligible: False ['target_not_single_token']
- J++ readout (top tokens per layer, last prompt position, before answering):
  - layer 4: '<u', '\\u', '<br', '<span', '<b', '\\n', '<strong', '\\s', '<a', '爲'
  - layer 8: '数学', 'Math', ' mathematical', ' Mathematical', 'ithmetic', 'Calcul', ' Euler', 'acers', 'geo', ' math'
  - layer 12: '’T', '’S', '1', ' Calculation', '2', '7', '’an', '’B', 'CT', '9'
  - layer 16: ' Calculator', ' Arithmetic', ' Calculation', 'Bitcoin', 'Calc', 'CT', 'RM', '推理', 'Logic', ' Logic'
  - layer 20: '计算', ' calculation', '@g', ' Calculation', '@end', ' calculations', ' Carn', ' Answer', 'Calc', '解答'
  - layer 24: '@c', '@m', '@g', '@d', '@s', '@end', '@a', '@if', '@include', '@test'
  - layer 28: ' Answer', ' answer', 'Answer', ' ANSW', ' Answers', '_answer', ' answers', '.answer', ' answering', '答案'
- Raw logit-lens readout (same activations):
  - layer 4: '1', 'd', 'C', ' C', '正', '一', 'A', 'eng', 'ly', 'x'
  - layer 8: '卓', 'math', '技', '横', '依法', '数学', ' math', 'inking', '粉', '晏'
  - layer 12: '昔', '牙', '完整的', '位', '法', '蜂', '先行', '绮', '计算', '程'
  - layer 16: '位数', '理', 'angen', '概', '埃尔', ' ans', ' jih', "'ve", '莱特', '尾'
  - layer 20: 'TTTT', '三八', ' radios', '熠', ' ans', 'OwnerId', 'ans', '</think>', 'uffle', 'engan'
  - layer 24: ' ans', ' Answer', 'Answer', 'ansible', ' Ans', 'ans', ' answer', '晗', 'engan', ' ANSW'
  - layer 28: ' Answer', 'Answer', ' answer', '_answer', '答案', ' ANSW', '.answer', '-answer', '答案是', 'answer'
- Condition `authentic_jpp`
  - Feedback prompt:
    Your previous answer was incorrect. Solve the problem again.
    Automatically extracted diagnostic tokens follow. They may be noisy or irrelevant.
    [{"layer": 4, "tokens": ["<u", "\\u", "<br", "<span", "<b", "\\n", "<strong", "\\s", "<a", "爲"]}, {"layer": 8, "tokens": ["数学", "Math", " mathematical", " Mathematical", "ithmetic", "Calcul", " Euler", "acers", "geo", " math"]}, {"layer": 12, "tokens": ["’T", "’S", "1", " Calculation", "2", "7", "’an", "’B", "CT", "9"]}, {"layer": 16, "tokens": [" Calculator", " Arithmetic", " Calculation", "Bitcoin", "Calc", "CT", "RM", "推理", "Logic", " Logic"]}, {"layer": 20, "tokens": ["计算", " calculation", "@g", " Calculation", "@end", " calculations", " Carn", " Answer", "Calc", "解答"]}, {"layer": 24, "tokens": ["@c", "@m", "@g", "@d", "@s", "@end", "@a", "@if", "@include", "@test"]}, {"layer": 28, "tokens": [" Answer", " answer", "Answer", " ANSW", " Answers", "_answer", " answers", ".answer", " answering", "答案"]}]
    Return only Answer: <integer>.
  - Second answer: 'Answer: 3482' -> wrong
- Condition `retry_only`
  - Feedback prompt:
    Review your previous answer. Keep it if correct; otherwise correct it.
    Return only Answer: <integer>.
  - Second answer: 'Answer: 3482' -> wrong
- Condition `shuffled_jpp` (donor dev-ec3c60dc9151be51)
  - Feedback prompt:
    Your previous answer was incorrect. Solve the problem again.
    Automatically extracted diagnostic tokens follow. They may be noisy or irrelevant.
    [{"layer": 4, "tokens": ["<u", "\\u", "<br", "<span", "<b", "\\n", "<strong", "\\s", "<a", "爲"]}, {"layer": 8, "tokens": ["数学", "Math", " mathematical", " Mathematical", "ithmetic", "Calcul", "acers", " Euler", " Adobe", "geo"]}, {"layer": 12, "tokens": ["’T", "’S", "1", "2", " Calculation", "’an", "7", "’B", "’n", "9"]}, {"layer": 16, "tokens": [" Calculator", "Bitcoin", " Calculation", " Arithmetic", "Calc", "RM", "CT", "IBM", "推理", "的计算"]}, {"layer": 20, "tokens": ["计算", " calculation", " Calculation", " calculations", "@g", " calculate", " Carn", "的计算", " Answer", "Calc"]}, {"layer": 24, "tokens": ["@c", "@m", "@g", "@s", "@d", "@a", "@end", "@if", "@include", "@test"]}, {"layer": 28, "tokens": [" Answer", " answer", "Answer", " ANSW", " Answers", "_answer", " answers", ".answer", " answering", "答案"]}]
    Return only Answer: <integer>.
  - Second answer: 'Answer: 3482' -> wrong
- Condition `error_only`
  - Feedback prompt:
    Your previous answer was incorrect. Solve the problem again.
    Return only Answer: <integer>.
  - Second answer: 'Answer: 3482' -> wrong

## Pre-registered protocol

# Frozen-model J++ recovery pilot

## Questions

1. On initially wrong answers, does J++ detect a known, unexpressed intermediate value more often than a raw logit lens or a readout from another problem?
2. Does authentic feedback improve second-attempt correction relative to both ordinary error feedback and shuffled feedback?
3. Within the detected-target subset, does removing or replacing that token reduce the benefit more than a matched sham edit?

These are separate hypotheses. Detection does not prove use; textual feedback effects do not establish that a neural feature caused the original answer; none independently establishes introspection.

## Population and measurement

The fixed sample is 200 development and 1,000 test **first attempts** by default, not 1,000 errors. Development and test use disjoint arithmetic operation templates. Each contains direct arithmetic and explicit variable assignments. Additional sequential operations control difficulty while preserving a known intermediate value. Problem counts, generator, and seed are saved before inference. There is no optional stopping on correction performance.

First generation is greedy, answer-only, with `enable_thinking=False`. Captures are the post-block residuals at the final prompt position during generation prefill. The default seven zero-based block indices are 4, 8, 12, 16, 20, 24, 28. Both readout methods receive the identical captures and vocabulary filter. Fixed top-10 entries per layer become feedback in the authentic arm without gold-guided selection or interpretation. Token scores are diagnostics, not calibrated probabilities that a concept is present.

A latent-information case is eligible only if the target has an exact individual vocabulary token, does not occur as a number in the question or visible response, and the response parses. Matching is on the whole decoded token after whitespace stripping: `143` does not match `43`. Unsupported multi-token targets are reported as ineligible rather than readout misses. Recall is a union over the fixed layers; MRR is truncated at the configured top-k. This is an operational detection measure, not proof of a complete correct internal calculation.

Formatting failures, truncations, and refusals are counted separately from parsed wrong answers. They are excluded from the primary conditional correction population; their counts remain in the overall first-attempt denominator. Second-attempt format failures count as unsuccessful corrections and, for initially correct cases, as harm.

## Feedback arms

All continuations preserve the exact original messages and first completion as conversation history. The generated prompts and token IDs are stored. There is no shared state between correction conditions and no condition sees another condition's answer.

| Arm | Initially wrong first attempt |
| --- | --- |
| retry_only | Neutral instruction to review and correct if necessary |
| error_only | Truthful indication that the answer was wrong |
| shuffled_jpp | Same error indication plus another problem's raw J++ readout |
| authentic_jpp | Same error indication plus the actual first attempt's raw J++ readout |

Donors are deterministically deranged within template, without reading grades, targets, gold answers, or readout contents. If a template has only one item, the donor is chosen from another item in that split. The full first-attempt bank, including format failures, supplies readouts. Accidental useful clues in shuffled notes remain in the data.

Authentic and shuffled notes use the same neutral framing, layer count, and readout slots. Serialization can still produce unequal token lengths; prompt lengths and generation costs are reported explicitly. All second attempts have the same maximum generation budget. This experiment isolates information-source effects; it does not yet establish superiority under equal total FLOPs against an optimized independent-sampling system.

Initially correct attempts receive truthful neutral review instructions in every arm. Their harm rates are an auxiliary analysis, separate from the verified-error population. No correct answer is falsely labeled wrong.

## Textual ablations

The subset is fixed by: parsed wrong answer, eligible target, target detected in the original J++ readout. All occurrences of the exact target token are edited.

- `remove_target`: replace target slots with `[removed]`.
- `replace_target`: replace target slots with an adjacent incorrect integer of the same digit length.
- `sham_remove`: replace the same number of non-target slots with `[removed]`, chosen deterministically.

The original authentic continuation is reused as the paired reference. Slot counts match, but serialized tokenizer lengths need not. Oracle use is explicitly recorded here and absent from primary authentic/shuffled payload construction. A sham cannot be constructed if no sufficient non-target slots exist; the runner fails visibly instead of silently changing the control. These are interventions on the feedback channel only.

## Locking and reporting

Development is complete before `freeze` opens test inference. The feasibility gate requires the configured minimum number of eligible errors and J++ target hit rates above both controls. It does not require a positive correction effect. A documented negative-readout study can override this feasibility gate. All protocol/source changes require a new run. Frozen development cannot receive new continuations.

The two primary behavioral contrasts are authentic minus shuffled and authentic minus error-only correction rates on paired, initially wrong attempts. Report exact McNemar p-values with Holm adjustment across these two comparisons, and percentile intervals from question-level paired resampling within templates. These intervals quantify uncertainty for this generator, conditional on the template mixture; they do not estimate variation across arbitrary unseen task families. No repeated continuations are treated as independent problems.

Report conditional contrasts for eligible cases with and without a detected intermediate. These strata are partly defined by the same signal being supplied as feedback, so a difference may simply mean that correct hints are more helpful. Ablation contrasts form a separate exploratory family with their own Holm adjustment.

Incomplete pairs are not imputed as failures or silently treated as complete. Reports show observed/expected coverage and are marked partial until first attempts and primary continuations are complete. All proposed ablations have a separate completion count. Zero-case rates and intervals are `null`, never fabricated zeros.

## Decision after the pilot

- No usable readout: study the measurement location, task difficulty, or decoder before a larger correction evaluation. A null readout is not evidence of absent latent knowledge.
- Useful readout but no correction benefit: investigate whether the feedback is too noisy or whether the solver can use it.
- Correction benefit that survives both controls and target/sham edits: extend to ordinary critiques, informationally matched external hints, additional models, and native activation interventions.

The old GSM8K project tests learning to use readouts through adapter training. This project tests immediate feedback to a frozen model. Their outputs and effect estimates must not be pooled.

