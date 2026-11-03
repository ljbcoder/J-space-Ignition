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
