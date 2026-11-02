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
