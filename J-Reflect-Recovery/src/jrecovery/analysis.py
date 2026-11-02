"""Paired item statistics; no scientific claims from the offline demonstration."""

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .data import donor_map
from .feedback import ABLATIONS, CONDITIONS, hits
from .pipeline import first_rows, load, predictions, problems
from .storage import exclusive


def mean(values):
    return sum(values) / len(values) if values else None


def paired_statistics(items, config):
    """items: (template, treatment_correct, control_correct); one pair per question."""
    n = len(items)
    if not n:
        return {"n": 0, "difference": None, "ci": [None, None], "wins": 0, "losses": 0, "p": None}
    differences = [int(a) - int(b) for _, a, b in items]
    wins, losses = differences.count(1), differences.count(-1)
    discordant = wins + losses
    p = (
        min(
            1.0,
            2
            * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
            / (2**discordant),
        )
        if discordant
        else 1.0
    )
    strata = defaultdict(list)
    for (template, _, _), difference in zip(items, differences, strict=True):
        strata[template].append(difference)
    rng = random.Random(config.seed)
    samples = sorted(
        sum(sum(rng.choices(values, k=len(values))) for values in strata.values()) / n
        for _ in range(config.analysis.bootstrap_samples)
    )
    alpha = (1 - config.analysis.confidence) / 2
    return {
        "n": n,
        "difference": mean(differences),
        "ci": [
            samples[int(alpha * (len(samples) - 1))],
            samples[int((1 - alpha) * (len(samples) - 1))],
        ],
        "wins": wins,
        "losses": losses,
        "p": p,
    }


def holm(comparisons):
    ordered = sorted((row["p"], name) for name, row in comparisons.items() if row["p"] is not None)
    prior = 0.0
    for rank, (p, name) in enumerate(ordered):
        prior = max(prior, min(1.0, p * (len(ordered) - rank)))
        comparisons[name]["holm_p"] = prior


def summarize(root, split):
    root = Path(root)
    manifest, config = load(root)
    items = problems(root, split)
    first = first_rows(root, split)
    oracle = {p["id"]: p for p in items}
    by_id = {r["id"]: r for r in first}
    donors = donor_map(first, config.seed) if len(first) >= 2 else {}
    first_complete = len(first) == len(items)
    correction = predictions(root, split, "correction")
    ablation = predictions(root, split, "ablation")
    indexed = {(r["id"], r["condition"]): r for r in correction + ablation}

    def detected(row, method="jpp"):
        return hits(row["readouts"], oracle[row["id"]]["target"], method)

    detection = {}
    for name, subset in {
        "all_eligible": [r for r in first if r["latent_eligible"]],
        "initially_wrong": [
            r for r in first if r["latent_eligible"] and r["grade"]["status"] == "wrong"
        ],
        "initially_correct": [r for r in first if r["latent_eligible"] and r["grade"]["correct"]],
    }.items():
        shuffled = [
            (hits(by_id[donors[r["id"]]]["readouts"], oracle[r["id"]]["target"]) is not None)
            for r in subset
            if r["id"] in donors
        ]
        detection[name] = {
            "n": len(subset),
            "jpp_hit_rate": mean([detected(r) is not None for r in subset]),
            "logit_hit_rate": mean([detected(r, "logit") is not None for r in subset]),
            "shuffled_hit_rate": mean(shuffled),
            "jpp_mrr": mean([1 / rank if (rank := detected(r)) else 0 for r in subset]),
            "jpp_vs_logit": paired_statistics(
                [
                    (r["template"], detected(r) is not None, detected(r, "logit") is not None)
                    for r in subset
                ],
                config,
            ),
            "jpp_vs_shuffled": paired_statistics(
                [
                    (
                        r["template"],
                        detected(r) is not None,
                        hits(by_id[donors[r["id"]]]["readouts"], oracle[r["id"]]["target"])
                        is not None,
                    )
                    for r in subset
                    if r["id"] in donors
                ],
                config,
            ),
        }

    arms = {}
    wrong = [r for r in first if r["grade"]["status"] == "wrong"]
    correct = [r for r in first if r["grade"]["correct"]]
    for condition in CONDITIONS:
        observed_wrong = [
            indexed[(r["id"], condition)] for r in wrong if (r["id"], condition) in indexed
        ]
        observed_correct = [
            indexed[(r["id"], condition)] for r in correct if (r["id"], condition) in indexed
        ]
        all_observed = observed_wrong + observed_correct
        arms[condition] = {
            "wrong_attempts_observed": len(observed_wrong),
            "wrong_attempts_expected": len(wrong),
            "wrong_to_right": sum(r["grade"]["correct"] for r in observed_wrong),
            "correction_rate": mean([r["grade"]["correct"] for r in observed_wrong]),
            "correct_attempts_observed": len(observed_correct),
            "correct_to_wrong": sum(not r["grade"]["correct"] for r in observed_correct),
            "harm_rate": mean([not r["grade"]["correct"] for r in observed_correct]),
            "second_format_failures": sum(
                r["grade"]["status"] == "format_failure" for r in all_observed
            ),
            "answer_change_rate": mean(
                [r["grade"]["answer"] != by_id[r["id"]]["grade"]["answer"] for r in all_observed]
            ),
            "prompt_tokens": sum(r["generation"]["prompt_tokens"] for r in all_observed),
            "generated_tokens": sum(r["generation"]["generated_tokens"] for r in all_observed),
            "seconds": sum(r["generation"]["seconds"] for r in all_observed),
        }

    def compare(treatment, control, subset):
        pairs = []
        for row in subset:
            a, b = indexed.get((row["id"], treatment)), indexed.get((row["id"], control))
            if a is not None and b is not None:
                pairs.append((row["template"], a["grade"]["correct"], b["grade"]["correct"]))
        return paired_statistics(pairs, config)

    comparisons = {
        f"{a}_minus_{b}": compare(a, b, wrong) for a, b in manifest["primary_comparisons"]
    }
    holm(comparisons)
    latent = {}
    for present in [True, False]:
        subset = [r for r in wrong if r["latent_eligible"] and bool(detected(r)) == present]
        latent["detected" if present else "not_detected"] = compare(
            "authentic_jpp", "shuffled_jpp", subset
        )
    eligible_ablation = [r for r in wrong if r["latent_eligible"] and detected(r)]
    ablations = {
        f"authentic_minus_{arm}": compare("authentic_jpp", arm, eligible_ablation)
        for arm in ABLATIONS
    }
    holm(ablations)
    all_generations = [r["generation"] for r in first + correction + ablation]
    expected_corrections = (len(wrong) + len(correct)) * len(CONDITIONS)
    return {
        "mode": manifest["mode"],
        "split": split,
        "status": "complete"
        if first_complete and len(correction) == expected_corrections
        else "partial",
        "first_attempts": len(first),
        "first_attempts_expected": len(items),
        "initial_correct": len(correct),
        "initial_wrong": len(wrong),
        "first_format_failures": len(first) - len(correct) - len(wrong),
        "first_accuracy": mean([r["grade"]["correct"] for r in first]),
        "corrections_observed": len(correction),
        "corrections_expected": expected_corrections,
        "ablations_observed": len(ablation),
        "ablations_expected": len(eligible_ablation) * len(ABLATIONS),
        "detection": detection,
        "conditions": arms,
        "primary_comparisons": comparisons,
        "latent_strata": latent,
        "ablations": ablations,
        "cost": {
            "generation_calls": len(all_generations),
            "prompt_tokens": sum(g["prompt_tokens"] for g in all_generations),
            "generated_tokens": sum(g["generated_tokens"] for g in all_generations),
            "generation_seconds": sum(g["seconds"] for g in all_generations),
            "readout_seconds": sum(r["readout_seconds"] for r in first),
        },
        "limitations": [
            "DEMO output is simulated and cannot be used as research evidence."
            if manifest["mode"] != "research"
            else "Pilot estimates; no claim of introspection or native-feature causality.",
            "Detection is best rank over the fixed layer set; identical search exposure in both lens baselines.",
            "Shuffled source is chosen without labels. With partial first collection, the donor bank is provisional.",
            "Intervals bootstrap independent questions within templates; they do not establish generalization to new task families.",
            "Feedback has matched readout slots, not exactly matched serialized prompt-token lengths; actual costs are recorded.",
            "Correct first attempts receive neutral review instructions, never a false error signal.",
            "Oracle-guided ablations are diagnostic only. Textual intervention does not establish neural causality.",
        ],
    }


def percentage(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def report(root, split):
    root = Path(root)
    with exclusive(root):
        result = summarize(root, split)
        output = root / "reports" / split
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        lines = [
            f"# J-Reflect recovery: {split}",
            "",
            f"**Mode: {result['mode']} — {result['status']}**",
            "",
        ]
        if result["mode"] != "research":
            lines += [
                "**SIMULATED WORKFLOW CHECK — NOT MODEL PERFORMANCE OR EXPERIMENTAL EVIDENCE.**",
                "",
            ]
        lines += [
            f"First attempts: {result['first_attempts']}/{result['first_attempts_expected']}. "
            f"Correct: {result['initial_correct']}; wrong: {result['initial_wrong']}; "
            f"format failures: {result['first_format_failures']}.",
            "",
            "## Readout detection",
            "",
            "Eligible initially wrong attempts only. Recall is at the configured top-k, across the fixed layers.",
            "",
            "| N | J++ hit rate | Logit hit rate | Shuffled hit rate |",
            "| --- | --- | --- | --- |",
        ]
        det = result["detection"]["initially_wrong"]
        lines += [
            f"| {det['n']} | {percentage(det['jpp_hit_rate'])} | {percentage(det['logit_hit_rate'])} | {percentage(det['shuffled_hit_rate'])} |",
            "",
            "## Correction and harm",
            "",
            "| Condition | Wrong cases observed | Correction rate | Correct cases audited | Harm rate |",
            "| --- | --- | --- | --- | --- |",
        ]
        for arm, values in result["conditions"].items():
            lines.append(
                f"| {arm} | {values['wrong_attempts_observed']}/{values['wrong_attempts_expected']} | {percentage(values['correction_rate'])} | {values['correct_attempts_observed']} | {percentage(values['harm_rate'])} |"
            )
        lines += [
            "",
            "## Paired contrasts",
            "",
            "Differences and intervals are percentage points. Incomplete pairs are excluded and N is shown.",
            "",
            "| Comparison | Pairs | Difference | Confidence interval | Holm p |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, values in result["primary_comparisons"].items():
            lines.append(
                f"| {name} | {values['n']} | {percentage(values['difference'])} | {percentage(values['ci'][0])} to {percentage(values['ci'][1])} | {values.get('holm_p', 'n/a')} |"
            )
        lines += ["", "## Interpretation and scope", ""] + [
            f"- {line}" for line in result["limitations"]
        ]
        lines += [
            "",
            f"Saved continuation records: {result['corrections_observed']}/{result['corrections_expected']}; ablations: {result['ablations_observed']}/{result['ablations_expected']}.",
            "",
            "Full detection comparisons, latent strata, ablations, and compute accounting are in `summary.json`.",
            "",
        ]
        (output / "report.md").write_text("\n".join(lines))
        first = {r["id"]: r for r in first_rows(root, split)}
        keys = {p["id"]: p for p in problems(root, split)}
        fields = [
            "id",
            "template",
            "condition",
            "first_status",
            "second_status",
            "latent_eligible",
            "target_detected",
            "prompt_tokens",
            "generated_tokens",
            "seconds",
            "mode",
        ]
        with (output / "trials.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in predictions(root, split, "correction") + predictions(
                root, split, "ablation"
            ):
                base = first[row["id"]]
                writer.writerow(
                    {
                        "id": row["id"],
                        "template": base["template"],
                        "condition": row["condition"],
                        "first_status": base["grade"]["status"],
                        "second_status": row["grade"]["status"],
                        "latent_eligible": base["latent_eligible"],
                        "target_detected": hits(base["readouts"], keys[row["id"]]["target"])
                        is not None,
                        **{
                            k: row["generation"][k]
                            for k in ["prompt_tokens", "generated_tokens", "seconds"]
                        },
                        "mode": result["mode"],
                    }
                )
    return str(output / "report.md")
