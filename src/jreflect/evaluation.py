"""Paired outcome statistics, without treating seeds as independent items."""

from .grading import changed
from .prompts import ARMS


def holm(p_values):
    ordered = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for i, name in enumerate(ordered):
        running = max(running, (len(ordered) - i) * p_values[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def mcnemar(a, b):
    import numpy as np
    from scipy.stats import binomtest

    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Paired comparisons need identical item counts")
    wins, losses = int((a & ~b).sum()), int((~a & b).sum())
    return {
        "readout_only_correct": wins,
        "control_only_correct": losses,
        "p": float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0,
    }


def outcome_metrics(first, second, change, valid):
    import numpy as np

    first, second, change, valid = [
        np.asarray(x, dtype=bool) for x in (first, second, change, valid)
    ]
    return {
        "first_accuracy": float(first.mean()),
        "second_accuracy": float(second.mean()),
        "net_accuracy_change_pp": 100 * float(second.mean() - first.mean()),
        "wrong_to_right": float(second[~first].mean()) if (~first).any() else None,
        "right_to_wrong": float((~second[first]).mean()) if first.any() else None,
        "change_rate": float(change.mean()),
        "invalid_rate": float((~valid).mean()),
        "initially_wrong": int((~first).sum()),
        "initially_right": int(first.sum()),
    }


def interval(values, confidence):
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    alpha = (1 - confidence) / 2
    return [float(x) for x in np.quantile(values, [alpha, 1 - alpha])]


def evaluate_predictions(
    base, predictions, seeds, *, bootstrap_samples=10000, confidence=0.95, seed=42
):
    import numpy as np

    ids = [row["id"] for row in base]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Evaluation needs distinct, nonempty base IDs")
    first = np.array([r["first_grade"]["correct"] for r in base])
    records, per_seed, primary_pass = {}, {}, []
    for training_seed in seeds:
        per_arm = {}
        for arm in ARMS:
            rows = predictions[(training_seed, arm)]
            lookup = {r["id"]: r for r in rows}
            if len(rows) != len(lookup) or set(lookup) != set(ids):
                raise ValueError(
                    f"Missing, extra, or duplicate predictions for {training_seed}/{arm}"
                )
            ordered = [lookup[key] for key in ids]
            second = np.array([r["grade"]["correct"] for r in ordered])
            valid = np.array([r["grade"]["valid"] for r in ordered])
            change = np.array(
                [
                    changed(a["first"]["text"], b["generation"]["text"])
                    for a, b in zip(base, ordered, strict=True)
                ]
            )
            records[(training_seed, arm)] = (second, change, valid)
            per_arm[arm] = outcome_metrics(first, second, change, valid)
        comparisons = {}
        for control in ["no_readout", "shuffled"]:
            a, b = records[(training_seed, "readout")][0], records[(training_seed, control)][0]
            comparisons[control] = {
                **mcnemar(a, b),
                "difference_pp": 100 * float(a.mean() - b.mean()),
            }
        adjusted = holm({name: row["p"] for name, row in comparisons.items()})
        for name in comparisons:
            comparisons[name]["p_holm"] = adjusted[name]
        passed = all(
            row["difference_pp"] > 0 and row["p_holm"] < 0.05 for row in comparisons.values()
        )
        primary_pass.append(passed)
        per_seed[str(training_seed)] = {
            "arms": per_arm,
            "comparisons": comparisons,
            "passes": passed,
        }
    draws = {key: [] for key in records}
    differences = {(s, arm): [] for s in seeds for arm in ["no_readout", "shuffled"]}
    mean_draws = {arm: [] for arm in ARMS}
    mean_differences = {arm: [] for arm in ["no_readout", "shuffled"]}
    rng = np.random.default_rng(seed)
    # Draw the same IDs for all arms and seeds; retain seed predictions as a cluster.
    for _ in range(bootstrap_samples):
        ix = rng.integers(0, len(ids), len(ids))
        for key, (second, change, valid) in records.items():
            draws[key].append(outcome_metrics(first[ix], second[ix], change[ix], valid[ix]))
        for s, control in differences:
            differences[(s, control)].append(
                100
                * float(records[(s, "readout")][0][ix].mean() - records[(s, control)][0][ix].mean())
            )
        for arm in ARMS:
            mean_draws[arm].append(
                100
                * float(
                    np.mean([records[(s, arm)][0][ix].mean() - first[ix].mean() for s in seeds])
                )
            )
        for control in mean_differences:
            mean_differences[control].append(
                float(np.mean([differences[(s, control)][-1] for s in seeds]))
            )
    for s in seeds:
        for arm in ARMS:
            metrics = per_seed[str(s)]["arms"][arm]
            metrics["ci95"] = {
                name: interval(
                    [d[name] if d[name] is not None else float("nan") for d in draws[(s, arm)]],
                    confidence,
                )
                for name in metrics
                if name not in {"initially_right", "initially_wrong"}
            }
        for control in ["no_readout", "shuffled"]:
            per_seed[str(s)]["comparisons"][control]["ci95_pp"] = interval(
                differences[(s, control)], confidence
            )
    mean = {
        arm: {
            "net_accuracy_change_pp": float(
                np.mean([per_seed[str(s)]["arms"][arm]["net_accuracy_change_pp"] for s in seeds])
            ),
            "ci95_pp": interval(mean_draws[arm], confidence),
        }
        for arm in ARMS
    }
    return {
        "n": len(ids),
        "seeds": seeds,
        "confidence": confidence,
        "bootstrap_samples": bootstrap_samples,
        "per_seed": per_seed,
        "seed_mean": mean,
        "seed_mean_comparisons": {
            arm: {
                "difference_pp": float(
                    np.mean([per_seed[str(s)]["comparisons"][arm]["difference_pp"] for s in seeds])
                ),
                "ci95_pp": interval(mean_differences[arm], confidence),
            }
            for arm in mean_differences
        },
        "success": len(seeds) >= 3 and all(primary_pass),
        "pilot_only": len(seeds) < 3,
    }
