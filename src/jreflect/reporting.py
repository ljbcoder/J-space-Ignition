"""Portable report artifacts, with explicit synthetic/pilot labels."""

import csv
from pathlib import Path

from .artifacts import write_json
from .prompts import ARMS


def write_report(output, result, *, synthetic=False, split="test"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {**result, "synthetic": synthetic, "split": split}
    if synthetic:
        result["success"] = None
    write_json(output / "metrics.json", result)
    title = (
        "SYNTHETIC PIPELINE CHECK — NOT EXPERIMENT RESULTS"
        if synthetic
        else f"J-Reflect {split} evaluation"
    )
    lines = [
        f"# {title}",
        "",
        f"Items: {result['n']}; seeds: {', '.join(map(str, result['seeds']))}.",
        "",
        "| Seed | Arm | First accuracy | Second accuracy | Net change (pp) | W→R | R→W | Change rate | Invalid |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def fmt(value):
        return "—" if value is None else f"{100 * value:.2f}%"

    flat = []
    for seed, record in result["per_seed"].items():
        for arm, row in record["arms"].items():
            lines.append(
                f"| {seed} | {arm} | {fmt(row['first_accuracy'])} | {fmt(row['second_accuracy'])} | {row['net_accuracy_change_pp']:+.2f} | {fmt(row['wrong_to_right'])} | {fmt(row['right_to_wrong'])} | {fmt(row['change_rate'])} | {fmt(row['invalid_rate'])} |"
            )
            flat.append({"seed": seed, "arm": arm, **{k: v for k, v in row.items() if k != "ci95"}})
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines += [
        "",
        "| Seed | Readout vs. control | Difference (pp) | Paired CI (pp) | Exact p | Holm p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for seed, record in result["per_seed"].items():
        for control, row in record["comparisons"].items():
            lo, hi = row["ci95_pp"]
            lines.append(
                f"| {seed} | {control} | {row['difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {row['p']:.4g} | {row['p_holm']:.4g} |"
            )
    lines += [
        "",
        "Intervals resample items together across arms and seeds. Full intervals and paired contingency counts are in metrics.json.",
        "",
    ]
    if synthetic:
        lines.append(
            "All answers and readouts in this check are synthetic fixtures. No scientific conclusion is supported."
        )
    elif split != "test":
        lines.append("Development evaluation only; this is not a final test result.")
    elif result["pilot_only"]:
        lines.append(
            "Pilot only: fewer than three seeds; the across-seed success criterion is not established."
        )
    else:
        lines.append(
            "Prespecified success criterion: " + ("met." if result["success"] else "not met.")
        )
    lines += ["", "![Net accuracy changes](accuracy.png)", ""]
    (output / "report.md").write_text("\n".join(lines))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    means = [result["seed_mean"][arm]["net_accuracy_change_pp"] for arm in ARMS]
    bounds = [result["seed_mean"][arm]["ci95_pp"] for arm in ARMS]
    ax.bar(ARMS, means, color=["#24766f", "#8295a8", "#c19a64"])
    # Draw intervals independently; percentile CIs need not contain the point estimate.
    for i, (lo, hi) in enumerate(bounds):
        ax.vlines(i, lo, hi, color="black", linewidth=1.4)
        ax.hlines([lo, hi], i - 0.08, i + 0.08, color="black", linewidth=1.4)
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_ylabel("Net accuracy change (percentage points)")
    ax.set_title(
        "Synthetic pipeline check"
        if synthetic
        else f"{split}: mean across {len(result['seeds'])} seeds"
    )
    fig.tight_layout()
    fig.savefig(output / "accuracy.png", dpi=160)
    plt.close(fig)
    return str(output / "report.md")
