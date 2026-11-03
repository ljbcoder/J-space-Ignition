"""Combine a downloaded run into one Markdown file to paste or upload into another LLM.

    python3 cloud/make_bundle.py runs/modal-pilot            # writes runs/modal-pilot/LLM_BUNDLE.md

Standard library only. Includes the experiment description, run configuration, every
split's report and summary, auto-detected data facts, and a few full example transcripts
(question, gold answer, hidden intermediate, readouts, each feedback prompt, each retry).
Upload reports/<split>/trials.csv alongside it for per-trial analysis.
"""

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXAMPLES = 4


def payload(path):
    return json.loads(Path(path).read_text())["payload"]


def readout_lines(readouts):
    lines = []
    for entry in readouts:
        tokens = ", ".join(repr(t["text"]) for t in entry["tokens"])
        lines.append(f"  - layer {entry['layer']}: {tokens}")
    return lines


def facts(run, split):
    rows = [payload(p) for p in sorted((run / "first" / split).glob("*.json"))]
    if not rows:
        return [], []
    status = Counter(r["grade"]["status"] for r in rows)
    reasons = Counter(x for r in rows for x in r["ineligible_reasons"])
    eligible = sum(r["latent_eligible"] for r in rows)
    lines = [
        f"- First attempts saved: {len(rows)}; grading: {dict(status)}",
        f"- Latent-eligible attempts: {eligible}; ineligibility reasons: {dict(reasons)}",
    ]
    return lines, rows


def examples(run, split, rows):
    problems = {p["id"]: p for p in payload(run / "data" / f"{split}.json")}
    plan_path = run / "correction" / split / "plan.json"
    jobs = payload(plan_path)["jobs"] if plan_path.exists() else []
    out = []
    for row in rows[:EXAMPLES]:
        problem = problems[row["id"]]
        out += [
            f"\n#### Example {row['id']} ({row['template']})",
            f"- Question: {problem['question']}",
            f"- Gold answer: {problem['gold']}; hidden intermediate (target): {problem['target']}",
            f"- Oracle steps: {'; '.join(problem['oracle_steps'])}",
            f"- First answer: {row['generation']['text'].strip()!r} -> {row['grade']['status']}",
            f"- Latent eligible: {row['latent_eligible']} {row['ineligible_reasons'] or ''}",
            "- J++ readout (top tokens per layer, last prompt position, before answering):",
            *readout_lines(row["readouts"]["jpp"]),
            "- Raw logit-lens readout (same activations):",
            *readout_lines(row["readouts"]["logit"]),
        ]
        for job in [j for j in jobs if j["id"] == row["id"]]:
            path = run / "correction" / split / row["id"] / f"{job['condition']}.json"
            second = payload(path) if path.exists() else None
            answer = (
                f"{second['generation']['text'].strip()!r} -> {second['grade']['status']}"
                if second
                else "(not run)"
            )
            feedback = job["messages"][-1]["content"].replace("\n", "\n    ")
            out += [
                f"- Condition `{job['condition']}`"
                + (f" (donor {job['donor_id']})" if job.get("donor_id") else ""),
                f"  - Feedback prompt:\n    {feedback}",
                f"  - Second answer: {answer}",
            ]
    return out


def main(run):
    run = Path(run)
    manifest = payload(run / "run.json")
    config = manifest["config"]
    out = [
        "# J-Reflect Recovery: run results for analysis",
        "",
        "## What this experiment is",
        "",
        "Question: can decoded internal information help a frozen language model correct its own "
        "mistakes? A frozen model (no training) answers program-generated arithmetic and "
        "variable-tracking problems answer-only, with thinking disabled. Before it answers, its "
        "residual-stream activations at the last prompt token are decoded with a released J++ "
        "lens (and a raw logit lens as baseline) into top-k tokens per layer. Each problem has a "
        "known hidden intermediate value. The model then gets a second attempt under four "
        "independent conditions: retry_only (neutral review), error_only (told it was wrong), "
        "shuffled_jpp (wrong + another problem's J++ readout), authentic_jpp (wrong + its own "
        "J++ readout). Primary contrasts: authentic minus shuffled, and authentic minus "
        "error_only, on initially wrong answers. Ablations edit the hidden value out of the "
        "feedback text. A development split gates a held-out test split.",
        "",
        "Key definition: an attempt is 'latent eligible' only if the hidden intermediate is a "
        "single vocabulary token, absent from the question and the visible answer, and the "
        "answer parses. Detection = that exact token appears in the readout top-k at any layer.",
        "",
        "## Run configuration",
        "",
        f"- Model: {config['model']['name']} @ {config['model']['revision']}",
        f"- Lens: {config['lens']['repo']}/{config['lens']['filename']} @ "
        f"{config['lens']['revision']}; layers {config['lens']['layers']}; top-k "
        f"{config['lens']['top_k']}",
        f"- Data: {config['data']}; decoding greedy, thinking={config['model']['enable_thinking']}, "
        f"max_new_tokens={config['model']['max_new_tokens']}",
        f"- Mode: {manifest['mode']}; frozen: {(run / 'frozen.json').exists()}",
        "",
        "## Notes from the researcher",
        "",
        "(Add your own notes or questions here before sharing.)",
    ]
    for split in ["dev", "test"]:
        lines, rows = facts(run, split)
        if not rows:
            continue
        out += ["", f"## {split} split", "", "### Data facts (computed from saved records)", *lines]
        report = run / "reports" / split / "report.md"
        summary = run / "reports" / split / "summary.json"
        if report.exists():
            out += ["", "### Report", "", report.read_text().replace("\n# ", "\n#### ")]
        if summary.exists():
            out += ["", "### summary.json", "", "```json", summary.read_text().strip(), "```"]
        out += ["", f"### Example transcripts (first {EXAMPLES} problems)", *examples(run, split, rows)]
    protocol = PROJECT / "docs" / "protocol.md"
    if protocol.exists():
        out += ["", "## Pre-registered protocol", "", protocol.read_text().replace("\n# ", "\n### ")]
    target = run / "LLM_BUNDLE.md"
    target.write_text("\n".join(out) + "\n")
    print(f"Wrote {target} ({target.stat().st_size / 1024:.0f} KB)")
    for split in ["dev", "test"]:
        csv = run / "reports" / split / "trials.csv"
        if csv.exists():
            print(f"Also upload: {csv}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
