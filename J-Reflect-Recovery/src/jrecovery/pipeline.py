"""Preparation, resumable paired trials, and a frozen held-out protocol."""

import random
from datetime import UTC, datetime
from pathlib import Path

from .config import Config
from .data import donor_map, generate, grade, numbers
from .feedback import ABLATIONS, CONDITIONS, ablation_messages, correction_messages, hits
from .storage import digest, environment, exclusive, file_digest, read, source_digest, write


def prepare(config, root, *, demo=False):
    root = Path(root)
    config = Config.from_dict(config.to_dict())
    with exclusive(root):
        if (root / "run.json").exists():
            raise ValueError("Run already exists; resume it or choose a new run folder")
        if not demo:
            from huggingface_hub import HfApi

            api = HfApi()
            config.model.revision = api.model_info(
                config.model.name, revision=config.model.revision
            ).sha
            info = api.model_info(config.lens.repo, revision=config.lens.revision)
            config.lens.revision = info.sha
            if config.lens.filename not in {f.rfilename for f in info.siblings}:
                raise ValueError("Configured lens file is absent from the resolved Hub revision")
        datasets = generate(config)
        for split, rows in datasets.items():
            write(root / "data" / f"{split}.json", rows)
        manifest = {
            "schema": 1,
            "mode": "DEMO_NOT_RESEARCH" if demo else "research",
            "created": datetime.now(UTC).isoformat(),
            "config": config.to_dict(),
            "source_sha256": source_digest(),
            "environment": environment(),
            "datasets": {split: digest(rows) for split, rows in datasets.items()},
            "conditions": list(CONDITIONS),
            "ablations": list(ABLATIONS),
            "primary_comparisons": [
                ["authentic_jpp", "shuffled_jpp"],
                ["authentic_jpp", "error_only"],
            ],
        }
        write(root / "run.json", manifest)
    return manifest


def load(root):
    manifest = read(Path(root) / "run.json")
    if manifest["source_sha256"] != source_digest():
        raise ValueError(
            "Source code changed since prepare; create a new run rather than mix protocols"
        )
    return manifest, Config.from_dict(manifest["config"])


def problems(root, split):
    manifest, _ = load(root)
    if split not in {"dev", "test"}:
        raise ValueError("Unknown split")
    rows = read(Path(root) / "data" / f"{split}.json")
    if digest(rows) != manifest["datasets"][split]:
        raise ValueError("Dataset changed after preparation")
    return rows


def first_rows(root, split, *, complete=False):
    manifest, _ = load(root)
    result = []
    items = problems(root, split)
    for item in items:
        path = Path(root) / "first" / split / f"{item['id']}.json"
        if not path.exists():
            if complete:
                raise ValueError(f"Collect all {split} first attempts before continuing")
            continue
        row = read(path)
        if row["run_sha256"] != digest(manifest) or row["problem_sha256"] != digest(item):
            raise ValueError("First-attempt provenance mismatch")
        if row["grade"] != grade(row["generation"], item["gold"]):
            raise ValueError("First-attempt grade mismatch")
        if row.get("activation_sha256"):
            activation_path = Path(root) / "activations" / split / f"{item['id']}.pt"
            if file_digest(activation_path) != row["activation_sha256"]:
                raise ValueError("Saved activation checksum mismatch")
        result.append(row)
    return result


def check_access(root, split):
    if split == "test":
        path = Path(root) / "frozen.json"
        if not path.exists():
            raise ValueError("Held-out test is closed; complete development and run freeze first")
        frozen = read(path)
        manifest, _ = load(root)
        if frozen["run_sha256"] != digest(manifest):
            raise ValueError("Frozen protocol no longer matches this run")
        if frozen["dev_first_sha256"] != digest(first_rows(root, "dev", complete=True)):
            raise ValueError("Development first attempts changed after freezing")
        if frozen["dev_corrections_sha256"] != digest(predictions(root, "dev", "correction")):
            raise ValueError("Development corrections changed after freezing")


def get_backend(root, manifest, config, supplied=None):
    if manifest["environment"] != environment():
        raise ValueError(
            "Python or model-package versions changed; use the original environment or a new run"
        )
    if supplied is None:
        if manifest["mode"] != "research":
            from .demo import DemoBackend

            supplied = DemoBackend(config, problems(root, "dev") + problems(root, "test"))
        else:
            from .backend import HFBackend

            supplied = HFBackend(config)
    is_demo = supplied.identity["kind"] == "demo"
    if is_demo != (manifest["mode"] != "research"):
        raise ValueError("Cannot mix demonstration and real model outputs")
    path = Path(root) / "backend.json"
    if path.exists() and read(path) != supplied.identity:
        raise ValueError("Model, lens, device, or execution backend changed during this run")
    write(path, supplied.identity)
    return supplied


def collect(root, split, limit=None, backend=None):
    root = Path(root)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    with exclusive(root):
        manifest, config = load(root)
        check_access(root, split)
        done = {row["id"] for row in first_rows(root, split)}
        pending = [p for p in problems(root, split) if p["id"] not in done]
        if limit is not None:
            pending = pending[:limit]
        if not pending:
            return {"saved": 0, "total": len(done)}
        if split == "dev" and (root / "frozen.json").exists():
            raise ValueError("Development data is frozen")
        backend = get_backend(root, manifest, config, backend)
        for item in pending:
            first, activations = backend.first(item["question"])
            grading = grade(first["generation"], item["gold"])
            reasons = []
            if not backend.supports(item["target"]):
                reasons.append("target_not_single_token")
            if item["target"] in numbers(item["question"]):
                reasons.append("target_in_prompt")
            if item["target"] in numbers(first["generation"]["text"]):
                reasons.append("target_in_visible_output")
            if grading["status"] == "format_failure":
                reasons.append("format_failure")
            activation_hash = None
            if config.lens.save_activations and manifest["mode"] == "research":
                activation_hash = backend.save_activations(
                    root / "activations" / split / f"{item['id']}.pt",
                    activations,
                    first["generation"]["prompt_ids"],
                )
            row = {
                "id": item["id"],
                "template": item["template"],
                "mode": manifest["mode"],
                "run_sha256": digest(manifest),
                "problem_sha256": digest(item),
                "backend_sha256": digest(backend.identity),
                "activation_sha256": activation_hash,
                **first,
                "grade": grading,
                "latent_eligible": not reasons,
                "ineligible_reasons": reasons,
            }
            write(root / "first" / split / f"{item['id']}.json", row)
            print(f"{split}: first attempt saved {item['id']} ({grading['status']})", flush=True)
    return {"saved": len(pending), "total": len(done) + len(pending)}


def plan(root, split, stage):
    root = Path(root)
    _, config = load(root)
    rows = first_rows(root, split, complete=True)
    by_id = {row["id"]: row for row in rows}
    keys = {p["id"]: p for p in problems(root, split)}
    donors = donor_map(rows, config.seed)
    jobs = []
    if stage not in {"correction", "ablation"}:
        raise ValueError("Unknown stage")
    for first in rows:
        if first["grade"]["status"] == "format_failure":
            continue
        if stage == "ablation" and (
            first["grade"]["correct"]
            or not first["latent_eligible"]
            or not hits(first["readouts"], keys[first["id"]]["target"])
        ):
            continue
        conditions = list(CONDITIONS if stage == "correction" else ABLATIONS)
        seed = int(digest([config.seed, first["id"], stage])[:12], 16)
        random.Random(seed).shuffle(conditions)
        for condition in conditions:
            edit = None
            if stage == "correction":
                messages = correction_messages(first, condition, by_id[donors[first["id"]]])
            else:
                messages, edit = ablation_messages(
                    first, keys[first["id"]]["target"], condition, seed
                )
            jobs.append(
                {
                    "id": first["id"],
                    "condition": condition,
                    "messages": messages,
                    "first_sha256": digest(first),
                    "donor_id": donors[first["id"]] if condition == "shuffled_jpp" else None,
                    "donor_sha256": digest(by_id[donors[first["id"]]])
                    if condition == "shuffled_jpp"
                    else None,
                    "oracle_assisted": stage == "ablation",
                    "edit": edit,
                }
            )
    identity = {"first_sha256": digest(rows), "stage": stage, "jobs": jobs}
    path = root / stage / split / "plan.json"
    if path.exists() and read(path) != identity:
        raise ValueError("Feedback plan changed; refusing to reuse cached predictions")
    write(path, identity)
    return identity


def predictions(root, split, stage):
    path = Path(root) / stage / split / "plan.json"
    if not path.exists():
        return []
    saved_plan = read(path)
    result = []
    for job in saved_plan["jobs"]:
        path = Path(root) / stage / split / job["id"] / f"{job['condition']}.json"
        if path.exists():
            row = read(path)
            if row["job_sha256"] != digest(job):
                raise ValueError("Prediction no longer matches its frozen prompt")
            result.append(row)
    return result


def correct(root, split, *, stage="correction", limit=None, backend=None):
    root = Path(root)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    with exclusive(root):
        manifest, config = load(root)
        check_access(root, split)
        if stage == "ablation":
            primary = plan(root, split, "correction")
            if len(predictions(root, split, "correction")) != len(primary["jobs"]):
                raise ValueError("Complete primary correction before running ablations")
        saved_plan = plan(root, split, stage)
        done = {(r["id"], r["condition"]) for r in predictions(root, split, stage)}
        pending = [j for j in saved_plan["jobs"] if (j["id"], j["condition"]) not in done]
        if limit is not None:
            pending = pending[:limit]
        if not pending:
            return {"saved": 0, "total": len(done)}
        if split == "dev" and (root / "frozen.json").exists():
            raise ValueError("Development continuations are frozen; use a new development run")
        backend = get_backend(root, manifest, config, backend)
        gold = {p["id"]: p["gold"] for p in problems(root, split)}
        for job in pending:
            generation = backend.complete(job["messages"])
            row = {
                "id": job["id"],
                "condition": job["condition"],
                "mode": manifest["mode"],
                "job_sha256": digest(job),
                "generation": generation,
                "grade": grade(generation, gold[job["id"]]),
            }
            write(root / stage / split / job["id"] / f"{job['condition']}.json", row)
            print(f"{split}: {job['condition']} saved {job['id']}", flush=True)
    return {"saved": len(pending), "total": len(done) + len(pending)}


def freeze(root, allow_weak_readout=None):
    from .analysis import summarize

    root = Path(root)
    with exclusive(root):
        manifest, config = load(root)
        first = first_rows(root, "dev", complete=True)
        primary = plan(root, "dev", "correction")
        corrected = predictions(root, "dev", "correction")
        if len(corrected) != len(primary["jobs"]):
            raise ValueError("Complete all development correction conditions before freeze")
        summary = summarize(root, "dev")
        detection = summary["detection"]["initially_wrong"]
        gate = (
            detection["n"] >= config.analysis.minimum_dev_errors
            and detection["jpp_hit_rate"] > detection["shuffled_hit_rate"]
            and detection["jpp_hit_rate"] > detection["logit_hit_rate"]
        )
        if not gate and not (allow_weak_readout and allow_weak_readout.strip()):
            raise ValueError(
                "Development readout gate failed. Improve a NEW development run, or freeze with --allow-weak-readout 'negative-result study rationale'."
            )
        frozen = {
            "run_sha256": digest(manifest),
            "dev_first_sha256": digest(first),
            "dev_corrections_sha256": digest(corrected),
            "readout_gate_passed": gate,
            "weak_readout_rationale": allow_weak_readout,
            "detection": detection,
        }
        path = root / "frozen.json"
        if path.exists() and read(path) != frozen:
            raise ValueError("This protocol has already been frozen")
        write(path, frozen)
    return frozen
