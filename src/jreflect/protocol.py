"""Freeze all analysis and model choices before opening the test split."""

from pathlib import Path

from .artifacts import directory_hash, file_hash, fingerprint, provenance, read_json, write_json
from .data import load_run
from .prompts import ARMS
from .sanity import require_gate


def freeze(root):
    root = Path(root)
    if (root / "protocol.json").exists():
        raise ValueError("Protocol is already frozen")
    run, config = load_run(root)
    gate = require_gate(root)
    lens = root / "lens" / "lens.pt"
    if file_hash(lens) != gate["lens_sha256"]:
        raise ValueError("Lens differs from the development gate")
    adapters = {}
    for seed in config.training.seeds:
        for arm in ARMS:
            path = root / "adapters" / str(seed) / arm
            identity = read_json(path / "identity.json")
            if identity["run"] != fingerprint(run) or identity["examples"] != fingerprint(
                read_json(root / "examples.json")
            ):
                raise ValueError("Adapter provenance differs from this run")
            if (
                identity["provenance"] != provenance()
                or identity["arm"] != arm
                or identity["seed"] != seed
            ):
                raise ValueError(
                    "All adapters must match their declared arm, seed, and current software"
                )
            if not read_json(path / "training.json")["complete"]:
                raise ValueError(f"Training incomplete for {seed}/{arm}")
            adapters[f"{seed}/{arm}"] = directory_hash(path / "adapter")
    protocol = {
        "schema": 1,
        "run": fingerprint(run),
        "config": config.to_dict(),
        "lens_sha256": file_hash(lens),
        "gate_hash": fingerprint(gate),
        "adapters": adapters,
        "examples_sha256": file_hash(root / "examples.json"),
        "provenance": provenance(),
        "statistics": {
            "test": "exact_two_sided_mcnemar",
            "correction": "Holm across two controls per seed",
            "bootstrap_unit": "item, paired across arms and seeds",
            "success": "both contrasts positive and adjusted p<0.05 in every seed; >=3 seeds",
        },
    }
    write_json(root / "protocol.json", {"protocol": protocol, "sha256": fingerprint(protocol)})
    return protocol


def verify_protocol(root):
    root = Path(root)
    envelope = read_json(root / "protocol.json")
    protocol = envelope["protocol"]
    run, _ = load_run(root)
    if fingerprint(protocol) != envelope["sha256"] or fingerprint(run) != protocol["run"]:
        raise ValueError("Frozen protocol or run was modified")
    if provenance() != protocol["provenance"]:
        raise ValueError("Software differs from the frozen protocol")
    if file_hash(root / "lens" / "lens.pt") != protocol["lens_sha256"]:
        raise ValueError("Frozen lens was modified")
    if (
        file_hash(root / "examples.json") != protocol["examples_sha256"]
        or fingerprint(read_json(root / "gate.json")) != protocol["gate_hash"]
    ):
        raise ValueError("Frozen development artifacts were modified")
    for name, expected in protocol["adapters"].items():
        path = root / "adapters" / name / "adapter"
        if not path.is_dir() or directory_hash(path) != expected:
            raise ValueError(f"Frozen adapter changed: {name}")
    return protocol
