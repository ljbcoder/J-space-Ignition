"""Dataset preparation never loads GSM8K test until the protocol is frozen."""

import random
from pathlib import Path

from .artifacts import fingerprint, read_json, write_json
from .config import Config
from .grading import gold_answer


def split_training(records, config):
    if len(records) != config.data.train_count:
        raise ValueError(f"Expected {config.data.train_count} train items, got {len(records)}")
    indices = list(range(len(records)))
    random.Random(config.seed).shuffle(indices)
    dev = indices[: config.data.dev_size]
    groups = {
        "dev_tune": dev[config.data.gate_size :],
        "dev_gate": dev[: config.data.gate_size],
        "train": indices[config.data.dev_size :],
    }
    return {
        split: [make_item(records[i], i, "train", split) for i in sorted(ids)]
        for split, ids in groups.items()
    }


def make_item(record, index, source_split, split):
    return {
        "id": f"gsm8k:{source_split}:{index:05d}",
        "split": split,
        "question": record["question"],
        "gold": gold_answer(record["answer"]),
    }


def prepare(config, root):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    root = Path(root)
    if (root / "run.json").exists():
        raise ValueError("Run already prepared; reuse it or choose a new run directory")
    api = HfApi()
    config = Config.from_dict(config.to_dict())
    config.model.revision = api.model_info(config.model.name, revision=config.model.revision).sha
    config.data.revision = api.dataset_info(config.data.name, revision=config.data.revision).sha
    # Eager load_dataset prepares every split before returning the requested one.
    # Streaming confines row reads to train and avoids materializing test in cache.
    records = list(
        load_dataset(
            config.data.name,
            config.data.subset,
            split="train",
            revision=config.data.revision,
            streaming=True,
        )
    )
    splits = split_training(records, config)
    for name, rows in splits.items():
        write_json(root / "data" / f"{name}.json", rows)
    # Reference reasoning stays separate from every inference/training input table.
    write_json(
        root / "data" / "references.json",
        {f"gsm8k:train:{i:05d}": row["answer"] for i, row in enumerate(records)},
    )
    write_json(
        root / "run.json",
        {
            "schema": 1,
            "config": config.to_dict(),
            "splits": {name: fingerprint(rows) for name, rows in splits.items()},
            "synthetic": False,
        },
    )


def load_run(root):
    run = read_json(Path(root) / "run.json")
    return run, Config.from_dict(run["config"])


def load_split(root, split):
    root = Path(root)
    run, _ = load_run(root)
    if split == "test":
        from .protocol import verify_protocol

        protocol = verify_protocol(root)
        manifest = read_json(root / "data" / "test_manifest.json")
        if manifest["protocol"] != fingerprint(protocol):
            raise ValueError("Test data belongs to a different frozen protocol")
        expected = manifest["hash"]
    else:
        if split not in run["splits"]:
            raise ValueError(f"Unknown split: {split}")
        expected = run["splits"][split]
    rows = read_json(root / "data" / f"{split}.json")
    if fingerprint(rows) != expected:
        raise ValueError(f"Dataset checksum mismatch for {split}")
    return rows


def prepare_test(root):
    from datasets import load_dataset

    from .protocol import verify_protocol

    root = Path(root)
    protocol = verify_protocol(root)
    _, config = load_run(root)
    if (root / "data" / "test_manifest.json").exists():
        return load_split(root, "test")
    records = list(
        load_dataset(
            config.data.name,
            config.data.subset,
            split="test",
            revision=config.data.revision,
            streaming=True,
        )
    )
    if len(records) != config.data.test_count:
        raise ValueError("Unexpected test item count")
    rows = [make_item(record, i, "test", "test") for i, record in enumerate(records)]
    write_json(root / "data" / "test.json", rows)
    write_json(
        root / "data" / "test_manifest.json",
        {"hash": fingerprint(rows), "protocol": fingerprint(protocol)},
    )
    return rows
