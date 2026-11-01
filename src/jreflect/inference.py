"""Second attempts always consume the same frozen-base first answers."""

from pathlib import Path

from .artifacts import RowCache, directory_hash, exclusive, fingerprint, load_cache
from .data import load_run
from .examples import build_examples
from .grading import grade
from .model import HFBackend
from .prompts import ARMS, first_prompt
from .sanity import checked_base


def predict(root, split, arm, seed, *, ablation=None):
    root = Path(root)
    run, config = load_run(root)
    if arm not in ARMS or seed not in config.training.seeds:
        raise ValueError("Unknown arm or seed")
    if split not in {"dev_tune", "test"}:
        raise ValueError("Adapter evaluation is restricted to dev_tune and the frozen test split")
    if split == "test":
        from .protocol import verify_protocol

        verify_protocol(root)
    meta, base = checked_base(root, split)
    examples = build_examples(base, config.seed, balance=False)["arms"][arm]
    adapter = root / "adapters" / str(seed) / arm / "adapter"
    if not adapter.is_dir():
        raise ValueError("Completed adapter checkpoint is missing")
    condition = f"{arm}__{ablation}" if ablation else arm
    path = root / "predictions" / split / str(seed) / condition
    identity = {
        "run": fingerprint(run),
        "base": fingerprint(base),
        "base_identity": meta,
        "adapter": directory_hash(adapter),
        "arm": arm,
        "seed": seed,
        "ablation": ablation,
    }
    with exclusive(path / ".lock"):
        cache = RowCache(path, identity)
        pending = [r for r in base if cache.get(r["id"]) is None]
        backend = HFBackend(config, adapter=str(adapter)) if pending else None
        by_source = {e["source_id"]: e for e in examples}
        if ablation == "readout_removed" and arm != "readout":
            raise ValueError("Readout-removal ablation applies to the Readout adapter")
        if ablation == "readout_removed":
            by_source = {
                e["source_id"]: e
                for e in build_examples(base, config.seed, balance=False)["arms"]["no_readout"]
            }
        for row in pending:
            prompt = (
                first_prompt(row["question"])
                if ablation == "tuned_first"
                else by_source[row["id"]]["prompt"]
            )
            generation = backend.generate(prompt)
            cache.put(
                {
                    "id": row["id"],
                    "generation": generation,
                    "grade": grade(
                        generation["text"], row["gold"], truncated=generation["truncated"]
                    ),
                }
            )
            print(f"{condition}/{seed}: saved {row['id']}", flush=True)
        cache.finish([r["id"] for r in base])
    return str(path)


def collect_predictions(root, split, base, seeds):
    result = {}
    for seed in seeds:
        for arm in ARMS:
            identity, rows = load_cache(Path(root) / "predictions" / split / str(seed) / arm)
            if (
                identity["base"] != fingerprint(base)
                or identity["arm"] != arm
                or identity["seed"] != seed
                or identity["ablation"] is not None
            ):
                raise ValueError("Prediction provenance mismatch")
            if identity["adapter"] != directory_hash(
                Path(root) / "adapters" / str(seed) / arm / "adapter"
            ):
                raise ValueError("Predictions came from a different adapter checkpoint")
            result[(seed, arm)] = rows
    return result
