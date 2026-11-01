"""Stage orchestration, with lazy model loading and validated cache identities."""

import time
from pathlib import Path

from .artifacts import RowCache, exclusive, file_hash, fingerprint, provenance, write_json
from .data import load_run, load_split
from .grading import grade
from .prompts import first_prompt


def generate_split(root, split, *, lens_path=None, limit=None, backend=None):
    root = Path(root)
    run, config = load_run(root)
    rows = load_split(root, split)
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    if split == "test":
        from .protocol import verify_protocol

        protocol = verify_protocol(root)
        if limit is not None:
            raise ValueError("Final test cache must contain the complete test split")
        if lens_path is None or file_hash(lens_path) != protocol["lens_sha256"]:
            raise ValueError("Test readouts must use the frozen J++ lens")
    method = "jpp" if lens_path else "logit"
    path = root / "base" / method / split
    identity = {
        "run": fingerprint(run),
        "split": split,
        "data": fingerprint(rows),
        "lens": file_hash(lens_path) if lens_path else None,
        "method": method,
        "synthetic": run["synthetic"],
        "provenance": provenance(),
    }
    started = time.monotonic()
    with exclusive(path / ".lock"):
        cache = RowCache(path, identity)
        selected = rows[:limit] if limit else rows
        pending = [r for r in selected if cache.get(r["id"]) is None]
        if pending and backend is None:
            from .model import HFBackend

            backend = HFBackend(config, lens_path)
        for index, row in enumerate(pending, 1):
            first = backend.generate(first_prompt(row["question"]))
            readouts, concepts = backend.read(row["question"], first)
            cache.put(
                {
                    **row,
                    "first": first,
                    "first_grade": grade(first["text"], row["gold"], truncated=first["truncated"]),
                    "readouts": readouts,
                    "concepts": concepts,
                }
            )
            print(f"{split}: saved {index}/{len(pending)} pending items", flush=True)
        all_ids = [r["id"] for r in rows]
        complete = all(cache.get(key) is not None for key in all_ids)
        if complete:
            cache.finish(all_ids)
        write_json(
            path / "last_execution.json",
            {
                "new_items": len(pending),
                "elapsed_seconds": time.monotonic() - started,
                "complete": complete,
            },
        )
    return {"path": str(path), "new_items": len(pending), "complete": complete}
