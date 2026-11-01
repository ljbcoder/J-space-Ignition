"""Content-addressed stage manifests and crash-safe per-item caches."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hash(path):
    root = Path(path)
    return fingerprint(
        {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}
    )


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def provenance():
    versions = {}
    for package in [
        "jreflect",
        "torch",
        "transformers",
        "datasets",
        "peft",
        "numpy",
        "scipy",
        "scikit-learn",
        "jpp-lens",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {
        "python": platform.python_version(),
        "packages": versions,
        "git_commit": commit,
        "source_hash": fingerprint(source),
    }


@contextmanager
def exclusive(path):
    """OS releases the advisory lock on crash; no stale PID file recovery needed."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another process is writing {path.parent}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class RowCache:
    def __init__(self, path, identity):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.manifest = self.path / "manifest.json"
        expected = {"schema": 1, "identity": identity, "fingerprint": fingerprint(identity)}
        if self.manifest.exists():
            if read_json(self.manifest) != expected:
                raise ValueError(f"Incompatible cache: {self.path}; use a new output directory")
        else:
            write_json(self.manifest, expected)

    def item_path(self, item_id):
        return self.path / "rows" / f"{fingerprint(item_id)}.json"

    def get(self, item_id):
        path = self.item_path(item_id)
        if not path.exists():
            return None
        record = read_json(path)
        if record["id"] != item_id or fingerprint(record["row"]) != record["sha256"]:
            raise ValueError(f"Corrupt cache row: {path}")
        return record["row"]

    def put(self, row):
        old = self.get(row["id"])
        if old is not None and old != row:
            raise ValueError(f"Attempt to replace completed item {row['id']}")
        write_json(
            self.item_path(row["id"]), {"id": row["id"], "row": row, "sha256": fingerprint(row)}
        )

    def finish(self, ids):
        rows = self.require(ids)
        write_json(self.path / "complete.json", {"ids": ids, "rows_hash": fingerprint(rows)})
        return rows

    def require(self, ids):
        rows = [self.get(item_id) for item_id in ids]
        if any(row is None for row in rows):
            raise ValueError(f"Incomplete cache: {self.path}")
        return rows


def load_cache(path, *, complete=True):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    cache = RowCache(path, manifest["identity"])
    if not complete:
        return cache
    done = read_json(path / "complete.json")
    rows = cache.require(done["ids"])
    if fingerprint(rows) != done["rows_hash"]:
        raise ValueError("Completion checksum mismatch")
    return manifest["identity"], rows
