"""Atomic, checksummed caches and exclusive stage execution."""

import hashlib
import importlib.metadata
import json
import os
import platform
from contextlib import contextmanager
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_digest():
    return digest({p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def environment():
    versions = {"python": platform.python_version()}
    for package in ["torch", "transformers", "huggingface-hub", "tokenizers", "accelerate"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(
            {"sha256": digest(payload), "payload": payload}, stream, indent=2, allow_nan=False
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read(path):
    record = json.loads(Path(path).read_text())
    if digest(record["payload"]) != record["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")
    return record["payload"]


@contextmanager
def exclusive(root):
    import fcntl

    Path(root).mkdir(parents=True, exist_ok=True)
    with (Path(root) / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another command is writing to this run") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
