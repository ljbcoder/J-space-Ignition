import pytest

from jreflect.artifacts import file_hash, fingerprint, provenance, write_json
from jreflect.config import Config
from jreflect.protocol import freeze, verify_protocol
from jreflect.sanity import probe_gate, require_gate


def feature_rows(start, count):
    rows = []
    for i in range(start, start + count):
        label = i % 2 == 0
        rows.append(
            {
                "id": str(i),
                "first_grade": {"correct": label},
                "readouts": {
                    "jpp": [
                        {
                            "layer": 1,
                            "position": "last_prompt",
                            "filtered": [{"token_id": 1, "score": 1.0 if label else -1.0}],
                        }
                    ],
                    "logit": [
                        {
                            "layer": 1,
                            "position": "last_prompt",
                            "filtered": [{"token_id": 1, "score": 0.0}],
                        }
                    ],
                },
            }
        )
    return rows


def test_probe_gate_uses_separate_items_and_detects_signal():
    result = probe_gate(feature_rows(0, 40), feature_rows(40, 20), samples=50)
    assert result["go"]
    assert result["methods"]["jpp"]["auc"] == 1
    assert result["methods"]["logit"]["auc"] == 0.5
    with pytest.raises(ValueError, match="disjoint"):
        probe_gate(feature_rows(0, 40), feature_rows(0, 20), samples=20)
    assert not probe_gate(feature_rows(0, 4), feature_rows(4, 4), samples=20)["go"]


def test_synthetic_cannot_pass_real_training_gate(tmp_path):
    run = {"schema": 1, "config": Config().to_dict(), "synthetic": True, "splits": {}}
    write_json(tmp_path / "run.json", run)
    write_json(tmp_path / "gate.json", {"run": fingerprint(run), "go": True})
    with pytest.raises(ValueError, match="real development"):
        require_gate(tmp_path)
    with pytest.raises(ValueError):
        freeze(tmp_path)


def test_protocol_requires_frozen_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_protocol(tmp_path)


def test_freeze_detects_adapter_mutation_and_closes_training(tmp_path):
    from jreflect.training import train

    config = Config()
    config.training.seeds = [42]
    run = {"config": config.to_dict(), "synthetic": False, "splits": {}}
    write_json(tmp_path / "run.json", run)
    write_json(tmp_path / "examples.json", {"fixture": True})
    lens = tmp_path / "lens" / "lens.pt"
    lens.parent.mkdir()
    lens.write_bytes(b"test fixture, not model weights")
    write_json(
        tmp_path / "gate.json",
        {"go": True, "run": fingerprint(run), "lens_sha256": file_hash(lens)},
    )
    for arm in ["readout", "no_readout", "shuffled"]:
        path = tmp_path / "adapters" / "42" / arm
        (path / "adapter").mkdir(parents=True)
        (path / "adapter" / "adapter_model.safetensors").write_bytes(b"test fixture")
        write_json(path / "training.json", {"complete": True})
        write_json(
            path / "identity.json",
            {
                "run": fingerprint(run),
                "examples": fingerprint({"fixture": True}),
                "provenance": provenance(),
                "arm": arm,
                "seed": 42,
            },
        )
    frozen = freeze(tmp_path)
    assert verify_protocol(tmp_path) == frozen
    with pytest.raises(ValueError, match="closed"):
        train(tmp_path, "readout", 42)
    (
        tmp_path / "adapters" / "42" / "readout" / "adapter" / "adapter_model.safetensors"
    ).write_bytes(b"modified")
    with pytest.raises(ValueError, match="adapter changed"):
        verify_protocol(tmp_path)
