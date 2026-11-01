import json
from fractions import Fraction

import pytest

from jreflect.artifacts import RowCache, fingerprint, load_cache, read_json, write_json
from jreflect.config import Config, load_config
from jreflect.data import load_split, split_training
from jreflect.diagnostics import FixtureBackend
from jreflect.evaluation import evaluate_predictions, holm, mcnemar
from jreflect.examples import build_examples, derangement
from jreflect.grading import changed, gold_answer, grade, parse_number
from jreflect.pipeline import generate_split
from jreflect.prompts import correction_prompt
from jreflect.readouts import band_concepts


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1,234.50", Fraction(2469, 2)),
        (" -0.50 ", Fraction(-1, 2)),
        (".25", Fraction(1, 4)),
        ("+7", Fraction(7)),
    ],
)
def test_numeric_normalization(value, expected):
    assert parse_number(value) == expected


@pytest.mark.parametrize(
    "value", ["1,23", "The answer is 3", "3 4", "NaN", "inf", "", "3/4", "$5", "1e3"]
)
def test_strict_answer_format(value):
    assert parse_number(value) is None


def test_grading_and_truncation():
    assert grade("12.00", "12")["correct"]
    assert not grade("12", "12", truncated=True)["correct"]
    assert not grade("Answer: 12", "12")["correct"]
    assert grade("Answer: 12", "12")["permissive_correct"]
    assert gold_answer("1+2=3. #### 3") == "3"
    with pytest.raises(ValueError):
        gold_answer("No delimiter")
    assert not changed("01", "1.0")
    assert changed("invalid", "1")


def test_config_rejects_errors():
    assert load_config("configs/initial.toml").readout.layers == [3, 7, 11, 15, 19, 23]
    with pytest.raises(TypeError):
        Config.from_dict({"typo": 1})
    with pytest.raises(ValueError):
        Config.from_dict({"training": {"micro_batch": 3}})
    with pytest.raises(ValueError):
        Config.from_dict({"readout": {"middle": [999]}})


@pytest.fixture
def run_dir(tmp_path):
    config = Config()
    config.data.train_count, config.data.dev_size, config.data.gate_size = 40, 12, 4
    records = [
        {"question": f"Fixture {i}: return index", "answer": f"SECRET REASONING #### {i}"}
        for i in range(40)
    ]
    splits = split_training(records, config)
    for name, rows in splits.items():
        write_json(tmp_path / "data" / f"{name}.json", rows)
    write_json(
        tmp_path / "run.json",
        {
            "schema": 1,
            "config": config.to_dict(),
            "synthetic": True,
            "splits": {name: fingerprint(rows) for name, rows in splits.items()},
        },
    )
    return tmp_path


def test_splits_disjoint_reproducible_and_no_reference_reasoning(run_dir):
    groups = [load_split(run_dir, name) for name in ["train", "dev_tune", "dev_gate"]]
    assert [len(g) for g in groups] == [28, 8, 4]
    ids = [r["id"] for group in groups for r in group]
    assert len(ids) == len(set(ids)) == 40
    assert "SECRET" not in json.dumps(groups)
    assert groups[0] == load_split(run_dir, "train")
    with pytest.raises(FileNotFoundError):
        load_split(run_dir, "test")


def test_prepare_streams_only_train_and_pins_revisions(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import datasets
    import huggingface_hub

    from jreflect.data import prepare

    calls = []

    def fake_loader(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["streaming"] is True and kwargs["split"] == "train"
        return iter([{"question": str(i), "answer": f"reference #### {i}"} for i in range(12)])

    monkeypatch.setattr(datasets, "load_dataset", fake_loader)
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(
            model_info=lambda *a, **kw: SimpleNamespace(sha="model-sha"),
            dataset_info=lambda *a, **kw: SimpleNamespace(sha="dataset-sha"),
        ),
    )
    config = Config()
    config.data.train_count, config.data.dev_size, config.data.gate_size = 12, 4, 2
    prepare(config, tmp_path)
    assert len(calls) == 1
    assert not (tmp_path / "data" / "test.json").exists()
    assert read_json(tmp_path / "run.json")["config"]["model"]["revision"] == "model-sha"


def test_split_tampering_is_detected(run_dir):
    path = run_dir / "data" / "train.json"
    rows = read_json(path)
    rows[0]["gold"] = "999"
    write_json(path, rows)
    with pytest.raises(ValueError, match="checksum"):
        load_split(run_dir, "train")


def test_cache_identity_integrity_and_resume(tmp_path):
    cache = RowCache(tmp_path, {"model": "v1"})
    row = {"id": "a", "answer": 2}
    cache.put(row)
    cache.put(row)
    with pytest.raises(ValueError, match="replace"):
        cache.put({"id": "a", "answer": 3})
    with pytest.raises(ValueError, match="Incompatible"):
        RowCache(tmp_path, {"model": "v2"})
    with pytest.raises(ValueError, match="Incomplete"):
        cache.finish(["a", "b"])
    cache.finish(["a"])
    assert load_cache(tmp_path)[1] == [row]
    record = read_json(cache.item_path("a"))
    record["row"]["answer"] = 99
    write_json(cache.item_path("a"), record)
    with pytest.raises(ValueError, match="Corrupt"):
        cache.get("a")


def test_generation_resume_does_not_reload_model(run_dir):
    first = generate_split(run_dir, "dev_tune", limit=3, backend=FixtureBackend())
    assert first["new_items"] == 3 and not first["complete"]
    second = generate_split(run_dir, "dev_tune", backend=FixtureBackend())
    assert second["new_items"] == 5 and second["complete"]
    third = generate_split(run_dir, "dev_tune")
    assert third["new_items"] == 0 and third["complete"]


def test_balancing_and_arm_control(run_dir):
    generate_split(run_dir, "train", backend=FixtureBackend())
    _, rows = load_cache(run_dir / "base" / "logit" / "train")
    result = build_examples(rows, 42, method="logit")
    arms = result["arms"]
    reference = [(r["id"], r["source_id"], r["target"]) for r in arms["readout"]]
    for arm in arms:
        assert [(r["id"], r["source_id"], r["target"]) for r in arms[arm]] == reference
        assert all("SECRET" not in r["prompt"] for r in arms[arm])
    assert all(r["donor_id"] != r["source_id"] for r in arms["shuffled"])
    assert all("internal workspace" not in r["prompt"] for r in arms["no_readout"])
    labels = {r["id"]: r["first_grade"]["correct"] for r in rows}
    assert sum(labels[r["source_id"]] for r in arms["readout"]) * 2 == len(reference)
    assert result == build_examples(rows, 42, method="logit")


def test_right_oversampling_keeps_all_wrong():
    rows = [
        {
            "id": str(i),
            "split": "train",
            "question": "q",
            "gold": "1",
            "first": {"text": "2"},
            "first_grade": {"correct": i == 0},
            "concepts": {"jpp": {"middle": [str(i)], "late": []}},
        }
        for i in range(6)
    ]
    result = build_examples(rows, 3)
    ids = [r["source_id"] for r in result["arms"]["readout"]]
    assert ids.count("0") == 5 and set(ids) == {str(i) for i in range(6)}
    assert len(result["donors"]) == 6
    with pytest.raises(ValueError):
        build_examples([{**r, "split": "dev_tune"} for r in rows], 42)


def test_derangement_and_narrow_prompt_api():
    mapping = derangement(["a", "b", "c"], 42)
    assert all(k != v for k, v in mapping.items())
    assert set(mapping) == set(mapping.values())
    with pytest.raises(ValueError):
        derangement(["a"], 42)
    with pytest.raises(TypeError):
        correction_prompt("q", "1", "no_readout", correctness=True)


def test_band_ranks_ignore_logit_scale():
    config = Config().readout
    rows = [
        {
            "layer": layer,
            "filtered": [
                {"text": " Total ", "rank": 1, "score": value},
                {"text": "half", "rank": 2, "score": value / 2},
            ],
        }
        for layer, value in [(11, 1), (15, 1000), (19, 2), (23, 3)]
    ]
    assert band_concepts(rows, config) == {"middle": ["total", "half"], "late": ["total", "half"]}


def test_paired_statistics_known_values():
    assert mcnemar([True] * 10, [False] * 10)["p"] == pytest.approx(2 / 1024)
    assert mcnemar([True, False], [True, False])["p"] == 1
    assert holm({"a": 0.01, "b": 0.03}) == {"a": 0.02, "b": 0.03}


def test_evaluation_aligns_ids_and_rejects_missing():
    base = [
        {"id": str(i), "first_grade": {"correct": False}, "first": {"text": "0"}} for i in range(8)
    ]
    predictions = {}
    for seed in [42, 43, 44]:
        for arm in ["readout", "no_readout", "shuffled"]:
            correct = arm == "readout"
            predictions[(seed, arm)] = [
                {
                    "id": str(i),
                    "generation": {"text": "1" if correct else "0"},
                    "grade": {"correct": correct, "valid": True},
                }
                for i in reversed(range(8))
            ]
    result = evaluate_predictions(base, predictions, [42, 43, 44], bootstrap_samples=20)
    assert result["success"]
    assert result["per_seed"]["42"]["arms"]["readout"]["right_to_wrong"] is None
    assert result["seed_mean"]["readout"]["ci95_pp"] == [100, 100]
    predictions[(42, "readout")].pop()
    with pytest.raises(ValueError, match="Missing"):
        evaluate_predictions(base, predictions, [42, 43, 44], bootstrap_samples=20)
