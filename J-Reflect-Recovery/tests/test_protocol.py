import copy
import json

import pytest

from jrecovery.analysis import paired_statistics, report, summarize
from jrecovery.config import Config
from jrecovery.data import donor_map, generate, grade, numbers
from jrecovery.demo import DemoBackend
from jrecovery.feedback import (
    ABLATIONS,
    CONDITIONS,
    ablation_messages,
    correction_messages,
    hits,
    payload,
)
from jrecovery.pipeline import collect, correct, first_rows, freeze, plan, prepare, problems
from jrecovery.storage import read, write


@pytest.fixture
def experiment(tmp_path):
    config = Config()
    config.data.dev_count = config.data.test_count = 18
    config.analysis.bootstrap_samples = 100
    prepare(config, tmp_path, demo=True)
    return tmp_path, config


def test_generated_oracles_and_template_separation():
    config = Config()
    config.data.dev_count = config.data.test_count = 90
    data = generate(config)
    assert data == generate(config)
    assert not {p["template"] for p in data["dev"]} & {p["template"] for p in data["test"]}
    for row in data["dev"] + data["test"]:
        assert row["target"] not in numbers(row["question"])
        assert row["target"] != row["gold"]
        assert 10 <= int(row["target"]) <= 999
        if row["template"].startswith("track"):
            values = {}
            # Execute only the test's trusted generator output with no builtins.
            code = row["question"].split(": ")[1].split(". What")[0]
            exec(code, {"__builtins__": {}}, values)
            assert str(values["y"]) == row["gold"]
            assert str(values["x"]) == row["target"]
        else:
            for step in row["oracle_steps"]:
                expression, expected = step.split(" = ")
                assert str(eval(expression, {"__builtins__": {}})) == expected


@pytest.mark.parametrize(
    "text,truncated,status",
    [
        ("Answer: 129", False, "correct"),
        ("Answer: +0129\n", False, "correct"),
        ("Answer: 132", False, "wrong"),
        ("Answer: 129", True, "format_failure"),
        ("I calculate 43. Answer: 129", False, "format_failure"),
        ("Answer: 129\nAnswer: 132", False, "format_failure"),
        ("Answer: 129.5", False, "format_failure"),
        ("129", False, "format_failure"),
    ],
)
def test_strict_grading(text, truncated, status):
    assert grade({"text": text, "truncated": truncated}, "129")["status"] == status


def test_donor_assignment_does_not_read_labels(experiment):
    root, config = experiment
    items = problems(root, "dev")
    donors = donor_map(items, config.seed)
    assert all(source != donor for source, donor in donors.items())
    assert len(set(donors.values())) == len(donors)
    modified = [{**p, "target": "HIDDEN", "gold": "SECRET"} for p in items]
    assert donor_map(modified, config.seed) == donors


def test_shared_history_no_oracle_feedback_and_neutral_correct_audit(experiment):
    root, _ = experiment
    collect(root, "dev")
    rows = first_rows(root, "dev")
    first = next(r for r in rows if r["grade"]["status"] == "wrong")
    donor = next(r for r in rows if r["id"] != first["id"])
    for condition in CONDITIONS:
        messages = correction_messages(first, condition, donor)
        assert messages[:2] == first["messages"]
        assert messages[2]["content"] == first["generation"]["text"]
        assert "oracle" not in messages[-1]["content"]
        if condition == "authentic_jpp":
            assert (
                json.dumps(payload(first["readouts"]), ensure_ascii=False)
                in messages[-1]["content"]
            )
    first = next(r for r in rows if r["grade"]["correct"])
    for condition in CONDITIONS:
        messages = correction_messages(
            first, condition, donor if donor["id"] != first["id"] else rows[-1]
        )
        assert "was incorrect" not in messages[-1]["content"]


def test_ablation_changes_exact_tokens_and_keeps_original(experiment):
    root, _ = experiment
    collect(root, "dev")
    items = {p["id"]: p for p in problems(root, "dev")}
    first = next(
        r for r in first_rows(root, "dev") if hits(r["readouts"], items[r["id"]]["target"])
    )
    original = copy.deepcopy(first)
    target = items[first["id"]]["target"]
    for condition in ABLATIONS:
        messages, edits = ablation_messages(first, target, condition, 42)
        notes = json.loads(messages[-1]["content"].split("irrelevant.\n")[1].split("\nReturn")[0])
        occurrences = sum(t.strip() == target for row in notes for t in row["tokens"])
        assert occurrences == (len(first["readouts"]["jpp"]) if condition == "sham_remove" else 0)
        assert len(edits["edited_slots"]) == len(first["readouts"]["jpp"])
    assert first == original
    assert hits({"jpp": [{"tokens": [{"text": "143"}]}]}, "43") is None


def test_partial_collection_resume_and_no_recomputation(experiment):
    root, config = experiment
    assert collect(root, "dev", limit=4)["saved"] == 4
    with pytest.raises(ValueError, match="Collect all"):
        correct(root, "dev")
    assert collect(root, "dev")["saved"] == 14
    backend = DemoBackend(config, problems(root, "dev"))
    backend.first = lambda _: pytest.fail("completed first attempt was regenerated")
    assert collect(root, "dev", backend=backend)["saved"] == 0
    correct(root, "dev", limit=3)
    assert summarize(root, "dev")["status"] == "partial"
    correct(root, "dev")
    backend.complete = lambda _: pytest.fail("completed correction was regenerated")
    assert correct(root, "dev", backend=backend)["saved"] == 0
    assert summarize(root, "dev")["status"] == "complete"


def test_test_split_gate_and_frozen_dev(experiment):
    root, _ = experiment
    with pytest.raises(ValueError, match="closed"):
        collect(root, "test")
    with pytest.raises(ValueError, match="Collect all"):
        freeze(root)
    collect(root, "dev")
    with pytest.raises(ValueError, match="correction"):
        freeze(root)
    correct(root, "dev")
    freeze(root, "Test fixture")
    collect(root, "test", limit=1)
    assert len(first_rows(root, "test")) == 1
    with pytest.raises(ValueError, match="frozen"):
        correct(root, "dev", stage="ablation")


def test_readout_gate_fails_without_evidence(experiment):
    root, config = experiment
    backend = DemoBackend(config, problems(root, "dev"))
    original = backend.first

    def no_signal(question):
        row, activations = original(question)
        for method in row["readouts"]:
            for layer in row["readouts"][method]:
                for token in layer["tokens"]:
                    token["text"] = "irrelevant"
        return row, activations

    backend.first = no_signal
    collect(root, "dev", backend=backend)
    correct(root, "dev", backend=backend)
    with pytest.raises(ValueError, match="gate failed"):
        freeze(root)
    assert not freeze(root, "Registered negative readout evaluation")["readout_gate_passed"]


def test_tampering_and_protocol_changes_fail_closed(experiment):
    root, _ = experiment
    data = root / "data" / "dev.json"
    rows = read(data)
    rows[0]["gold"] = "BAD"
    write(data, rows)
    with pytest.raises(ValueError, match="Dataset changed"):
        collect(root, "dev")


def test_checksum_corruption_fails(experiment):
    root, _ = experiment
    path = root / "run.json"
    record = json.loads(path.read_text())
    record["payload"]["config"]["seed"] = 123
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Checksum"):
        collect(root, "dev")


def test_format_failures_excluded_and_ablation_subset(experiment):
    root, config = experiment
    backend = DemoBackend(config, problems(root, "dev"))
    original = backend.first
    bad_question = problems(root, "dev")[0]["question"]

    def bad_first(question):
        row, captures = original(question)
        if question == bad_question:
            row["generation"]["text"] = "There are several possible answers"
        return row, captures

    backend.first = bad_first
    collect(root, "dev", backend=backend)
    correct(root, "dev", backend=backend)
    correct(root, "dev", stage="ablation", backend=backend)
    summary = summarize(root, "dev")
    assert summary["first_format_failures"] == 1
    assert summary["corrections_observed"] == 17 * len(CONDITIONS)
    assert summary["ablations_observed"] == summary["ablations_expected"]
    for job in plan(root, "dev", "ablation")["jobs"]:
        assert job["oracle_assisted"]
    report_path = report(root, "dev")
    assert "NOT MODEL PERFORMANCE" in __import__("pathlib").Path(report_path).read_text()


def test_known_paired_statistics():
    config = Config()
    config.analysis.bootstrap_samples = 100
    stats = paired_statistics([("a", True, False)] * 6, config)
    assert stats["difference"] == 1
    assert stats["ci"] == [1, 1]
    assert stats["p"] == 2 / 64
    assert stats["wins"] == 6
    assert paired_statistics([], config)["difference"] is None
    assert paired_statistics([("a", False, False)], config)["p"] == 1
