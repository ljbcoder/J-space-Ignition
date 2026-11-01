"""Fit on development items, measure once on separate gate items."""

from pathlib import Path

from .artifacts import fingerprint, load_cache, read_json, write_json
from .data import load_run, load_split
from .evaluation import interval
from .readouts import content_overlap, probe_features


def checked_base(root, split):
    root = Path(root)
    run, _ = load_run(root)
    identity, rows = load_cache(root / "base" / "jpp" / split)
    expected = load_split(root, split)
    if identity["run"] != fingerprint(run) or identity["method"] != "jpp" or identity["synthetic"]:
        raise ValueError("Base cache is not a real J++ run for this configuration")
    if identity["data"] != fingerprint(expected) or [r["id"] for r in rows] != [
        r["id"] for r in expected
    ]:
        raise ValueError("Base cache does not cover the complete split")
    return identity, rows


def probe_gate(tune, heldout, *, seed=42, samples=10000, confidence=0.95):
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    a_ids, b_ids = {r["id"] for r in tune}, {r["id"] for r in heldout}
    if a_ids & b_ids or len(a_ids) != len(tune) or len(b_ids) != len(heldout):
        raise ValueError("Probe fitting and gate IDs must be distinct and disjoint")
    y_train = np.array([r["first_grade"]["correct"] for r in tune], dtype=int)
    y_test = np.array([r["first_grade"]["correct"] for r in heldout], dtype=int)
    counts = {
        "tune": np.bincount(y_train, minlength=2).tolist(),
        "gate": np.bincount(y_test, minlength=2).tolist(),
    }
    if min(counts["tune"]) < 5 or min(counts["gate"]) < 5:
        return {"go": False, "status": "insufficient_class_counts", "counts": counts}
    scores, methods = {}, {}
    for method in ["jpp", "logit"]:
        pipeline = Pipeline(
            [
                ("features", DictVectorizer()),
                ("scale", StandardScaler(with_mean=False)),
                ("probe", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        )
        search = GridSearchCV(
            pipeline,
            {"probe__C": [0.01, 0.1, 1.0, 10.0]},
            scoring="roc_auc",
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            error_score="raise",
        )
        search.fit([probe_features(r, method) for r in tune], y_train)
        x_test = [probe_features(r, method) for r in heldout]
        scores[method] = search.predict_proba(x_test)[:, 1]
        methods[method] = {
            "auc": float(roc_auc_score(y_test, scores[method])),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, search.predict(x_test))),
            "C": search.best_params_["probe__C"],
            "cv_auc": float(search.best_score_),
        }
    rng, values, gaps = np.random.default_rng(seed), [], []
    for _ in range(samples):
        ix = rng.integers(0, len(heldout), len(heldout))
        if len(set(y_test[ix])) < 2:
            continue
        auc = {m: float(roc_auc_score(y_test[ix], scores[m][ix])) for m in scores}
        values.append(auc["jpp"])
        gaps.append(auc["jpp"] - auc["logit"])
    auc_ci, gap_ci = interval(values, confidence), interval(gaps, confidence)
    go = auc_ci is not None and gap_ci is not None and auc_ci[0] > 0.5 and gap_ci[0] > 0
    return {
        "go": go,
        "status": "go" if go else "inconclusive_or_no_signal",
        "counts": counts,
        "methods": methods,
        "jpp_auc_ci": auc_ci,
        "auc_difference_ci": gap_ci,
        "prediction_rows": [
            {"id": r["id"], "correct": bool(y_test[i]), **{m: float(scores[m][i]) for m in scores}}
            for i, r in enumerate(heldout)
        ],
    }


def run_gate(root):
    root = Path(root)
    if (root / "gate.json").exists() or (root / "gate_started.json").exists():
        raise ValueError(
            "The held-out gate has already been opened; preserve it and create a new experiment for changes"
        )
    run, config = load_run(root)
    tune_meta, tune = checked_base(root, "dev_tune")
    gate_meta, heldout = checked_base(root, "dev_gate")
    if tune_meta["lens"] != gate_meta["lens"] or tune_meta["provenance"] != gate_meta["provenance"]:
        raise ValueError("Dev caches were generated with different lenses or software")
    validation = read_json(root / "lens" / "validation.json")
    if not validation["go"] or validation["lens_sha256"] != tune_meta["lens"]:
        raise ValueError("Lens must pass its held-out latent-task check first")
    write_json(
        root / "gate_started.json",
        {"run": fingerprint(run), "tune": fingerprint(tune), "heldout": fingerprint(heldout)},
    )
    result = probe_gate(
        tune,
        heldout,
        seed=config.seed,
        samples=config.evaluation.bootstrap_samples,
        confidence=config.evaluation.confidence,
    )
    references = read_json(root / "data" / "references.json")
    write_json(
        root / "content_audit.json",
        [
            {"id": r["id"], "methods": content_overlap(r, references[r["id"]])}
            for r in tune + heldout
        ],
    )
    accuracy = sum(r["first_grade"]["correct"] for r in tune + heldout) / (len(tune) + len(heldout))
    result.update(
        {
            "run": fingerprint(run),
            "lens_sha256": tune_meta["lens"],
            "tune_hash": fingerprint(tune),
            "gate_hash": fingerprint(heldout),
            "base_dev_accuracy": accuracy,
            "estimated_test_errors": config.data.test_count * (1 - accuracy),
            "interpretation": "Resource gate only; a no-go does not establish absence of useful representations.",
        }
    )
    write_json(root / "gate.json", result)
    return result


def require_gate(root, lens_sha=None):
    root = Path(root)
    run, _ = load_run(root)
    gate = read_json(root / "gate.json")
    if not gate["go"] or gate["run"] != fingerprint(run) or run["synthetic"]:
        raise ValueError("Full training requires a successful real development gate for this run")
    if lens_sha is not None and gate["lens_sha256"] != lens_sha:
        raise ValueError("Gate belongs to a different lens")
    return gate
