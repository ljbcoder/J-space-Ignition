"""Pinned upstream integration: router -> resumable Jacobians -> weights -> validation."""

import importlib.util
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import exclusive, file_hash, fingerprint, provenance, read_json, write_json
from .data import load_run
from .lenses import JPP_COMMIT


def load_upstream(checkout):
    checkout = Path(checkout).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if commit != JPP_COMMIT or dirty:
        raise ValueError(f"J++ checkout must be unmodified at {JPP_COMMIT}")
    source = str(checkout / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location(
        "jreflect_upstream_cli", checkout / "scripts" / "jpp_cli.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fit(root, checkout, stage="all"):
    import torch

    from .model import HFBackend

    root, checkout = Path(root).resolve(), Path(checkout).resolve()
    run, config = load_run(root)
    if run["synthetic"] or (root / "gate_started.json").exists():
        raise ValueError(
            "Lens fitting is closed for synthetic runs or after opening the development gate"
        )
    upstream = load_upstream(checkout)
    output = root / "lens"
    with exclusive(output / ".lock"):
        identity = {
            "run": fingerprint(run),
            "upstream_commit": JPP_COMMIT,
            "provenance": provenance(),
            "router_records": 1000,
            "jacobian_records": 64,
            "experts": 8,
            "max_tokens": 128,
            "skip_first": 16,
            "weight_fit_fraction": 0.2,
            "layers": config.readout.layers,
        }
        path = output / "identity.json"
        if path.exists():
            saved = read_json(path)
            if saved["identity"] != identity:
                raise ValueError("Lens stage inputs or software changed")
        else:
            saved = {"identity": identity, "created": datetime.now(UTC).isoformat()}
            write_json(path, saved)
        backend = HFBackend(config)
        from jlens import from_hf

        model = from_hf(backend.model, backend.tokenizer, force_bos=False)
        stages = ["router", "jacobians", "weights", "validate"] if stage == "all" else [stage]
        for current in stages:
            print(f"J++ stage: {current}", flush=True)
            if current == "router":
                _router(upstream, model, config, output)
            elif current == "jacobians":
                _jacobians(upstream, model, config, output, saved["created"])
            elif current == "weights":
                _weights(upstream, model, config, checkout, output)
            elif current == "validate":
                _validate(upstream, model, config, checkout, output)
            else:
                raise ValueError(f"Unknown J++ stage {current}")
        return {
            "stage": stage,
            "output": str(output),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
        }


def _router(upstream, model, config, output):
    if (output / "router.pt").exists():
        return
    from datasets import load_dataset
    from huggingface_hub import HfApi

    prompt_path = output / "wikitext.json"
    if not prompt_path.exists():
        revision = HfApi().dataset_info("Salesforce/wikitext").sha
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="train",
            revision=revision,
            streaming=True,
        )
        prompts = []
        for row in dataset:
            if len(row["text"]) >= 600:
                prompts.append(row["text"])
            if len(prompts) == 1000:
                break
        if len(prompts) != 1000:
            raise ValueError("Not enough qualifying WikiText records")
        write_json(
            prompt_path, {"revision": revision, "prompts": prompts, "hash": fingerprint(prompts)}
        )
    records = read_json(prompt_path)
    if fingerprint(records["prompts"]) != records["hash"]:
        raise ValueError("WikiText checksum mismatch")
    router = upstream.fit_router(
        model,
        records["prompts"],
        layers=config.readout.layers,
        num_clusters=8,
        projection_dim=64,
        skip_first_n_positions=16,
        max_seq_len=128,
        seed=config.seed,
    )
    router.save(str(output / "router.pt"))


def _jacobians(upstream, model, config, output, created):
    from workspace_lens.config import LensConfig
    from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
    from workspace_lens.routing.router import ActivationRouterCollection

    if (output / "experts.pt").exists():
        return
    prompts = read_json(output / "wikitext.json")["prompts"][:64]
    router = ActivationRouterCollection.load(str(output / "router.pt"))
    fit_config = LensConfig(
        hf_model_name=config.model.name,
        checkpoint_name="qwen25-jpp",
        creation_time=created,
        artifacts_base_dir=str(output / "checkpoints"),
        source_layers=config.readout.layers,
        relative_end_transport_layer=-1,
        d_model=model.d_model,
        lrp_mode="rlens",
        jacobian_rows_per_pass=16,
        max_seq_len=128,
        skip_first_n_positions=16,
        checkpoint_every_n_prompts=4,
    )
    checkpoint = Path(fit_config.checkpoint_path) / "experts_K8_checkpoint.pt"
    if checkpoint.exists():
        trainer = ExpertJacobianRelPTrainer.from_checkpoint_dir(
            str(checkpoint.parent),
            prompts,
            router_collections_K_dict={8: router},
            model=model,
            device=config.model.device,
        )
    else:
        trainer = ExpertJacobianRelPTrainer(
            fit_config,
            model,
            prompts,
            router_collections_K_dict={8: router},
            device=config.model.device,
        )
    trainer.fit()
    experts = upstream.merge_experts(
        [trainer.config.checkpoint_path],
        num_clusters=8,
        min_kept_positions=50,
        checkpoint_name="qwen25-jpp-merged",
        device=config.model.device,
    )
    experts.save(str(output / "experts.pt"))


def _items(upstream, model, config, checkout, output):
    from lens_evals.readout_evals.readout_eval_items import load_readout_eval_items

    eval_dir = checkout / "data" / "jlens" / "evaluations"
    correctness = output / "correctness.csv"
    if not correctness.exists():
        items = load_readout_eval_items(str(eval_dir))
        runner = upstream.readout_runner(model, layers=config.readout.layers, max_seq_len=512)
        runner.grade_model_correctness(items, hf_model_name=config.model.name).to_csv(
            correctness, index=False
        )
    items, fitting, excluded = upstream.load_items_and_exclusions(
        model,
        eval_data_dir=str(eval_dir),
        correctness_csv=str(correctness),
        hf_model_name=config.model.name,
        layers=config.readout.layers,
        readout_max_seq_len=512,
        items_spec=f"fit:{config.seed}",
        fit_fraction=0.2,
    )
    scoreable = upstream.readout_runner(
        model, layers=config.readout.layers, max_seq_len=512
    ).scoreable_items(items)
    heldout = upstream.select_items(scoreable, f"held-out:{config.seed}", fit_fraction=0.2)
    manifest = {
        "fitting": [i.name for i in fitting],
        "heldout": [i.name for i in heldout],
        "correctness_sha256": file_hash(correctness),
        "scoreable": len(scoreable),
    }
    if not fitting or not heldout or set(manifest["fitting"]) & set(manifest["heldout"]):
        raise ValueError("Latent fitting and evaluation splits must be nonempty and disjoint")
    path = output / "latent_split.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Latent evaluation split changed")
    write_json(path, manifest)
    return items, fitting, heldout, excluded


def _weights(upstream, model, config, checkout, output):
    import torch
    from workspace_lens.fitting.condense_experts import ExpertJacobians

    if (output / "lens.meta.json").exists():
        return
    items, fitting, _, excluded = _items(upstream, model, config, checkout, output)
    experts = ExpertJacobians.load(str(output / "experts.pt"))
    weighting, lens = upstream.fit_weights(
        model,
        experts,
        items,
        layers=config.readout.layers,
        excluded_token_ids=excluded,
        cache_path=str(output / "latent_residuals.pt"),
        checkpoint_name="qwen25-jpp",
        hf_model_name=config.model.name,
        fitting_item_names=[i.name for i in fitting],
    )
    lens.save(str(output / "lens.pt"), dtype=torch.float32)
    weighting.save(str(output / "weights.json"))
    write_json(
        output / "lens.meta.json",
        {
            "model": config.model.name,
            "revision": config.model.revision,
            "lens_sha256": file_hash(output / "lens.pt"),
            "method": "jpp",
            "upstream_commit": JPP_COMMIT,
            "latent_split_sha256": file_hash(output / "latent_split.json"),
        },
    )


def _validate(upstream, model, config, checkout, output):
    from workspace_lens.config import LensConfig
    from workspace_lens.lenses.base_lens import BaseLens
    from workspace_lens.lenses.logit_lens import LogitLens

    if (output / "validation.json").exists():
        return
    _, _, heldout, excluded = _items(upstream, model, config, checkout, output)
    lens = BaseLens.load(str(output / "lens.pt"))
    logit = LogitLens(
        config.readout.layers,
        config=LensConfig(
            hf_model_name=config.model.name,
            checkpoint_name="logit",
            lens_type="logit",
            source_layers=config.readout.layers,
            d_model=model.d_model,
        ),
    )
    ranks, scores = upstream.evaluate(
        model,
        {"jpp": lens, "logit": logit},
        heldout,
        layers=config.readout.layers,
        excluded_token_ids=excluded,
        ks=[1, 10],
    )
    ranks.to_csv(output / "heldout_ranks.csv", index=False)
    scores.to_csv(output / "heldout_scores.csv", index=False)
    selected = scores[
        (scores["eval"] == "macro") & (scores["k"] == 10) & (scores["weighting"] == "item")
    ]
    values = {row["lens"]: float(row["pass_at_k"]) for _, row in selected.iterrows()}
    go = (
        all(math.isfinite(values.get(m, float("nan"))) for m in ["jpp", "logit"])
        and values["jpp"] > values["logit"]
    )
    qualitative = []
    for prompt in [
        "Fact: The number of legs on the animal that spins webs is",
        "Fact: In humans, the organ that pumps blood through the body has this many chambers: ",
    ]:
        logits, _, _ = lens.apply(
            model, prompt, layers=config.readout.layers, token_positions_for_residuals=[-1]
        )
        for layer, scores_at_layer in logits.items():
            score = scores_at_layer[0].clone()
            score[excluded] = -float("inf")
            qualitative.append(
                {
                    "prompt": prompt,
                    "layer": layer,
                    "tokens": [
                        model.tokenizer.decode([i]) for i in score.topk(10).indices.tolist()
                    ],
                }
            )
    write_json(output / "qualitative.json", qualitative)
    write_json(
        output / "validation.json",
        {
            "go": bool(go),
            "lens_sha256": file_hash(output / "lens.pt"),
            "heldout_count": len(heldout),
            "recall_at_10": {k: v if math.isfinite(v) else None for k, v in values.items()},
            "rule": "Held-out item-weighted macro recall@10 must exceed the logit lens; GSM8K gate follows.",
        },
    )
