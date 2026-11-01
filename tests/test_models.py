from pathlib import Path

import pytest

from jreflect.artifacts import file_hash, write_json
from jreflect.diagnostics import tiny_components
from jreflect.lenses import JPP_COMMIT, JppLens
from jreflect.model import HFBackend, capture_positions
from jreflect.readouts import VocabularyFilter
from jreflect.training import collate_targets, encode_target, fit_adapter

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.model


def test_hooks_capture_post_block_positions_and_preserve_forward():
    model, _, _ = tiny_components()
    inputs = torch.tensor([[12, 13, 14, 15]])
    with torch.no_grad():
        baseline = model(inputs, output_hidden_states=True)
        stock_hooks = dict(model.model.layers[0]._forward_hooks)
        with capture_positions(model, [0, 2], [2, 3]) as captured:
            hooked = model(inputs)
    torch.testing.assert_close(baseline.logits, hooked.logits)
    # HF hidden_states[0] is embedding output; block index 0 is hidden_states[1].
    torch.testing.assert_close(captured[0], baseline.hidden_states[1][:, [2, 3]])
    assert dict(model.model.layers[0]._forward_hooks) == stock_hooks
    with pytest.raises(RuntimeError), capture_positions(model, [0], [0]):
        raise RuntimeError("interrupted")
    assert dict(model.model.layers[0]._forward_hooks) == stock_hooks


def test_target_mask_and_padding():
    _, tokenizer, config = tiny_components()
    a = encode_target(tokenizer, "Problem: 2 total", "7", config.model.max_length)
    b = encode_target(tokenizer, "Problem: 2", "8", config.model.max_length)
    assert [v for v in a["labels"] if v != -100] == tokenizer.encode(
        "7", add_special_tokens=False
    ) + [tokenizer.eos_token_id]
    batch = collate_targets([a, b], tokenizer.pad_token_id)
    assert batch["labels"][1, -1] == -100
    assert batch["attention_mask"][1, -1] == 0
    with pytest.raises(ValueError, match="context"):
        encode_target(tokenizer, "Problem: 2 total", "7", 2)


def test_semantic_filter_precedes_topk_and_keeps_digits():
    model, tokenizer, _ = tiny_components()
    filt = VocabularyFilter(tokenizer, model.config.vocab_size)
    scores = torch.zeros(len(tokenizer))
    special, copied, digit, other = (
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("total"),
        tokenizer.convert_tokens_to_ids("2"),
        tokenizer.convert_tokens_to_ids("half"),
    )
    scores[special], scores[copied], scores[digit], scores[other] = 10, 9, 8, 7
    result = filt.rank(scores, "total 2", 2)
    assert [r["token_id"] for r in result["semantic"]] == [copied, digit]
    assert [r["token_id"] for r in result["filtered"]] == [digit, other]


def test_generation_replay_has_exact_first_token():
    model, tokenizer, config = tiny_components()
    backend = HFBackend.__new__(HFBackend)
    backend.model, backend.tokenizer, backend.config, backend.lens = model, tokenizer, config, None
    backend.vocab = VocabularyFilter(tokenizer, model.config.vocab_size)
    generation = backend.generate("Problem: 2 total")
    rows, _ = backend.read("2 total", generation)
    assert len(rows["logit"]) == len(config.readout.layers) * 2
    assert {row["position"] for row in rows["logit"]} == {"last_prompt", "first_answer"}
    assert generation["answer_ids"]


def test_local_pretrained_backend_and_adapter_reload(tmp_path):
    model, tokenizer, config = tiny_components()
    model_path = tmp_path / "base"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    config.model.name = str(model_path)
    backend = HFBackend(config)
    assert backend.generate("Problem: 2 total")["answer_ids"]
    examples = [{"prompt": "Problem: 2 total", "target": "7"}] * 4
    fit_adapter(backend.model, backend.tokenizer, examples, config, 42, tmp_path / "trained")
    restored = HFBackend(config, adapter=str(tmp_path / "trained" / "adapter"))
    assert restored.generate("Problem: 2 total")["answer_ids"]


def test_lens_geometry_provenance_and_orientation(tmp_path):
    _, _, config = tiny_components()
    matrices = {
        layer: torch.arange(16, dtype=torch.float).reshape(4, 4) for layer in config.readout.layers
    }
    path = tmp_path / "lens.pt"
    torch.save(
        {
            "config": {"hf_model_name": config.model.name, "relative_end_transport_layer": -1},
            "parameters": {"jacobians": matrices},
        },
        path,
    )
    meta = {
        "model": config.model.name,
        "revision": config.model.revision,
        "method": "jpp",
        "upstream_commit": JPP_COMMIT,
        "lens_sha256": file_hash(path),
    }
    write_json(path.with_suffix(".meta.json"), meta)
    lens = JppLens(path, config, 4, 4)
    vector = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(lens.transport(vector, 0), torch.tensor([[0.0, 4.0, 8.0, 12.0]]))
    config.model.revision = "different"
    with pytest.raises(ValueError, match="provenance"):
        JppLens(path, config, 4, 4)


def test_lora_resume_matches_uninterrupted_and_base_is_frozen(tmp_path):
    examples = [{"prompt": f"Problem: {i} total", "target": "7"} for i in range(8)]
    model, tokenizer, config = tiny_components()
    original = {name: value.clone() for name, value in model.named_parameters()}
    whole = fit_adapter(model, tokenizer, examples, config, 42, tmp_path / "whole")
    for name, value in model.named_parameters():
        # PEFT wraps projections but leaves the original tensor values untouched.
        old_name = name.replace(".base_layer", "")
        if old_name in original:
            torch.testing.assert_close(value, original[old_name], rtol=0, atol=0)
    model2, tokenizer2, config2 = tiny_components()
    partial = fit_adapter(
        model2, tokenizer2, examples, config2, 42, tmp_path / "resume", max_steps=1
    )
    assert not partial["complete"]
    model3, tokenizer3, config3 = tiny_components()
    resumed = fit_adapter(model3, tokenizer3, examples, config3, 42, tmp_path / "resume")
    assert resumed["complete"] and whole["complete"]
    assert resumed["losses"] == pytest.approx(whole["losses"], abs=1e-6)
    assert whole["losses"][-1] < whole["losses"][0]
    from safetensors.torch import load_file

    a = load_file(tmp_path / "whole" / "adapter" / "adapter_model.safetensors")
    b = load_file(tmp_path / "resume" / "adapter" / "adapter_model.safetensors")
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=1e-6)


@pytest.mark.jpp
def test_upstream_lrp_forward_gradient_and_restore():
    checkout = Path("vendor/jpp_lens")
    if not checkout.exists():
        pytest.skip("Pinned upstream checkout not installed")
    from jreflect.jpp import load_upstream

    load_upstream(checkout)
    from workspace_lens.lrp import apply_lrp_rules_for_mode

    model, _, _ = tiny_components()
    x = torch.randn(1, 4, 32, requires_grad=True)
    baseline = model(inputs_embeds=x).logits
    ordinary = torch.autograd.grad(baseline.sum(), x)[0]
    with apply_lrp_rules_for_mode(model, "rlens") as report:
        patched = model(inputs_embeds=x).logits
        relevance = torch.autograd.grad(patched.sum(), x)[0]
        assert len(report.mlp_rule) == 4 and len(report.ln_rule) > 0
    torch.testing.assert_close(patched, baseline, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(relevance).all() and not torch.allclose(ordinary, relevance)
    torch.testing.assert_close(model(inputs_embeds=x).logits, baseline)


@pytest.mark.jpp
def test_tiny_qwen_jpp_fit_weighting_and_readout_roundtrip(tmp_path):
    checkout = Path("vendor/jpp_lens")
    if not checkout.exists():
        pytest.skip("Pinned upstream checkout not installed")
    from jreflect.jpp import load_upstream

    upstream = load_upstream(checkout)
    from jlens import from_hf
    from lens_evals.readout_evals.readout_eval_items import ReadoutEvalItem
    from workspace_lens.config import LensConfig
    from workspace_lens.fitting.condense_experts import CondenseConfig
    from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
    from workspace_lens.lenses.base_lens import BaseLens

    model, tokenizer, config = tiny_components()
    model.config.name_or_path = config.model.name
    wrapped = from_hf(model, tokenizer, force_bos=False)
    prompts = ["2 3 4 5 6 7 8 9", "9 8 7 6 5 4 3 2"]
    router = upstream.fit_router(
        wrapped,
        prompts,
        layers=[0],
        num_clusters=2,
        projection_dim=4,
        skip_first_n_positions=1,
        max_seq_len=8,
        seed=42,
    )
    fit_config = LensConfig(
        hf_model_name=config.model.name,
        checkpoint_name="tiny",
        artifacts_base_dir=str(tmp_path),
        source_layers=[0],
        relative_end_transport_layer=-1,
        lrp_mode="rlens",
        max_seq_len=8,
        skip_first_n_positions=1,
        jacobian_rows_per_pass=8,
        checkpoint_every_n_prompts=1,
    )
    trainer = ExpertJacobianRelPTrainer(
        fit_config, wrapped, prompts, router_collections_K_dict={2: router}, device="cpu"
    )
    trainer.fit()
    resumed = ExpertJacobianRelPTrainer.from_checkpoint_dir(
        trainer.config.checkpoint_path,
        prompts,
        router_collections_K_dict={2: router},
        model=wrapped,
        device="cpu",
    )
    resumed.fit()
    assert resumed.next_prompt_idx == 2
    experts = upstream.merge_experts(
        [trainer.config.checkpoint_path],
        num_clusters=2,
        min_kept_positions=1,
        checkpoint_name="merged",
        device="cpu",
    )
    items = [
        ReadoutEvalItem("association", f"association-{i}", f"2 3 {i}", None, ("half",))
        for i in range(4)
    ]
    weighting, lens = upstream.fit_weights(
        wrapped,
        experts,
        items,
        layers=[0],
        excluded_token_ids=[],
        cache_path=None,
        checkpoint_name="weighted",
        hf_model_name=config.model.name,
        fitting_item_names=[items[0].name, items[1].name],
        condense_config=CondenseConfig(steps=3, top_n=2),
    )
    assert weighting.weights_L_dict_E[0].shape == (2,)
    lens.save(str(tmp_path / "lens.pt"), dtype=torch.float32)
    restored = BaseLens.load(str(tmp_path / "lens.pt"))
    residual = torch.randn(1, 1, 32)
    torch.testing.assert_close(lens.transport(residual, 0), restored.transport(residual, 0))
    ranks, _ = upstream.evaluate(
        wrapped, {"jpp": restored}, items[2:], layers=[0], excluded_token_ids=[], ks=[10]
    )
    assert set(ranks["item"]) == {items[2].name, items[3].name}
