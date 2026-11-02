from pathlib import Path

import pytest

from jrecovery.backend import HFBackend, NativeLens, capture_prefill, text_decoder
from jrecovery.config import Config
from jrecovery.feedback import first_messages

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytestmark = pytest.mark.model


def components(tmp_path, kind="qwen35"):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
    )

    torch.manual_seed(123)
    torch.set_num_threads(1)
    tokens = [
        "<pad>",
        "<eos>",
        "<unk>",
        "system",
        "user",
        "assistant",
        "Answer",
        ":",
        "multiply",
    ] + [str(i) for i in range(1000)]
    raw = Tokenizer(WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    settings = dict(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if kind == "qwen35":
        model = Qwen3_5ForConditionalGeneration(
            Qwen3_5Config(
                text_config={
                    **settings,
                    "head_dim": 8,
                    "linear_key_head_dim": 8,
                    "linear_value_head_dim": 8,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "partial_rotary_factor": 1.0,
                    "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]},
                },
                vision_config={
                    "depth": 1,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_heads": 4,
                    "out_hidden_size": 32,
                    "num_position_embeddings": 16,
                },
                image_token_id=len(tokenizer) + 1,
                video_token_id=len(tokenizer) + 2,
                vision_start_token_id=len(tokenizer) + 3,
                vision_end_token_id=len(tokenizer) + 4,
            )
        ).eval()
    else:
        model = Qwen2ForCausalLM(Qwen2Config(**settings)).eval()
    config = Config()
    config.model.name = f"tiny-{kind}"
    config.model.revision = "random-fixture"
    config.model.device, config.model.dtype = "cpu", "float32"
    config.model.max_context, config.model.max_new_tokens = 512, 4
    config.lens.layers, config.lens.top_k = [0, 1, 2], 5
    maps = {i: torch.eye(32) + torch.randn(32, 32) * 0.05 for i in config.lens.layers}
    path = tmp_path / f"{kind}.pt"
    torch.save(
        {
            "config": {
                "hf_model_name": config.model.name,
                "relative_end_transport_layer": -1,
                "lens_type": "jacobian",
            },
            "source_layers": config.lens.layers,
            "parameters": {"jacobians": maps},
        },
        path,
    )
    return model, tokenizer, config, path


@pytest.mark.parametrize("kind", ["qwen2", "qwen35"])
def test_actual_prefill_readouts_match_manual_unembedding(tmp_path, kind):
    model, tokenizer, config, path = components(tmp_path, kind)
    backend = HFBackend(config, model=model, tokenizer=tokenizer, lens_path=path)
    question = "multiply 26 17"
    plain = backend.complete(first_messages(question))
    result, captures = backend.first(question)
    assert result["generation"]["answer_ids"] == plain["answer_ids"]
    assert set(captures) == {0, 1, 2}
    assert all(v.shape == (32,) for v in captures.values())
    assert all(not layer._forward_hooks for layer in backend.decoder.layers)
    inputs = torch.tensor([result["generation"]["prompt_ids"]])
    with (
        torch.no_grad(),
        capture_prefill(backend.decoder, config.lens.layers, inputs.shape[1]) as direct,
    ):
        model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=True)
    for layer in config.lens.layers:
        torch.testing.assert_close(direct[layer], captures[layer])
    with torch.no_grad():
        transported = backend.lens.transport(captures[0], 0)
        expected = backend.head(backend.decoder.norm(transported))
    for token in result["readouts"]["jpp"][0]["tokens"]:
        assert token["score"] == pytest.approx(expected[token["id"]].item(), abs=1e-5)
    activation_path = tmp_path / "activation.pt"
    backend.save_activations(activation_path, captures, result["generation"]["prompt_ids"])
    saved = torch.load(activation_path, weights_only=True)
    assert saved["position"] == len(result["generation"]["prompt_ids"]) - 1
    assert backend.supports("129") and not backend.supports("9999")


def test_hooks_clean_up_after_exception(tmp_path):
    model, _, _, _ = components(tmp_path)
    decoder = text_decoder(model)
    with pytest.raises(RuntimeError), capture_prefill(decoder, [0, 1], 3):
        raise RuntimeError("interrupted")
    assert all(not layer._forward_hooks for layer in decoder.layers)


def test_lens_provenance_geometry_and_finiteness(tmp_path):
    _, _, config, path = components(tmp_path)
    original = torch.load(path, weights_only=True)
    wrong = tmp_path / "wrong.pt"
    modified = dict(original)
    modified["config"] = {**original["config"], "hf_model_name": "other-model"}
    torch.save(modified, wrong)
    with pytest.raises(ValueError, match="different model"):
        NativeLens(wrong, config, 32, 4)
    modified = {
        **original,
        "parameters": {"jacobians": {0: torch.eye(16), 1: torch.eye(32), 2: torch.eye(32)}},
    }
    torch.save(modified, wrong)
    with pytest.raises(ValueError, match="Invalid lens"):
        NativeLens(wrong, config, 32, 4)
    original["parameters"]["jacobians"][0][0, 0] = float("nan")
    torch.save(original, wrong)
    with pytest.raises(ValueError, match="Invalid lens"):
        NativeLens(wrong, config, 32, 4)


def test_no_context_truncation_or_missing_lens_fallback(tmp_path):
    model, tokenizer, config, path = components(tmp_path)
    with pytest.raises(ValueError, match="real lens"):
        HFBackend(config, model=model, tokenizer=tokenizer)
    backend = HFBackend(config, model=model, tokenizer=tokenizer, lens_path=path)
    with pytest.raises(ValueError, match="context limit"):
        backend.complete(first_messages("multiply " * 600))
    assert Path(path).exists()
