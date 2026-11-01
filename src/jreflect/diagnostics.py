"""Explicitly synthetic fixtures plus a real, randomly initialized tiny Qwen2."""

import re
from pathlib import Path

from .artifacts import fingerprint, write_json
from .config import Config
from .data import split_training
from .grading import grade


def tiny_components(seed=42):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tokens = [
        "<pad>",
        "<eos>",
        "<unk>",
        "<system>",
        "<user>",
        "<assistant>",
        "Problem",
        ":",
        "number",
        "total",
        "remaining",
        "half",
    ] + [str(i) for i in range(100)]
    tokenizer = Tokenizer(
        WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=["<system>", "<user>", "<assistant>"],
    )
    tokenizer.chat_template = "{% for message in messages %}{{ '<' + message['role'] + '> ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    ).eval()
    config = Config()
    config.model.name, config.model.revision = "tiny-random-qwen2", "fixture"
    config.model.device, config.model.dtype = "cpu", "float32"
    config.model.max_length, config.model.max_new_tokens = 256, 8
    config.readout.layers, config.readout.middle, config.readout.late = [0, 1, 2], [1], [2]
    config.training.rank, config.training.alpha = 4, 8
    config.training.epochs, config.training.effective_batch = 2, 4
    config.training.learning_rate = 0.01
    config.training.gradient_checkpointing = False
    return model, tokenizer, config.validate()


def smoke(output, upstream=None):
    import torch

    from .model import HFBackend
    from .readouts import VocabularyFilter
    from .training import fit_adapter

    model, tokenizer, config = tiny_components()
    backend = HFBackend.__new__(HFBackend)
    backend.config, backend.model, backend.tokenizer, backend.lens = config, model, tokenizer, None
    backend.vocab = VocabularyFilter(tokenizer, model.config.vocab_size)
    first = backend.generate("Problem: 2 3 total")
    readouts, concepts = backend.read("2 3 total", first)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "tiny_readout.json",
        {
            "synthetic": True,
            "random_weights": True,
            "generation": first,
            "readouts": readouts,
            "concepts": concepts,
        },
    )
    lrp = None
    if upstream:
        from .jpp import load_upstream

        load_upstream(upstream)
        from workspace_lens.lrp import apply_lrp_rules_for_mode

        x = torch.tensor([[12, 13, 14]])
        before = model(x).logits.detach()
        with apply_lrp_rules_for_mode(model, "rlens") as patched:
            after = model(x).logits.detach()
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
        lrp = {
            "forward_preserved": True,
            "norms": len(patched.ln_rule),
            "mlps": len(patched.mlp_rule),
        }
    examples = [{"prompt": f"Problem: {i} total", "target": "7"} for i in range(8)]
    result = fit_adapter(model, tokenizer, examples, config, 42, output / "training")
    summary = {
        "synthetic": True,
        "random_weights": True,
        "readout_positions": ["last_prompt", "first_answer"],
        "lrp": lrp,
        "training": result,
        "scientific_result": False,
    }
    write_json(output / "smoke.json", summary)
    return summary


class FixtureBackend:
    """Never presented as a model or J++ lens; deterministic plumbing check only."""

    def generate(self, prompt):
        index = int(re.search(r"Fixture (\d+)", prompt)[1])
        answer = index if index % 2 == 0 else index + 1
        return {
            "text": str(answer),
            "prompt_ids": [1],
            "answer_ids": [answer],
            "stop_reason": "fixture",
            "truncated": False,
        }

    def read(self, question, generation):
        rows = [
            {
                "layer": 11,
                "position": "last_prompt",
                "semantic": [{"token_id": 1, "text": "total", "score": 1.0, "rank": 1}],
                "filtered": [{"token_id": 1, "text": "total", "score": 1.0, "rank": 1}],
            }
        ]
        return {"logit": rows}, {"logit": {"middle": ["total"], "late": ["remaining"]}}


def demo(output):
    from .artifacts import load_cache
    from .evaluation import evaluate_predictions
    from .examples import build_examples
    from .pipeline import generate_split
    from .reporting import write_report

    output = Path(output)
    config = Config()
    config.data.train_count, config.data.dev_size, config.data.gate_size = 60, 20, 8
    config.training.seeds, config.evaluation.bootstrap_samples = [42], 200
    records = [
        {"question": f"Fixture {i}: return its index.", "answer": f"Synthetic reference. #### {i}"}
        for i in range(60)
    ]
    splits = split_training(records, config)
    run = {
        "schema": 1,
        "config": config.to_dict(),
        "synthetic": True,
        "splits": {k: fingerprint(v) for k, v in splits.items()},
    }
    if (output / "run.json").exists():
        from .artifacts import read_json

        if read_json(output / "run.json") != run:
            raise ValueError("Demo output contains another experiment")
    else:
        for name, rows in splits.items():
            write_json(output / "data" / f"{name}.json", rows)
        write_json(output / "run.json", run)
    generate_split(output, "dev_tune", limit=10, backend=FixtureBackend())
    generate_split(output, "dev_tune", backend=FixtureBackend())
    generate_split(output, "train", backend=FixtureBackend())
    _, training = load_cache(output / "base" / "logit" / "train")
    write_json(
        output / "synthetic_examples.json", build_examples(training, config.seed, method="logit")
    )
    _, base = load_cache(output / "base" / "logit" / "dev_tune")
    predictions = {}
    for arm in ["readout", "no_readout", "shuffled"]:
        predictions[(42, arm)] = [
            {
                "id": row["id"],
                "generation": {"text": row["first"]["text"]},
                "grade": grade(row["first"]["text"], row["gold"]),
            }
            for row in base
        ]
    result = evaluate_predictions(base, predictions, [42], bootstrap_samples=200)
    return write_report(output / "report", result, synthetic=True, split="dev_tune")
