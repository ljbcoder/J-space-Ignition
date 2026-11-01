"""Target-only LoRA training, deterministic example order, resumable optimizer steps."""

import math
import random
import time
from pathlib import Path

from .artifacts import exclusive, fingerprint, load_cache, provenance, read_json, write_json
from .data import load_run
from .examples import build_examples
from .prompts import ARMS, messages
from .sanity import checked_base, require_gate


def encode_target(tokenizer, prompt, target, max_length):
    prefix = tokenizer.apply_chat_template(
        messages(prompt), tokenize=True, add_generation_prompt=True, return_dict=False
    )
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids or tokenizer.eos_token_id is None:
        raise ValueError("Training needs target tokens and an assistant EOS token")
    ids = prefix + target_ids + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        raise ValueError("Training example exceeds context limit; refusing to truncate targets")
    return {
        "input_ids": ids,
        "labels": [-100] * len(prefix) + target_ids + [tokenizer.eos_token_id],
    }


def collate_targets(batch, pad_id):
    import torch

    length = max(len(x["input_ids"]) for x in batch)
    return {
        "input_ids": torch.tensor(
            [r["input_ids"] + [pad_id] * (length - len(r["input_ids"])) for r in batch]
        ),
        "labels": torch.tensor([r["labels"] + [-100] * (length - len(r["labels"])) for r in batch]),
        "attention_mask": torch.tensor(
            [[1] * len(r["input_ids"]) + [0] * (length - len(r["input_ids"])) for r in batch]
        ),
    }


def prepare_examples(root):
    root = Path(root)
    if (root / "protocol.json").exists():
        raise ValueError("Cannot change training data after protocol freeze")
    _, config = load_run(root)
    identity, rows = checked_base(root, "train")
    gate = require_gate(root, identity["lens"])
    examples = build_examples(rows, config.seed)
    payload = {"base_hash": fingerprint(rows), "gate_hash": fingerprint(gate), **examples}
    path = root / "examples.json"
    if path.exists() and read_json(path) != payload:
        raise ValueError("Examples already exist with incompatible inputs")
    write_json(path, payload)
    return {"instances_per_arm": len(examples["arms"]["readout"]), "unique_train_items": len(rows)}


def fit_adapter(model, tokenizer, examples, config, seed, output, *, max_steps=None):
    """Also used by the real tiny-model integration test. No remote access here."""
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from peft.utils.save_and_load import get_peft_model_state_dict
    from transformers import get_cosine_schedule_with_warmup, set_seed

    set_seed(seed)
    if not examples or (max_steps is not None and max_steps < 1):
        raise ValueError("Training requires examples and a positive step limit")
    t = config.training
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if not set(t.target_modules) <= names:
        raise ValueError("Model lacks one or more configured LoRA projections")
    model = get_peft_model(
        model,
        LoraConfig(
            r=t.rank,
            lora_alpha=t.alpha,
            lora_dropout=t.dropout,
            target_modules=t.target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    model.config.use_cache = False
    if t.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()
    encoded = [
        encode_target(tokenizer, r["prompt"], r["target"], config.model.max_length)
        for r in examples
    ]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=t.learning_rate
    )
    batches_per_epoch = math.ceil(len(encoded) / t.effective_batch)
    total_steps = batches_per_epoch * t.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * t.warmup_ratio), total_steps
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "optimizer.pt"
    identity = fingerprint({"examples": examples, "config": config.to_dict(), "seed": seed})
    completed, losses = 0, []
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Training checkpoint belongs to different examples or settings")
        set_peft_model_state_dict(model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        completed, losses = saved["completed"], saved["losses"]
        torch.set_rng_state(saved["rng"])
        if torch.cuda.is_available() and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    device = next(model.parameters()).device
    step, started = 0, time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(t.epochs):
        order = list(range(len(encoded)))
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), t.effective_batch):
            group = [encoded[i] for i in order[start : start + t.effective_batch]]
            step += 1
            if step <= completed:
                continue
            if max_steps is not None and step > max_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            # Token-normalized loss across microbatches, including the final partial group.
            target_count = sum(sum(v != -100 for v in row["labels"][1:]) for row in group)
            accumulated = 0.0
            for offset in range(0, len(group), t.micro_batch):
                batch = collate_targets(
                    group[offset : offset + t.micro_batch],
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id,
                )
                batch = {k: v.to(device) for k, v in batch.items()}
                weight = (batch["labels"][:, 1:] != -100).sum() / target_count
                loss = model(**batch).loss * weight
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                accumulated += float(loss.detach())
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(accumulated)
            completed = step
            saved = {
                "identity": identity,
                "adapter": {
                    k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()
                },
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "completed": completed,
                "losses": losses,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            temporary = output / "optimizer.tmp"
            torch.save(saved, temporary)
            temporary.replace(state_path)
            print(
                f"seed {seed}: optimizer step {completed}/{total_steps}, loss={accumulated:.4f}",
                flush=True,
            )
        if max_steps is not None and completed >= max_steps:
            break
    complete = completed == total_steps
    if complete:
        model.save_pretrained(output / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(output / "adapter")
    summary = {
        "completed_steps": completed,
        "total_steps": total_steps,
        "complete": complete,
        "losses": losses,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    write_json(output / "training.json", summary)
    return summary


def train(root, arm, seed, *, max_steps=None):
    from .model import HFBackend

    root = Path(root)
    run, config = load_run(root)
    if (root / "protocol.json").exists():
        raise ValueError("Training is closed after protocol freeze")
    if arm not in ARMS or seed not in config.training.seeds:
        raise ValueError("Arm or seed is outside the configured experiment")
    gate = require_gate(root)
    examples = read_json(root / "examples.json")
    _, base = load_cache(root / "base" / "jpp" / "train")
    if examples["gate_hash"] != fingerprint(gate) or examples["base_hash"] != fingerprint(base):
        raise ValueError("Training examples have stale provenance")
    output = root / "adapters" / str(seed) / arm
    identity = {
        "run": fingerprint(run),
        "examples": fingerprint(examples),
        "arm": arm,
        "seed": seed,
        "provenance": provenance(),
    }
    with exclusive(output / ".lock"):
        if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
            raise ValueError("Training software or inputs changed; start a new run")
        write_json(output / "identity.json", identity)
        backend = HFBackend(config)
        return fit_adapter(
            backend.model,
            backend.tokenizer,
            examples["arms"][arm],
            config,
            seed,
            output,
            max_steps=max_steps,
        )
