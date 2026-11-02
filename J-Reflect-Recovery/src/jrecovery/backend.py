"""Frozen HF model, generation-prefill captures, and native J++ checkpoint readout."""

import os
import time
from contextlib import contextmanager
from pathlib import Path

from .feedback import first_messages
from .storage import file_digest


def text_decoder(model):
    for path in ("model", "model.language_model", "language_model"):
        module = model
        try:
            for part in path.split("."):
                module = getattr(module, part)
        except AttributeError:
            continue
        if all(hasattr(module, name) for name in ("layers", "norm", "embed_tokens")):
            return module
    raise ValueError("Unsupported model text-decoder layout")


@contextmanager
def capture_prefill(decoder, layers, prompt_length):
    captures, handles = {}, []
    try:
        for layer in layers:

            def hook(_module, _inputs, output, index=layer):
                hidden = output if hasattr(output, "shape") else output[0]
                if index not in captures and hidden.shape[1] == prompt_length:
                    captures[index] = hidden[0, -1, :].detach().float().cpu().clone()

            handles.append(decoder.layers[layer].register_forward_hook(hook))
        yield captures
    finally:
        for handle in handles:
            handle.remove()


class NativeLens:
    def __init__(self, path, config, hidden_size, layer_count):
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        metadata = checkpoint.get("config", {})
        if metadata.get("hf_model_name") != config.model.name:
            raise ValueError("Lens was fitted for a different model")
        if metadata.get("relative_end_transport_layer") != -1:
            raise ValueError("Lens must target the final decoder block")
        if metadata.get("lens_type") != "jacobian":
            raise ValueError("Expected a native J++ Jacobian checkpoint")
        maps = checkpoint.get("parameters", {}).get("jacobians", checkpoint.get("J", {}))
        self.maps = {}
        for layer in config.lens.layers:
            if layer not in maps or not 0 <= layer < layer_count - 1:
                raise ValueError(f"Missing or invalid source layer {layer}")
            matrix = maps[layer]
            if (
                tuple(matrix.shape) != (hidden_size, hidden_size)
                or not torch.isfinite(matrix).all()
            ):
                raise ValueError(f"Invalid lens matrix at layer {layer}")
            self.maps[layer] = matrix.float()
        self.sha256 = file_digest(path)
        self.metadata = metadata

    def transport(self, residual, layer):
        return residual.float() @ self.maps[layer].to(residual.device).T


class HFBackend:
    def __init__(self, config, *, model=None, tokenizer=None, lens_path=None):
        import torch

        self.config = config
        if model is None:
            if config.model.device.startswith("cuda") and not torch.cuda.is_available():
                raise ValueError(
                    "CUDA is unavailable. Run real inference on a suitable GPU host; use demo locally."
                )
            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

            if config.model.name != "Qwen/Qwen3.5-9B":
                raise ValueError("The production loader currently supports Qwen/Qwen3.5-9B only")
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.name, revision=config.model.revision
            )
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                config.model.name,
                revision=config.model.revision,
                dtype=getattr(torch, config.model.dtype),
                attn_implementation="sdpa",
            ).to(config.model.device)
            lens_path = hf_hub_download(
                config.lens.repo, config.lens.filename, revision=config.lens.revision
            )
        if tokenizer is None or lens_path is None:
            raise ValueError(
                "A tokenizer and real lens checkpoint are required; no logit-lens fallback"
            )
        self.model, self.tokenizer = model.eval(), tokenizer
        self.model.requires_grad_(False)
        self.decoder = text_decoder(model)
        text_config = model.config.get_text_config()
        self.lens = NativeLens(lens_path, config, text_config.hidden_size, len(self.decoder.layers))
        self.device = self.decoder.embed_tokens.weight.device
        self.head = model.get_output_embeddings()
        self.token_text = []
        vocab_size = self.head.weight.shape[0]
        for start in range(0, vocab_size, 4096):
            self.token_text.extend(
                tokenizer.batch_decode(
                    [[i] for i in range(start, min(start + 4096, vocab_size))],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
        special = set(tokenizer.all_special_ids)
        self.allowed = [
            i not in special and any(c.isalnum() for c in text)
            for i, text in enumerate(self.token_text)
        ]
        self.numeric_tokens = {
            text.strip()
            for i, text in enumerate(self.token_text)
            if self.allowed[i] and text.strip().isdigit()
        }
        self.identity = {
            "kind": "hf",
            "model": config.model.name,
            "model_revision": config.model.revision,
            "model_class": type(model).__name__,
            "lens_sha256": self.lens.sha256,
            "lens_revision": config.lens.revision,
            "device": str(self.device),
            "dtype": str(self.head.weight.dtype),
            "capture": "last_prompt_token_generation_prefill",
        }

    def supports(self, target):
        return target in self.numeric_tokens

    def _generate(self, messages, capture=False):
        import torch

        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            enable_thinking=False,
        )
        if len(ids) + self.config.model.max_new_tokens > self.config.model.max_context:
            raise ValueError("Prompt exceeds the frozen context limit; truncation is prohibited")
        inputs = torch.tensor([ids], device=self.device)
        start = time.perf_counter()
        layers = self.config.lens.layers if capture else []
        with torch.inference_mode(), capture_prefill(self.decoder, layers, len(ids)) as captures:
            output = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=self.config.model.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id,
            )
        generated = output[0, len(ids) :].tolist()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tokenizer.eos_token_id}
        ended = bool(generated and generated[-1] in eos_ids)
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "prompt_ids": ids,
            "answer_ids": generated,
            "prompt_tokens": len(ids),
            "generated_tokens": len(generated),
            "truncated": not ended,
            "stop_reason": "eos" if ended else "length",
            "seconds": time.perf_counter() - start,
        }, captures

    def complete(self, messages):
        return self._generate(messages)[0]

    def first(self, question):
        import torch

        messages = first_messages(question)
        generation, captures = self._generate(messages, capture=True)
        if set(captures) != set(self.config.lens.layers):
            raise ValueError("Generation did not capture every requested prefill layer")
        readouts = {"jpp": [], "logit": []}
        start = time.perf_counter()
        with torch.inference_mode():
            mask = torch.tensor(self.allowed, dtype=torch.bool, device=self.head.weight.device)
            if int(mask.sum()) < self.config.lens.top_k:
                raise ValueError("Vocabulary has fewer semantic tokens than top_k")
            for layer in self.config.lens.layers:
                residual = captures[layer].to(self.device)
                for method in readouts:
                    transported = (
                        self.lens.transport(residual, layer) if method == "jpp" else residual
                    )
                    logits = self.head(
                        self.decoder.norm(transported.to(self.head.weight.dtype))
                    ).float()
                    if not torch.isfinite(logits).all():
                        raise ValueError("Non-finite readout logits")
                    values, indices = logits.masked_fill(~mask, -float("inf")).topk(
                        self.config.lens.top_k
                    )
                    readouts[method].append(
                        {
                            "layer": layer,
                            "position": len(generation["prompt_ids"]) - 1,
                            "tokens": [
                                {"id": i, "text": self.token_text[i], "score": v}
                                for i, v in zip(indices.tolist(), values.tolist(), strict=True)
                            ],
                        }
                    )
        return {
            "messages": messages,
            "generation": generation,
            "readouts": readouts,
            "readout_seconds": time.perf_counter() - start,
        }, captures

    def save_activations(self, path, captures, prompt_ids):
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "activations": captures,
                "prompt_ids": prompt_ids,
                "position": len(prompt_ids) - 1,
                "backend": self.identity,
            },
            temporary,
        )
        os.replace(temporary, path)
        return file_digest(path)
