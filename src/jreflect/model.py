"""Sequential greedy inference and exact-prefix activation replay."""

from contextlib import contextmanager

from .lenses import JppLens
from .prompts import messages
from .readouts import VocabularyFilter, band_concepts


@contextmanager
def capture_positions(model, layers, positions):
    captures = {}
    handles = []
    try:
        for layer in layers:

            def hook(_module, _inputs, output, index=layer):
                hidden = output if hasattr(output, "shape") else output[0]
                captures[index] = hidden[:, positions, :].detach().clone()

            handles.append(model.model.layers[layer].register_forward_hook(hook))
        yield captures
    finally:
        for handle in handles:
            handle.remove()


class HFBackend:
    def __init__(self, config, lens_path=None, adapter=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if config.model.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use a CUDA host for the configured experiment.")
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.name, revision=config.model.revision
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                config.model.name,
                revision=config.model.revision,
                torch_dtype=getattr(torch, config.model.dtype),
                attn_implementation="sdpa",
            )
            .to(config.model.device)
            .eval()
        )
        if not hasattr(self.model.model, "layers") or not hasattr(self.model.model, "norm"):
            raise ValueError("Only the tested Qwen-style decoder layout is supported")
        if max(config.readout.layers) >= len(self.model.model.layers) - 1:
            raise ValueError("Readout source layer must precede the final target block")
        self.lens = (
            JppLens(lens_path, config, self.model.config.hidden_size, len(self.model.model.layers))
            if lens_path
            else None
        )
        self.vocab = VocabularyFilter(self.tokenizer, self.model.config.vocab_size)
        if adapter is not None:
            from peft import PeftModel

            if lens_path:
                raise ValueError(
                    "Primary readouts must come from the frozen base, without an adapter"
                )
            self.model = PeftModel.from_pretrained(self.model, adapter).eval()

    def prompt_ids(self, prompt):
        ids = self.tokenizer.apply_chat_template(
            messages(prompt), tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if len(ids) + self.config.model.max_new_tokens > self.config.model.max_length:
            raise ValueError(
                "Prompt exceeds the frozen context limit; no silent truncation is allowed"
            )
        return ids

    def generate(self, prompt):
        import torch

        prompt_ids = self.prompt_ids(prompt)
        device = next(self.model.parameters()).device
        inputs = torch.tensor([prompt_ids], device=device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.config.model.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id,
            )
        answer_ids = output[0, len(prompt_ids) :].tolist()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        eos_ids.add(self.tokenizer.eos_token_id)
        ended = bool(answer_ids and answer_ids[-1] in eos_ids)
        return {
            "text": self.tokenizer.decode(answer_ids, skip_special_tokens=True),
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
            "stop_reason": "eos" if ended else "length",
            "truncated": not ended,
        }

    def read(self, question, generation):
        import torch

        prefix = generation["prompt_ids"]
        answer = generation["answer_ids"]
        if not answer:
            raise ValueError("Generation returned no token to replay")
        inputs = torch.tensor([prefix + answer[:1]], device=next(self.model.parameters()).device)
        positions = [len(prefix) - 1, len(prefix)]
        with (
            torch.no_grad(),
            capture_positions(self.model, self.config.readout.layers, positions) as hidden,
        ):
            # Call the decoder only: avoid allocating full sequence x vocabulary logits.
            self.model.model(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False
            )
        readouts = {"logit": []}
        if self.lens:
            readouts["jpp"] = []
        with torch.no_grad():
            for layer in self.config.readout.layers:
                residual = hidden.pop(layer)
                for method in readouts:
                    transported = (
                        self.lens.transport(residual, layer) if method == "jpp" else residual
                    )
                    logits = self.model.lm_head(
                        self.model.model.norm(transported.to(residual.dtype))
                    )[0]
                    for position, scores in zip(
                        ["last_prompt", "first_answer"], logits, strict=True
                    ):
                        ranked = self.vocab.rank(scores, question, self.config.readout.top_k)
                        readouts[method].append({"layer": layer, "position": position, **ranked})
        concepts = {
            method: band_concepts(rows, self.config.readout) for method, rows in readouts.items()
        }
        return readouts, concepts
