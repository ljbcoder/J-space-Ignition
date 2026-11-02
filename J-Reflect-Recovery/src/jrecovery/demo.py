"""Deterministic simulated fixtures. These outputs never measure model behavior."""

import json

from .feedback import first_messages
from .storage import digest


class DemoBackend:
    def __init__(self, config, items):
        self.config = config
        self.items = {p["question"]: p for p in items}
        self.identity = {"kind": "demo", "warning": "SIMULATED_NOT_RESEARCH"}

    def supports(self, target):
        return True

    def generation(self, text, messages):
        return {
            "text": text,
            "prompt_ids": [],
            "answer_ids": [],
            "prompt_tokens": sum(len(m["content"].split()) for m in messages),
            "generated_tokens": len(text.split()),
            "truncated": False,
            "stop_reason": "simulated",
            "seconds": 0.0,
        }

    def first(self, question):
        item = self.items[question]
        key = int(digest(question)[:8], 16)
        answer = item["gold"] if key % 3 == 0 else str(int(item["gold"]) + 1)
        messages = first_messages(question)
        readouts = {"jpp": [], "logit": []}
        for method in readouts:
            for layer in self.config.lens.layers:
                tokens = [f"note{i}" for i in range(self.config.lens.top_k)]
                if (method == "jpp" and key % 2 == 0) or (method == "logit" and key % 7 == 0):
                    tokens[0] = item["target"]
                readouts[method].append(
                    {
                        "layer": layer,
                        "position": 0,
                        "tokens": [
                            {"id": i, "text": token, "score": float(-i)}
                            for i, token in enumerate(tokens)
                        ],
                    }
                )
        return {
            "messages": messages,
            "generation": self.generation(f"Answer: {answer}", messages),
            "readouts": readouts,
            "readout_seconds": 0.0,
        }, {}

    def complete(self, messages):
        item = self.items[messages[1]["content"]]
        key = int(digest(json.dumps(messages, sort_keys=True))[:8], 16)
        # Deliberately artificial, varying outcomes to exercise paired statistics.
        answer = item["gold"] if key % 2 == 0 else str(int(item["gold"]) + 2)
        return self.generation(f"Answer: {answer}", messages)
