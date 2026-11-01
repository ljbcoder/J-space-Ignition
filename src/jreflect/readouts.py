"""Filtered vocabulary rankings and scale-independent band aggregation."""

import re
from collections import defaultdict


def words(text):
    return set(re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE))


class VocabularyFilter:
    def __init__(self, tokenizer, vocab_size):
        self.text = [tokenizer.decode([i]) if i < len(tokenizer) else "" for i in range(vocab_size)]
        special = set(tokenizer.all_special_ids)
        self.allowed = [
            i not in special and any(c.isalnum() for c in token)
            for i, token in enumerate(self.text)
        ]
        self.word_tokens = defaultdict(list)
        for i, token in enumerate(self.text):
            # Numeric tokens are deliberately retained, even if copied from the question.
            if not any(c.isdigit() for c in token):
                for word in words(token):
                    self.word_tokens[word].append(i)

    def rank(self, logits, question, k):
        import torch

        scores = logits.detach().float().reshape(-1).cpu()
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite readout logits")
        allowed = torch.tensor(self.allowed, dtype=torch.bool)

        def top(mask):
            ids = torch.where(mask)[0]
            if not len(ids):
                return []
            # Stable sort resolves equal-score tokens by vocabulary ID.
            order = torch.argsort(scores[ids], descending=True, stable=True)[:k]
            return [
                {"token_id": int(i), "text": self.text[i], "rank": rank, "score": float(scores[i])}
                for rank, i in enumerate(ids[order].tolist(), 1)
            ]

        semantic = top(allowed)
        for word in words(question):
            allowed[self.word_tokens.get(word, [])] = False
        return {"semantic": semantic, "filtered": top(allowed)}


def band_concepts(readouts, config):
    result = {}
    for band, layers in [("middle", config.middle), ("late", config.late)]:
        scores = defaultdict(float)
        for readout in readouts:
            if readout["layer"] in layers:
                seen = set()
                for token in readout["filtered"]:
                    text = " ".join(token["text"].casefold().split())
                    if text and text not in seen:
                        scores[text] += 1 / token["rank"]
                        seen.add(text)
        result[band] = sorted(scores, key=lambda text: (-scores[text], text))[: config.concepts]
    return result


def probe_features(row, method):
    # Scores are retained per position/layer; scaling is learned only inside CV folds.
    return {
        f"{r['layer']}:{r['position']}:{t['token_id']}": t["score"]
        for r in row["readouts"][method]
        for t in r["filtered"]
    }


def content_overlap(row, reference):
    target = words(reference.split("####")[0]) - words(row["question"])
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "is",
        "are",
        "so",
        "then",
        "he",
        "she",
        "it",
        "his",
        "her",
        "of",
        "to",
        "in",
    }
    target -= stop
    result = {}
    for method, readouts in row["readouts"].items():
        seen = set().union(*(words(t["text"]) for r in readouts for t in r["filtered"][:10]))
        result[method] = {
            "hits": sorted(target & seen),
            "targets": sorted(target),
            "recall": len(target & seen) / len(target) if target else None,
        }
    return result
