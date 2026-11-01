"""One balanced manifest, shared by all three arms and all adapter seeds."""

import random

from .prompts import ARMS, correction_prompt


def derangement(ids, seed):
    if len(ids) != len(set(ids)) or len(ids) < 2:
        raise ValueError("Donor assignment needs at least two distinct source IDs")
    rng = random.Random(seed)
    donors = list(ids)
    while True:
        rng.shuffle(donors)
        if all(a != b for a, b in zip(ids, donors, strict=True)):
            return dict(zip(ids, donors, strict=True))


def build_examples(rows, seed, *, balance=True, method="jpp"):
    if not rows or len({r["split"] for r in rows}) != 1:
        raise ValueError("Examples must come from one nonempty split")
    if balance and rows[0]["split"] != "train":
        raise ValueError("Only the training split may be balanced")
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate source IDs")
    donors = derangement(sorted(by_id), seed)
    chosen = sorted(by_id)
    if balance:
        wrong = [key for key in chosen if not by_id[key]["first_grade"]["correct"]]
        right = [key for key in chosen if by_id[key]["first_grade"]["correct"]]
        if not wrong or not right:
            raise ValueError("Balancing requires both correct and incorrect first answers")
        rng = random.Random(seed)
        right = (
            rng.sample(right, len(wrong))
            if len(right) >= len(wrong)
            else rng.choices(right, k=len(wrong))
        )
        chosen = wrong + right
        rng.shuffle(chosen)
    arms = {arm: [] for arm in ARMS}
    for index, source_id in enumerate(chosen):
        row = by_id[source_id]
        for arm in ARMS:
            donor = donors[source_id] if arm == "shuffled" else source_id
            concepts = None if arm == "no_readout" else by_id[donor]["concepts"][method]
            arms[arm].append(
                {
                    "id": f"{row['split']}:{index:06d}",
                    "source_id": source_id,
                    "split": row["split"],
                    "arm": arm,
                    "donor_id": donor if arm == "shuffled" else None,
                    "prompt": correction_prompt(
                        row["question"], row["first"]["text"], arm, concepts
                    ),
                    "target": row["gold"],
                }
            )
    return {"arms": arms, "donors": donors, "method": method, "balanced": balance}
