"""Ordinary feedback never receives the answer key. Ablations explicitly do."""

import copy
import json
import random

CONDITIONS = ("retry_only", "error_only", "shuffled_jpp", "authentic_jpp")
ABLATIONS = ("remove_target", "replace_target", "sham_remove")
SYSTEM = "Solve the problem. Return only Answer: <integer>, with no explanation."


def first_messages(question):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def payload(readouts):
    """Same fixed layers/top-k slots; no oracle ranking, filtering, or interpretation."""
    return [
        {"layer": row["layer"], "tokens": [t["text"] for t in row["tokens"]]}
        for row in readouts["jpp"]
    ]


def hits(readouts, target, method="jpp"):
    ranks = [
        rank
        for row in readouts[method]
        for rank, token in enumerate(row["tokens"], 1)
        if token["text"].strip() == target
    ]
    return min(ranks) if ranks else None


def correction_messages(first, condition, donor=None):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if condition == "retry_only" or first["grade"]["correct"]:
        instruction = "Review your previous answer. Keep it if correct; otherwise correct it."
    else:
        instruction = "Your previous answer was incorrect. Solve the problem again."
    notes = None
    if condition == "authentic_jpp":
        notes = payload(first["readouts"])
    elif condition == "shuffled_jpp":
        if donor is None or donor["id"] == first["id"]:
            raise ValueError("Shuffled feedback requires a different first attempt")
        notes = payload(donor["readouts"])
    return with_feedback(first, instruction, notes)


def with_feedback(first, instruction, notes):
    content = instruction
    if notes is not None:
        content += (
            "\nAutomatically extracted diagnostic tokens follow. They may be noisy or irrelevant.\n"
            + json.dumps(notes, ensure_ascii=False)
        )
    content += "\nReturn only Answer: <integer>."
    return copy.deepcopy(first["messages"]) + [
        {"role": "assistant", "content": first["generation"]["text"]},
        {"role": "user", "content": content},
    ]


def ablation_messages(first, target, condition, seed):
    if condition not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {condition}")
    notes = payload(first["readouts"])
    locations = [
        (i, j)
        for i, row in enumerate(notes)
        for j, text in enumerate(row["tokens"])
        if text.strip() == target
    ]
    if not locations:
        raise ValueError("Ablation requires a detected target token")
    if condition == "sham_remove":
        candidates = [
            (i, j)
            for i, row in enumerate(notes)
            for j, text in enumerate(row["tokens"])
            if text.strip() != target
        ]
        if len(candidates) < len(locations):
            raise ValueError("Not enough non-target slots for a matched sham ablation")
        random.Random(seed).shuffle(candidates)
        locations = candidates[: len(locations)]
    wrong = str(int(target) + 1)
    if len(wrong) != len(target):
        wrong = str(int(target) - 1)
    for i, j in locations:
        notes[i]["tokens"][j] = wrong if condition == "replace_target" else "[removed]"
    messages = with_feedback(
        first, "Your previous answer was incorrect. Solve the problem again.", notes
    )
    return messages, {
        "edited_slots": [list(position) for position in locations],
        "replacement": wrong if condition == "replace_target" else "[removed]",
    }
