"""Program-verified arithmetic and variable traces with disjoint template splits."""

import random
import re

from .storage import digest

TEMPLATES = {
    "dev": ["add_multiply", "multiply_subtract", "track_add"],
    "test": ["subtract_multiply", "multiply_add", "track_subtract"],
}


def numbers(text):
    return set(re.findall(r"(?<![\w.])-?\d+(?![\w.])", text))


def make_problem(template, rng, tail_steps=0):
    a, b, c = rng.randint(21, 89), rng.randint(11, 39), rng.randint(3, 9)
    d = rng.randint(11, 37)
    if template == "add_multiply":
        target, gold = a + b, (a + b) * c
        question = (
            f"Start with {a}. Add {b}, then multiply the result by {c}. What is the final number?"
        )
        steps = [f"{a} + {b} = {target}", f"{target} * {c} = {gold}"]
    elif template == "subtract_multiply":
        a += 100
        target, gold = a - b, (a - b) * c
        question = f"Start with {a}. Subtract {b}, then multiply the result by {c}. What is the final number?"
        steps = [f"{a} - {b} = {target}", f"{target} * {c} = {gold}"]
    elif template in {"multiply_subtract", "multiply_add"}:
        target = a * c
        subtract = template == "multiply_subtract"
        gold = target - b if subtract else target + b
        operation, symbol = ("subtract", "-") if subtract else ("add", "+")
        question = f"Multiply {a} by {c}. Then {operation} {b}. What is the final number?"
        steps = [f"{a} * {c} = {target}", f"{target} {symbol} {b} = {gold}"]
    else:
        subtract = template == "track_subtract"
        target = a * c - b if subtract else a * c + b
        gold = target * d
        symbol = "-" if subtract else "+"
        question = f"Execute these assignments in order: x = {a}; x = x * {c}; x = x {symbol} {b}; y = x * {d}. What is y?"
        steps = [f"x = {a}", f"x = {a * c}", f"x = {target}", f"y = {gold}"]
    variable = template.startswith("track")
    ending = ". What is y?" if variable else " What is the final number?"
    question = question.removesuffix(ending)
    for index in range(tail_steps):
        operand = rng.randint(11, 37)
        symbol = ["*", "+", "-"][index % 3]
        previous = gold
        gold = (
            gold * operand if symbol == "*" else gold + operand if symbol == "+" else gold - operand
        )
        steps.append(f"y = {gold}" if variable else f"{previous} {symbol} {operand} = {gold}")
        if variable:
            question += f"; y = y {symbol} {operand}"
        else:
            operation = {
                "*": f"multiply the result by {operand}",
                "+": f"add {operand}",
                "-": f"subtract {operand}",
            }[symbol]
            question += f" Next {operation}."
    question += ending
    return {
        "template": template,
        "question": question,
        "target": str(target),
        "gold": str(gold),
        "oracle_steps": steps,
    }


def generate(config):
    rng = random.Random(config.seed)
    seen = set()
    result = {}
    for split in ["dev", "test"]:
        rows = []
        count = getattr(config.data, f"{split}_count")
        attempts = 0
        while len(rows) < count:
            attempts += 1
            if attempts > count * 100:
                raise ValueError("Requested more unique questions than this generator can supply")
            template = TEMPLATES[split][len(rows) % len(TEMPLATES[split])]
            row = make_problem(template, rng, config.data.tail_steps)
            if (
                row["question"] in seen
                or row["target"] in numbers(row["question"])
                or row["target"] == row["gold"]
                or not 10 <= int(row["target"]) <= 999
            ):
                continue
            seen.add(row["question"])
            rows.append({"id": f"{split}-{digest(row['question'])[:16]}", "split": split, **row})
        result[split] = rows
    return result


def grade(generation, gold):
    match = re.fullmatch(r"\s*Answer:\s*([+-]?\d+)\s*", generation["text"])
    if generation.get("truncated") or not match:
        return {"status": "format_failure", "correct": False, "answer": None}
    answer = str(int(match.group(1)))
    return {
        "status": "correct" if answer == gold else "wrong",
        "correct": answer == gold,
        "answer": answer,
    }


def donor_map(rows, seed):
    """Derange within template; do not inspect answers, gold labels, or readout contents."""
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        groups.setdefault(row["template"], []).append(row["id"])
    result = {}
    singletons = []
    for ids in groups.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        if len(ids) == 1:
            singletons.extend(ids)
        else:
            result.update(zip(ids, ids[1:] + ids[:1], strict=True))
    all_ids = sorted(row["id"] for row in rows)
    if len(all_ids) < 2:
        raise ValueError("Shuffled readouts need at least two first attempts")
    for source in singletons:
        result[source] = rng.choice([item for item in all_ids if item != source])
    return result
