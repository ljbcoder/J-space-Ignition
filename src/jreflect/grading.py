"""Strict primary grading: one bare number, compared exactly."""

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
BARE = re.compile(rf"{NUMBER}\Z")


def parse_number(text):
    text = str(text).strip()
    if not BARE.fullmatch(text):
        return None
    try:
        return Fraction(Decimal(text.replace(",", "")))
    except (InvalidOperation, ValueError):
        return None


def gold_answer(solution):
    if "####" not in solution:
        raise ValueError("Reference solution has no #### answer delimiter")
    answer = solution.rsplit("####", 1)[1].strip()
    if parse_number(answer) is None:
        raise ValueError(f"Invalid reference answer: {answer!r}")
    return answer.replace(",", "")


def grade(text, gold, *, truncated=False):
    target = parse_number(gold)
    if target is None:
        raise ValueError("Invalid gold number")
    parsed = parse_number(text)
    valid = parsed is not None and not truncated
    # Diagnostic only. This never changes primary correctness.
    candidates = re.findall(NUMBER, text)
    permissive = parse_number(candidates[-1]) if candidates else None
    return {
        "parsed": str(parsed) if parsed is not None else None,
        "valid": valid,
        "correct": valid and parsed == target,
        "permissive_correct": permissive == target and not truncated,
    }


def changed(first, second):
    a, b = parse_number(first), parse_number(second)
    if a is not None and b is not None:
        return a != b
    return " ".join(first.split()) != " ".join(second.split())
