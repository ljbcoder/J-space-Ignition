"""Inputs are deliberately narrow: no references or correctness fields."""

ARMS = ("readout", "no_readout", "shuffled")
SYSTEM = "Solve the problem. Reply with a number only, without explanation."


def first_prompt(question):
    return f"Problem: {question}\nGive your answer as a number only."


def correction_prompt(question, first_answer, arm, concepts=None):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    prompt = f"Problem: {question}\nYour previous answer: {first_answer}\n"
    if arm != "no_readout":
        if concepts is None or set(concepts) != {"middle", "late"}:
            raise ValueError("Readout arms require middle and late concepts")
        prompt += (
            "While you answered, these concepts were most active in your\n"
            "internal workspace (strongest first):\n"
            f"  middle layers: {', '.join(concepts['middle']) or '(none)'}\n"
            f"  late layers:   {', '.join(concepts['late']) or '(none)'}\n"
        )
    return prompt + "Give your final answer as a number only."


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
