"""Extract explicit scalar answers without using the reference to choose one."""

from fractions import Fraction
import re


SCORER_VERSION = "scalar-final-v2"
NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?"


def extract_answer(response):
    clean = response.replace("−", "-").replace("**", "").replace("__", "").replace("`", "")
    # Use the last explicit answer label, rather than a number in the reasoning.
    labels = list(re.finditer(r"final\s+answer\s*:", clean, flags=re.IGNORECASE))
    if not labels:
        return None, "missing_final_answer_label"
    tail = clean[labels[-1].end():].strip().splitlines()
    if not tail:
        return None, "missing_final_answer_value"
    answer = tail[0].replace("*", "").strip()
    answer = re.sub(r"\\(?:dfrac|frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", r"\1/\2", answer)
    answer = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", answer)
    for marker in ("\\(", "\\)", "\\[", "\\]", "$", "£", "€"):
        answer = answer.replace(marker, "")
    answer = answer.strip()
    match = re.match(NUMBER, answer)
    if not match:
        return None, "unsupported_final_answer_format"
    remainder = answer[match.end():]
    if re.search(r"\d", remainder):
        return None, "multiple_numbers_in_final_answer"
    # Do not accept an arithmetic expression as if its first operand were the answer.
    if re.search(r"[=+*/^]|\\(?:times|div)|\s-\s", remainder):
        return None, "unevaluated_expression"
    value = match.group().replace(",", "").replace(" ", "")
    try:
        Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None, "invalid_number"
    return value, "final_answer_label"


def score_response(response, reference):
    predicted, method = extract_answer(response)
    correct = predicted is not None and Fraction(predicted) == Fraction(str(reference).replace(",", ""))
    return {"predicted_answer": predicted, "reference_answer": str(reference),
            "parse_failed": predicted is None, "correct": correct,
            "answer_parse_method": method, "scorer_version": SCORER_VERSION}
