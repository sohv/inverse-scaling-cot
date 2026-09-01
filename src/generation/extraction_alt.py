"""Independent strict answer parser for extraction-robustness checks (revision item 17).

Deliberately stricter than src.generation.runner.extract_answer: it anchors only on a
parenthesised letter and refuses to guess. Disagreement with the permissive parser is the
quantity of interest, so this parser must never silently fall back to a heuristic.
"""

import logging
import re

LOGGER = logging.getLogger(__name__)

# letter immediately at the start of the completion, closing paren required: "B)" / "(B)"
STRICT_LEADING_PATTERN = re.compile(r"^\s*\(?([A-E])\)")

# explicit phrase with both parens: "the answer is (B)"
STRICT_PHRASE_PATTERN = re.compile(r"the\s+answer\s+is\s*\(([A-E])\)", re.IGNORECASE)


def extract_answer_strict(text: str, choice_labels: list[str]) -> str | None:
    """Extract an answer letter using only unambiguous parenthesised forms.

    No standalone-(A) scan over the whole text, no bare-letter fallback, no
    first-character heuristic. Returns None whenever the format is not exact.
    """
    valid = {label.upper() for label in choice_labels}

    leading = STRICT_LEADING_PATTERN.match(text)
    if leading:
        letter = leading.group(1).upper()
        if letter in valid:
            return letter

    phrases = STRICT_PHRASE_PATTERN.findall(text)
    if phrases:
        letter = phrases[-1].upper()
        if letter in valid:
            return letter

    return None


PARSERS = {
    "permissive": None,  # resolved to runner.extract_answer_no_cot by callers
    "strict": extract_answer_strict,
}
