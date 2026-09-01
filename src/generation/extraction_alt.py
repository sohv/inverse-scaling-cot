"""Independent strict answer parser for extraction-robustness checks (revision item 17).

Stricter than src.generation.runner.extract_answer: it accepts only an unambiguous
parenthesised option and never guesses. It is deliberately NOT restricted to A-E, and it
is case-insensitive -- the 200-row manual audit (results/extraction/manual_audit_summary.json)
showed that an [A-E]-only, case-sensitive version disagreed with the permissive parser on
200/200 sampled cases and was wrong on every one of them: 188 were the ARC-Challenge
questions whose choice labels are 1-4, and 12 were models emitting a lowercase "b)".
"""

import logging
import re

LOGGER = logging.getLogger(__name__)


def _alternation(choice_labels: list[str]) -> str:
    """Regex alternation over the question's real labels, longest first."""
    return "|".join(re.escape(label) for label in sorted(choice_labels, key=len, reverse=True))


def extract_answer_strict(text: str, choice_labels: list[str]) -> str | None:
    """Extract an answer using only unambiguous parenthesised forms.

    Accepts a leading "B)" / "(B)" / "3)" at the very start of the completion, or an
    explicit "the answer is (B)". No bare-letter fallback, no scan for a stray "(A)"
    mid-text, and no first-character heuristic.
    """
    if not choice_labels:
        return None
    canonical = {label.upper(): label for label in choice_labels}
    alt = _alternation(choice_labels)

    leading = re.match(rf"\s*\(?\s*({alt})\s*\)", text, re.IGNORECASE)
    if leading:
        return canonical[leading.group(1).upper()]

    phrases = re.findall(rf"the\s+answer\s+is\s*\(\s*({alt})\s*\)", text, re.IGNORECASE)
    if phrases:
        return canonical[phrases[-1].upper()]

    return None
