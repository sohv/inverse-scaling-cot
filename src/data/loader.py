import logging

from datasets import load_dataset
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)


class Question(BaseModel):
    """Universal question schema across all datasets."""

    id: str  # unique ID: "{dataset_name}_{original_index}"
    dataset_name: str  # one of: aqua, logiqa, arc_challenge, openbookqa, hellaswag
    question_text: str  # the full question/context text
    choices: list[str]  # choice texts, always ordered A, B, C, D, (E)
    choice_labels: list[str]  # ["A", "B", "C", "D"] or ["A", "B", "C", "D", "E"]
    correct_label: str  # the letter label of the correct answer
    raw_index: int  # index in the original HF dataset split


DATASET_CONFIGS: dict[str, dict] = {
    "aqua": {"hf_id": "deepmind/aqua_rat", "hf_config": "raw", "split": "test"},
    "logiqa": {"hf_id": "dmayhem93/agieval-logiqa-en", "hf_config": None, "split": "test"},
    "arc_challenge": {"hf_id": "allenai/ai2_arc", "hf_config": "ARC-Challenge", "split": "test"},
    "openbookqa": {"hf_id": "allenai/openbookqa", "hf_config": "main", "split": "test"},
    "hellaswag": {"hf_id": "Rowan/hellaswag", "hf_config": None, "split": "validation"},
}

LETTER_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _parse_aqua(row: dict, idx: int) -> Question:
    """Parse AQuA-RAT row.

    Options come as ['A)125', 'B)150', ...].
    Split on first ')' to separate label from text.
    """
    raw_options = row["options"]
    choices = []
    choice_labels = []
    for opt in raw_options:
        parts = opt.split(")", 1)
        label = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else opt
        choice_labels.append(label)
        choices.append(text)
    return Question(
        id=f"aqua_{idx}",
        dataset_name="aqua",
        question_text=row["question"],
        choices=choices,
        choice_labels=choice_labels,
        correct_label=row["correct"],
        raw_index=idx,
    )


def _parse_logiqa(row: dict, idx: int) -> Question:
    """Parse LogiQA row from dmayhem93/agieval-logiqa-en.

    'query' contains context + question + embedded choices + an answer prompt.
    Split on 'Answer Choices:' to extract just context + question.
    'choices' is a list of strings like '(A)text'; strip the letter prefix.
    'gold' is a single-element list with the 0-indexed correct answer.
    """
    query = row["query"]
    if "Answer Choices:" in query:
        question_text = query.split("Answer Choices:")[0].strip()
    else:
        question_text = query.strip()

    choices = []
    for choice in row["choices"]:
        # Strip "(A)", "(B)", … prefix
        if len(choice) > 3 and choice[0] == "(" and choice[2] == ")":
            choices.append(choice[3:].strip())
        else:
            choices.append(choice.strip())

    n_choices = len(choices)
    choice_labels = LETTER_LABELS[:n_choices]
    correct_idx = row["gold"][0]
    return Question(
        id=f"logiqa_{idx}",
        dataset_name="logiqa",
        question_text=question_text,
        choices=choices,
        choice_labels=choice_labels,
        correct_label=choice_labels[correct_idx],
        raw_index=idx,
    )


def _parse_arc(row: dict, idx: int) -> Question:
    """Parse ARC-Challenge row.

    Choices are nested dict with 'text' and 'label' lists.
    """
    choice_labels = row["choices"]["label"]
    choices = row["choices"]["text"]
    return Question(
        id=f"arc_challenge_{idx}",
        dataset_name="arc_challenge",
        question_text=row["question"],
        choices=choices,
        choice_labels=choice_labels,
        correct_label=row["answerKey"],
        raw_index=idx,
    )


def _parse_openbookqa(row: dict, idx: int) -> Question:
    """Parse OpenBookQA row. Same nested choices structure as ARC."""
    choice_labels = row["choices"]["label"]
    choices = row["choices"]["text"]
    return Question(
        id=f"openbookqa_{idx}",
        dataset_name="openbookqa",
        question_text=row["question_stem"],
        choices=choices,
        choice_labels=choice_labels,
        correct_label=row["answerKey"],
        raw_index=idx,
    )


def _parse_hellaswag(row: dict, idx: int) -> Question:
    """Parse HellaSwag row.

    'endings' is a list of 4 completions, 'label' is '0'-'3'.
    Map numeric label to letter: 0->A, 1->B, 2->C, 3->D.
    """
    choices = row["endings"]
    n_choices = len(choices)
    choice_labels = LETTER_LABELS[:n_choices]
    correct_idx = int(row["label"])
    return Question(
        id=f"hellaswag_{idx}",
        dataset_name="hellaswag",
        question_text=row["ctx"],
        choices=choices,
        choice_labels=choice_labels,
        correct_label=choice_labels[correct_idx],
        raw_index=idx,
    )


PARSERS: dict[str, callable] = {
    "aqua": _parse_aqua,
    "logiqa": _parse_logiqa,
    "arc_challenge": _parse_arc,
    "openbookqa": _parse_openbookqa,
    "hellaswag": _parse_hellaswag,
}


def load_dataset_questions(dataset_name: str) -> list[Question]:
    """Load full dataset from HuggingFace, parse all rows into Question objects.

    Args:
        dataset_name: One of 'aqua', 'logiqa', 'arc_challenge', 'openbookqa', 'hellaswag'

    Returns:
        List of Question objects, one per row in the dataset split.
    """
    cfg = DATASET_CONFIGS[dataset_name]
    parser = PARSERS[dataset_name]

    LOGGER.info(f"Loading {dataset_name} from {cfg['hf_id']} (split={cfg['split']})")

    if cfg["hf_config"]:
        ds = load_dataset(cfg["hf_id"], cfg["hf_config"], split=cfg["split"])
    else:
        ds = load_dataset(cfg["hf_id"], split=cfg["split"])

    questions = []
    for idx, row in enumerate(ds):
        q = parser(row, idx)
        questions.append(q)

    LOGGER.info(f"Loaded {len(questions)} questions from {dataset_name}")
    return questions
