import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def round_floats(obj: object, ndigits: int = 4) -> object:
    """Recursively round all floats in a nested dict/list structure."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(item, ndigits) for item in obj]
    return obj


def write_jsonl(path: str | Path, records: list[dict], round_digits: int = 4) -> None:
    """Write list of dicts to JSONL with float rounding. Prints path to stdout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            rounded = round_floats(record, round_digits)
            f.write(json.dumps(rounded, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read JSONL file, return list of dicts."""
    path = Path(path)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, data: dict, indent: int = 2) -> None:
    """Write dict to JSON with indentation. Prints path to stdout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write("\n")
    print(f"Saved config to {path}")


def read_json(path: str | Path) -> dict:
    """Read JSON file, return dict."""
    path = Path(path)
    with open(path) as f:
        return json.load(f)
