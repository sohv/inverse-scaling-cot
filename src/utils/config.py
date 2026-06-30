import json
import logging
import subprocess
from dataclasses import asdict
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def get_git_hash() -> str:
    """Return first 8 chars of current git commit hash."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:8]


def normalize_name(name: str) -> str:
    """Replace / and whitespace with underscores for use in filenames."""
    return name.replace("/", "_").replace(" ", "_")


def make_output_dir(base: str, model_id: str, dataset_name: str) -> Path:
    """Create and return output directory: base/model_name__dataset_name/."""
    model_name = normalize_name(model_id)
    dir_path = Path(base) / f"{model_name}__{dataset_name}"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_run_config(output_dir: Path, config: object, extra_metadata: dict | None = None) -> Path:
    """Save config.json with git hash, all config fields, and optional metadata."""
    data: dict = {}
    data["git_hash"] = get_git_hash()
    if hasattr(config, "__dataclass_fields__"):
        data["config"] = asdict(config)
    else:
        data["config"] = str(config)
    if extra_metadata:
        data["metadata"] = extra_metadata
    path = output_dir / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    LOGGER.info(f"Saved run config to {path}")
    return path
