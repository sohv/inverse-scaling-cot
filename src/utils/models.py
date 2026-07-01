from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    family: str  # "qwen" or "llama"
    size_b: float  # parameter count in billions
    is_instruct: bool
    is_base: bool


INSTRUCT_MODELS: list[ModelInfo] = [
    # Qwen2.5-Instruct family (7 sizes)
    ModelInfo("Qwen/Qwen2.5-0.5B-Instruct", "qwen", 0.5, True, False),
    ModelInfo("Qwen/Qwen2.5-1.5B-Instruct", "qwen", 1.5, True, False),
    ModelInfo("Qwen/Qwen2.5-3B-Instruct", "qwen", 3.0, True, False),
    ModelInfo("Qwen/Qwen2.5-7B-Instruct", "qwen", 7.0, True, False),
    ModelInfo("Qwen/Qwen2.5-14B-Instruct", "qwen", 14.0, True, False),
    ModelInfo("Qwen/Qwen2.5-32B-Instruct", "qwen", 32.0, True, False),
    ModelInfo("Qwen/Qwen2.5-72B-Instruct", "qwen", 72.0, True, False),
    ModelInfo("Qwen/Qwen2.5-72B-Instruct-AWQ", "qwen", 72.0, True, False),
    # Llama-3-Instruct family (4 sizes)
    ModelInfo("meta-llama/Llama-3.2-1B-Instruct", "llama", 1.0, True, False),
    ModelInfo("meta-llama/Llama-3.2-3B-Instruct", "llama", 3.0, True, False),
    ModelInfo("meta-llama/Llama-3.1-8B-Instruct", "llama", 8.0, True, False),
    ModelInfo("meta-llama/Llama-3.1-70B-Instruct", "llama", 70.0, True, False),
    ModelInfo("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", "llama", 70.0, True, False),
]

BASE_MODELS: list[ModelInfo] = [
    # Qwen2.5 base (non-instruct) checkpoints for Exp 6
    ModelInfo("Qwen/Qwen2.5-0.5B", "qwen", 0.5, False, True),
    ModelInfo("Qwen/Qwen2.5-1.5B", "qwen", 1.5, False, True),
    ModelInfo("Qwen/Qwen2.5-3B", "qwen", 3.0, False, True),
    ModelInfo("Qwen/Qwen2.5-7B", "qwen", 7.0, False, True),
    ModelInfo("Qwen/Qwen2.5-14B", "qwen", 14.0, False, True),
    ModelInfo("Qwen/Qwen2.5-32B", "qwen", 32.0, False, True),
    ModelInfo("Qwen/Qwen2.5-72B", "qwen", 72.0, False, True),
]

MODEL_REGISTRY: dict[str, ModelInfo] = {m.model_id: m for m in INSTRUCT_MODELS + BASE_MODELS}

ALL_DATASETS = ["aqua", "logiqa", "arc_challenge", "openbookqa", "hellaswag"]


def get_model_info(model_id: str) -> ModelInfo:
    """Look up model metadata by HuggingFace ID."""
    return MODEL_REGISTRY[model_id]


def get_family(model_id: str) -> str:
    """Return 'qwen' or 'llama' for a given model_id."""
    return MODEL_REGISTRY[model_id].family


def normalize_model_name(model_id: str) -> str:
    """Replace / and whitespace with underscores for use in filenames."""
    return model_id.replace("/", "_").replace(" ", "_")
