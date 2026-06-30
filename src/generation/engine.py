"""vLLM engine wrapper for batched generation.

Handles model loading, chat template application, and batch inference.
Uses llm.chat() for instruct models (applies native chat template automatically)
and llm.generate() for base models (raw text prompts).
"""

import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Generation parameters matching experimental protocol."""

    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 1024
    seed: int | None = None
    n: int = 1  # number of completions per prompt


class VLLMEngine:
    """Wrapper around vLLM's offline LLM for batched inference."""

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
    ):
        from vllm import LLM

        self.model_id = model_id
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        LOGGER.info(f"Loaded model {model_id} with tp={tensor_parallel_size}")

    def generate_chat(
        self,
        conversations: list[list[dict[str, str]]],
        gen_config: GenerationConfig,
        continue_final_message: bool = False,
    ) -> list[list[str]]:
        """Generate completions for a batch of conversations.

        Uses llm.chat() which automatically applies the model's native chat template.

        Args:
            conversations: List of message lists, each [{"role": ..., "content": ...}, ...]
            gen_config: Generation parameters

        Returns:
            List of lists of generated texts. Outer = one per conversation.
            Inner = gen_config.n completions per conversation.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            max_tokens=gen_config.max_tokens,
            seed=gen_config.seed,
            n=gen_config.n,
        )
        outputs = self.llm.chat(
            messages=conversations,
            sampling_params=sampling_params,
            use_tqdm=True,
            continue_final_message=continue_final_message,
            add_generation_prompt=not continue_final_message,
        )
        results = []
        for output in outputs:
            completions = [o.text for o in output.outputs]
            results.append(completions)
        return results

    def generate_text(
        self,
        prompts: list[str],
        gen_config: GenerationConfig,
    ) -> list[list[str]]:
        """Generate completions for raw text prompts (base models).

        Args:
            prompts: List of raw text strings
            gen_config: Generation parameters

        Returns:
            List of lists of generated texts.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            max_tokens=gen_config.max_tokens,
            seed=gen_config.seed,
            n=gen_config.n,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        results = []
        for output in outputs:
            completions = [o.text for o in output.outputs]
            results.append(completions)
        return results
