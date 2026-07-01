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


class HFEngine:
    """HuggingFace Transformers engine for batched generation.

    Uses device_map="auto" to automatically distribute the model across all
    available GPUs without any tensor-parallel configuration. Drop-in
    replacement for VLLMEngine for large models (70B/72B) that exceed vLLM's
    memory overhead budget.
    """

    def __init__(
        self,
        model_id: str,
        torch_dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        dtype = getattr(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.eval()
        # Input tensors must go to the device hosting the embedding/first layer.
        self._input_device = next(self.model.parameters()).device
        LOGGER.info(f"Loaded model {model_id} with device_map=auto, input_device={self._input_device}")

    def _sample(
        self,
        input_ids,
        gen_config: GenerationConfig,
    ) -> list[str]:
        """Run generation for a single tokenised input, returning n completions."""
        import torch

        input_len = input_ids.shape[1]
        inputs = {"input_ids": input_ids.to(self._input_device)}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=gen_config.temperature > 0,
                temperature=gen_config.temperature if gen_config.temperature > 0 else 1.0,
                top_p=gen_config.top_p,
                max_new_tokens=gen_config.max_tokens,
                num_return_sequences=gen_config.n,
            )
        return [
            self.tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            for seq in out
        ]

    def generate_chat(
        self,
        conversations: list[list[dict[str, str]]],
        gen_config: GenerationConfig,
        continue_final_message: bool = False,
    ) -> list[list[str]]:
        results = []
        for messages in conversations:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=not continue_final_message,
                continue_final_message=continue_final_message,
            )
            input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
            results.append(self._sample(input_ids, gen_config))
        return results

    def generate_text(
        self,
        prompts: list[str],
        gen_config: GenerationConfig,
    ) -> list[list[str]]:
        results = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            results.append(self._sample(input_ids, gen_config))
        return results


class VLLMEngine:
    """Wrapper around vLLM's offline LLM for batched inference."""

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        enforce_eager: bool = False,
        quantization: str | None = None,
    ):
        from vllm import LLM

        self.model_id = model_id
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            quantization=quantization,
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
