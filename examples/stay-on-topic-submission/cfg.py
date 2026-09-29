"""
Classifier-Free-Guidance from Sanchez et al. 2024, ICML 2024 Spotlight.

Many references made to author's code:
https://github.com/Vermeille/transformers/blob/ee21d2549cc886f63d6e43b83de5cfbe7a6c385a/src/transformers/generation/logits_process.py#L1239-L1352
https://github.com/huggingface/transformers/pull/24654
"""

from typing import Optional

import torch
import transformers
from torch.nn import functional as F


@torch.no_grad()
def classifier_free_guidance(
    model: transformers.PreTrainedModel,
    guidance_scale: float,
    cond_input_ids: torch.LongTensor,
    uncond_input_ids: Optional[torch.LongTensor] = None,
    use_cache: bool = False,
    cond_past_key_values: Optional[tuple] = None,
    uncond_past_key_values: Optional[tuple] = None,
    input_scores: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, tuple, tuple]:
    """

    Args:
        model: transformers.PreTrainedModel
        guidance_scale: float
        cond_input_ids: torch.LongTensor of shape (B, L)
        uncond_input_ids: Optional[torch.LongTensor] (B, S)
        use_cache: bool, whether to use the cache for the model forward pass,
            defaults to False
        cond_past_key_values: Optional[tuple], the past key values for the conditioned logits
        uncond_past_key_values: Optional[tuple], the past key values for the
            unconditioned logits
        input_scores: Optional[torch.Tensor] (B, V), the conditioned logits for the
            predicted next token

    Returns:
        adjusted_logits: torch.Tensor (B, V) the adjusted logits for the
            predicted next token
        cond_past_key_values: tuple, the past key values for the conditioned logits
        uncond_past_key_values: tuple, the past key values for the unconditioned logits
    """

    if input_scores is None:
        cond_output = model(
            cond_input_ids, past_key_values=cond_past_key_values, use_cache=use_cache
        )
        cond_past_key_values = cond_output.past_key_values
        # (B, L, V)
        cond_logits = cond_output.logits
        # (B, V), the conditioned logits for the predicted next token
        cond_logits_next = cond_logits[:, -1, :]
    else:
        cond_logits_next = input_scores
        cond_past_key_values = None

    if guidance_scale == 1:
        return (
            F.log_softmax(cond_logits_next, dim=-1),
            cond_past_key_values,
            uncond_past_key_values,
        )

    # take the last token of the input ids as the unconditional input if not specified
    if uncond_input_ids is None:
        uncond_input_ids = cond_input_ids[:, -1:]

    uncond_output = model(
        uncond_input_ids,
        past_key_values=uncond_past_key_values,
        use_cache=use_cache,
    )
    uncond_past_key_values = uncond_output.past_key_values
    # (B, S, V), where S is the number of tokens in the unconditional input
    uncond_logits = uncond_output.logits
    # (B, V), the uncondtioned logits for the predicted next token
    uncond_logits_next = uncond_logits[:, -1, :]

    # work in log probability space
    cond_log_probs = F.log_softmax(cond_logits_next, dim=-1)
    uncond_log_probs = F.log_softmax(uncond_logits_next, dim=-1)

    # (B, V)
    adjusted_logits = uncond_log_probs + guidance_scale * (cond_log_probs - uncond_log_probs)

    return adjusted_logits, cond_past_key_values, uncond_past_key_values


class CFGLogitsProcessor(transformers.LogitsProcessor):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        guidance_scale: float,
        uncond_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.uncond_state = {
            "uncond_ids": uncond_ids,
            "past_key_values": None,
            "first_pass": True,
        }
        self.cond_state = {
            "past_key_values": None,
        }
        self.use_cache = use_cache

    def __call__(self, input_ids, scores) -> torch.Tensor:
        uncond_ids = self._get_uncond_ids(input_ids)

        adjusted_scores, cond_past_key_values, uncond_past_key_values = classifier_free_guidance(
            model=self.model,
            guidance_scale=self.guidance_scale,
            cond_input_ids=input_ids,
            uncond_input_ids=uncond_ids,
            use_cache=self.use_cache,
            cond_past_key_values=self.cond_state["past_key_values"],
            uncond_past_key_values=self.uncond_state["past_key_values"],
            input_scores=scores,
        )
        self.cond_state["past_key_values"] = cond_past_key_values
        self.uncond_state["past_key_values"] = uncond_past_key_values

        return adjusted_scores

    def _get_uncond_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Gets the unconditioned token IDs for the current generation step.
        Will set the unconditioned token IDs if they are not set yet as a side effect.
        """
        # on the first generation step, we need to init uncond_ids
        if self.uncond_state["first_pass"]:
            if self.uncond_state["uncond_ids"] is None:
                uncond_ids = input_ids[:, -1:]
            else:
                uncond_ids = self.uncond_state["uncond_ids"]
            self.uncond_state["first_pass"] = False
        # on subsequent gen steps, we simply append to previous uncond_ids
        else:
            if not self.use_cache:
                uncond_ids = torch.cat([self.uncond_state["uncond_ids"], input_ids[:, -1:]], dim=1)
            else:
                uncond_ids = input_ids[:, -1:]

        # set the uncond_ids for the next generation step
        self.uncond_state["uncond_ids"] = uncond_ids
        return uncond_ids


if __name__ == "__main__":
    # test the usage in docstring
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopPLogitsWarper,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = AutoModelForCausalLM.from_pretrained("gpt2")

    prompt = tokenizer(
        "Today a dragon flew over Paris, France, the parisians were very",
        return_tensors="pt",
    )
    # either provide a negative prompt:
    neg_prompt = tokenizer("A sad event happened,", return_tensors="pt")["input_ids"]
    # or don't:
    # neg_prompt = prompt['input_ids'][:, -1:]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    neg_prompt = neg_prompt.to(device)
    outputs = model.generate(
        input_ids=prompt["input_ids"].to(device),
        attention_mask=prompt["attention_mask"].to(device),
        max_new_tokens=125,
        logits_processor=LogitsProcessorList(
            [
                CFGLogitsProcessor(
                    model=model,
                    guidance_scale=2,
                    uncond_ids=neg_prompt,
                    use_cache=True,
                )
            ]
        ),
        do_sample=True,
        stop_strings=None,
        tokenizer=tokenizer,
        temperature=0.8,
        top_p=0.95,
    )

    print(tokenizer.decode(outputs[0]))
