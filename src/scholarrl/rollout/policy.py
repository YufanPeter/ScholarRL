"""Policy interface and implementations for trajectory rollout.

A policy maps conversation history -> next action text. It receives the FULL
messages list (not just the latest observation) because a language model is
stateless: it needs the whole history (question + past searches/reads) to decide.
"""
from __future__ import annotations

import re
from typing import List, Protocol

from scholarrl.env.actions import STOP_TOKENS


class Policy(Protocol):
    """Interface for action generation policies."""

    def generate(self, messages: List[dict]) -> str:
        """Generate next action text from the full conversation history."""
        ...


class StubPolicy:
    """Scripted policy for testing rollout without a model.

    Fixed pattern: search -> read first result -> select it -> finish.
    Uses the number of assistant turns so far to know which step it's on
    (no internal counter -> safe to reuse across episodes).
    """

    def generate(self, messages: List[dict]) -> str:
        # step = how many actions we've already taken this episode
        step = sum(1 for m in messages if m["role"] == "assistant")
        last_obs = messages[-1]["content"]  # latest env observation

        if step == 0:
            # First action: pull a keyword from the question and search
            if "Question:" in last_obs:
                words = last_obs.split("Question:")[-1].strip().split()[:3]
                return f"<search>{' '.join(words)}</search>"
            return "<search>machine learning</search>"

        if step == 1:
            # Read the first paper id from the search results ('id=<pid>  Title' lines)
            m = re.search(r"id=(\S+)", last_obs)
            if m:
                return f"<read>{m.group(1)}</read>"
            return "<finish/>"

        if step == 2:
            # Select the paper we just read
            if last_obs.startswith("[read]"):
                pid = last_obs.split("[read]")[1].split()[0].strip()
                return f"<select>{pid}</select>"
            return "<finish/>"

        return "<finish/>"


class HFPolicy:
    """Real language-model policy backed by transformers.

    Loads a chat model (default Qwen2.5-3B-Instruct), feeds it the conversation
    via the tokenizer's chat template, and returns the generated action text.
    Unlike StubPolicy, this actually understands the question and abstracts.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        device: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # pick device: cuda on the server, mps on Mac, else cpu
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        ).to(device)
        self.model.eval()

    def generate(self, messages: List[dict]) -> str:
        import torch

        # chat template turns messages -> a single prompt string the model expects,
        # with add_generation_prompt=True appending the "assistant:" cue.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                # Stop as soon as one action's closing tag fires, so each turn emits
                # exactly ONE action (matches the contract documented in env.actions).
                # Without this the model runs to max_new_tokens and may emit several
                # actions or hallucinate the env's reply.
                stop_strings=STOP_TOKENS,
                tokenizer=self.tokenizer,
            )
        # decode only the newly generated tokens (strip the prompt)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
