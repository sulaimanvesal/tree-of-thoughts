"""LLM backends for Tree of Thoughts.

The agent programs against a tiny two-method interface, mirroring the two
LM calls in the paper:

  * ``generate``  - thought generation (paper Sec. 3.2): given a state,
    produce candidate next thoughts. Covers both of the paper's generation
    strategies: independent sampling and joint proposal (the mock below
    implements the "propose" style, returning a ranked candidate set).
  * ``evaluate``  - state evaluation (paper Sec. 3.3): score how promising a
    state is. Returns one of "sure" / "likely" / "impossible", the value
    labels the paper uses in its prompts.

``MockBackend`` is deterministic and offline: it stands in for both the
propose and value prompts with exact Game-of-24 arithmetic, so the demo runs
with no API key and exercises the full search loop. Swap in
``OpenAICompatibleBackend`` for real models (it issues the paper's propose /
value prompts verbatim in spirit).
"""
from __future__ import annotations

import json
import os
import urllib.request

# Paper's value labels, mapped to numeric scores for ranking.
VALUE_SCORES = {"sure": 20, "likely": 10, "impossible": 0}


class LLMBackend:
    """Interface the ToT agent programs against."""

    def generate(self, prompt: str) -> list[str]:
        """Propose candidate next thoughts for a state. Returns thought strings."""
        raise NotImplementedError

    def evaluate(self, prompt: str) -> str:
        """Evaluate a state. Returns one of 'sure', 'likely', 'impossible'."""
        raise NotImplementedError


class MockBackend(LLMBackend):
    """Deterministic offline stand-in for the paper's propose/value prompts.

    The propose-style generation enumerates every distinct arithmetic move
    from the current Game-of-24 numbers in a fixed order (pairs in sorted
    order, ops ``+ - * /``), formatted as::

        "9 - 3 = 6 (left: 4 6)"

    The value evaluation uses exact reachability analysis: a state is "sure"
    when it is already 24, "likely" when 24 can still be built from the
    remaining numbers, and "impossible" otherwise. This is the programmatic
    twin of the paper's self-evaluation prompt ("sure/likely/impossible"),
    which lets the demo exercise generation -> evaluation -> search without
    an API key.
    """

    def generate(self, prompt: str) -> list[str]:
        # Prompt format from GameOf24.propose_prompt: "Current numbers: a b c d"
        numbers = _parse_numbers(prompt)
        return _candidate_thoughts(numbers)

    def evaluate(self, prompt: str) -> str:
        # Prompt format from GameOf24.evaluate_prompt: "Numbers left: a b ..."
        numbers = _parse_numbers(prompt)
        return _value_label(numbers)


def _parse_numbers(prompt: str) -> list[int]:
    for line in prompt.splitlines():
        if line.startswith(("Current numbers:", "Numbers left:")):
            return [int(x) for x in line.split(":", 1)[1].split()]
    raise ValueError(f"could not parse numbers from prompt: {prompt!r}")


def _candidate_thoughts(numbers: list[int]) -> list[str]:
    """Enumerate every distinct next arithmetic move (propose-style)."""
    from fractions import Fraction

    thoughts: list[str] = []
    seen: set[tuple] = set()
    vals = [Fraction(n) for n in numbers]
    ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
           ("*", lambda a, b: a * b), ("/", lambda a, b: a / b if b != 0 else None)]
    for i in range(len(vals)):
        for j in range(len(vals)):
            if i == j:
                continue
            for sym, fn in ops:
                result = fn(vals[i], vals[j])
                if result is None or result.denominator != 1:
                    continue  # keep the demo in integers, like the paper
                r = int(result)
                rest = [int(vals[k]) for k in range(len(vals)) if k not in (i, j)]
                key = (min(int(vals[i]), int(vals[j])), max(int(vals[i]), int(vals[j])), sym, r)
                if key in seen:
                    continue
                seen.add(key)
                rest_sorted = sorted(rest + [r])
                thoughts.append(
                    f"{int(vals[i])} {sym} {int(vals[j])} = {r} "
                    f"(left: {' '.join(map(str, rest_sorted))})"
                )
    return sorted(thoughts)


def _value_label(numbers: list[int]) -> str:
    """Exact stand-in for the paper's value prompt: sure/likely/impossible."""
    if len(numbers) == 1:
        return "sure" if numbers[0] == 24 else "impossible"
    return "likely" if _can_make_24(numbers) else "impossible"


def _can_make_24(numbers: list[int]) -> bool:
    from fractions import Fraction

    vals = [Fraction(n) for n in numbers]

    def search(nums: list[Fraction]) -> bool:
        if len(nums) == 1:
            return nums[0] == 24
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i == j:
                    continue
                rest = [nums[k] for k in range(len(nums)) if k not in (i, j)]
                for r in (nums[i] + nums[j], nums[i] - nums[j],
                          nums[i] * nums[j],
                          nums[i] / nums[j] if nums[j] != 0 else None):
                    if r is None:
                        continue
                    if search(rest + [r]):
                        return True
        return False

    return search(vals)


class OpenAICompatibleBackend(LLMBackend):
    """Backend for any OpenAI-compatible chat-completions endpoint.

    Uses the paper's prompt styles: a *propose* prompt asking for candidate
    next thoughts, and a *value* prompt asking for a sure/likely/impossible
    judgement with reasoning.

    Set ``OPENAI_API_KEY`` and optionally ``OPENAI_BASE_URL``
    (defaults to https://api.openai.com/v1) and ``TOT_MODEL``.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("TOT_MODEL", "gpt-4o-mini")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def _chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.load(resp)
        return body["choices"][0]["message"]["content"].strip()

    def generate(self, prompt: str) -> list[str]:
        out = self._chat(
            "You are a Tree of Thoughts proposer. Given the current state of a "
            "Game of 24 puzzle, propose up to 5 diverse, promising next arithmetic "
            "steps. Reply with one step per line, each exactly in the form "
            "'a OP b = c (left: n1 n2 ...)' listing the numbers remaining after "
            "the step. Do not repeat equivalent steps.",
            prompt,
        )
        return [line.strip() for line in out.splitlines() if line.strip()]

    def evaluate(self, prompt: str) -> str:
        out = self._chat(
            "You are a Tree of Thoughts evaluator. Given the remaining numbers "
            "of a Game of 24 puzzle, judge whether 24 is still reachable. "
            "Reply with exactly one word: 'sure' if the state already equals 24 "
            "or 24 is clearly reachable, 'likely' if it might be reachable, "
            "'impossible' if it cannot be reached. Briefly reason first, then "
            "put the word on the last line.",
            prompt,
        ).lower()
        for label in ("sure", "likely", "impossible"):
            if label in out:
                return label
        return "likely"
