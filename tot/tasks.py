"""Task definitions: the problems ToT searches over.

The paper evaluates on three tasks: Game of 24, Creative Writing, and Mini
Crosswords. This package implements Game of 24 (paper Sec. 4.1) as the bundled
task because its thought decomposition is crisp: a *thought* is one
intermediate arithmetic step, a *state* is the remaining numbers plus the
equation history, and a solution is reached when a single number 24 remains.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

THOUGHT_RE = re.compile(
    r"(?P<a>-?\d+)\s*(?P<op>[+\-*/])\s*(?P<b>-?\d+)\s*=\s*(?P<c>-?\d+)\s*"
    r"\(left:\s*(?P<left>[\d\s-]+)\)"
)


@dataclass
class GameOf24:
    """Game of 24: combine four numbers with + - * / to make 24.

    Paper Sec. 4.1: thought decomposition = one intermediate equation per
    step (depth 3 for 4 input numbers); the search stops when the remaining
    numbers are a single 24.
    """

    numbers: tuple[int, ...] = (4, 9, 10, 13)

    @staticmethod
    def initial_state(numbers: tuple[int, ...] | list[int]) -> dict:
        """Paper Sec. 3.1 thought decomposition: the root state."""
        return {"numbers": tuple(sorted(numbers)), "history": []}

    @staticmethod
    def propose_prompt(state: dict) -> str:
        """Paper Sec. 3.2 'propose' strategy: ask for candidate next steps."""
        nums = " ".join(map(str, state["numbers"]))
        history = "\n".join(state["history"]) or "(no steps yet)"
        return (
            "Game of 24. Use each number once with + - * / to reach 24.\n"
            f"Current numbers: {nums}\n"
            f"Steps so far:\n{history}\n"
            "Propose possible next steps."
        )

    @staticmethod
    def evaluate_prompt(state: dict) -> str:
        """Paper Sec. 3.3 value prompt: judge the state's promise."""
        nums = " ".join(map(str, state["numbers"]))
        return (
            "Game of 24. Use each number once with + - * / to reach 24.\n"
            f"Numbers left: {nums}\n"
            "Evaluate if given the numbers above can reach 24 (sure/likely/impossible)."
        )

    @staticmethod
    def apply_thought(state: dict, thought: str) -> dict:
        """Transition: parse a thought string into the next state."""
        m = THOUGHT_RE.fullmatch(thought.strip())
        if not m:
            raise ValueError(f"malformed thought: {thought!r}")
        left = tuple(sorted(int(x) for x in m.group("left").split()))
        return {
            "numbers": left,
            "history": state["history"] + [f"{m.group('a')} {m.group('op')} {m.group('b')} = {m.group('c')}"],
        }

    @staticmethod
    def is_solution(state: dict) -> bool:
        return len(state["numbers"]) == 1 and state["numbers"][0] == 24

    @staticmethod
    def state_label(state: dict) -> str:
        nums = " ".join(map(str, state["numbers"]))
        return f"[{nums}]"
