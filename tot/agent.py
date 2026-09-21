"""The Tree of Thoughts agent: decompose -> generate -> evaluate -> search.

Paper Sec. 3 breaks ToT into four modules; this class wires them together:

1. **Thought decomposition** (Sec. 3.1): ``task.initial_state`` defines the
   root (for Game of 24, the four input numbers).
2. **Thought generation** (Sec. 3.2): ``expand`` calls the backend's
   ``generate`` (the paper's propose-style generation) and turns each
   thought string into a child node.
3. **State evaluation** (Sec. 3.3): ``evaluate_state`` scores each new state
   with the backend's ``evaluate`` ("sure"/"likely"/"impossible"). The
   ``evaluator`` mode selects how scores pick winners: ``"value"`` ranks each
   state independently (paper's value strategy); ``"vote"`` re-ranks the whole
   candidate set comparatively (paper's vote strategy) - here implemented as
   a pairwise tournament on the value labels, deterministic and offline.
4. **Search** (Sec. 3.4): ``BFSSearch`` or ``DFSSearch`` explores the tree.
"""
from __future__ import annotations

from .llm import VALUE_SCORES, LLMBackend
from .search import BFSSearch, DFSSearch, Node
from .tasks import GameOf24


class ToTAgent:
    def __init__(
        self,
        backend: LLMBackend,
        numbers: tuple[int, ...] | list[int] = (4, 9, 10, 13),
        search: str = "bfs",
        width: int = 5,
        evaluator: str = "value",
    ) -> None:
        self.backend = backend
        self.numbers = tuple(numbers)
        self.task = GameOf24(numbers=self.numbers)
        self.evaluator = evaluator
        if search == "bfs":
            self.searcher = BFSSearch(width=width, max_depth=len(self.numbers) - 1)
        elif search == "dfs":
            self.searcher = DFSSearch(max_depth=len(self.numbers) - 1)
        else:
            raise ValueError(f"unknown search: {search!r}")

    # -- paper Sec. 3.3: state evaluation ---------------------------------
    def evaluate_state(self, state: dict) -> str:
        return self.backend.evaluate(self.task.evaluate_prompt(state))

    def score(self, label: str) -> float:
        return float(VALUE_SCORES.get(label, 0))

    # -- paper Sec. 3.2: thought generation --------------------------------
    def expand(self, node: Node) -> list[Node]:
        thoughts = self.backend.generate(self.task.propose_prompt(node.state))
        children = []
        for thought in thoughts:
            try:
                state = self.task.apply_thought(node.state, thought)
            except ValueError:
                continue  # skip malformed thoughts, like the paper's parser
            label = self.evaluate_state(state)
            children.append(
                Node(
                    state=state,
                    thought=thought,
                    value=self.score(label),
                    label=label,
                    depth=node.depth + 1,
                )
            )
        return self._select(children)

    def _select(self, children: list[Node]) -> list[Node]:
        """Paper Sec. 3.3: 'value' ranks states independently, 'vote'
        re-ranks them comparatively. Both are deterministic here."""
        if self.evaluator == "vote" and len(children) > 1:
            # pairwise tournament: a thought wins a "vote" for every rival
            # whose value label it strictly beats
            wins = {id(c): 0 for c in children}
            for i, a in enumerate(children):
                for j, b in enumerate(children):
                    if i != j and a.value > b.value:
                        wins[id(a)] += 1
            children.sort(key=lambda c: (-wins[id(c)], -c.value, c.thought))
        return children

    def is_solution(self, state: dict) -> bool:
        return self.task.is_solution(state)

    # -- run ---------------------------------------------------------------
    def solve(self) -> Node | None:
        """Run the search and return the solution node, or None."""
        return self.searcher.run(self)

    @staticmethod
    def format_solution(node: Node) -> str:
        lines = ["Solution found:"]
        for i, step in enumerate(node.state["history"], 1):
            lines.append(f"  {i}. {step}")
        return "\n".join(lines)
