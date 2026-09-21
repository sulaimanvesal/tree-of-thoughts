"""Search algorithms over the tree of thoughts (paper Sec. 3.4).

The paper explores with two classical strategies, both implemented here:

* ``BFSSearch`` - breadth-first search. At each depth, generate the children
  of every frontier state, evaluate them, and keep the top-``b`` most
  promising (the paper uses b=5 for Game of 24).
* ``DFSSearch`` - depth-first search with backtracking. Recurse into the most
  promising child first; when a state's value drops below ``threshold``,
  prune it and backtrack to try the next sibling (the paper's "look ahead or
  backtrack when necessary").
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    """One node in the tree of thoughts."""

    state: dict
    thought: str = "<root>"
    value: float = 0.0
    label: str = "impossible"
    children: list["Node"] = field(default_factory=list)
    depth: int = 0

    def is_solution(self, is_solution_fn) -> bool:
        return is_solution_fn(self.state)


class BFSSearch:
    """BFS over thoughts: keep the top-b states at every depth.

    Matches the paper's BFS (Game of 24, Sec. 4.1): S_0 = {root}; for each
    step, S_t = top-b of {generate(s) for s in S_{t-1}} ranked by the
    evaluator. Returns the first solution found, or None.
    """

    def __init__(self, width: int = 5, max_depth: int = 3) -> None:
        self.width = width
        self.max_depth = max_depth
        self.nodes_expanded = 0
        self.levels: list[list[Node]] = []

    def run(self, agent) -> Node | None:
        root = Node(state=agent.task.initial_state(agent.numbers), depth=0)
        root.label = agent.evaluate_state(root.state)
        root.value = agent.score(root.label)
        frontier = [root]
        self.levels = [[root]]
        for _ in range(self.max_depth):
            candidates: list[Node] = []
            for node in frontier:
                if agent.is_solution(node.state):
                    return node
                children = agent.expand(node)
                self.nodes_expanded += len(children)
                candidates.extend(children)
            if not candidates:
                return None
            # keep the top-b states by value (paper's pruning step)
            candidates.sort(key=lambda n: (-n.value, n.thought))
            frontier = candidates[: self.width]
            self.levels.append(frontier)
            for node in frontier:
                if agent.is_solution(node.state):
                    return node
        return None


class DFSSearch:
    """DFS over thoughts with value-based pruning and backtracking.

    Recurse into children ordered by value; prune (backtrack) any state whose
    value falls below ``threshold`` - the paper's mechanism for abandoning
    dead-end reasoning paths and trying alternatives.
    """

    def __init__(self, max_depth: int = 3, threshold: float = 1.0) -> None:
        self.max_depth = max_depth
        self.threshold = threshold
        self.nodes_expanded = 0
        self.pruned = 0

    def run(self, agent) -> Node | None:
        root = Node(state=agent.task.initial_state(agent.numbers), depth=0)
        root.label = agent.evaluate_state(root.state)
        root.value = agent.score(root.label)
        return self._dfs(agent, root)

    def _dfs(self, agent, node: Node) -> Node | None:
        if agent.is_solution(node.state):
            return node
        if node.depth >= self.max_depth:
            return None
        children = agent.expand(node)
        self.nodes_expanded += len(children)
        children.sort(key=lambda n: (-n.value, n.thought))
        for child in children:
            if child.value < self.threshold:
                self.pruned += 1
                continue  # backtrack: abandon this reasoning path
            node.children.append(child)
            found = self._dfs(agent, child)
            if found is not None:
                return found
        return None
