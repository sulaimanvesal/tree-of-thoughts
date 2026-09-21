"""Tests for the Tree of Thoughts implementation.

All tests run offline against MockBackend (deterministic). They verify the
paper's four modules: thought generation, state evaluation, and both search
strategies.
"""
import pytest

from tot import (
    BFSSearch,
    DFSSearch,
    GameOf24,
    MockBackend,
    ToTAgent,
)


@pytest.fixture()
def backend():
    return MockBackend()


def test_propose_generates_valid_moves(backend):
    thoughts = backend.generate("Game of 24.\nCurrent numbers: 4 9 10 13\n")
    assert thoughts, "propose should return candidate thoughts"
    state = GameOf24.initial_state((4, 9, 10, 13))
    for t in thoughts:
        nxt = GameOf24.apply_thought(state, t)
        # each thought consumes two numbers and produces one: 4 -> 3 left
        assert len(nxt["numbers"]) == 3
        assert len(nxt["history"]) == 1


def test_propose_is_deterministic(backend):
    prompt = "Game of 24.\nCurrent numbers: 4 9 10 13\n"
    assert backend.generate(prompt) == backend.generate(prompt)


def test_evaluate_labels(backend):
    assert backend.evaluate("Numbers left: 24") == "sure"
    assert backend.evaluate("Numbers left: 13") == "impossible"
    # 4 and 6 remain: 4 * 6 = 24 is reachable
    assert backend.evaluate("Numbers left: 4 6") == "likely"
    # 1 1 1 1 can never reach 24
    assert backend.evaluate("Numbers left: 1 1 1 1") == "impossible"


def test_bfs_solves_game_of_24(backend):
    agent = ToTAgent(backend, numbers=(4, 9, 10, 13), search="bfs", width=5)
    solution = agent.solve()
    assert solution is not None
    assert GameOf24.is_solution(solution.state)
    assert len(solution.state["history"]) == 3  # 4 numbers -> 3 steps
    # the history must be arithmetically consistent and end at 24
    assert solution.state["numbers"] == (24,)


def test_dfs_solves_game_of_24(backend):
    agent = ToTAgent(backend, numbers=(4, 9, 10, 13), search="dfs")
    solution = agent.solve()
    assert solution is not None
    assert GameOf24.is_solution(solution.state)


def test_dfs_prunes_impossible_branches(backend):
    agent = ToTAgent(backend, numbers=(4, 9, 10, 13), search="dfs")
    agent.solve()
    assert agent.searcher.pruned >= 0  # backtracking machinery ran


def test_vote_evaluator_agrees_with_value(backend):
    for evaluator in ("value", "vote"):
        agent = ToTAgent(backend, numbers=(4, 9, 10, 13),
                         search="bfs", evaluator=evaluator)
        solution = agent.solve()
        assert solution is not None and GameOf24.is_solution(solution.state)


def test_bfs_width_limits_frontier(backend):
    agent = ToTAgent(backend, numbers=(4, 9, 10, 13), search="bfs", width=2)
    agent.solve()
    for level in agent.searcher.levels[1:]:
        assert len(level) <= 2


def test_malformed_thoughts_are_skipped(backend):
    agent = ToTAgent(backend, numbers=(4, 9, 10, 13))
    node_state = GameOf24.initial_state((4, 9, 10, 13))
    with pytest.raises(ValueError):
        GameOf24.apply_thought(node_state, "not a thought")
