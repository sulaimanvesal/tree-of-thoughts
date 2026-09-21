"""Offline demo: solve Game of 24 with Tree of Thoughts.

Runs entirely without an API key using MockBackend, which stands in for the
paper's propose/value prompts with exact arithmetic. Shows the search tree
level by level, then prints the solution.

Usage:
    python examples/run_demo.py                # default puzzle: 4 9 10 13
    python examples/run_demo.py 4 9 10 13      # any four numbers
    python examples/run_demo.py --search dfs   # DFS with backtracking
    python examples/run_demo.py --evaluator vote
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from tot import BFSSearch, MockBackend, ToTAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree of Thoughts demo: Game of 24")
    parser.add_argument("numbers", nargs="*", default=["4", "9", "10", "13"],
                        help="four numbers for the puzzle")
    parser.add_argument("--search", choices=["bfs", "dfs"], default="bfs")
    parser.add_argument("--evaluator", choices=["value", "vote"], default="value")
    parser.add_argument("--width", type=int, default=5)
    args = parser.parse_args()

    numbers = tuple(int(n) for n in args.numbers)
    agent = ToTAgent(MockBackend(), numbers=numbers, search=args.search,
                     width=args.width, evaluator=args.evaluator)

    print(f"Solving Game of 24 for {list(numbers)} "
          f"({args.search.upper()}, evaluator={args.evaluator})\n")

    solution = agent.solve()

    searcher = agent.searcher
    if isinstance(searcher, BFSSearch):
        for depth, level in enumerate(searcher.levels):
            print(f"--- depth {depth}: {len(level)} frontier state(s) ---")
            for node in level:
                print(f"  {agent.task.state_label(node.state)} "
                      f"value={node.label} ({node.value:g})")
                if node.thought != "<root>":
                    print(f"      via: {node.thought}")
        print(f"\nnodes expanded: {searcher.nodes_expanded}")
    else:
        print(f"nodes expanded: {searcher.nodes_expanded}, "
              f"pruned by backtracking: {searcher.pruned}")

    print()
    if solution is None:
        print("No solution found.")
    else:
        print(agent.format_solution(solution))


if __name__ == "__main__":
    main()
