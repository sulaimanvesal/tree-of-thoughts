"""Tree of Thoughts (ToT): deliberate problem solving with language models.

A runnable implementation of Yao et al., "Tree of Thoughts: Deliberate Problem
Solving with Large Language Models" (NeurIPS 2023, arXiv:2305.10601).

Instead of left-to-right token decoding, ToT searches over coherent *thoughts*:
the LM proposes candidate next steps, self-evaluates them, and a search
algorithm (BFS/DFS) explores the most promising branches, with backtracking.
"""

from .agent import ToTAgent
from .llm import LLMBackend, MockBackend, OpenAICompatibleBackend
from .search import BFSSearch, DFSSearch, Node
from .tasks import GameOf24

__all__ = [
    "ToTAgent",
    "LLMBackend",
    "MockBackend",
    "OpenAICompatibleBackend",
    "BFSSearch",
    "DFSSearch",
    "Node",
    "GameOf24",
]

__version__ = "0.1.0"
