# Tree of Thoughts

A runnable Python implementation of **"Tree of Thoughts: Deliberate Problem Solving with Large Language Models"** — Yao, Yu, Zhao, Shafran, Griffiths, Cao & Narasimhan (Princeton / Google DeepMind), NeurIPS 2023 · [arXiv:2305.10601](https://arxiv.org/abs/2305.10601).

Standard LLM decoding is token-level and left-to-right: once the model commits to a reasoning step, it can't take it back. Tree of Thoughts (ToT) generalizes Chain-of-Thought into **search over coherent "thought" units** — the model proposes multiple candidate next steps, *self-evaluates* how promising each partial state is, and explores the most promising branches with backtracking. On Game of 24 the paper reports GPT-4 jumping from **4% (CoT) to 74% (ToT)**.

This repo implements the paper's four modules and both search strategies, with the paper's Game of 24 task as the bundled demo. It runs **fully offline** — no API key needed.

## Setup

```bash
git clone https://github.com/sulaimanvesal/tree-of-thoughts.git
cd tree-of-thoughts
pip install -r requirements.txt   # pytest only
```

## Usage

Solve Game of 24 out of the box (offline `MockBackend` stands in for the paper's propose/value prompts with exact arithmetic):

```bash
python examples/run_demo.py                 # BFS on 4 9 10 13
python examples/run_demo.py 4 9 10 13       # any four numbers
python examples/run_demo.py --search dfs    # DFS with backtracking
python examples/run_demo.py --evaluator vote  # comparative voting evaluator
```

Run the tests:

```bash
pytest -q
```

### Use it as a library

```python
from tot import ToTAgent, MockBackend, OpenAICompatibleBackend

# offline, deterministic
agent = ToTAgent(MockBackend(), numbers=(4, 9, 10, 13), search="bfs", width=5)
solution = agent.solve()
print(agent.format_solution(solution))

# with a real model (OpenAI-compatible endpoint)
agent = ToTAgent(OpenAICompatibleBackend(), numbers=(4, 9, 10, 13))
solution = agent.solve()
```

To plug in your own task (e.g. the paper's Creative Writing), subclass the pattern in `tot/tasks.py`: define `initial_state`, `propose_prompt`, `evaluate_prompt`, `apply_thought`, and `is_solution` — the agent and both searchers work unchanged.

To use a real LLM, set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL` / `TOT_MODEL`); `OpenAICompatibleBackend` issues the paper's propose-style and value-style prompts.

## Architecture

```mermaid
flowchart TB
    subgraph ToTAgent["ToTAgent (tot/agent.py)"]
        E["evaluate_state()\n§3.3 state evaluation"]
        X["expand()\n§3.2 thought generation"]
    end
    subgraph Search["Search (tot/search.py)"]
        BFS["BFSSearch\nkeep top-b states per depth"]
        DFS["DFSSearch\nprune + backtrack below threshold"]
    end
    subgraph Backend["LLMBackend (tot/llm.py)"]
        Mock["MockBackend\npropose/value, deterministic"]
        OAI["OpenAICompatibleBackend\nreal propose/value prompts"]
    end
    Task["GameOf24 (tot/tasks.py)\n§4.1: state = remaining numbers\nthought = one equation"]

    ToTAgent -->|chooses| Search
    X -->|generate()| Backend
    E -->|evaluate()| Backend
    X <-->|thought strings| Task
    Search -->|expands nodes via| ToTAgent
```

## Paper → code mapping

| Paper (arXiv:2305.10601) | This repo |
|---|---|
| §3.1 Thought decomposition — a thought is one intermediate step; a state holds the partial solution | `tot/tasks.py` — `GameOf24.initial_state()`; a state is `{numbers, history}` |
| §3.2 Thought generation — "propose" strategy: LM proposes candidate next steps jointly | `tot/llm.py` — `LLMBackend.generate()`; `MockBackend` enumerates candidate moves propose-style; `OpenAICompatibleBackend` issues the propose prompt |
| §3.3 State evaluation — "value each state independently" vs "vote across states" | `tot/llm.py` — `LLMBackend.evaluate()` returns `sure` / `likely` / `impossible`; `tot/agent.py` — `evaluator="value"` ranks independently, `evaluator="vote"` runs a pairwise tournament |
| §3.4 Search — BFS keeping top-b states per step | `tot/search.py` — `BFSSearch(width=5)` (paper's Game-of-24 setting) |
| §3.4 Search — DFS with backtracking when value is low | `tot/search.py` — `DFSSearch(threshold=...)` prunes and backtracks |
| §4.1 Game of 24 — depth-3 search over intermediate equations | `examples/run_demo.py` — solves 4 9 10 13 offline, prints the tree level by level |

Notes on fidelity: the paper's value prompt asks the LM to judge reachability heuristically; `MockBackend` substitutes exact reachability analysis so the demo is deterministic and offline — swap in `OpenAICompatibleBackend` for the genuine LM-judgement behavior. Division is restricted to exact integer results to keep the demo aligned with the paper's examples.

## Project layout

```
tree-of-thoughts/
├── tot/
│   ├── __init__.py      # public API
│   ├── agent.py         # ToTAgent: the four paper modules wired together
│   ├── llm.py           # LLMBackend, MockBackend, OpenAICompatibleBackend
│   ├── tasks.py         # GameOf24 task (paper §4.1)
│   └── search.py        # Node, BFSSearch, DFSSearch (paper §3.4)
├── examples/
│   └── run_demo.py      # offline Game-of-24 demo (no API key)
├── tests/
│   └── test_tot.py      # 9 tests: generation, evaluation, BFS, DFS
├── requirements.txt
├── LICENSE              # MIT
└── README.md
```

## Citation

```bibtex
@inproceedings{yao2023tree,
  title={Tree of Thoughts: Deliberate Problem Solving with Large Language Models},
  author={Yao, Shunyu and Yu, Dian and Zhao, Jeffrey and Shafran, Izhak
          and Griffiths, Thomas L. and Cao, Yuan and Narasimhan, Karthik},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## License

MIT — see [LICENSE](LICENSE).
