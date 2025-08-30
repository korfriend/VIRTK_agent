# Repository Guidelines

Agent workflow and style guide: see `docs/AgentStyle.md` (Korean).

## Project Structure & Module Organization
- `agent/`: CLI and core modules (`main.py`, `tools_itk.py`, `tools_vtk.py`, `tools_rtk.py`).
- `docs/`, `docs_api_index/`: API quick refs and generated index files used by the agent.
- `examples/`: Small runnable samples (e.g., `examples/plan_ct_fdk.json`).
- `scripts/`: Utilities like `gen_api_index.py`, `print_agent_config.py`.
- `configs/`: Agent configuration (e.g., `configs/agent.config.json`).
- `tests/` (optional): Pytest suite when present.

## Build, Test, and Development Commands
- Create venv (Windows): `python -m venv .venv && .venv\Scripts\activate`
- Create venv (macOS/Linux): `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt` (add if missing).
- Run the agent: `python agent/main.py "<task prompt>"`
- Use index (optional): `python agent/main.py --use-index "<task>"`
- Build API index: `python scripts/gen_api_index.py`
- Run tests: `pytest -q` (if `tests/` exists).

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, UTF-8.
- PEP 8 + type hints; concise English comments and docstrings.
- Naming: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE`.
- Prefer `argparse`, `logging`, small testable functions; avoid side effects in module scope.

## Testing Guidelines
- Framework: `pytest`; place tests under `tests/` as `test_*.py`.
- Use small synthetic data; avoid large binaries in the repo.
- Cover core logic in `agent/` (tools and CLI flows); add regression tests for bugs.
- Examples: `pytest -q` or `pytest tests/test_tools_itk.py -q`.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (≤72 chars); explain why in body when helpful.
- Reference issues (`#<id>`); one logical change per commit.
- PRs: clear description, testing steps, sample commands, and any relevant logs.
- Keep diffs focused; update docs/examples when behavior changes.

## Agent-Specific Notes
- Prefer runnable, minimal examples; avoid speculative APIs.
- When uncertain, use safe defaults and leave clear TODOs.
- Keep `docs/` and `docs_api_index/` consistent with code changes.

