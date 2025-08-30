# VIRTK Agent

A domain-specific Python coding agent focused on ITK/VTK/RTK pipelines. It provides a CLI, small runnable examples, and a light RAG workflow over local docs.

## Quick Start

- Create venv
  - Windows: `python -m venv .venv && .venv\Scripts\activate`
  - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt` (add if missing)
- Run the agent: `python agent/main.py "<task prompt>"`
- Optional index mode: `python agent/main.py --use-index "<task>"`
- Build API index: `python scripts/gen_api_index.py`
- Tests (if present): `pytest -q`

## Documentation

- Repository guide: `AGENTS.md`
- Agent workflow/style: `docs/AgentStyle.md`
- Quick refs: `docs/ITK_API.md`, `docs/VTK_API.md`, `docs/RTK_API.md`
- Generated API index: `docs_api_index/*.md`

## Project Structure

- `agent/`: CLI and tools (`main.py`, `tools_itk.py`, `tools_vtk.py`, `tools_rtk.py`)
- `docs/`: Guides and quick refs
- `docs_api_index/`: Generated index files
- `examples/`: Small runnable JSON plans
- `scripts/`: Utilities (`gen_api_index.py`, etc.)
- `configs/`: Agent configuration
- `tests/` (optional): Pytest suite

## Requirements

- Python 3.10+
- UTF-8 encoding for all text files (use `scripts/convert_to_utf8.ps1` on Windows if needed)

## Contributing

- Follow `AGENTS.md` guidelines.
- Keep changes focused; include tests for core logic when possible.

