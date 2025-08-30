#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import agent.main as m
except Exception as e:
    print("ERROR: failed to import agent.main:", e)
    sys.exit(1)

print("DOC_PATHS (ordered):")
for i, p in enumerate(m.DOC_PATHS):
    print(f"  {i}: {p}")

first = Path(m.DOC_PATHS[0]).name if m.DOC_PATHS else None
print("HAS_AGENTS_FIRST:", first == "AGENTS.md")

print("GEN_MODEL:", getattr(m, 'GEN_MODEL', None))
print("EMBED_MODEL:", getattr(m, 'EMBED_MODEL', None))
