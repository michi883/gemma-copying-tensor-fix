#!/usr/bin/env python3
"""Check the current state of all seven task repos (intentionally failing initially)."""
import subprocess
from pathlib import Path

base = Path(__file__).resolve().parent / "tasks"
for task in sorted(base.iterdir()):
    if not task.is_dir():
        continue
    p = subprocess.run(["python", "-m", "pytest", "-q"], cwd=task, capture_output=True, text=True)
    print(f"{task.name}: {'PASS' if p.returncode == 0 else 'FAIL (expected before fix)'}")
