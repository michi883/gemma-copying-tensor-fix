# Gemma E4B: seven synthetic edit tasks

Seven minimal Python repositories testing whether original E4B and E4B + rank-128 LoRA can apply edits involving previously troublesome identifiers.

## Run a task

```bash
python -m pip install pytest
cd tasks/06_payloads
python -m pytest -q  # should fail before repair
```

## Evaluate fairly

1. Create a fresh copy of the same task repository for each model and trial.
2. Give the agent the content of `ISSUE.md` as its user task, plus the same repo tools and budgets.
3. Do not expose the reference solution or test expectations beyond files in the repository. (The included tests are visible to the agent; use separate hidden tests for a formal benchmark.)
4. Record the whole trajectory, especially `edit_file` arguments and tool errors.
5. After the attempt, run `python -m pytest -q` in that trial's working tree.
6. Score both functional test passing and edit/copy error categories. Any valid patch is acceptable, even if it differs from the intended small edit.
7. Use the same prompts, decoding, token budgets and seeds across base and LoRA; run multiple trials if possible.

## Scope and limitations

These deliberately small tasks use *known* problematic identifiers. They measure targeted editing behavior, not broad SWE-agent ability. Some agents may avoid copying a target word by finding another valid fix. Tests alone do not establish whether a success difference was caused by copying, so review exact generated edits. Existing test files are intentionally visible for easy local debugging; hide independent tests in a publication-grade benchmark.

This repository does not call Gemma or claim any measured model improvement.
