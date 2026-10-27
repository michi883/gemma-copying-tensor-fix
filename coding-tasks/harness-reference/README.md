# Harness scripts for the seven coding tasks (reference only)

The seven tasks (`../tasks/`, one Python repository each) were run by an agent loop that reuses, unchanged, the tool implementations and prompt builder of the competition's **`swegemma` 0.2.11** package (installed with `adk_submission` 0.2.13). That package carries no licence information and is not redistributed here, so **these scripts cannot be run from this repository alone**. They are included so the exact procedure is inspectable.

| script | role |
|---|---|
| `harness.py` | one trial: local sandbox directory + baseline git commit, model loop of ~60 lines against an OpenAI-compatible server, grading with `python -m pytest -q` on the final tree. Budgets 28 tool calls / 80 turns / 8 min; temperature 0, seed 1, thinking off (`enable_thinking=false` and server `--reasoning-budget 0`). Reads `tools.json` (the competition's tool declarations, **not included**) |
| `run_arm.py`, `serve.sh` | run all tasks for one arm (`original` = Google's QAT GGUF, `lora` = the GGUF with the rank-128 adapter merged in) against a llama-server on port 8110, one server at a time |
| `analyze.py` | builds `../results/analysis.json|md` from the run records |
| `cross_replay.py`, `probe_edit.py`, `summarize_replays.py` | send the request the original arm made at its decisive edit to both models (replays, token-level probes, 10-seed samples) |

What *is* included and checked: the tasks (each fails its own tests before repair: 1 failed + 1 passed, 2 failed ×5, 2 failed + 2 passed; reproduced with `python3 run_tests.py`), the per-run records (`../results/runs/{original,lora}/<task>/{result.json,final.patch,events.jsonl}`), the replays and the two probes behind Figure 3. `python3 ../../verify_release.py --data` recomputes 1 of 7 vs 5 of 7 from the `result.json` files. The harness's first user message (the competition's prompt wrapper plus each task's `ISSUE.md`) is omitted with `transcript.json`; the system message was `You are an agent. Your internal name is "swe_agent".`

The adapter arm is the rank-128 adapter **merged into the GGUF** (`../../adapter/merge_adapter_gguf.py`), run in llama.cpp. These tasks were **not** run on Kaggle vLLM.
