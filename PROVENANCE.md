# Provenance

Every file in this release came from the private research repository the experiments were run in, except the files listed under "New in this release". `PROVENANCE.tsv` has one row per file (222 rows): release path, source path in the research repository, sha256 of the source file, sha256 of the release file, and the transformation. `SHA256SUMS` covers every file of the release.

## Where the directories came from

| release | research repository |
|---|---|
| `copying/discovery/`, `copying/copy_page/` | `experiments/identifier_fidelity/discovery/`, `…/copy_page/` (kept as siblings because `run.py` imports `../copy_page/server.py`) |
| `tensor/apply_fix.py` | `experiments/identifier_fidelity/gguf_scale_patch.py` |
| `tensor/audit/` | `experiments/identifier_fidelity/{norm_input_check,dose_response,root_cause,mlx_scan}.py`, `experiments/tensor_audit/*.py`, `docs/figures/verify_dose_gain.py` |
| `tensor/results/` | `experiments/tensor_audit/*.json|log`, `experiments/identifier_fidelity/rc_data/` (dose-response, probe scans), `docs/figures/source/` |
| `adapter/` | `experiments/lora_feasibility/` (adapter `ple_proj_svd_r128`, `build_adapter.py`, `lowrank.py`, three result files, plan), `gemma-edit-tasks/experiments/{merge_adapter_gguf.py,validation/}` |
| `coding-tasks/` | `gemma-edit-tasks/` (tasks, `tasks.json`, README), `gemma-edit-tasks/experiments/` (run records, replays, two probes, analysis, harness scripts) |
| `kaggle/` | `docs/kaggle-vllm-evidence/` |
| `figures/` | `docs/figures/` |

## Transformations (the only differences from the research files)

| count | kind | what was changed | how it was verified |
|---|---|---|---|
| 189 | verbatim | nothing (includes the 6 members of the Kaggle archive, which are byte-identical to the archive's contents) | source and release sha256 equal in `PROVENANCE.tsv` |
| 16 | **metadata path scrub** (13 plain, 3 also gzip'd) | the `file` field of code-block records (`plan_blocks.json`, `plan_retry.json`, `results_blocks_*`, `results_retry_*`) and the `model_path` field of `*.meta.json` and of the four 427-item result files in `adapter/results/`. Prefixes replaced: the author's virtualenv `site-packages/` → `site-packages/`, the Homebrew Python 3.14 library → `stdlib/`, a scratch snapshot of `requests` → `requests-snapshot/`, other scratch directories → `scratch/`, the models folder → `models/` | the build script parsed source and release as JSON and asserted that every leaf value is identical except the scrubbed strings, each of which equals the scrub function applied to the original; no other field differs |
| 4 | verbatim + gzip | `results_{words,plurals}_{orig,patched}_nothink.jsonl` compressed (gzip level 9, no timestamp); with the 3 scrubbed + gzip'd files the 81 MB of per-item results become 11 MB | `copying/discovery/UNCOMPRESSED.sha256` holds the sha256 of the unzipped bytes of all 7 gzip'd files; `verify_release.py --hashes` checks it |
| 12 | **path placeholders in scripts** (10 plain, 2 with an environment variable, next row) | absolute paths of the author's machine replaced by `/path/to/…` (llama.cpp `gguf-py`, Hugging Face cache, the models folder, the identifier_fidelity directory), by `python` (interpreter), or by `/tmp/gemma-edit-work` (scratch directory in `run_arm.py`, `cross_replay.py`) | `diff` against the source shows only those lines |
| (2 of the 12) | scripts with an environment variable | `tensor/apply_fix.py` and `adapter/merge_adapter_gguf.py`: the hard-coded `sys.path.insert(…gguf-py)` became `if os.environ.get('GGUF_PY'): sys.path.insert(0, os.environ['GGUF_PY'])` (+ `import os`) so the scripts run with the PyPI package `gguf` | output of both scripts compared byte-for-byte with the GGUFs used in the experiments (see `REPRODUCE.md`) |
| 1 | banner | `figures/FIGURES-NOTES.md` (the research repository's `docs/figures/README.md`) with a 4-line note that its paths name the research layout | — |

No experiment result value, number, model output, prompt or setting was changed. The scripts that produced the data (`run.py`, `analyze.py`, `server.py`, `generate_figures.py`, `build_adapter.py`, `lowrank.py`) are byte-identical to the originals.

## Equivalence checks run on the release itself

- `generate_figures.py --check` run on a layout rebuilt from the **released** (scrubbed, gzip'd) files re-extracts every plotted number and equals the shipped `figure-data.json` (26 of 26 claim checks); the six figure files regenerate byte-identically.
- `copying/discovery/analyze.py words|plurals|blocks nothink --json` on the unzipped release data reproduces `analysis_*_nothink.json` byte-for-byte; the text tables reproduce with `--list 500`, `--list 40` and `--list 60` respectively.
- `inspect_tensor.py` → `build_adapter.py` rebuilds `adapter_model.safetensors` and `adapter_config.json` byte-for-byte; `apply_fix.py` and `merge_adapter_gguf.py` rebuild the ×2 and merged GGUFs byte-for-byte.

## Deliberately left out

| omitted | reason |
|---|---|
| `coding-tasks` `tools.json`, `transcript.json`, `runs/_discarded/` | `tools.json` and the first message of every `transcript.json` are text of the competition's `swegemma` package, which has no licence information and is competition-supplied code that the rules (§4(b), as relayed by the author) forbid redistributing to nonparticipants; `_discarded` holds a superseded pilot and a re-run. `events.jsonl`, `result.json` and `final.patch` of every run are kept. |
| lab narrative reports (`ROOT_CAUSE.md`, `tensor_audit/REPORT.md`, `identifier_fidelity/REPORT.md`, `gemma-edit-tasks/experiments/REPORT.md`, `lora_feasibility/REPORT.md`) | working notes of the lab; `gemma-edit-tasks/experiments/REPORT.md` still describes the Kaggle vLLM run as blocked and not run, which is outdated (the evidence in `kaggle/` exists). The paper and this README carry the claims; the data and scripts are the evidence. |
| default-template ("page") runs, 11 MB `plural_population.json`, 22 MB `pool_words.json`, plan-building scripts | not used by the paper; the pools depend on the author's installed packages and cannot be rebuilt elsewhere. The frozen plans (`plan_*.json`) are the test inputs. |
| `lora_feasibility` variants other than the original, ×2 and rank-128 SVD runs (rank 16/512, data-aware, competition-tensor variants), llama-server logs | not in the paper's claims; the llama-server logs contain local paths |
| 24 of 26 probe files in `gemma-edit-tasks/experiments/probes/` (27 MB) | only the two files behind Figure 3's `diagrams` probabilities are needed by `generate_figures.py` |
| `exact_copy_probe/`, `real_edits/`, `copy_page/` UI and logs, `compare_*`, 31B and other-model material | other experiments of the lab, not used for the paper's claims |

## New in this release

| file | purpose |
|---|---|
| `verify_release.py` | hash, data and figure checks (standard library; `--figures` needs numpy and matplotlib) |
| `tensor/inspect_tensor.py` | measures the tensor in the three public checkpoints by HTTP range requests; reuses the unpacking logic of `experiments/lora_feasibility/stepb/{fetch,deq_compare}.py`, without their dependency on the 5 GB GGUF |
| `README.md`, `REPRODUCE.md`, `PROVENANCE.md`, `PROVENANCE.tsv`, `SHA256SUMS`, `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `requirements.txt`, `adapter/SHA256SUMS`, `copying/discovery/UNCOMPRESSED.sha256`, `tensor/audit/README.md`, `coding-tasks/harness-reference/README.md` | documentation and bookkeeping |
