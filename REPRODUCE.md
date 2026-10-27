# Reproducing the results

There is **no one-command reproduction**. Some steps need nothing but Python; others need multi-gigabyte checkpoints, a local llama.cpp, a Kaggle GPU session, or a competition package that is not redistributed. Every step below says what it needs and whether it was actually run for this release.

**Status key.** ✅ run on 2026-10-27 in a fresh virtual environment with the output shown or described · 🔌 only the script plumbing was run (a stub server, no model) · ⛔ not run here (needs resources we did not use) · 🚫 cannot be run from this repository.

**Test environment.** macOS (Darwin 25.6.0, arm64), Python 3.14.8, numpy 2.5.4, matplotlib 3.11.2, gguf 0.19.0, pytest 9.x. Other platforms should work for ✅ steps; PNG anti-aliasing and the last bits of the SVD may differ on other BLAS/matplotlib builds.

## 0. Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy, matplotlib, gguf
```

## 1. Check the package — seconds, no downloads ✅

```bash
python3 verify_release.py --all
```
Checks every file against `SHA256SUMS`; the adapter against `adapter/SHA256SUMS`; the 7 gzip'd result files against the hashes of their unzipped bytes; recomputes 268 → 1 (7,688 words), 274 → 1 (2,000 plurals), 368 → 15 (1,278 blocks), the Kaggle result (18 inputs, 15 → 1, 14 fixed, 0 regressions), the seven tasks (1 vs 5 passed), the 665-tensor outlier, the adapter's tensor names and shapes, and that the Kaggle manifest, evidence archive and executed notebook agree with the released adapter; regenerates the three figures. Exit status 0 means everything passed. `python3 verify_release.py` without `--figures` needs only the standard library.

## 2. The tensor, measured from public files — ✅ 12 s, 125 MB downloaded

```bash
python3 tensor/inspect_tensor.py --save-w4a16 W_w4a16_deq.npy
```
Reads the Hugging Face safetensors headers and, by HTTP range requests, only the bytes of `per_layer_model_projection` in three pinned revisions. Output (tested):

```
checkpoint    stored as                   shape             Frobenius        rms  vs standard
qat-w4a16-ct  I32+BF16 (int4, dequantised)(10752, 2560)        2.0508  3.909e-04       0.0158
qat-unquant   BF16                        (10752, 2560)        2.0144  3.840e-04       0.0155
standard      BF16                        (10752, 2560)      129.9061  2.476e-02       1.0000
standard / QAT (bf16) = 64.5x;  standard / QAT (W4A16, dequantised) = 63.3x
```
This confirms the key observation for the **competition checkpoint** too. It does *not* redo the comparison of all 665 tensors (§7).

## 3. The training-free adapter, rebuilt — ✅ 4 s after §2

```bash
python3 adapter/build_adapter.py W_w4a16_deq.npy rebuilt_r128 128 svd
(cd rebuilt_r128 && shasum -a 256 adapter_model.safetensors adapter_config.json)   # compare with adapter/SHA256SUMS
```
The adapter is the best rank-128 approximation (truncated SVD) of the dequantised tensor, stored as bf16 factors `B` (10752 × 128) and `A` (128 × 2560); LoRA scaling `alpha/r = 1`, so it adds `B·A` to the tensor. No data, no gradient updates. Tested result: both files **byte-identical** to the released ones (`9ce83a7c…c888`, `3786855f…4dfbe1`). An exact ×2 would need rank 2,560; the serving limit is 512.

## 4. The direct ×2 correction and the merged-adapter model — ✅ 5 GB files, about a minute each

Get Google's GGUF `gemma-4-E4B_q4_0-it.gguf` from <https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf> (revision `4b4a2c1d584be7264f87aac328a1bc739ce81b6c`, 5,154,941,280 bytes, sha256 `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee` — the value Hugging Face lists, and the hash of the file used in all experiments). Then:

```bash
python3 tensor/apply_fix.py gemma-4-E4B_q4_0-it.gguf x2.gguf per_layer_model_proj.weight 2.0
shasum -a 256 x2.gguf     # 3a6ad54ed7a46e5fb9b6c6ccd6b16060f105fcc77ad4d7936c721dcdb4e9dc7c

python3 adapter/merge_adapter_gguf.py adapter gemma-4-E4B_q4_0-it.gguf merged.gguf
shasum -a 256 merged.gguf # ce8b12447330ebfdc41186c99aa10f7c2d969f1071c6ddd9835333ab8889585e
```
Tested with the PyPI `gguf` 0.19.0 on a local copy of that file: both outputs were **byte-identical** to the GGUFs used in the experiments (`gemma-4-E4B_q4_0-it-plproj-x2.gguf`, `…-lora-r128-merged.gguf`). The download command itself was not run here. `apply_fix.py` copies the whole file, then multiplies the one F16 tensor in the copy. Note that the merged tensor has 1.33× the original Frobenius norm (the adapter approximates ×2 but is not ×2).

## 5. Copy tests (Figure 1; configuration A)

**Analysis from the shipped records — ✅.**
```bash
cp -R copying /tmp/copying-run && cd /tmp/copying-run/discovery
gunzip -k results_*.jsonl.gz
python3 analyze.py words nothink --json          # also: plurals, blocks; --list 500 / 40 / 60 reproduce the shipped .txt tables
```
`analysis_{words,plurals,blocks}_nothink.json` are reproduced byte-for-byte, and the text tables with `--list 500` (words), `--list 40` (plurals), `--list 60` (blocks). `analyze.py` reports 1,277 paired blocks (367 → 15) because block `b00835` timed out in the first original run; the paper's 1,278 / 368 includes the separate re-run (`results_retry_orig_nothink_b00835.jsonl.gz`), which `verify_release.py` and `generate_figures.py` apply.

**Re-running the probes — runner plumbing 🔌, model results ⛔.** Start one llama-server per model (flags as recorded in the research notes, which are not part of this package; llama.cpp build b11146, commit `7fe450e19`, Metal):

```bash
llama-server -m gemma-4-E4B_q4_0-it.gguf             --alias gemma-4-e4b --host 127.0.0.1 --port 8101 -ngl 99 -c 32768 -np 1 --flash-attn on --jinja --reasoning-budget 4096
llama-server -m gemma-4-E4B_q4_0-it-plproj-x2.gguf   --alias gemma-4-e4b --host 127.0.0.1 --port 8102 -ngl 99 -c 32768 -np 1 --flash-attn on --jinja --reasoning-budget 4096

# a fresh working directory: run.py resumes and appends to any results file it finds there
mkdir -p /tmp/probe/copying/discovery && cp -R copying/copy_page /tmp/probe/copying/
cp copying/discovery/run.py copying/discovery/analyze.py copying/discovery/plan_*.json /tmp/probe/copying/discovery/
cd /tmp/probe/copying/discovery
python3 run.py --plan plan_words.json --server orig    --mode nothink
python3 run.py --plan plan_words.json --server patched --mode nothink     # likewise plan_plurals.json and plan_blocks.json
```

`run.py` checks that the server's reported model file is named `gemma-4-E4B_q4_0-it.gguf` / `…-plproj-x2.gguf`. Each request is the fixed prompt `Copy the following text exactly. Output only the copied text:\n` + text, temperature 0, seed 1, thinking off (`chat_template_kwargs.enable_thinking=false`). Measured on Apple silicon: words ≈ 32 min, plurals ≈ 16 min, blocks ≈ 37 min per model. In the original block run, block `b00835` exceeded `run.py`'s 180 s request timeout and was re-run once separately with a longer limit (`plan_retry.json`, `results_retry_orig_nothink_b00835.jsonl.gz`; the command of that single re-run is not recorded in the package). We ran the script against a stub server that copies perfectly (25 items of `plan_words.json`: 25 records, all exact) to test the plumbing; **we did not re-run the model**. Results from another llama.cpp build or backend are not guaranteed to be identical.

## 6. Figures — ✅ 2 s

```bash
python3 verify_release.py --figures      # or:  cd figures && python3 generate_figures.py --plot-only
```
`--plot-only` renders the six files from `figures/figure-data.json` with no raw data and no model. `verify_release.py --figures` additionally rebuilds the research repository's layout from the shipped files in a temporary directory and runs `generate_figures.py --check`, which re-extracts every plotted number from the raw per-item records and compares it with `figure-data.json` (26 claim checks). Tested: all pass and all six figure files regenerate byte-identically on the test machine. Figure 1 and 2 are configuration A (direct ×2); Figure 3 is configuration B (merged adapter). The Kaggle result (configuration C) is not plotted; it is in `figure-data.json` → `kaggle_vllm_adapter`.

## 7. Tensor audit, RMSNorm floor, dose-response — ⛔ (data checked ✅)

The 665-tensor comparison (`weights_scale_audit`), the mean-square-versus-ε measurement (`norm_input_check`), the activation audit, the dose-response with perplexity, and the healthy-model reversal were produced by the scripts in `tensor/audit/` on the author's machine. They need the 5 GB GGUF, ≈16 GB of bf16 weights (a community mlx conversion of `google/gemma-4-E4B-it` was used as the standard release), mlx, and for the dose-response a running llama-server and llama-perplexity. See `tensor/audit/README.md`; none was re-run. What can be checked from the records: `verify_release.py --data --figures` recomputes the 665-tensor outlier, all dose-response counts, the reversal counts and the weight statistics from the shipped output files. **The text used for the perplexity numbers (61.57 ± 2.10 at ×1, 60.57 ± 2.06 at ×2) is not recorded in the package, so those two numbers cannot be recomputed.**

## 8. Kaggle vLLM comparison (configuration C) — ⛔ re-run; evidence ✅ checked

What the notebook needs: Kaggle with **4×L4 GPUs**; the competition's **Gemma 4 Developer Agent wheelhouse** dataset (`metric/gemma-4-developer-agent-wheelhouse`, vLLM 0.19.1, 41 wheels); the model **`google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct`, version 2**; and the adapter as a Kaggle dataset at `/kaggle/input/datasets/michi883/ple-proj-svd-r128` containing exactly `adapter_config.json` and `adapter_model.safetensors`. Internet stays off, and that is not optional: as stated by the author from the competition materials (not re-checked here), 4×L4 sessions for this competition are restricted to notebooks attached to the competition `gemma-4-developer-agent` with internet disabled. The saved run used exactly this setup (the saved notebook's metadata: competition attached, internet off, GPU on, NVIDIA L4), so re-running it means attaching that competition. Steps: publish the two adapter files as that dataset (the two files and their hashes are in `adapter/`), open `kaggle/copy-test-lora.ipynb`, attach the three inputs, run all cells, download `copy_lora_evidence.zip` from the Output tab, and compare it with `kaggle/evidence/`.

The notebook starts one vLLM server serving the base model and, with `--enable-lora --max-loras 1 --max-lora-rank 128 --lora-modules ple=<adapter dir>`, the adapter as a second model ID, then sends the same 18 requests to both IDs (the fixed copy prompt, temperature 0, seed 1, `max_tokens` 128, thinking off) and compares each reply with its input by exact string equality. The shipped notebook is the executed original (cells 1–6 run in order, outputs preserved); it has not been edited.

Checked here without a GPU (`verify_release.py --data`): the archive's members equal the extracted files; the 18 paired outputs give 15 → 1 failures, 14 fixed, 0 regressions; thinking off and every reply `finish_reason: stop`; the manifest's adapter hashes equal the released files; the server log shows adapter `ple` loaded and 36 chat requests (18 × 2); the notebook's model path, adapter dataset and printed summary agree. **Limits:** 18 hand-picked inputs, no control set, text only (no log-probabilities), and the full base-checkpoint hash was not recorded (only its `config.json`, whose sha256 `7f0cd2d1…` equals the `config.json` of Hugging Face `google/gemma-4-E4B-it-qat-w4a16-ct` at revision `6cd26aaa…`, checked 2026-10-27). The seven coding tasks were not run on Kaggle vLLM.

## 9. Seven coding tasks (configuration B) — tasks ✅, agent runs 🚫

`python3 coding-tasks/run_tests.py` (needs `pytest` on the path; run it on a copy, it leaves cache files) shows that every task fails its own tests before repair (tested: 1 failed + 1 passed, then 2 failed ×5, then 2 failed + 2 passed). The agent runs need the competition's `swegemma` 0.2.11 package (not redistributed), llama.cpp and the two GGUFs of §4; see `coding-tasks/harness-reference/README.md`. The shipped run records are checked by `verify_release.py --data` (1 of 7 vs 5 of 7 passed; four fail → pass, none pass → fail).

## Summary of what was and was not re-run

| claim | re-derived from shipped records | re-run from public inputs | not re-run |
|---|---|---|---|
| 268 → 1, 274 → 1, 368 → 15 copy failures | ✅ | | model inference (≈4 h on Apple silicon) |
| tensor ≈ 1/65 of standard (E4B QAT, W4A16 competition checkpoint) | ✅ | ✅ (§2) | 665-tensor comparison |
| RMSNorm floor, dose-response, healthy-model reversal | ✅ | | scripts need local weights, mlx, llama.cpp |
| ×2 file and merged-adapter file | | ✅ byte-identical (§4) | |
| adapter file | ✅ | ✅ byte-identical (§3) | |
| Kaggle 15 → 1 on 18 inputs | ✅ (log, manifest, notebook) | | GPU run |
| seven coding tasks 1 → 5 | ✅ (run records) | | agent runs (competition package) |
| figures | ✅ | ✅ | |
