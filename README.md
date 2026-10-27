# Fixing Gemma's copying errors by changing one tensor — evidence package

**Author:** Michi Yamamoto, UserLab · **Research contact:** michi@userlab.org · **GitHub:** [michi883](https://github.com/michi883)

Data, scripts and checks behind the paper *"Fixing Gemma's Copying Errors by Changing One Tensor"* (Kaggle Gemma 4 Developer Agent Paper Track). The package is meant for independent checking: the per-item records behind every number, the scripts that produced them, and a checker you can run in seconds without a GPU.

> **Scope.** One model release, one tensor, selected test sets. These results show a targeted repair of a specific copying weakness. They do **not** measure general coding ability, and the test sets were enriched for the vulnerable pattern (see [Limits](#limits)).

## What was found, in plain language

**The problem.** Asked to copy the text `directs`, the 4-bit-ready (QAT) release of Gemma 4 E4B sometimes writes `direct`. A coding agent that has just read `document.get("diagrams", [])` can then send its edit tool the old text `document.get("diagram", [])`; the tool cannot find it and the edit fails. The failure concentrates where the tokenizer splits a word into a stem plus a lone `s` (`directs` = `direct` + `s`): 260 of 432 such identifiers in the scanned Python corpus (60%) were copied wrongly, against 8 of 7,224 (0.11%) of all other tested words.

**Which release.** Google's QAT release of **Gemma 4 E4B**. The ordinary (non-QAT) E4B and the E2B QAT model copied the probe contexts correctly in our controls. No other model sizes or families were tested in this package.

| checkpoint | observed here |
|---|---|
| `google/gemma-4-E4B-it-qat-q4_0-gguf` → `gemma-4-E4B_q4_0-it.gguf` (sha256 `676c3507…`) | fails: 268 of 7,688 words, 274 of 2,000 plurals, 368 of 1,278 code blocks (`copying/`) |
| `google/gemma-4-E4B-it-qat-w4a16-ct` (the competition checkpoint) | stores the same small tensor (norm 2.05 vs 129.9 in the standard release, `tensor/inspect_tensor.py`); the unmodified model failed 15 of 18 copy inputs on Kaggle vLLM (`kaggle/`) |
| `google/gemma-4-E4B-it` (standard release) | correct final `s` is the top choice in 10 of 10 probe contexts (4-bit-rounded llama.cpp and full-precision mlx) (`tensor/results/rc_data/`) |
| Gemma 4 E2B QAT | 10 of 10 probe contexts correct; its signal sits about 3× above the normalization floor |

**The cause we traced.** Comparing 665 tensors with the standard release (`tensor/results/weights_scale_audit.json`), exactly one, `per_layer_model_proj.weight`, is far smaller (ratio 0.0155 ≈ 1/65; all others lie between 0.46 and 1.20). Its output feeds an RMSNorm whose safety constant is 1e-6. The signal's mean square is about 7e-7 to 9e-7, so the constant is as large as the signal and the normalization no longer cancels the tensor's small size. Multiplying the tensor by 0.25 made copying worse; at ×2 the correct `s` was the top prediction in 10 of 10 selected contexts (1 of 10 as shipped), and larger multipliers behaved the same. We do not know why the final `s` is the affected token, or whether the small tensor came from training or from conversion to the released format.

## Three configurations — never pooled

| | what it is | where it was run | supports |
|---|---|---|---|
| **A. Direct ×2** | Google's QAT Q4_0 GGUF with `per_layer_model_proj.weight` multiplied by 2 in place (`tensor/apply_fix.py`; sha256 `3a6ad54e…`) | local llama.cpp b11146, Apple Silicon | copy tables (268 → 1, 274 → 1, 368 → 15), dose-response, Figures 1 and 2 |
| **B. Rank-128 adapter, merged into the GGUF** | the same GGUF with the adapter file added into that tensor, `W + B·A` (`adapter/merge_adapter_gguf.py`; sha256 `ce8b1244…`) | local llama.cpp b11146 | the **seven coding tasks** (1 of 7 → 5 of 7), the 427-item copy check, token probabilities, Figure 3 |
| **C. Rank-128 adapter, loaded dynamically** | the unmodified `gemma-4-e4b-it-qat-w4a16-ct` plus the same adapter file as a vLLM LoRA | Kaggle, vLLM 0.19.1, 4×L4 | the 18 copy inputs only (15 → 1 failures) |

The seven coding tasks were **not** run on Kaggle vLLM. The adapter only approximates ×2 (the merged tensor has 1.33× the original norm, not 2×), so B and C are not the same model as A.

## Reproducing each part

Full commands, expected output and what was actually tested: [`REPRODUCE.md`](REPRODUCE.md). There is no one-command reproduction: the full experiments need multi-gigabyte checkpoints, a local llama.cpp or a Kaggle GPU session, and (for the coding tasks) a competition package that is not redistributed.

1. **Check the package (seconds, no GPU).** `python3 verify_release.py` checks every file hash and recomputes the headline numbers from the per-item records.
2. **Reproduce the ×2 correction.** `tensor/apply_fix.py` doubles one tensor of Google's public GGUF; the output is byte-identical to the file used in the experiments (needs the 5 GB GGUF). `tensor/inspect_tensor.py` re-measures the tensor in the public checkpoints with 125 MB of range requests (12 s) and prints the 1/64.5 ratio.
3. **Rebuild the adapter.** The adapter is the best rank-128 approximation (truncated SVD) of the dequantised W4A16 tensor, stored as bf16 LoRA factors; no data, no gradient steps. An exact ×2 would need rank 2,560, above the serving limit of 512. `inspect_tensor.py --save-w4a16` followed by `adapter/build_adapter.py` rebuilds `adapter/adapter_model.safetensors` byte-for-byte on the machine we tested (macOS arm64, numpy 2.5.4); other BLAS builds may differ in the last bits.
4. **Repeat the Kaggle comparison.** [`kaggle/copy-test-lora.ipynb`](kaggle/copy-test-lora.ipynb) is the executed notebook (outputs kept). To rerun it you need the competition wheelhouse, the E4B W4A16 model (version 2), 4×L4 GPUs, and the adapter as a Kaggle dataset. Recorded evidence: [`kaggle/evidence/`](kaggle/evidence/).
5. **Regenerate the three figures.** `pip install -r requirements.txt` then `python3 verify_release.py --figures`; or `cd figures && python3 generate_figures.py --plot-only`. No model inference is involved.

## What needs substantial resources

| step | resources | time we measured |
|---|---|---|
| Re-run the copy probes (`copying/run.py`) | llama.cpp b11146 with the original and the ×2 GGUF (5 GB each), Apple Silicon or GPU | words 32 min, plurals 16 min, blocks 37 min per model |
| Re-run the 665-tensor comparison and activation audit (`tensor/audit/`) | the GGUF plus the standard E4B in bf16 (~16 GB), mlx on Apple Silicon | not re-run for this release |
| Kaggle vLLM comparison | Kaggle notebook, 4×L4, competition datasets | one server start plus 36 requests |
| Seven coding tasks (`coding-tasks/harness-reference/`) | llama.cpp with GGUFs A and B **and** the competition's `swegemma` 0.2.11 package | not reproducible from this repository alone |

## Limits

- The copy test sets are **enriched** for the vulnerable pattern (every lone-`s` name in the corpus, plus samples of other classes). Percentages are not everyday copying error rates.
- Each prompt was run once (temperature 0, seed 1, thinking off). Several results are single deterministic trajectories.
- The Kaggle check is 18 hand-picked inputs, text only, no control set; "0 regressions" means 3 inputs.
- The seven coding tasks are hand-built around known failures, one run per task and arm, sign test p = 0.125. Two tasks did not flip (`aligns`, `extracts`).
- The healthy-model reversal is partial (6 of 10 probe contexts fail when the tensor is shrunk in the 4-bit-rounded standard model, 2 of 10 at full precision). Thinking-on, standard benchmarks and other models are untested.
- `figures/FIGURES-NOTES.md` lists known inconsistencies in the evidence (for example 1,277 vs 1,278 blocks in `analyze.py` output, caused by one timed-out block that was re-run).

## Layout

```
copying/        probe runner (run.py), analyzer, frozen test inputs (plan_*.json), per-item results (gzip), analysis tables, run logs
tensor/         apply_fix.py (×2), inspect_tensor.py (public-data check), results/ (audits, dose-response, probe scans), audit/ (reference scripts)
adapter/        the adapter (config + weights + SHA256SUMS), build_adapter.py, lowrank.py, merge_adapter_gguf.py, results/ (427-item copy checks)
coding-tasks/   the seven tasks, run records, replays and probes, harness-reference/ (not runnable without the competition package)
kaggle/         executed notebook and the evidence archive (also unpacked)
figures/        generate_figures.py, figure-data.json, the three figures (PNG + SVG), source-data/, FIGURES-NOTES.md
verify_release.py   hash, data and figure checks          PROVENANCE.md / PROVENANCE.tsv   where every file came from and what was changed
```

Scripts are the research scripts. They are unchanged except for local-path placeholders (listed per file in `PROVENANCE.tsv`); two small scripts are new for this release: `verify_release.py` and `tensor/inspect_tensor.py`.

## Licence

Original code, documentation, data and figures: Copyright 2026 Michi Yamamoto, Apache-2.0 (`LICENSE`, `NOTICE`). Questions about the research: michi@userlab.org. Third-party material — Gemma 4 weights and the adapter derived from them, code excerpts used as test inputs, competition resources — is covered by [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Original Gemma checkpoints are not included.
