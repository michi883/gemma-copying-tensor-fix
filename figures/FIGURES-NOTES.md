> **Note for this release.** The paths in this file (`experiments/...`, `docs/...`, `gemma-edit-tasks/...`) name the layout of the private research repository the figures were
> produced in. `../PROVENANCE.tsv` maps each of them to its place in this release, and `python ../verify_release.py --figures` rebuilds that layout in a temporary
> directory and runs `generate_figures.py --check` against it. The text below is the original file, otherwise unchanged except for local-path placeholders.

# Figures for "Fixing Gemma's Copying Errors by Changing One Tensor"

Three deliberately plain graphics, one finding each. The images contain only the visual, short axis labels and exact numbers: **no titles, subtitles or footnotes**. The title, setup and limitations are the captions below, which sit outside the figure in the paper. Methods, sampling designs and the mechanism are kept in "Details for the paper", with their numbers and sources. Every number in the figures is recomputed from the raw experiment files in this repository (see "Where each number comes from"). No model was run to make them.

| file | finding |
|---|---|
| `figure1-copying-errors.{png,svg}` | doubling one tensor: 268 → 1 failed copies out of 7,688 |
| `figure2-tensor-mechanism.{png,svg}` | copying flips from failing to working between ×1 and ×2 |
| `figure3-coding-agent.{png,svg}` | seven coding tasks: 1 passed before, 5 after |
| `generate_figures.py` | extracts every number from the raw files, writes `figure-data.json`, checks it against the paper and reports, draws the six files |
| `figure-data.json` | all extracted numbers with source paths and basis, including everything the simplified graphics leave out (subgroups, intervals, RMSNorm statistics, healthy-model test, probabilities). No timestamps; regenerates byte-for-byte |
| `verify_dose_gain.py` | optional static check of the norm-gain numbers in the dose-response data (needs the local GGUF) |
| `source/` | two archived outputs of static weight calculations that `generate_figures.py` reads |

PNG: 300 dpi (2,580 × 1,410 px, 2,580 × 1,650 px and 2,280 × 1,890 px, i.e. 8.6, 8.6 and 7.6 in wide; designed so that text stays above 9 pt when scaled to a 6.5 in text width). SVG: text is converted to outlines, so it looks the same everywhere but is not selectable. Fonts: DejaVu Sans / Sans Mono (ship with Matplotlib). Colours: Okabe-Ito vermillion `#D55E00` = original, blue `#0072B2` = corrected, grey = other multipliers. The vermillion/blue pair passes the dataviz validator's colour-blind, lightness and contrast checks; values are also written as text, so colour is never the only channel.

## Regenerate

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install matplotlib==3.11.2 numpy==2.5.4          # tested with Python 3.14.8 on macOS
python generate_figures.py                            # extract -> figure-data.json -> check claims -> render
python generate_figures.py --plot-only                # render from figure-data.json only (no raw experiment files needed)
python generate_figures.py --check                    # re-extract, compare with figure-data.json, exit 1 on any difference
```

Two full runs and a `--plot-only` run produced byte-identical PNG, SVG and JSON files (same machine and library versions; other platforms may differ in anti-aliasing).

## Verify

1. **Claim table.** Every run prints 26 checks of recomputed values against numbers stated in `docs/identifier-fidelity-paper-draft-v2.md` and the experiment reports (all `ok`). The run stops without drawing if one fails. They are stored in `figure-data.json` under `claim_checks`.
2. **Independent re-derivation.** Before the script was written the same totals were recomputed with separate throwaway code and agreed (268→1 of 7,688; 274→1 of 2,000; 368→15 of 1,278; 1/7 vs 5/7; 0.88% vs 99.7%).
3. **Static weight check (optional; needs the 5 GB GGUF and `gguf-py`).** `verify_dose_gain.py` recomputes the norm-gain, share-below-floor and mean-square values at all eight multipliers from the stored tensor; they matched the recorded `dose_response.jsonl` values to better than 1e-3. `source/norm_input_check_stdout.txt` is the unedited stdout of `experiments/identifier_fidelity/norm_input_check.py` (13 s, no inference), run for this task.
4. **Visual.** All three PNGs were inspected at full size for clipping, overlap and legibility after the last change, and again scaled to a 6.5 in text width (smallest text 12 pt in the file, about 9 pt at that width). The final files are the ones inspected.

## The three different "corrected" models (never pooled)

| arm | what it is | used in |
|---|---|---|
| **×2 tensor** | Google's QAT E4B Q4_0 GGUF with `per_layer_model_proj.weight` multiplied by 2 directly (sha256 `3a6ad54e…`), local llama.cpp b11146 | Figures 1, 2 |
| **merged adapter (local)** | the same GGUF with the rank-128 truncated-SVD adapter (3,408,208 bytes, sha256 `9ce83a7c…`) merged into that tensor, local llama.cpp | Figure 3 |
| **Kaggle vLLM adapter** | the same adapter file served as a LoRA next to `gemma-4-e4b-it-qat-w4a16-ct` by vLLM 0.19.1, 4×L4; 18 hand-picked copy inputs: base failed 15, adapter 1, 14 fixed, 0 regressions, no control set | **not plotted**; `figure-data.json` → `kaggle_vllm_adapter` |

The rank-128 adapter only approximates ×2 (an exact ×2 has rank 2,560; the serving limit is 512), so the merged adapter is not the ×2 model.

## Captions (paste-ready; they sit outside the figure)

**Figure 1. Doubling one tensor fixed almost every copying error.** Failed exact copies out of 7,688 selected words and identifiers, for the original model (268) and for the same model with one tensor doubled (1); the box shows the original writing `direct` when asked to copy `directs`. Both models are Google's Gemma 4 E4B QAT (4-bit-ready) release run in llama.cpp (Q4_0 GGUF, temperature 0, thinking off, each item copied once); the corrected model is the same file with `per_layer_model_proj.weight` multiplied by 2 directly, not the adapter. The words are a selected, enriched sample (every name ending in a separate "s" token found in Python source plus stratified samples of other words), so the counts are not a population estimate.

**Figure 2. Copying flips from failing to working at ×2.** Probability that the model writes the correct final "s" for ten test cases (one dot each; line = median) as the multiplier on `per_layer_model_proj.weight` changes; the numbers above the plot count the cases in which the correct "s" is the model's top choice (0, 1 and 10 of 10). The model is the original QAT E4B Q4_0 in llama.cpp with only that tensor scaled: ×1 is the shipped setting and ×2 the intervention; probabilities are teacher-forced, each setting was measured once, and the line joins measured points only. The ten cases are a selected set (three are variants of `directs`), not a random sample, so the figure shows the direction of the effect, not a rate.

**Figure 3. Seven coding tasks: 1 passed before the fix, 5 after.** Pass or fail of seven small Python repair tasks for the original model and for the same model with the adapter, with the number passed in the bottom row. "Original" is the QAT E4B Q4_0; "With adapter" is the same file with the rank-128 adapter (built from the tensor by truncated SVD) merged into its weights, run locally in llama.cpp with thinking off and temperature 0, one deterministic run per task. It is neither the directly doubled tensor of Figures 1 and 2 nor the Kaggle vLLM run. The tasks target identifiers known to fail, so this is not a general coding benchmark (four tasks flipped from fail to pass, none the other way; sign test p = 0.125).

## Details for the paper (removed from the graphics, kept here)

### Figure 1: subgroups and sampling

The failures concentrate in one tokenization shape: the tokenizer splits off a lone "s" piece (`directs` = `direct` + `s`; `directing` = `direct` + `ing`; `graphs` = one piece). Failure counts, original → ×2:

| test | group | n | original | ×2 | selection |
|---|---|---|---|---|---|
| words and identifiers in Python source | ends in lone "s" | 432 | 260 (60.2%) | 0 | **census**: all such names in the scanned corpus (Python 3.14 stdlib + site-packages); 32 more names ending in a literal "_s" had 0 failures and are left out |
| | everything else | 7,224 | 8 (0.11%) | 1 (0.01%) | stratified: 2,224 items from exhaustively tested strata, 5,000 random draws; by token class one piece 1/1,297 → 1, other ending 6/3,327 → 0, other 1/2,600 → 0 |
| English plurals from a dictionary | ends in lone "s" | 1,500 | 273 (18.2%, 95% CI 16.3–20.2) | 1 (0.07%, 0.01–0.38) | seeded random draw of 12,801 such plurals |
| | everything else | 500 | 1 (0.20%, 0.04–1.12) | 0 (0%, 0–0.76) | seeded random draw of 136,073 |
| Python snippets (1, 4 or 10 lines) | contains a lone "s" token | 528 | 358 (67.8%, 63.7–71.7) | 6 (1.1%, 0.5–2.5) | balanced by design; random windows within each group |
| | no lone "s" | 750 | 10 (1.3%, 0.7–2.4) | 9 (1.2%, 0.6–2.3) | |

Pooled: words 268 → 1 of 7,688; plurals 274 → 1 of 2,000 (lone-"s" plurals are 8.6% of the dictionary but 75% of this sample); snippets 368 → 15 of 1,278. Intervals are 95% Wilson intervals, meaningful only for the random draws. Outside the lone-"s" group failures are rare (0.1–1.3%). In snippets they do not change (10 → 9), and 9 of the 15 remaining snippet failures are a second, unrelated family (opening `"""`, last character of a block); in the word test the few `-ing`-type failures are also repaired (8 → 1). Run settings: temperature 0, seed 1, thinking off, llama.cpp b11146, original sha256 `676c3507…`.

### Figure 2: the mechanism, in plain words

`per_layer_model_proj.weight` turns each token into a small signal *x* that is fed into each of the model's 42 layers. Before use, an RMSNorm rescales it: output = x ÷ √(mean(x²) + 0.000001). If mean(x²) is much larger than the 0.000001 "floor", the floor is negligible and the tensor's scale cancels out. The shipped QAT tensor is about 65× smaller than in the standard release (Frobenius norm 2.014 vs 129.908; 665 tensors compared, the only one outside 0.46–1.20 of the standard scale), so mean(x²) ≈ 0.00000085 sits at the floor: the norm's output shrinks to 0.68 of its intended size (5th percentile 0.42), differently for every token and layer. At ×2 it is 0.88 (0.68), the same as the E2B QAT sibling as shipped (0.87), which copies correctly. Of the model's RMSNorm families only this one is near the floor; the other seven in the text tower sit at least 490× above it (an eighth, `post_per_layer_input_norm`, has median 3.8 and one structural zero).

Dose-response (`figure-data.json` → `figure2.dose_response`; "gain" = √(mean(x²)/(mean(x²)+0.000001)) computed from the stored weights for 2,000 random tokens × 42 layers, not measured during generation):

| multiplier | contexts with correct "s" on top | median p("s") | min–max p("s") | gain median (5th pct) | pairs below floor | perplexity ± se |
|---|---|---|---|---|---|---|
| ×0.25 | 0/10 | 0.0000 | 0.00000–0.00000 | 0.223 (0.116) | 100.0% | 72.85 ± 2.62 |
| **×1 (shipped)** | **1/10** | 0.0297 | 0.00089–0.77355 | 0.676 (0.423) | 61.4% | 61.57 ± 2.10 |
| **×2 (fix)** | **10/10** | 0.9997 | 0.99893–0.99998 | 0.878 (0.682) | 6.5% | 60.57 ± 2.06 |
| ×4 | 10/10 | 1.0000 | 0.99994–1.00000 | 0.965 (0.881) | 0.0% | 60.54 ± 2.06 |
| ×8 | 10/10 | 1.0000 | 0.99995–1.00000 | 0.991 (0.966) | 0.0% | 60.56 ± 2.06 |
| ×16 | 10/10 | 1.0000 | 0.99995–1.00000 | 0.998 (0.991) | 0.0% | 60.56 ± 2.06 |
| ×32 | 10/10 | 1.0000 | 0.99995–1.00000 | 0.999 (0.998) | 0.0% | 60.57 ± 2.06 |
| ×50.6 (=√2560) | 10/10 | 1.0000 | 0.99995–1.00000 | 1.000 (0.999) | 0.0% | 60.57 ± 2.06 |

The ten contexts: `directs`, `directs now`, `directs then forks`, `forks`, `allow_redirects`, `redirects`, `treats`, `payloads`, `pulls`, `yields`. The ×0.25 probabilities are stored as 0.0 (rounded to 5 decimals; ranks 179,316–262,144 of 262,144). Perplexity: 60 chunks × 512 tokens of one text; each run's standard error exceeds the ×1 → ×2 difference, so the data support "no harm seen", not "improved".

Mean square entering the norm (median over 2,000 random tokens): E4B QAT ×1 8.5e-7, ×2 3.4e-6; E2B QAT 3.0e-6; standard E4B 4.6e-3; standard E2B 1.4e-2.

Reverse experiment (contexts, of 10, where the correct "s" stays the top choice): standard E4B 4-bit rounded (llama.cpp) 10/10 → tensor made 65× smaller (×0.0153) 4/10; standard E4B full precision (mlx) 10/10 → 8/10; shipped QAT 1/10 → ×2 10/10. Shrinking a healthy model recreates the failure only partly; the two rows differ in software as well as precision, and a shrunk standard model is not the shipped QAT model.

### Figure 3: what happened in each task, and the `diagrams` edit

| task | original | adapter | final patch (checked against `final.patch`) |
|---|---|---|---|
| directs | fail | pass | original renames to `direct` everywhere |
| aligns | fail | fail | no edit in either arm; both repeat themselves to the 8,192-token limit |
| extracts | fail | fail | both write `extract_s` |
| treats | pass | pass | copied correctly by both |
| diagrams | fail | pass | original writes `diagram_list`; adapter `diagrams` |
| payloads | fail | pass | original keeps `payload`; adapter `payloads` |
| accepts | fail | pass | original writes `accept` |

For `diagrams`, both models received the identical request (rebuilt byte for byte from the original run, turn 4) and identical generated text up to token 85, where the file's `document.get("diagrams"` is quoted. Next-token probabilities after `document.get("diagram`:

| next piece | original | adapter |
|---|---|---|
| `",` (ends the string, no "s") | 79.3% (chosen) | 0.26% |
| `_` | 14.6% | 0.03% |
| `s` (correct) | 0.88% | 99.7% (chosen) |

The original's `old_string` was `"diagram": document.get("diagram", []),`; the file contains `"diagram": document.get("diagrams", []),`, so the edit tool reported "old_string not found". The adapter sent the file's own line; the edit applied and both tests passed. Repeated 10 times at temperature 0.2: edit applied 0/10 (the original produced no edit call in one of the ten) versus 10/10. Over all six tasks that have an edit, 2 contexts × 10 seeds each: tests pass for 23 of 120 sampled edits (original) vs 68 of 120 (adapter); repeated edits in fixed contexts, not independent tasks. Copied alone, the original gets `diagrams` right; the error appears inside the agent's long context. Thinking off was checked (0 reasoning characters in all 14 runs).

## Where each number comes from

All paths are relative to the repository root; JSON paths are in `figure-data.json`.

| figure | numbers | raw source | basis |
|---|---|---|---|
| 1 | 268 / 1 / 7,688; `directs` → `direct`; subgroup tables | `experiments/identifier_fidelity/discovery/results_{words,plurals,blocks}_{orig,patched}_nothink.jsonl` (+ `results_retry_orig_nothink_b00835.jsonl`), `plan_*.json`, `*.meta.json` | recomputed per item |
| 2 | probabilities, counts of 10, perplexity | `…/identifier_fidelity/rc_data/dose_response.jsonl` | recorded raw values; counts recomputed |
| 2 | gain, floor shares, mean squares, Frobenius norms | same file; `docs/figures/source/norm_input_check_stdout.txt`; `experiments/tensor_audit/*.json` | recomputed from stored weights |
| 2 | reverse experiment | `…/rc_data/{nonqat_q4_0_llamacpp,ptq_proj_x0153}/scan.jsonl`, `…/rc_data/mlx_scan.jsonl`, `…/rc_data/e2b_q4_0/scan.jsonl` | recomputed (arg-max counts) |
| 3 | pass/fail, tests, patches | `gemma-edit-tasks/experiments/runs/{original,lora}/<task>/{result.json,final.patch}`, `analysis.json` | recomputed |
| 3 | probabilities, edit calls, 0/10 vs 10/10, 23/120 vs 68/120 | `…/probes/edit_{original,lora}__ctx-original__05_diagrams.json`, `…/replays/`, `…/replays_sampled/` | recomputed |
| – | Kaggle 18 inputs, 15 → 1 | `docs/kaggle-vllm-evidence/copy_lora_evidence.zip` | recomputed; not drawn |

**Report-derived (not recomputed here):** temperature 0 and no thinking budget for the seven tasks (`gemma-edit-tasks/experiments/REPORT.md`; 0 reasoning characters in every run is checked), perplexity settings (from `dose_response.py`), the Kaggle server configuration, the Frobenius norm 2.05 of the Kaggle W4A16 tensor (not plotted), and "free greedy generation agrees with the arg-max of every row" (`ROOT_CAUSE.md`; the figures use teacher-forced arg-max).

## Missing, inconsistent or ambiguous evidence

*Inconsistent between files*

1. **Code blocks, 1,277 vs 1,278.** `discovery/analysis_blocks_nothink.json` (from `analyze.py`) pairs 1,277 blocks (367 → 15) because block `b00835` timed out in the first original run. The paper's 1,278 / 368 needs the separate retry file, where that block never stops copying (counted as a failure). Correct, but not reproducible from `analyze.py` alone.
2. **Mean square at the floor.** Paper: about 0.0000007. `tensor_audit` (6 sequences): 7.1e-7. `ROOT_CAUSE.md` (named tokens): 4–8e-7. Recomputed over 2,000 random tokens: 8.5e-7. Same regime.
3. **"60% of lone-"s" words".** 260/432 excludes 32 names ending in a literal "_s" (0 failures). Including them it is 260/464 = 56%.
4. `gemma-edit-tasks/experiments/REPORT.md` still says no record of the Kaggle vLLM run exists; the evidence bundle now exists (`docs/kaggle-vllm-evidence/`). The paper is not affected.

*Weak or ambiguous*

5. **The healthy-model reversal is partial** (6 of 10 contexts fail when 4-bit rounded, 2 of 10 at full precision; different software; not the shipped QAT model).
6. **The 10 dose-response contexts are not independent** (three are the same word) and are hand-picked.
7. **Perplexity** differences are inside each run's standard error; the paired difference was not recorded.
8. **×0.25 probabilities** are stored as 0.0 (rounded) and plotted as 0.
9. **Gain and floor shares** are computed from weights, not observed during generation.
10. **Seven tasks, one run each, sign test p = 0.125.** The `diagrams` failure appears only in the agent's context. Task `extracts` was not tried with the direct ×2 tensor.
11. **Kaggle:** 18 hand-picked inputs, no control set, no logprobs, none of the seven coding tasks; its `diagrams` and `dumping` inputs pass on the base model.
12. **Not run anywhere:** the seven tasks on vLLM, a thinking-on arm, a standard coding benchmark, a random (non-enriched) sample for the word test.

## Suggested paper changes (only where the evidence exposes a genuine gap)

The draft itself was not modified.

- **§3:** "reproduced copying errors" → say it recreated them in 6 of 10 probe contexts (4-bit rounded) and 2 of 10 (full precision).
- **§2 table:** the "2,000 English plurals" row is 1,500 lone-"s" plurals plus 500 others (75% lone-"s" by design, 8.6% in the dictionary), and the code-block row is 528 + 750 by design. Say so in the row or a note.
- **§5:** add one clause that the original copies `diagrams` correctly on its own and fails only inside the agent's context.
- **§8 / reproducibility:** note that 1,278 / 368 includes the re-run of block `b00835`.
- **§3:** "mean square about 0.0000007" → "about 0.0000007 to 0.0000009, depending on the tokens".
- **§3 (optional):** "repaired most failures" is conservative: ×2 fixed 10/10 probe contexts and 267 of 268 failing words.
