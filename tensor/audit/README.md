# Reference scripts for the tensor analysis

These are the research scripts that produced the files in `../results/`. They are shipped **as evidence of how the numbers were made**; they were written for one Apple-silicon machine and are **not** runnable as they stand. None of them was re-run for this release. What *was* checked is the recorded output (`python3 ../../verify_release.py --data --figures`) and the one public re-measurement in `../inspect_tensor.py`.

Absolute paths of the author's machine were replaced by placeholders (list in `../../PROVENANCE.tsv`); edit these constants before use:

| placeholder | meaning |
|---|---|
| `/path/to/llama.cpp/gguf-py` | `gguf-py` of llama.cpp commit `7fe450e19` (build b11146); `pip install gguf==0.19.0` is an equivalent for reading/writing GGUF, but also needs `pyyaml` |
| `/path/to/models/gemma-4/gemma-4-E4B_q4_0-it.gguf` | Google's QAT GGUF (sha256 `676c3507…`, see `../../REPRODUCE.md` for the source); `gemma-4-E2B_q4_0-it.gguf` likewise |
| `/path/to/.cache/huggingface/hub/` | local Hugging Face cache holding `mlx-community/gemma-4-E4B-it-bf16` (the "standard" E4B in bf16, ≈16 GB) and, for E2B, `google/gemma-4-E2B-it` and `…-qat-q4_0-unquantized` |
| `/path/to/identifier_fidelity` | directory containing `mlx_scan.py`, `copy_page/` (here: this directory and `../../copying/copy_page`) |

| script | produces | needs |
|---|---|---|
| `weights_scale_audit.py` | `../results/weights_scale_audit.json|log` — RMS of all 665 comparable text-tower tensors in the QAT GGUF vs the standard bf16 release (28 s) | mlx, the GGUF, the bf16 snapshot |
| `norm_input_check.py` | `../results/norm_input_check_stdout.txt` — mean square at the RMSNorm input against ε = 1e-6, E4B and E2B (13 s) | mlx, GGUFs, HF caches |
| `activation_audit.py`, `mlx_scan.py` | `../results/activation_audit.json|log`, `../results/rc_data/mlx_scan.jsonl` — activations in mlx with the tensor ×1 vs ×2; the full-precision half of the "shrink the healthy model" test | mlx-lm, the GGUF, the bf16 snapshot; ≈30 GB memory (run nothing else) |
| `root_cause.py` | `../results/rc_data/*/scan.jsonl` — teacher-forced probability of the correct final `s` in ten contexts, one request per context | a running llama-server |
| `dose_response.py` | `../results/dose_response.jsonl` — scans, perplexity and gain at ×0.25 … ×50.6 | llama-server, llama-perplexity, the GGUF, and a text file `PPL_TEXT` — **the text used is not recorded anywhere in this package, so the perplexity column cannot be recomputed** |
| `verify_dose_gain.py` | `../../figures/source-data/dose_gain_recompute.json` — norm gain at the eight multipliers from the stored weights | the GGUF, numpy, gguf-py |

For an independent check of the central observation that needs none of this, run `../inspect_tensor.py` (125 MB of downloads, 12 s).
