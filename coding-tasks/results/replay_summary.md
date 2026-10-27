## A. Temperature 0: decisive edit replayed in a fixed context

| task | context: original trajectory | | context: LoRA trajectory | |
|---|---|---|---|---|
| | model=original | model=LoRA | model=original | model=LoRA |
| 01_directs | NOT applied (Failed to replace: old_string ) | applied, tests PASS | NOT applied (Failed to replace: old_string ) | applied, tests PASS |
| 02_aligns | n/a (no edit call in that trajectory) | n/a (no edit call in that trajectory) | n/a (no edit call in that trajectory) | n/a (no edit call in that trajectory) |
| 03_extracts | applied, tests fail | applied, tests fail | no edit call | applied, tests fail |
| 04_treats | applied, tests PASS | applied, tests PASS | applied, tests PASS | applied, tests PASS |
| 05_diagrams | NOT applied (Failed to replace: old_string ) | applied, tests PASS | NOT applied (Failed to replace: old_string ) | applied, tests PASS |
| 06_payloads | applied, tests fail | NOT applied (Failed to replace: old_string ) | applied, tests fail | NOT applied (Failed to replace: old_string ) |
| 07_accepts | NOT applied (Failed to replace: old_string ) | NOT applied (Failed to replace: old_string ) | NOT applied (Failed to replace: old_string ) | applied, tests PASS |
| **tests pass (of 6 tasks with an edit)** | 1 | 3 | 1 | 4 |

## B. Lab sampling (temperature 0.2, top_p 0.95), 10 seeds per cell, exploratory

| task | ctx=original: model=original | ctx=original: model=LoRA | ctx=lora: model=original | ctx=lora: model=LoRA |
|---|---|---|---|---|
| 01_directs | applied 0/10, pass 0/10 | applied 10/10, pass 10/10 | applied 3/10, pass 3/10 | applied 10/10, pass 10/10 |
| 02_aligns | n/a | n/a | n/a | n/a |
| 03_extracts | applied 9/10, pass 0/10 | applied 10/10, pass 0/10 | applied 7/10, pass 0/10 | applied 9/10, pass 0/10 |
| 04_treats | applied 10/10, pass 10/10 | applied 10/10, pass 10/10 | applied 10/10, pass 10/10 | applied 10/10, pass 10/10 |
| 05_diagrams | applied 0/10, pass 0/10 | applied 10/10, pass 10/10 | applied 0/10, pass 0/10 | applied 10/10, pass 10/10 |
| 06_payloads | applied 0/10, pass 0/10 | applied 0/10, pass 0/10 | applied 3/10, pass 0/10 | applied 0/10, pass 0/10 |
| 07_accepts | applied 0/10, pass 0/10 | applied 0/10, pass 0/10 | applied 0/10, pass 0/10 | applied 8/10, pass 8/10 |
| **all samples** | applied 19/60, pass 10/60 | applied 40/60, pass 30/60 | applied 23/60, pass 13/60 | applied 47/60, pass 38/60 |
