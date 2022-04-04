# Train
```bash
# vanilla
bash slurm/vanilla/vanilla_tsm_ek100.sh
# dann
bash slurm/arr_submit.sh slurm/dann/dann_tsm_ek100.sh

# begins 6 hours later
bash slurm/arr_submit.sh slurm/dann/dann_tsm_ek100.sh now+6hour  
```

<br><br>
---

# Test

## `vanilla/vanilla_test_by_jids.sh`
```bash
# vanilla
. slurm/vanilla/vanilla_test_by_jids.sh {16279..16284}
# vanilla model-selected by MCA  
. slurm/vanilla/vanilla_test_by_jids.sh {16847..16852}
```
## `test_by_jids.sh`
```bash
# dann
. slurm/test_by_jids.sh {16748..16753}
```

<br><br>

# Collect Scores (no GPUs needed)

## From Train Results: `print_best_scores.py`
```python
python slurm/print_best_scores.py -j {16279..16284}  # vanilla
python slurm/print_best_scores.py -j {16383..16388}  # vanilla
python slurm/print_best_scores.py -j {16748..16753}  # dann

python slurm/print_best_scores.py -j 16279 -o  # one-line by a single jid
python slurm/print_best_scores.py -smb mca -d ek100 -b tsm -m vanilla -dom P02 -t source-only -o  # one-line by config vars
```

## From Test Results: `print_best_scores_test.py`
```python
```

<br><br>

# Trained Model Descriptions

## Vanilla

### Tasks for Each JID
| JID | Domain | Task | Note |
|---|:---:|---|---|
| `$jid` | P02 | source-only | trained only with shared labels |
| `$jid + 1` | P02 | target-only | trained with all labels |
| `$jid + 2` | P04 | source-only |  |
| `$jid + 3` | P04 | target-only |  |
| `$jid + 4` | P22 | source-only |  |
| `$jid + 5` | P22 | target-only |  |

### Trained Models
| JIDs | Model-Selection Metric | Note |
|---|---|---|
| `{16279..16284}` | Top-1 Acc. |  |
| `{16383..16388}` | Top-1 Acc. |  |
| `{16847..16852}` | Mean-class Acc. |  |

<br>

## DANN
### Trained Models
| JIDs | Model-Selection Metric | Note |
|---|---|---|
| `{16748..16753}` | Top-1 Acc. |  |
| `{16770..16775}` | Top-1 Acc. |  |
| `{x..x}` | Mean-class Acc. |  |

<br>

## OSBP
| JIDs | Model-Selection Metric | Note |
|---|---|---|
| `{16748..16753}` | Top-1 Acc. |  |
| `{x..x}` | Mean-class Acc. |  |
