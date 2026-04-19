# Experiment Tracker — COMP560 ReID

Shared table for all parallel training runs. **Push updates to the `results` branch** to avoid main-branch merge conflicts.

Columns:
- **A:** evaluated on `datasets/dataset_a/our_test.parquet` (animals, no same-camera exclusion, self-match suppressed)
- **M:** evaluated on `datasets/market1501/market1501_test.parquet` (persons, same-camera exclusion ON)
- **avg** = unweighted mean of A:mAP and M:mAP — **primary model-selection metric**
- Leave cells blank (`—`) when a metric wasn't run

| exp_id | owner | status | backbone | data | epochs | A: R-1 | A: mAP | M: R-1 | M: mAP | avg | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_resnet50_a_only | Alex | done | ResNet50 | a only | 20 | 70.65 | 68.17 | — | — | — | MPS, pre-plan baseline |
| exp01_teacher_dinov2_large |  | todo | DinoV2-L | a + market | 20 |  |  |  |  |  | main teacher |
| exp02_teacher_dinov2_base |  | todo | DinoV2-B | a + market | 20 |  |  |  |  |  | backup teacher |
| exp03_baseline_resnet50_cross |  | todo | ResNet50 | a + market | 25 |  |  |  |  |  | no-KD baseline |
| exp04_baseline_efficientnet_cross |  | todo | EffNet-B0 | a + market | 25 |  |  |  |  |  | no-KD baseline |
| exp05_student_resnet50_kd |  | todo | ResNet50 | a + market | 25 |  |  |  |  |  | distilled from exp01 |
| exp06_student_efficientnet_kd |  | todo | EffNet-B0 | a + market | 25 |  |  |  |  |  | distilled from exp01 |

## Bars (kill criteria)
- exp03 after 4 epochs on full cross-domain: if <65% R-1 on **both** A and M → unified label space is hurting us. Switch to multi-task heads.
- exp01 teacher final: ≥80% R-1 on A, ≥70% R-1 on M.
- exp05/exp06 student: within 3 pts R-1 of teacher on both domains (else KD is broken).
- Final ensemble: beat best single model by ≥1 pt R-1 / ≥2 pts mAP.

## How to update
```bash
git checkout -b results-$(date +%Y%m%d-%H%M)
# edit RESULTS.md
git add RESULTS.md && git commit -m "results: exp0X <short>"
git push -u origin HEAD
```
Merge to `results` branch when ready; keep main clean.

## Commands cheat-sheet

Train:
```
python train.py --config configs/experiments/<exp>.yaml
```

Predict (both domains per checkpoint):
```
python scripts/make_predictions.py --checkpoint checkpoints/<exp>/best.pth \
  --dataset_root datasets/dataset_a --parquet our_test.parquet \
  --protocol dataset_a --out predictions/<exp>_a.csv

python scripts/make_predictions.py --checkpoint checkpoints/<exp>/best.pth \
  --dataset_root datasets/market1501 --parquet market1501_test.parquet \
  --protocol dataset_b --out predictions/<exp>_m.csv
```

Evaluate:
```
python evaluate_copy.py --student_id <id> --prediction predictions/<exp>_a.csv \
  --datasets dataset_a --parquet our_test.parquet

python evaluate_copy.py --student_id <id> --prediction predictions/<exp>_m.csv \
  --datasets dataset_b --datasets_root datasets \
  # (requires datasets/dataset_b/test.parquet; for Market as dataset_b, symlink)
```

Ensemble:
```
python scripts/ensemble.py --checkpoints checkpoints/exp01/best.pth checkpoints/exp05/best.pth \
  --dataset_root datasets/dataset_a --parquet our_test.parquet \
  --protocol dataset_a --out predictions/ensemble_a.csv
```
