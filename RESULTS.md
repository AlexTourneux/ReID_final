# Experiment Tracker — COMP560 ReID

Shared table for all parallel training runs. **Push updates on the `results` branch** to avoid main-branch merge conflicts.

Row conventions:
- Each trained model contributes up to 3 rows: `A` (dataset_a animals), `M` (Market-1501 persons), `combined` (mean of A and M).
- **A:** `datasets/dataset_a/our_test.parquet`, no same-camera exclusion, self-match suppressed.
- **M:** `datasets/market1501/market1501_test.parquet`, same-camera exclusion ON.
- **combined:** unweighted mean of A and M for each accuracy metric — primary model-selection metric.
- **throughput / peak_mem / params:** model-level (same for A and M rows); reported on the `combined` row for readability. Units: images/sec at batch=64 (T4), peak GPU MB at batch=64 forward, millions of parameters.
- Leave cells blank when a number wasn't run yet.

| exp_id | dataset | owner | backbone | status | R-1 | R-5 | R-10 | R-20 | mAP | mINP | throughput | peak_mem | params | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_resnet50_a_only | A | Alex | ResNet50 | done | 70.65 |  |  |  | 68.17 |  |  |  |  | MPS, pre-plan baseline |
| baseline_resnet50_a_only | M | Alex | ResNet50 | skipped |  |  |  |  |  |  |  |  |  | dataset_a-only model |
| baseline_resnet50_a_only | combined | Alex | ResNet50 | done |  |  |  |  |  |  |  |  | 25.0 | — |
| exp01_teacher_dinov2_large | A |  | DinoV2-L | todo |  |  |  |  |  |  |  |  |  |  |
| exp01_teacher_dinov2_large | M |  | DinoV2-L | todo |  |  |  |  |  |  |  |  |  |  |
| exp01_teacher_dinov2_large | combined |  | DinoV2-L | todo |  |  |  |  |  |  |  |  | ~305 | main teacher |
| exp02_teacher_dinov2_base | A |  | DinoV2-B | todo |  |  |  |  |  |  |  |  |  |  |
| exp02_teacher_dinov2_base | M |  | DinoV2-B | todo |  |  |  |  |  |  |  |  |  |  |
| exp02_teacher_dinov2_base | combined |  | DinoV2-B | todo |  |  |  |  |  |  |  |  | ~86 | backup teacher |
| exp05_student_resnet50_kd | A |  | ResNet50 | todo |  |  |  |  |  |  |  |  |  |  |
| exp05_student_resnet50_kd | M |  | ResNet50 | todo |  |  |  |  |  |  |  |  |  |  |
| exp05_student_resnet50_kd | combined |  | ResNet50 | todo |  |  |  |  |  |  |  |  | ~25 | distilled from exp01 |
| exp06_student_efficientnet_kd | A |  | EffNet-B0 | todo |  |  |  |  |  |  |  |  |  |  |
| exp06_student_efficientnet_kd | M |  | EffNet-B0 | todo |  |  |  |  |  |  |  |  |  |  |
| exp06_student_efficientnet_kd | combined |  | EffNet-B0 | todo |  |  |  |  |  |  |  |  | ~5 | distilled from exp01 |

## Bars (kill criteria)
- exp01 teacher final (combined row): ≥80% R-1 on A, ≥70% R-1 on M.
- exp05/exp06 student: within 3 pts R-1 of teacher on both domains (else KD is broken).
- Final ensemble: beat best single model by ≥1 pt R-1 / ≥2 pts mAP.

## How to update
```bash
git checkout -b results-$(date +%Y%m%d-%H%M)
# edit RESULTS.md (manually or via scripts/update_results.py)
git add RESULTS.md && git commit -m "results: exp0X <short>"
git push -u origin HEAD
```
Merge to `results` when ready; keep `main` clean.

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

Evaluate (use the prof's grader — outputs CSV summary with Rank-1/5/10/20, mAP, mINP):
```
python evaluate_copy.py --student_id <id> --prediction predictions/<exp>_a.csv \
  --datasets dataset_a --parquet our_test.parquet

python evaluate_copy.py --student_id <id> --prediction predictions/<exp>_m.csv \
  --datasets dataset_b --parquet market1501_test.parquet
# (symlink datasets/dataset_b -> datasets/market1501 so the grader finds the parquet)
```

Benchmark efficiency (throughput + peak memory):
```
python scripts/benchmark.py --checkpoint checkpoints/<exp>/best.pth
```

Record rows in this table. Recommended: auto-ingest the summary CSV written by `evaluate_copy.py`:
```
python scripts/update_results.py --exp_id <exp> --dataset A \
  --from_summary results/<student>_<ts>_summary.csv --dataset_in_summary dataset_a

python scripts/update_results.py --exp_id <exp> --dataset M \
  --from_summary results/<student>_<ts>_summary.csv --dataset_in_summary dataset_b

python scripts/update_results.py --exp_id <exp> --dataset combined \
  --throughput 230 --peak_mem 5400 --params 305 --status done --notes "teacher, 10 ep"
# combined's R-1..mINP are auto-averaged from the A and M rows
```

Or pass metrics manually:
```
python scripts/update_results.py --exp_id <exp> --dataset A \
  --r1 94.5 --r5 98.2 --r10 98.9 --r20 99.4 --map 93.7 --minp 85.2 --status done
```
