# COMP 560 Object Re-ID Project Notes

## Scope
- Working with **dataset_a only**. Ignore all dataset_b references in the README and scripts.
- dataset_a lives at `datasets/dataset_a/` — images are gitignored and must be obtained separately.

## Dataset Status
- `train.parquet`: ~110K image entries across ~10K identities
- `test.parquet`: ~31K image entries
- Images on disk: ~24K images currently present under `datasets/dataset_a/images/`
- Images are gitignored — teammates must download separately and place in `datasets/dataset_a/images/`

## Path Convention
- Parquet stores paths relative to the dataset root (e.g. `images/AAUZebraFish/data/filename.png`)
- Scripts join with `--data_root ./datasets/dataset_a`, producing `./datasets/dataset_a/images/...`
- No absolute paths — works across machines as long as images are in the right relative location
