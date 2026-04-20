#!/usr/bin/env python3
"""Update one (exp_id, dataset) row of RESULTS.md.

RESULTS.md holds a markdown table with `exp_id` and `dataset` as the
composite primary key. Each trained model has up to 3 rows:
    (exp, A)  -- dataset_a metrics
    (exp, M)  -- Market-1501 metrics
    (exp, combined)  -- mean of A and M (and model-level efficiency)

This script finds the matching (exp_id, dataset) row and overwrites the
specified columns. Columns not passed are left untouched.

Two ways to populate accuracy metrics:
    1. --from_summary <path>  -- parse the CSV written by evaluate_copy.py
       (expects columns: dataset, Rank-1, Rank-5, Rank-10, Rank-20,
        mAP, mINP, combined). Pick the right row with
        --dataset_in_summary (dataset_a or dataset_b).
    2. --r1 / --r5 / --r10 / --r20 / --map / --minp  -- pass numbers directly.

When --dataset combined is used and no accuracy flags / summary are
given, the script auto-averages the (exp_id, A) and (exp_id, M) rows
already in the table. Efficiency columns (throughput, peak_mem, params)
are best recorded on the `combined` row.

Examples:
    # Record dataset_a evaluation
    python scripts/update_results.py --exp_id exp01_teacher_dinov2_large \\
        --dataset A --status done \\
        --from_summary results/alex_20260420_104523_summary.csv \\
        --dataset_in_summary dataset_a

    # Record efficiency + auto-average combined row
    python scripts/update_results.py --exp_id exp01_teacher_dinov2_large \\
        --dataset combined --status done \\
        --throughput 230 --peak_mem 5400 --params 305 --notes "teacher, 10 ep"
"""

import argparse
import csv
from pathlib import Path


COLUMNS = [
    "exp_id", "dataset", "owner", "backbone", "status",
    "r1", "r5", "r10", "r20", "map", "minp",
    "throughput", "peak_mem", "params", "notes",
]

HEADER_PREFIX = "| exp_id "


def parse_table(text):
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith(HEADER_PREFIX):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"Could not find RESULTS.md header starting with '{HEADER_PREFIX}'.")
    sep_idx = header_idx + 1
    end_idx = len(lines)
    for j in range(sep_idx + 1, len(lines)):
        if not lines[j].startswith("|"):
            end_idx = j
            break
    return lines, header_idx, sep_idx, end_idx


def split_row(line):
    parts = line.strip().strip("|").split("|")
    return [p.strip() for p in parts]


def join_row(cells):
    return "| " + " | ".join(cells) + " |"


def fmt(v):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def parse_summary(path, dataset_key):
    """Parse evaluate_copy.py's summary CSV and return dict of metrics.

    Expected columns: dataset,Rank-1,Rank-5,Rank-10,Rank-20,mAP,mINP,combined
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset"].strip() == dataset_key:
                return {
                    "r1": float(row["Rank-1"]),
                    "r5": float(row["Rank-5"]),
                    "r10": float(row["Rank-10"]),
                    "r20": float(row["Rank-20"]),
                    "map": float(row["mAP"]),
                    "minp": float(row["mINP"]),
                }
    raise RuntimeError(f"Row with dataset='{dataset_key}' not found in {path}")


def find_row(lines, sep_idx, end_idx, exp_id, dataset):
    for i in range(sep_idx + 1, end_idx):
        cells = split_row(lines[i])
        if len(cells) >= 2 and cells[0] == exp_id and cells[1] == dataset:
            return i, cells
    return None, None


def cell_to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_md", default="RESULTS.md")
    ap.add_argument("--exp_id", required=True)
    ap.add_argument("--dataset", required=True, choices=["A", "M", "combined"])
    ap.add_argument("--owner")
    ap.add_argument("--backbone")
    ap.add_argument("--status", choices=["todo", "running", "done", "failed", "skipped"])
    ap.add_argument("--r1", type=float)
    ap.add_argument("--r5", type=float)
    ap.add_argument("--r10", type=float)
    ap.add_argument("--r20", type=float)
    ap.add_argument("--map", type=float)
    ap.add_argument("--minp", type=float)
    ap.add_argument("--throughput", type=float, help="images/sec at batch=64")
    ap.add_argument("--peak_mem", type=float, help="peak GPU memory in MB at batch=64")
    ap.add_argument("--params", type=float, help="parameter count in millions")
    ap.add_argument("--notes")

    # Bulk ingest from evaluate_copy.py summary CSV
    ap.add_argument("--from_summary",
                    help="Parse accuracy metrics from evaluate_copy.py summary CSV")
    ap.add_argument("--dataset_in_summary", default=None,
                    help="Dataset name inside the summary CSV (dataset_a / dataset_b). "
                    "Default: dataset_a if --dataset=A else dataset_b.")
    args = ap.parse_args()

    # Start with flag values
    updates = {
        "owner": args.owner,
        "backbone": args.backbone,
        "status": args.status,
        "r1": args.r1,
        "r5": args.r5,
        "r10": args.r10,
        "r20": args.r20,
        "map": args.map,
        "minp": args.minp,
        "throughput": args.throughput,
        "peak_mem": args.peak_mem,
        "params": args.params,
        "notes": args.notes,
    }

    # Merge in CSV-derived metrics (flags take precedence if both given)
    if args.from_summary:
        key = args.dataset_in_summary or ("dataset_a" if args.dataset == "A" else "dataset_b")
        csv_metrics = parse_summary(args.from_summary, key)
        for k, v in csv_metrics.items():
            if updates.get(k) is None:
                updates[k] = v

    updates = {k: v for k, v in updates.items() if v is not None}

    path = Path(args.results_md)
    text = path.read_text()
    lines, header_idx, sep_idx, end_idx = parse_table(text)

    header_cells = split_row(lines[header_idx])
    if len(header_cells) != len(COLUMNS):
        raise RuntimeError(
            f"RESULTS.md has {len(header_cells)} columns, expected {len(COLUMNS)}: {COLUMNS}. "
            f"Header: {header_cells}")

    row_idx, cells = find_row(lines, sep_idx, end_idx, args.exp_id, args.dataset)
    if row_idx is None:
        print(f"WARN: row ({args.exp_id}, {args.dataset}) not found — appending.")
        cells = [""] * len(COLUMNS)
        cells[0] = args.exp_id
        cells[1] = args.dataset
        row_idx = end_idx
        lines.insert(row_idx, join_row(cells))
        end_idx += 1

    for key, val in updates.items():
        cells[COLUMNS.index(key)] = fmt(val)

    # Auto-average A + M into the combined row when user didn't supply accuracy
    if args.dataset == "combined":
        supplied_acc = any(k in updates for k in ["r1", "r5", "r10", "r20", "map", "minp"])
        if not supplied_acc:
            a_idx, a_cells = find_row(lines, sep_idx, end_idx, args.exp_id, "A")
            m_idx, m_cells = find_row(lines, sep_idx, end_idx, args.exp_id, "M")
            if a_cells and m_cells:
                for metric in ["r1", "r5", "r10", "r20", "map", "minp"]:
                    j = COLUMNS.index(metric)
                    a_v = cell_to_float(a_cells[j])
                    m_v = cell_to_float(m_cells[j])
                    if a_v is not None and m_v is not None:
                        cells[j] = f"{(a_v + m_v) / 2:.2f}"

    lines[row_idx] = join_row(cells)
    path.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))
    print(f"Updated {path}: ({args.exp_id}, {args.dataset})")


if __name__ == "__main__":
    main()
