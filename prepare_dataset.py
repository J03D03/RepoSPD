"""
Convert dataset.JSONL + CSV splits into RepoSPD pipeline format.

Reads:
    datasets/dataset2-mr-advisory-cpp.jsonl
    datasets/dataset2-mr-advisory-cpp-{split}-{train,val,test}.csv

Writes:
    dataset/example.json              <- full converted dataset
    dataset/dataset2-fq-splits.json   <- {commit_id: "train"|"valid"|"test"}
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

JSONL_PATH = Path("datasets/dataset2-mr-advisory-cpp.jsonl")
SPLITS_DIR = Path("datasets")
OUT_DIR = Path("dataset")


def load_split(split_name):
    """Return {full_commit_id: "train"|"valid"|"test"}"""
    mapping = {}
    for subset, label in [("train", "train"), ("val", "valid"), ("test", "test")]:
        path = SPLITS_DIR / f"dataset2-mr-advisory-cpp-{split_name}-{subset}.csv"
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row["commit_id"]] = label
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="groupstrat-seed3",
        help="Split variant to use (default: groupstrat-seed3)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print(f"Loading split: {args.split}")
    split_map = load_split(args.split)  # full commit_id -> split label
    print(f"  {len(split_map)} entries in split files")

    print(f"Loading JSONL...")
    records = []
    skipped = 0

    with open(JSONL_PATH) as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Processing JSONL"):
        item = json.loads(line)
        commit_id = item["commit_id"]

        if commit_id not in split_map:
            skipped += 1
            continue

        project_url = item["project_url"].rstrip("/")
        ori = project_url.split("/")[-1]
        records.append(
            {
                "idx": len(records),
                "category": "security"
                if str(item["is_vfc"]).lower() == "true"
                else "non-security",
                "commit_message": item["commit_message"],
                "diff_code": item["commit_diff"],
                "ori_dataset": ori,
                "project_url": project_url,
                "commit_id": commit_id,
            }
        )

    print(f"  {len(records)} records kept, {skipped} skipped")

    # Save full dataset
    out_json = OUT_DIR / "example.json"
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {out_json}")

    seen_ids = {r["commit_id"] for r in records}
    split_by_id = {cid: label for cid, label in split_map.items() if cid in seen_ids}
    out_splits = OUT_DIR / "dataset2-fq-splits.json"
    with open(out_splits, "w") as f:
        json.dump(split_by_id, f, indent=2)
    print(f"Wrote {out_splits}")

    # Summary
    counts = Counter(split_by_id.values())
    total = sum(counts.values())
    sec = sum(1 for r in records if r["category"] == "security")
    print(f"\nSummary:")
    print(f"  Total: {total}  (security: {sec}, non-security: {total - sec})")
    print(
        f"  Train: {counts['train']}  Valid: {counts['valid']}  Test: {counts['test']}"
    )


if __name__ == "__main__":
    main()
