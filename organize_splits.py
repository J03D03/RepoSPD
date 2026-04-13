import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def organize():
    # Paths
    split_json = Path("dataset/dataset2-fq-splits.json")
    source_dir = Path("dataset/data")
    target_base = Path("dataset/example_dataset")

    if not split_json.exists():
        print(f"Error: {split_json} not found. Run prepare_dataset.py first.")
        return

    with open(split_json, 'r') as f:
        splits = json.load(f)

    print(f"Organizing {len(splits)} commits into train/valid/test splits...")

    # Create target directories
    for label in ["train", "valid", "test"]:
        (target_base / label).mkdir(parents=True, exist_ok=True)

    moved_count = 0
    missing_count = 0

    for commit_id, label in tqdm(splits.items()):
        # add_dep.py produces directories named by commit_id in dataset/data/
        src_commit_dir = source_dir / commit_id
        dest_commit_dir = target_base / label / commit_id

        if src_commit_dir.exists() and src_commit_dir.is_dir():
            # If destination exists, remove it first to avoid nesting
            if dest_commit_dir.exists():
                shutil.rmtree(dest_commit_dir)
            
            # Copy or Move. Move is faster if on same filesystem.
            shutil.copytree(src_commit_dir, dest_commit_dir)
            moved_count += 1
        else:
            missing_count += 1

    print(f"\nSuccess: Organized {moved_count} commits.")
    if missing_count > 0:
        print(f"Warning: {missing_count} commits were in the split JSON but missing from {source_dir}.")

if __name__ == "__main__":
    organize()
