"""Build train/val/test folds of paired CT/MRI cases for RIED-Net.

Reads 3D NIfTI volumes from two separate folders (one per modality) and pairs
them up by filename similarity (falling back to sorted order if the two
folders have exactly the same number of files and nothing better matches).
No files are copied or symlinked -- each fold is just a ``pairs.csv`` that
points at the original files, so the dataset stays paired even when the CT
and MRI filenames don't literally match (e.g. ``..._t2f.nii.gz`` vs.
``..._t1c.nii.gz``).

Usage
-----
    python prepare_data.py --ct_dir ./CT --mri_dir ./MRI --out_dir ./dataset \
        --fold_name fold1 --val_frac 0.2 --test_frac 0.2 --seed 42

Output
------
    dataset/fold1/train/pairs.csv
    dataset/fold1/val/pairs.csv
    dataset/fold1/test/pairs.csv
    dataset/fold1/split_summary.json
"""
import argparse
import csv
import json
import os
import random
from difflib import SequenceMatcher
from glob import glob

NII_PATTERNS = ("*.nii", "*.nii.gz")


def list_nifti(folder):
    files = []
    for pat in NII_PATTERNS:
        files.extend(glob(os.path.join(folder, "**", pat), recursive=True))
    return sorted(set(files))


def strip_ext(path):
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return os.path.splitext(name)[0]


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def match_pairs(ct_files, mri_files):
    """Greedily pair CT/MRI files by filename similarity (best match first).

    Falls back to exact-name matching first (common case), then to greedy
    similarity matching so datasets like BraTS demo files
    (``..._t2f.nii.gz`` <-> ``..._t1c.nii.gz``) still pair correctly by
    shared case-id prefix.
    """
    ct_by_name = {strip_ext(f): f for f in ct_files}
    mri_by_name = {strip_ext(f): f for f in mri_files}

    pairs = []
    used_ct, used_mri = set(), set()

    # Pass 1: identical basenames.
    for name in list(ct_by_name):
        if name in mri_by_name:
            pairs.append((ct_by_name[name], mri_by_name[name], name))
            used_ct.add(name)
            used_mri.add(name)

    remaining_ct = {n: f for n, f in ct_by_name.items() if n not in used_ct}
    remaining_mri = {n: f for n, f in mri_by_name.items() if n not in used_mri}

    # Pass 2: greedy best-similarity matching for what's left.
    if remaining_ct and remaining_mri:
        candidates = []
        for cn, cf in remaining_ct.items():
            for mn, mf in remaining_mri.items():
                candidates.append((similarity(cn, mn), cn, mn, cf, mf))
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_ct, matched_mri = set(), set()
        for score, cn, mn, cf, mf in candidates:
            if cn in matched_ct or mn in matched_mri:
                continue
            # longest common substring as case id, fallback to CT name
            case_id = os.path.commonprefix([cn, mn]) or cn
            pairs.append((cf, mf, case_id))
            matched_ct.add(cn)
            matched_mri.add(mn)

        unmatched_ct = set(remaining_ct) - matched_ct
        unmatched_mri = set(remaining_mri) - matched_mri
        for n in unmatched_ct:
            print(f"WARNING: no MRI match found for CT file {remaining_ct[n]}")
        for n in unmatched_mri:
            print(f"WARNING: no CT match found for MRI file {remaining_mri[n]}")

    pairs.sort(key=lambda p: p[2])
    return pairs


def split_cases(pairs, val_frac, test_frac, seed):
    n = len(pairs)
    if n == 0:
        raise ValueError("No CT/MRI pairs found -- check --ct_dir/--mri_dir.")

    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n_val = round(n * val_frac)
    n_test = round(n * test_frac)

    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
    while n_val + n_test >= n:
        if n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    val = shuffled[:n_val]
    test = shuffled[n_val:n_val + n_test]
    train = shuffled[n_val + n_test:]
    return {"train": train, "val": val, "test": test}


def write_pairs_csv(pairs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "target", "case_id"])
        for ct_path, mri_path, case_id in pairs:
            writer.writerow([os.path.abspath(ct_path), os.path.abspath(mri_path), case_id])


def main():
    parser = argparse.ArgumentParser(description="Prepare CT/MRI paired folds for RIED-Net")
    parser.add_argument("--ct_dir", default="./CT", type=str, help="Folder of CT (input) NIfTI volumes")
    parser.add_argument("--mri_dir", default="./MRI", type=str, help="Folder of MRI (target) NIfTI volumes")
    parser.add_argument("--out_dir", default="./dataset", type=str, help="Where to write the fold CSVs")
    parser.add_argument("--fold_name", default="fold1", type=str)
    parser.add_argument("--val_frac", default=0.2, type=float)
    parser.add_argument("--test_frac", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    ct_files = list_nifti(args.ct_dir)
    mri_files = list_nifti(args.mri_dir)
    print(f"Found {len(ct_files)} CT files in {args.ct_dir}")
    print(f"Found {len(mri_files)} MRI files in {args.mri_dir}")

    pairs = match_pairs(ct_files, mri_files)
    print(f"Matched {len(pairs)} CT/MRI pairs")
    for ct_path, mri_path, case_id in pairs:
        print(f"  [{case_id}] {os.path.basename(ct_path)}  <->  {os.path.basename(mri_path)}")

    splits = split_cases(pairs, args.val_frac, args.test_frac, args.seed)

    fold_dir = os.path.join(args.out_dir, args.fold_name)
    summary = {}
    for split_name, split_pairs in splits.items():
        csv_path = os.path.join(fold_dir, split_name, "pairs.csv")
        write_pairs_csv(split_pairs, csv_path)
        summary[split_name] = {
            "n_cases": len(split_pairs),
            "case_ids": [p[2] for p in split_pairs],
            "pairs_csv": os.path.abspath(csv_path),
        }
        print(f"{split_name}: {len(split_pairs)} case(s) -> {csv_path}")

    with open(os.path.join(fold_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Fold written to {os.path.abspath(fold_dir)}")


if __name__ == "__main__":
    main()
