"""
Converts a Roboflow YOLO-format detection export into the classification
folder-per-class layout expected by train_denomination_classifier.py.

Skip this script entirely if your dataset is ALREADY organized as:
    data_dir/10/*.jpg
    data_dir/20/*.jpg
    data_dir/500/*.jpg
    ...
(this is the typical layout for the Mendeley dataset, and for Roboflow
exports created with the "Folder Structure" / classification export format
instead of YOLO).

Expected Roboflow YOLO export layout (what you get from "YOLOv8" or
similar detection export formats):
    roboflow_dir/
        train/
            images/*.jpg
            labels/*.txt      (each line: class_id x_center y_center width height,
                                 OR a variable-length polygon/segmentation line —
                                 either way, only the leading class_id is read)
        valid/
            images/... labels/...
        test/            (optional)
        data.yaml         (contains 'names: [class0, class1, ...]')

This script:
    1. Reads the class name list from data.yaml
    2. For each image, reads its FIRST annotation line to get the class id
       (assumes one denomination per image, which is true for whole-note
       photos — if an image has multiple boxes for some other reason,
       only the first box's class is used)
    3. Copies the image into output_dir/<split>/<class_name>/

--- CHANGED IN THIS REVISION (see project audit — validation leakage) ---
Earlier versions of this script merged train/valid/test into a single
flat output_dir/<class_name>/ pool, relying on train_denomination_classifier.py
to do its own random 85/15 re-split afterwards. That destroyed Roboflow's
original split boundary and let near-duplicate photos (see --dedupe below)
land on both sides of the split purely by chance. This version instead
preserves train/valid/test as THREE SEPARATE output trees
(output_dir/train/<class>/, output_dir/val/<class>/, output_dir/test/<class>/)
so no re-split ever happens — train_denomination_classifier.py now loads
each one directly (see its updated docstring/--data_dir handling).

Two new, independent, OPTIONAL safety flags:

  --dedupe
      Cross-split near-duplicate resolution via perceptual image hashing
      (not filename matching — Roboflow can assign different filenames to
      what is actually the same or a near-identical recapture of the same
      physical note; this is exactly how the original leak was missed).
      When two images across different splits hash as near-identical,
      only the copy in the higher-priority split is kept (priority:
      test > valid > train — the smaller, non-augmented sets are the ones
      we most need to trust for an honest accuracy number). This is safe
      to enable by default for any Roboflow export with train/valid/test
      folders; it does NOT touch near-duplicates that are both in the SAME
      split (e.g. Roboflow's own deliberate train-set augmentation copies
      of one source photo) — that kind of duplication is intentional and
      is left alone.

  --exclusion_list path/to/file.txt
      Plain text, one filename (basename, matched case-sensitively against
      the image filename Roboflow generated) per line. Images matching are
      skipped entirely, regardless of split. Intended for the harder,
      judgement-based problem the audit also found: watermarked
      stock-photo images (e.g. visible "Alamy"/"indiararecoins" watermarks)
      and specimen/promotional graphics mixed into the dataset. Detecting
      those reliably is NOT automated here — run dataset_audit.py first,
      which does a best-effort OCR keyword scan and writes a starter
      suggested_exclusions.txt (also pre-filled with the same --dedupe
      cross-split logic above, so you can use ONE file for both purposes),
      then review/extend it by hand before passing it here.

USAGE:
    python prepare_roboflow_data.py --roboflow_dir path/to/export \
        --output_dir ./data_for_training \
        --dedupe \
        --exclusion_list ./suggested_exclusions.txt
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

# Splits in priority order for --dedupe conflict resolution — see
# module docstring. Mirrors dataset_audit.py's SPLIT_PRIORITY exactly;
# keep these in sync if either changes.
SPLIT_PRIORITY = ("test", "valid", "train")
# Roboflow's own name for "valid" maps to our output "val" folder, to
# match the conventional train/val/test naming train_denomination_classifier.py
# looks for.
OUTPUT_SPLIT_NAME = {"train": "train", "valid": "val", "test": "test"}


def load_class_names(data_yaml_path: Path):
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)
    return data["names"]


def load_exclusion_list(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    lines = Path(path).read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}


def _get_image_class(image_path: Path, labels_dir: Path, class_names) -> Optional[str]:
    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        return None
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return None
    class_id = int(lines[0].split()[0])  # first annotation's class (bbox or polygon format)
    return str(class_names[class_id])


def _compute_cross_split_dedupe_set(
    roboflow_dir: Path, hamming_threshold: int = 6
) -> Set[str]:
    """
    Returns the set of filenames to DROP because a near-identical image
    (by perceptual hash) exists in a higher-priority split. Only called
    when --dedupe is passed. This intentionally re-derives the same logic
    as dataset_audit.py rather than importing it, so this script has no
    hard dependency on that file existing/being run first.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "--dedupe requires 'imagehash' and 'Pillow': pip install imagehash pillow"
        ) from exc

    entries: List[Tuple[str, Path, "imagehash.ImageHash"]] = []
    for split in SPLIT_PRIORITY:
        images_dir = roboflow_dir / split / "images"
        if not images_dir.exists():
            continue
        for p in sorted(images_dir.glob("*.*")):
            try:
                entries.append((split, p, imagehash.phash(Image.open(p))))
            except Exception:  # noqa: BLE001 — unreadable image, skip silently here;
                continue        # convert_split() will also just skip it downstream.

    to_drop: Set[str] = set()
    n = len(entries)
    for i in range(n):
        split_i, path_i, hash_i = entries[i]
        best_priority_seen = SPLIT_PRIORITY.index(split_i)
        for j in range(n):
            if i == j:
                continue
            split_j, path_j, hash_j = entries[j]
            if hash_i - hash_j <= hamming_threshold:
                best_priority_seen = min(best_priority_seen, SPLIT_PRIORITY.index(split_j))
        if SPLIT_PRIORITY.index(split_i) > best_priority_seen:
            # A near-duplicate exists in a strictly higher-priority split —
            # drop this (lower-priority) copy.
            to_drop.add(path_i.name)

    return to_drop


def convert_split(
    split_dir: Path,
    split_name: str,
    class_names,
    output_dir: Path,
    exclusions: Set[str],
    dedupe_drop_set: Set[str],
) -> Tuple[int, int]:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.exists():
        return 0, 0

    out_split_dir = output_dir / OUTPUT_SPLIT_NAME[split_name]

    copied, skipped = 0, 0
    for image_path in images_dir.glob("*.*"):
        if image_path.name in exclusions:
            skipped += 1
            continue
        if image_path.name in dedupe_drop_set:
            skipped += 1
            continue

        class_name = _get_image_class(image_path, labels_dir, class_names)
        if class_name is None:
            continue

        dest_dir = out_split_dir / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, dest_dir / image_path.name)
        copied += 1

    return copied, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roboflow_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--dedupe", action="store_true",
        help="Drop lower-priority-split copies of cross-split near-duplicate images "
        "(perceptual-hash based; see module docstring). Recommended for this project's "
        "dataset — it directly fixes the validation-leakage finding from the audit.",
    )
    parser.add_argument(
        "--dedupe_hamming_threshold", type=int, default=6,
        help="Only used with --dedupe. Max perceptual-hash Hamming distance to treat "
        "two images as the same underlying photo (default 6).",
    )
    parser.add_argument(
        "--exclusion_list", type=str, default=None,
        help="Optional path to a plain-text file of filenames (one per line) to skip "
        "entirely — e.g. stock-photo/specimen images flagged by dataset_audit.py.",
    )
    args = parser.parse_args()

    roboflow_dir = Path(args.roboflow_dir)
    output_dir = Path(args.output_dir)

    class_names = load_class_names(roboflow_dir / "data.yaml")
    print(f"Classes found in data.yaml: {class_names}")

    exclusions = load_exclusion_list(args.exclusion_list)
    if exclusions:
        print(f"Loaded {len(exclusions)} filename(s) from --exclusion_list.")

    dedupe_drop_set: Set[str] = set()
    if args.dedupe:
        print(
            f"Computing cross-split near-duplicates (Hamming <= "
            f"{args.dedupe_hamming_threshold})..."
        )
        dedupe_drop_set = _compute_cross_split_dedupe_set(
            roboflow_dir, args.dedupe_hamming_threshold
        )
        print(f"  -> {len(dedupe_drop_set)} image(s) will be dropped as cross-split duplicates.")
        overlap = dedupe_drop_set & exclusions
        if overlap:
            print(f"  ({len(overlap)} of those were already in --exclusion_list too.)")

    total_copied, total_skipped = 0, 0
    per_split_counts: Dict[str, int] = {}
    for split in SPLIT_PRIORITY:  # test, valid, train — order doesn't affect output, just log order
        split_dir = roboflow_dir / split
        if split_dir.exists():
            copied, skipped = convert_split(
                split_dir, split, class_names, output_dir, exclusions, dedupe_drop_set
            )
            out_name = OUTPUT_SPLIT_NAME[split]
            per_split_counts[out_name] = per_split_counts.get(out_name, 0) + copied
            print(f"  {split} -> {out_name}: copied {copied}, skipped {skipped}")
            total_copied += copied
            total_skipped += skipped

    print(f"\nDone. {total_copied} images written to {output_dir} "
          f"(skipped {total_skipped} excluded/deduplicated).")
    print(f"Per-split totals: {per_split_counts}")
    print(
        "\nNext: python train_denomination_classifier.py --data_dir "
        + str(output_dir)
        + "\n(train/val[/test] subfolders under --data_dir are now used directly — "
        "no internal re-split happens when they're present; see that script's docstring.)"
    )


if __name__ == "__main__":
    main()
