"""
Read-only audit tool for a raw Roboflow YOLO-format export, run BEFORE
prepare_roboflow_data.py. Produces a human-reviewable report — it never
copies, moves, or deletes anything itself.

Background (see project audit): the denomination classifier's reported
99.5% validation accuracy is not fully trustworthy because at least one
confirmed case of near-identical photos exists across the train/valid
split boundary (same physical note, same background, same hand — just a
slightly different capture), despite having different Roboflow-assigned
filenames. Roboflow's per-file augmentation naming (`<name>_jpg.rf.<hash>`)
cannot catch this, because the duplicate photos have different `<name>`
stems too. This script instead compares actual pixel content via
perceptual hashing (average/frequency-domain image similarity, robust to
minor re-compression/relighting), which is exactly what caught the
original 2000-rupee leak during the audit.

Two independent things are flagged here, and they are NOT auto-resolved
the same way:

1. CROSS-SPLIT NEAR-DUPLICATES (validation leakage)
   Deterministic and safe to auto-fix — if the same real-world photo (or
   a near-identical recapture of it) appears in more than one split,
   there is no legitimate reason to keep both copies. This script writes
   `suggested_exclusions.txt`: the train-side filenames of every
   cross-split duplicate group, safe to pass directly to
   prepare_roboflow_data.py's --exclusion_list (or, with --auto_dedupe,
   prepare_roboflow_data.py can resolve this itself without needing this
   script at all — see that file's docstring).

2. POSSIBLE STOCK/SPECIMEN/WATERMARKED IMAGES
   NOT auto-resolved — this needs human judgement, not a heuristic
   auto-delete. This script does a best-effort OCR keyword scan (only if
   pytesseract + the system tesseract-ocr binary are installed; silently
   skipped otherwise) for common stock-agency/promotional-graphic
   watermark text (e.g. "alamy", "shutterstock", "specimen", "istock").
   Matches are listed in the CSV report as REVIEW candidates only. You
   decide, by opening the flagged images, whether to add them to your own
   exclusion list.

USAGE:
    python dataset_audit.py --roboflow_dir path/to/roboflow_export \
        --report_csv ./dataset_audit_report.csv \
        --suggested_exclusions ./suggested_exclusions.txt

Then, inspect suggested_exclusions.txt (and any OCR-flagged rows in the
CSV you agree with), optionally merge in your own manually-identified
stock-photo filenames, and pass the final list to:
    python prepare_roboflow_data.py --exclusion_list ./suggested_exclusions.txt ...
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

try:
    import imagehash
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "dataset_audit.py requires the 'imagehash' package: "
        "pip install imagehash (see training/requirements-train.txt)"
    ) from exc

# Splits in priority order — a duplicate found in a HIGHER-priority split
# is kept; the SAME photo's copies in lower-priority splits are what get
# suggested for exclusion. test/valid are small, non-augmented, and are
# the sets we most need to trust for an honest accuracy number, so they
# outrank train.
SPLIT_PRIORITY = ("test", "valid", "train")

# Stock-photo / promotional-graphic watermark keywords (best-effort OCR
# scan only — see module docstring). Lowercase, matched as substrings
# against lowercased OCR output.
_WATERMARK_KEYWORDS = (
    "alamy", "shutterstock", "istock", "gettyimages", "getty images",
    "dreamstime", "123rf", "depositphotos", "stock photo",
    "specimen", "watermark", "sample only",
)


def _try_ocr_scan(image_path: Path) -> List[str]:
    """
    Best-effort watermark/specimen keyword scan. Returns a list of matched
    keywords, or an empty list if none matched OR if OCR isn't available
    on this machine (pytesseract / the tesseract-ocr binary are optional —
    this function must never raise or block the rest of the audit).
    """
    try:
        import pytesseract
    except ImportError:
        return []

    try:
        text = pytesseract.image_to_string(Image.open(image_path)).lower()
    except Exception:  # noqa: BLE001 — tesseract binary missing, corrupt image, etc.
        return []

    return [kw for kw in _WATERMARK_KEYWORDS if kw in text]


def _collect_images(roboflow_dir: Path) -> List[Tuple[str, Path]]:
    """Returns [(split_name, image_path), ...] for every image across
    train/valid/test that actually exists in this export."""
    found = []
    for split in SPLIT_PRIORITY:
        images_dir = roboflow_dir / split / "images"
        if not images_dir.exists():
            continue
        for p in sorted(images_dir.glob("*.*")):
            found.append((split, p))
    return found


def _group_near_duplicates(
    entries: List[Tuple[str, Path, "imagehash.ImageHash"]], hamming_threshold: int
) -> List[List[int]]:
    """
    Naive O(n^2) near-duplicate clustering by Hamming distance between
    perceptual hashes — fine for dataset sizes in the hundreds/low
    thousands typical of a Roboflow export. Returns a list of groups, each
    a list of indices into `entries`. Uses union-find for correctness when
    similarity isn't perfectly transitive.
    """
    n = len(entries)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if entries[i][2] - entries[j][2] <= hamming_threshold:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [members for members in groups.values() if len(members) > 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roboflow_dir", type=str, required=True)
    parser.add_argument("--report_csv", type=str, default="./dataset_audit_report.csv")
    parser.add_argument("--suggested_exclusions", type=str, default="./suggested_exclusions.txt")
    parser.add_argument(
        "--hamming_threshold", type=int, default=6,
        help="Max perceptual-hash Hamming distance to treat two images as the same "
        "underlying photo (default 6, matches the threshold used during the "
        "project audit that found the original leak).",
    )
    parser.add_argument(
        "--skip_ocr", action="store_true",
        help="Skip the best-effort stock/specimen watermark OCR scan entirely "
        "(it's auto-skipped anyway if pytesseract isn't installed).",
    )
    args = parser.parse_args()

    roboflow_dir = Path(args.roboflow_dir)
    all_images = _collect_images(roboflow_dir)
    if not all_images:
        raise SystemExit(f"No images found under {roboflow_dir}/{{train,valid,test}}/images/")

    print(f"Hashing {len(all_images)} images (perceptual hash, this may take a minute)...")
    entries: List[Tuple[str, Path, "imagehash.ImageHash"]] = []
    for split, path in all_images:
        try:
            h = imagehash.phash(Image.open(path))
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: could not hash {path} ({exc}) — skipping")
            continue
        entries.append((split, path, h))

    print("Grouping near-duplicates (Hamming distance <= "
          f"{args.hamming_threshold})...")
    groups = _group_near_duplicates(entries, args.hamming_threshold)

    cross_split_groups = [
        g for g in groups if len({entries[i][0] for i in g}) > 1
    ]
    same_split_groups = [g for g in groups if g not in cross_split_groups]

    print(f"  {len(cross_split_groups)} cross-split near-duplicate group(s) found "
          "(validation/test leakage risk).")
    print(f"  {len(same_split_groups)} same-split near-duplicate group(s) found "
          "(likely benign Roboflow augmentation copies — not excluded).")

    ocr_scan_enabled = not args.skip_ocr
    if ocr_scan_enabled:
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            print("  (pytesseract not installed — skipping watermark OCR scan; "
                  "structural/leakage checks above are unaffected.)")
            ocr_scan_enabled = False

    # Index -> group id, for CSV reporting.
    group_of_index: Dict[int, int] = {}
    for gid, g in enumerate(groups):
        for i in g:
            group_of_index[i] = gid

    exclusion_lines: List[str] = []
    report_rows = []
    for idx, (split, path, h) in enumerate(entries):
        gid = group_of_index.get(idx)
        group_spans_splits = gid is not None and (groups[gid] in cross_split_groups)
        ocr_hits = _try_ocr_scan(path) if ocr_scan_enabled else []

        suggested_action = "keep"
        if group_spans_splits:
            # This group has members in more than one split — keep only the
            # copy in the highest-priority split present; suggest excluding
            # every other (lower-priority) copy.
            splits_in_group = {entries[i][0] for i in groups[gid]}
            highest_present = next(s for s in SPLIT_PRIORITY if s in splits_in_group)
            if split != highest_present:
                suggested_action = "exclude (cross-split duplicate)"
                exclusion_lines.append(path.name)
        if ocr_hits:
            suggested_action = (
                "REVIEW (possible stock/specimen image)"
                if suggested_action == "keep"
                else suggested_action + " + REVIEW (possible stock/specimen image)"
            )

        report_rows.append(
            {
                "split": split,
                "filename": path.name,
                "width_height": f"{Image.open(path).size}",
                "duplicate_group_id": gid if gid is not None else "",
                "group_spans_splits": group_spans_splits,
                "ocr_watermark_keyword_hits": ";".join(ocr_hits),
                "suggested_action": suggested_action,
            }
        )

    report_path = Path(args.report_csv)
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"\nWrote full report: {report_path} ({len(report_rows)} rows)")

    exclusions_path = Path(args.suggested_exclusions)
    exclusions_path.write_text(
        "\n".join(sorted(set(exclusion_lines))) + ("\n" if exclusion_lines else "")
    )
    print(
        f"Wrote {len(set(exclusion_lines))} suggested exclusion(s) (train-side "
        f"cross-split duplicates only) to: {exclusions_path}"
    )

    review_count = sum(1 for r in report_rows if "REVIEW" in r["suggested_action"])
    if review_count:
        print(
            f"\n{review_count} image(s) flagged for manual REVIEW (possible "
            f"stock/specimen watermark text detected) — see '{report_path}', "
            "column 'suggested_action'. These are NOT in suggested_exclusions.txt "
            "automatically; add filenames you agree with yourself."
        )
    print(
        "\nNext: inspect suggested_exclusions.txt (and any REVIEW rows above), "
        "then run:\n"
        "  python prepare_roboflow_data.py --roboflow_dir "
        f"{roboflow_dir} --output_dir ./data_for_training "
        f"--exclusion_list {exclusions_path}"
    )


if __name__ == "__main__":
    main()
