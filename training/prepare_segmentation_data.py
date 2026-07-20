"""
Converts a Roboflow YOLO-format export's annotations into a binary
segmentation dataset (image + note-vs-background mask pairs), for
training the optional note-boundary segmenter
(app/models/note_boundary/onnx_backend.py).

Reuses the SAME raw --roboflow_dir input as prepare_roboflow_data.py (see
that file for the expected train/valid/test + images/ + labels/ +
data.yaml layout) — run this instead of / in addition to that script,
depending on which model you're training. Preserves Roboflow's own
train/valid/test split boundary the same way (see project audit —
validation leakage), for consistency with the rest of this project's data
preparation, even though split leakage matters less for a boundary/shape
task than it does for a classification accuracy number.

--- WHY BOTH BBOX AND POLYGON LABELS ARE USED (see project audit) ---
This dataset's labels are a mix: ~58% are variable-length polygon outlines
of the note, the rest are plain 5-value boxes (class x y w h). Both are
used here, not just the polygons:
    - polygon lines  -> filled exactly as annotated (tighter boundary)
    - bbox-only lines -> filled as the axis-aligned rectangle
An image with only a bbox annotation still gives the segmenter a
perfectly reasonable "note is roughly here" training signal — it's a
looser mask than a true polygon, but far better than discarding ~42% of
the available images outright. Whether a given label was bbox or polygon
is recorded in the output manifest so you can filter to polygon-only
training if you want the tightest possible masks instead.

SCOPE: masks encode note LOCATION only, from the same denomination-only
Roboflow annotations the rest of this project's dataset audit already
covers — there is nothing counterfeit/genuine-related here, and this
script must never be extended toward that.

USAGE:
    python prepare_segmentation_data.py --roboflow_dir path/to/roboflow_export \
        --output_dir ./data_for_segmentation \
        --exclusion_list ./suggested_exclusions.txt   # optional, reuse from dataset_audit.py
"""

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

SPLIT_PRIORITY = ("test", "valid", "train")
OUTPUT_SPLIT_NAME = {"train": "train", "valid": "val", "test": "test"}


def load_exclusion_list(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    lines = Path(path).read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}


def _parse_first_annotation(label_path: Path) -> Optional[Tuple[str, List[float]]]:
    """Returns ("bbox", [cx,cy,w,h]) or ("polygon", [x1,y1,...,xn,yn]),
    all values still normalized [0,1] — or None if the label file is
    missing/empty. Only the FIRST annotation line is used (see
    prepare_roboflow_data.py for the same one-denomination-per-image
    assumption)."""
    if not label_path.exists():
        return None
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return None
    tokens = lines[0].split()
    coords = [float(t) for t in tokens[1:]]
    if len(coords) == 4:
        return "bbox", coords
    if len(coords) >= 6 and len(coords) % 2 == 0:
        return "polygon", coords
    return None  # malformed line — skip rather than guess


def _rasterize_mask(
    shape_type: str, coords: List[float], width: int, height: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype="uint8")
    if shape_type == "bbox":
        cx, cy, w, h = coords
        x1 = int((cx - w / 2) * width)
        y1 = int((cy - h / 2) * height)
        x2 = int((cx + w / 2) * width)
        y2 = int((cy + h / 2) * height)
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)
    else:  # polygon
        pts = np.array(
            [(coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords), 2)],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], color=255)
    return mask


def convert_split(
    split_dir: Path, split_name: str, output_dir: Path, exclusions: Set[str], manifest_rows: list
) -> Tuple[int, int]:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.exists():
        return 0, 0

    out_name = OUTPUT_SPLIT_NAME[split_name]
    out_images_dir = output_dir / out_name / "images"
    out_masks_dir = output_dir / out_name / "masks"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    copied, skipped = 0, 0
    for image_path in sorted(images_dir.glob("*.*")):
        if image_path.name in exclusions:
            skipped += 1
            continue

        parsed = _parse_first_annotation(labels_dir / f"{image_path.stem}.txt")
        if parsed is None:
            skipped += 1
            continue
        shape_type, coords = parsed

        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue
        height, width = image.shape[:2]

        mask = _rasterize_mask(shape_type, coords, width, height)

        cv2.imwrite(str(out_images_dir / image_path.name), image)
        mask_name = f"{image_path.stem}.png"
        cv2.imwrite(str(out_masks_dir / mask_name), mask)

        manifest_rows.append(
            {"split": out_name, "image": image_path.name, "mask": mask_name, "label_type": shape_type}
        )
        copied += 1

    return copied, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roboflow_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--exclusion_list", type=str, default=None,
        help="Optional path to a plain-text filename exclusion list — reuse the same "
        "file dataset_audit.py / prepare_roboflow_data.py produced/consumed.",
    )
    args = parser.parse_args()

    roboflow_dir = Path(args.roboflow_dir)
    output_dir = Path(args.output_dir)
    exclusions = load_exclusion_list(args.exclusion_list)
    if exclusions:
        print(f"Loaded {len(exclusions)} filename(s) from --exclusion_list.")

    manifest_rows: list = []
    total_copied, total_skipped = 0, 0
    for split in SPLIT_PRIORITY:
        split_dir = roboflow_dir / split
        if split_dir.exists():
            copied, skipped = convert_split(split_dir, split, output_dir, exclusions, manifest_rows)
            print(f"  {split} -> {OUTPUT_SPLIT_NAME[split]}: copied {copied}, skipped {skipped}")
            total_copied += copied
            total_skipped += skipped

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "mask", "label_type"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    polygon_count = sum(1 for r in manifest_rows if r["label_type"] == "polygon")
    bbox_count = sum(1 for r in manifest_rows if r["label_type"] == "bbox")
    print(
        f"\nDone. {total_copied} image/mask pairs written to {output_dir} "
        f"(skipped {total_skipped}).\n"
        f"  {polygon_count} from polygon labels (tight masks), "
        f"{bbox_count} from bbox-only labels (rectangular masks)."
    )
    print(f"Manifest: {manifest_path}")
    print(
        "\nNext: python train_note_boundary_model.py --data_dir " + str(output_dir)
    )


if __name__ == "__main__":
    main()
