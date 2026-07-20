"""
Trains a real denomination classifier for the Counterfeit Currency
Identification Agent, and exports it to ONNX so it drops directly into
app/models/weights/denomination_classifier.onnx — this activates the
existing OnnxDenominationBackend automatically. No app code changes
needed; DenominationService already prefers the ONNX backend over the
heuristic one whenever it's available.

--- PREFERRED data_dir layout (pre-split — see project audit) ---
If --data_dir contains a `train/` subfolder, this script uses a PRE-SPLIT
layout and does NOT perform any internal train/val split:
    data_dir/
        train/  10/*.jpg  20/*.jpg  ...  500/*.jpg  2000/*.jpg
        val/    10/*.jpg  ...                                   (optional but recommended)
        test/   10/*.jpg  ...                                   (optional)
This is exactly what prepare_roboflow_data.py now produces. Using it is
strongly preferred: the original version of this script merged all
available images into one pool and did its own random 85/15 split, which
silently let near-duplicate/augmented copies of the same source photo
land on both sides of the split (see project audit — this is the root
cause of the previously-unverified 99.5% validation accuracy figure).
Loading a pre-split directory instead means whatever split boundary you
already cleaned (deduplicated, exclusion-filtered) in prepare_roboflow_data.py
is respected exactly, end to end. If val/ is present it is used for
per-epoch model selection; if test/ is also present, it's evaluated once
at the end as a final held-out sanity check (never used for checkpoint
selection).

--- LEGACY fallback layout (flat, single split) ---
If --data_dir does NOT contain a `train/` subfolder, it's treated as the
original flat torchvision ImageFolder layout:
    data_dir/
        10/   *.jpg
        20/   *.jpg
        50/   *.jpg
        100/  *.jpg
        200/  *.jpg
        500/  *.jpg
        2000/ *.jpg
and split internally via --val_split, exactly as before (with a printed
warning) — kept only for backward compatibility with datasets prepared
before this revision, or non-Roboflow sources you've organized this way
yourself. Prefer the pre-split layout above for anything new.

If your downloaded dataset isn't already in one of these two shapes (e.g.
a Roboflow YOLO-format detection export), run prepare_roboflow_data.py
first to convert it.

USAGE:
    python train_denomination_classifier.py --data_dir path/to/data --epochs 15

IMPORTANT — preprocessing must match app/models/denomination_classifier/onnx_backend.py:
    resize to 224x224, RGB channel order, pixel values scaled to [0,1]
    (NO ImageNet mean/std normalization). This script intentionally
    matches that exactly — if you change one side, change the other.
"""

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

INPUT_SIZE = (224, 224)  # must match OnnxDenominationBackend.INPUT_SIZE


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def get_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(INPUT_SIZE),
                # NOTE: deliberately NO horizontal/vertical flip — currency
                # text and serial numbers become mirrored/unreadable, which
                # would teach the model on physically impossible examples.
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomRotation(degrees=8),
                transforms.ToTensor(),  # scales to [0,1], matches onnx_backend preprocessing
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
        ]
    )


def build_datasets(data_dir: str, val_split: float):
    """
    Returns (train_dataset, val_dataset, test_dataset_or_None, class_names, using_presplit).

    Prefers the pre-split train/[val/][test/] layout (see module docstring)
    when a `train/` subfolder exists under data_dir; falls back to the
    legacy flat-folder + internal random_split behavior otherwise.
    """
    data_root = Path(data_dir)

    if (data_root / "train").is_dir():
        train_dataset = datasets.ImageFolder(
            str(data_root / "train"), transform=get_transforms(train=True)
        )
        class_names = train_dataset.classes

        val_dir = data_root / "val"
        if val_dir.is_dir():
            val_dataset = datasets.ImageFolder(str(val_dir), transform=get_transforms(train=False))
            if val_dataset.classes != class_names:
                raise ValueError(
                    f"Class mismatch between train/ ({class_names}) and val/ "
                    f"({val_dataset.classes}) — every denomination folder present under "
                    f"train/ must also exist under val/ (even if a class has very few "
                    f"validation images, an empty folder is still required) so class "
                    f"indices line up identically across both sets."
                )
        else:
            print(
                "WARNING: pre-split layout detected (train/ exists) but val/ is missing — "
                "falling back to an internal random split of train/ only. Add a val/ folder "
                "(e.g. re-run prepare_roboflow_data.py, which produces one from Roboflow's "
                "own valid/ split) for a trustworthy, non-leaked validation accuracy."
            )
            val_size = max(1, int(len(train_dataset) * val_split))
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            val_dataset.dataset = datasets.ImageFolder(
                str(data_root / "train"), transform=get_transforms(train=False)
            )

        test_dataset = None
        test_dir = data_root / "test"
        if test_dir.is_dir():
            test_dataset = datasets.ImageFolder(str(test_dir), transform=get_transforms(train=False))
            if test_dataset.classes != class_names:
                raise ValueError(
                    f"Class mismatch between train/ ({class_names}) and test/ "
                    f"({test_dataset.classes})."
                )

        return train_dataset, val_dataset, test_dataset, class_names, True

    # --- Legacy fallback: flat single-folder layout, internal random split ---
    print(
        "NOTE: no train/ subfolder found under --data_dir — using the legacy flat-layout "
        "path (single pool, internal random split). This is kept for backward "
        "compatibility; prefer running prepare_roboflow_data.py to get a clean, "
        "pre-split train/val/test layout instead (see this script's docstring)."
    )
    full_dataset = datasets.ImageFolder(data_dir, transform=get_transforms(train=True))
    class_names = full_dataset.classes
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset = datasets.ImageFolder(data_dir, transform=get_transforms(train=False))
    return train_dataset, val_dataset, None, class_names, False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def export_to_onnx(model, class_names, onnx_path: Path, labels_path: Path):
    # BUGFIX: the dummy trace input below is a plain CPU tensor, but `model`
    # may currently live on GPU (training uses model.to(device)). Exporting
    # requires model and input on the SAME device, so temporarily move the
    # model to CPU for export, then restore it to keep training on GPU.
    original_device = next(model.parameters()).device
    model = model.to("cpu")
    model.eval()
    # OnnxDenominationBackend treats the model's raw output as class
    # probabilities (argmax + direct confidence read) — wrap with Softmax
    # so that assumption holds after export.
    wrapped = nn.Sequential(model, nn.Softmax(dim=1))

    dummy_input = torch.randn(1, 3, *INPUT_SIZE)  # CPU tensor, matches model above
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        input_names=["input"],
        output_names=["probabilities"],
        dynamic_axes={"input": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=12,
    )
    try:
        # Newer torch (2.x+) defaults to a 'dynamo' exporter that requires
        # the extra 'onnxscript' package. Force the classic, broadly
        # compatible TorchScript-based exporter instead when available.
        torch.onnx.export(wrapped, dummy_input, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch versions don't have a 'dynamo' parameter at all (they
        # always use the classic exporter) — retry without it.
        torch.onnx.export(wrapped, dummy_input, str(onnx_path), **export_kwargs)
    print(f"Saved ONNX model to {onnx_path}")

    model.to(original_device)  # restore for the next training epoch

    # Auto-generate labels.py using the EXACT class order PyTorch used for
    # training. This matters a lot: torchvision's ImageFolder sorts class
    # folder names ALPHABETICALLY AS STRINGS, e.g. "100" < "20" < "2000" <
    # "500" — NOT numeric order. Hand-typing this mapping is a common and
    # easy-to-miss source of silently-wrong predictions, so it's generated
    # here directly from the trained dataset instead.
    #
    # NOTE: deliberately NOT using json.dumps() for the dict body — JSON
    # coerces integer keys to strings (e.g. {0: "10"} -> '{"0": "10"}'),
    # which would silently break DENOMINATION_LABELS.get(int_class_idx, ...)
    # lookups in onnx_backend.py. Python dict-literal syntax is written by hand instead.
    labels_dict = {i: name for i, name in enumerate(class_names)}
    dict_body = "\n".join(f"    {i}: {name!r}," for i, name in labels_dict.items())
    labels_path.write_text(
        '"""\n'
        "Auto-generated by training/train_denomination_classifier.py.\n"
        "Class index -> denomination label mapping, matching the exact\n"
        "training-time class order. Do not hand-edit the ordering -- rerun\n"
        "training (or edit both this file and the ONNX model consistently)\n"
        "if it needs to change.\n"
        '"""\n\n'
        f"DENOMINATION_LABELS = {{\n{dict_body}\n}}\n",
        encoding="utf-8",  # BUGFIX: without this, Windows writes using the
        # system's default codepage (often cp1252), which can silently
        # produce bytes that later fail to decode as UTF-8 when Python
        # imports the file (e.g. an em-dash character previously used here
        # crashed with "SyntaxError: 'utf-8' codec can't decode byte 0x97").
    )
    print(f"Wrote label mapping to {labels_path}")
    print(f"Class order used: {labels_dict}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "app" / "models" / "weights"),
        help="Where to write denomination_classifier.onnx (defaults to the app's weights folder).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset, class_names, using_presplit = build_datasets(
        args.data_dir, args.val_split
    )
    print(f"Found {len(class_names)} classes: {class_names}")
    print(
        f"Total images: train={len(train_dataset)} val={len(val_dataset)}"
        + (f" test={len(test_dataset)}" if test_dataset is not None else "")
        + f" (layout: {'pre-split' if using_presplit else 'legacy flat + internal split'})"
    )
    if len(train_dataset) < 50:
        print(
            "WARNING: very small training set — expect an overfit/unreliable model. "
            "Aim for at least a few hundred images per class if possible."
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    onnx_path = Path(args.weights_dir) / "denomination_classifier.onnx"
    labels_path = (
        Path(__file__).resolve().parent.parent
        / "app" / "models" / "denomination_classifier" / "labels.py"
    )

    best_val_acc = 0.0
    best_state_dict = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} — "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            export_to_onnx(model, class_names, onnx_path, labels_path)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Final model: {onnx_path}")
    print(f"Label mapping: {labels_path}")

    if test_dataset is not None:
        # Evaluate the BEST checkpoint (by val accuracy), not whichever
        # epoch happened to run last — this is the number reported in
        # docs/README as the trustworthy, held-out figure, since test/ was
        # never touched by training or by checkpoint selection.
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"\nFinal held-out TEST accuracy (best checkpoint, never used for training or "
            f"model selection): {test_acc:.3f} (loss={test_loss:.4f}, n={len(test_dataset)})"
        )
    else:
        print(
            "\nNo test/ set found under --data_dir — only train/val accuracy is available. "
            "For a fully trustworthy final number, re-run prepare_roboflow_data.py (it "
            "produces test/ from Roboflow's own test split) and retrain."
        )
    print("Restart the FastAPI server — the ONNX backend activates automatically.")


if __name__ == "__main__":
    main()
