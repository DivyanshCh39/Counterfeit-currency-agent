"""
Trains the OPTIONAL note-boundary segmentation model for the Counterfeit
Currency Identification Agent, and exports it to ONNX so it drops
directly into app/models/weights/note_boundary_segmenter.onnx — this
activates NoteBoundaryOnnxBackend automatically (see that file and
app/services/note_boundary_service.py). No app code changes needed;
PreprocessingService already delegates to NoteBoundaryService, which
already prefers the ONNX backend over the OpenCV heuristic whenever
weights are present. Skipping this training step entirely is completely
fine — the existing OpenCV contour heuristic keeps handling alignment on
its own, exactly as it does today.

SCOPE (see project audit): this model localizes the note boundary ONLY,
for perspective-alignment support. It is trained purely on the
denomination dataset's polygon/bbox annotations (rasterized into masks by
prepare_segmentation_data.py) — there are no counterfeit/genuine labels
anywhere in this pipeline, and this script must never be pointed at any.
It has no connection to app/services/decision_service.py and contributes
nothing to the counterfeit verdict.

Expected --data_dir layout (output of prepare_segmentation_data.py):
    data_dir/
        train/  images/*.jpg   masks/*.png  (0/255 single-channel)
        val/    images/...     masks/...
        test/   images/...     masks/...     (optional)
        manifest.csv

MODEL: a small MobileNetV2-encoder U-Net. Reusing MobileNetV2 (already
this project's transfer-learning backbone for denomination classification
— see train_denomination_classifier.py) keeps the tooling/mental-model
consistent across both trainers, and keeps the exported model small
enough for the same mobile/edge/POS-terminal deployment targets as the
rest of this project. This is deliberately a lightweight encoder-decoder,
not a large segmentation architecture (e.g. full U-Net/DeepLab) — the
target output is a single coarse note-vs-background mask, not fine
per-pixel semantic detail, so a heavier model would only cost inference
latency without meaningfully improving the boundary quad extracted from
it downstream (see app/utils/geometry_utils.mask_to_quad).

USAGE:
    python train_note_boundary_model.py --data_dir ./data_for_segmentation --epochs 20

IMPORTANT — preprocessing must match app/models/note_boundary/onnx_backend.py:
    resize to 256x256, RGB channel order, pixel values scaled to [0,1]
    (NO ImageNet mean/std normalization) — same convention already used by
    app/models/denomination_classifier/onnx_backend.py, kept identical
    here for consistency across every ONNX backend in this project. If you
    change one side, change the other.
"""

import argparse
import copy
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

INPUT_SIZE = (256, 256)  # must match NoteBoundaryOnnxBackend.INPUT_SIZE


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class NoteMaskDataset(Dataset):
    """Reads (image, mask) pairs from one prepare_segmentation_data.py
    split folder (data_dir/<split>/images, data_dir/<split>/masks)."""

    def __init__(self, split_dir: Path):
        self.images_dir = split_dir / "images"
        self.masks_dir = split_dir / "masks"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"No images/ folder under {split_dir}")
        self.image_paths = sorted(self.images_dir.glob("*.*"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.masks_dir / f"{image_path.stem}.png"

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, INPUT_SIZE)
        image_tensor = torch.from_numpy(image.astype("float32") / 255.0).permute(2, 0, 1)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy((mask > 127).astype("float32")).unsqueeze(0)

        return image_tensor, mask_tensor


# ----------------------------------------------------------------------
# Model: lightweight MobileNetV2-encoder U-Net
# ----------------------------------------------------------------------
class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip: Optional[torch.Tensor]):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class NoteBoundarySegmenter(nn.Module):
    """
    MobileNetV2 encoder (ImageNet-pretrained, same backbone family as
    train_denomination_classifier.py) + a small 4-stage decoder with skip
    connections -> single-channel logit map at input resolution. Sigmoid
    is applied at export time (see export_to_onnx), matching
    OnnxDenominationBackend's own predict-raw-then-wrap-Softmax pattern.
    """

    # torchvision MobileNetV2 feature indices whose outputs we tap as
    # encoder skip connections, shallowest to deepest.
    _SKIP_LAYER_INDICES = (1, 3, 6, 13)

    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.encoder = backbone

        # Channel counts at each tapped MobileNetV2 feature index, plus the
        # final feature map — fixed for mobilenet_v2, used to size the decoder.
        c1, c2, c3, c4, c_final = 16, 24, 32, 96, 1280

        self.decoder4 = _DecoderBlock(c_final, c4, 256)
        self.decoder3 = _DecoderBlock(256, c3, 128)
        self.decoder2 = _DecoderBlock(128, c2, 64)
        self.decoder1 = _DecoderBlock(64, c1, 32)
        self.final_upsample = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        skips = {}
        h = x
        for idx, layer in enumerate(self.encoder):
            h = layer(h)
            if idx in self._SKIP_LAYER_INDICES:
                skips[idx] = h
        # h is now the final MobileNetV2 feature map (1280 channels, /32 resolution)

        d = self.decoder4(h, skips[13])
        d = self.decoder3(d, skips[6])
        d = self.decoder2(d, skips[3])
        d = self.decoder1(d, skips[1])
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        d = self.final_upsample(d)
        logits = self.head(d)  # (N, 1, H, W) — raw logits, sigmoid applied at export/inference
        return logits


# ----------------------------------------------------------------------
# Loss / metric
# ----------------------------------------------------------------------
def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Combined BCE + Dice loss — standard, well-understood choice for binary
    segmentation. BCE alone tends to under-penalize small boundary errors
    on a mostly-foreground mask like a note filling most of the frame;
    Dice directly optimizes mask overlap, which is what mask_to_quad()'s
    downstream contour extraction actually cares about.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)

    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    dice_loss = 1 - dice.mean()

    return bce + dice_loss


@torch.no_grad()
def compute_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) >= 1).float().sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss, running_iou, n_batches = 0.0, 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = dice_bce_loss(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_iou += compute_iou(logits, masks)
        n_batches += 1
    return running_loss / n_batches, running_iou / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running_loss, running_iou, n_batches = 0.0, 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = dice_bce_loss(logits, masks)
        running_loss += loss.item()
        running_iou += compute_iou(logits, masks)
        n_batches += 1
    return running_loss / n_batches, running_iou / n_batches


def export_to_onnx(model: nn.Module, onnx_path: Path):
    # Same GPU/CPU export bugfix as train_denomination_classifier.py's
    # export_to_onnx(): the dummy trace input is a CPU tensor, so the model
    # must be temporarily moved to CPU regardless of which device it
    # trained on.
    original_device = next(model.parameters()).device
    model = model.to("cpu")
    model.eval()
    # Wrap with Sigmoid so the ONNX graph's output is already a [0,1]
    # probability map — NoteBoundaryOnnxBackend.detect() treats output[0]
    # as a ready-to-threshold probability mask, matching this exactly.
    wrapped = nn.Sequential(model, nn.Sigmoid())

    dummy_input = torch.randn(1, 3, *INPUT_SIZE)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        input_names=["input"],
        output_names=["mask"],
        dynamic_axes={"input": {0: "batch"}, "mask": {0: "batch"}},
        opset_version=12,
    )
    try:
        torch.onnx.export(wrapped, dummy_input, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapped, dummy_input, str(onnx_path), **export_kwargs)
    print(f"Saved ONNX note-boundary segmenter to {onnx_path}")

    model.to(original_device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "app" / "models" / "weights"),
        help="Where to write note_boundary_segmenter.onnx (defaults to the app's weights folder).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = Path(args.data_dir)
    train_dataset = NoteMaskDataset(data_root / "train")
    val_dir = data_root / "val"
    if not val_dir.exists():
        raise SystemExit(
            f"No val/ folder found under {data_root} — run prepare_segmentation_data.py "
            "first (it produces train/val[/test] from Roboflow's own split)."
        )
    val_dataset = NoteMaskDataset(val_dir)

    test_dataset = None
    test_dir = data_root / "test"
    if test_dir.exists():
        test_dataset = NoteMaskDataset(test_dir)

    print(
        f"Total images: train={len(train_dataset)} val={len(val_dataset)}"
        + (f" test={len(test_dataset)}" if test_dataset is not None else "")
    )
    if len(train_dataset) < 50:
        print(
            "WARNING: very small training set — expect a rough/unreliable segmenter. "
            "The OpenCV heuristic fallback remains available regardless."
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = NoteBoundarySegmenter().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    onnx_path = Path(args.weights_dir) / "note_boundary_segmenter.onnx"

    best_val_iou = 0.0
    best_state_dict = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_iou = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} — "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.3f} "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.3f}"
        )

        if val_iou >= best_val_iou:
            best_val_iou = val_iou
            best_state_dict = copy.deepcopy(model.state_dict())
            export_to_onnx(model, onnx_path)

    print(f"\nTraining complete. Best val IoU: {best_val_iou:.3f}")
    print(f"Final model: {onnx_path}")

    if test_dataset is not None:
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        test_loss, test_iou = evaluate(model, test_loader, device)
        print(
            f"\nFinal held-out TEST IoU (best checkpoint, never used for training or "
            f"model selection): {test_iou:.3f} (loss={test_loss:.4f}, n={len(test_dataset)})"
        )
    else:
        print(
            "\nNo test/ set found under --data_dir — only train/val IoU is available."
        )

    print(
        "\nRestart the FastAPI server — NoteBoundaryOnnxBackend activates automatically "
        "and takes priority over the OpenCV heuristic; the heuristic remains available "
        "as a fallback regardless (see app/services/note_boundary_service.py)."
    )


if __name__ == "__main__":
    main()
