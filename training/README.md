# Training a real denomination classifier

This replaces the prototype's heuristic color-histogram denomination
matcher with a real, trained CNN — fixing the `denomination_confidence: 0.00`
/ `"unknown"` results you get today when `data/reference_notes/` has no
templates for a denomination.

> **Scope note:** this fixes denomination classification only. It does
> **not** fix serial-number OCR problems (e.g. OCR only reading "AM"
> instead of the full serial) — that's a separate ROI-calibration issue,
> unrelated to denomination classification. See the main README's
> "Next Improvements" section for that fix.
>
> This dataset/pipeline is **not** suitable for counterfeit/genuine
> classification and this script does not attempt it — there are no
> counterfeit labels here, only denomination labels. Counterfeit
> detection stays a separate, explainable multi-signal consistency check
> in `app/services/decision_service.py`; a more reliable denomination
> classifier only helps that system indirectly, by making the ROI
> template selection it depends on more trustworthy.

---

## 1. Get your data into the right shape

**Recommended layout — pre-split train/val/test** (see project audit: this
avoids the validation-leakage problem the old flat-layout + internal
random-split approach had):
```
data_for_training/
    train/  10/*.jpg  20/*.jpg  50/*.jpg  100/*.jpg  200/*.jpg  500/*.jpg  2000/*.jpg
    val/    10/*.jpg  ...
    test/   10/*.jpg  ...   (optional, but gives you a trustworthy final accuracy number)
```
`train_denomination_classifier.py` uses this directly when a `train/`
subfolder is present — no internal re-split happens, so whatever
train/val/test boundary you set up here is respected exactly.

You don't need all 7 denominations — train on whichever your dataset
actually covers. Missing ones just keep falling back to the heuristic/
`DEFAULT` behavior, same as today.

**A flat single-folder layout** (`data_for_training/10/*.jpg`, no `train/`
subfolder) is still supported for backward compatibility — the script
falls back to its original internal random-split behavior automatically,
with a printed warning. Prefer the pre-split layout above for anything new.

**If your dataset is a Roboflow YOLO-format detection export** (images/ +
labels/ folders + a data.yaml), convert it first. Two optional safety
flags are available and worth using — see the module docstrings for
details on what each one actually does:

```bash
# Step 1 (optional but recommended): audit for near-duplicate photos across
# splits and possible stock/specimen images, BEFORE converting anything.
python dataset_audit.py --roboflow_dir path/to/roboflow_export

# Step 2: review dataset_audit_report.csv and suggested_exclusions.txt,
# then convert — --dedupe additionally re-derives the same cross-split
# duplicate check on its own (so step 1 is optional, not required):
python prepare_roboflow_data.py \
  --roboflow_dir path/to/roboflow_export \
  --output_dir ./data_for_training \
  --dedupe \
  --exclusion_list ./suggested_exclusions.txt
```
This produces the pre-split `train/`, `val/`, `test/` layout above
directly from Roboflow's own split (Roboflow's `valid/` becomes `val/`),
reading each image's annotated class from its YOLO label file (bbox or
polygon format — only the leading class id is read either way).

You can combine images from multiple datasets into the same folder
structure before training — more data per class generally helps — just
make sure you don't introduce the same cross-source duplication problem
`--dedupe`/`dataset_audit.py` exists to catch.

---

## 2. Install training dependencies

```bash
cd training
pip install -r requirements-train.txt

# Install PyTorch separately (pick the right command for your machine at
# https://pytorch.org/get-started/locally/). CPU-only example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 3. Train

```bash
python train_denomination_classifier.py --data_dir ./data_for_training --epochs 15
```

Useful flags:
- `--epochs` (default 15)
- `--batch_size` (default 32 — lower this if you run out of memory)
- `--lr` (default 1e-4)
- `--val_split` (default 0.15 — **only used in the legacy flat-layout fallback**;
  ignored when `train/`/`val/` subfolders are present, since that split is
  already fixed by prepare_roboflow_data.py)

What happens automatically, every time validation accuracy improves:
- The model is exported to `../app/models/weights/denomination_classifier.onnx`
- `../app/models/denomination_classifier/labels.py` is regenerated to
  match the **exact** class-index order PyTorch used during training

That second point matters more than it sounds: `ImageFolder` sorts class
folder names **alphabetically as strings** — `"100" < "20" < "2000" < "500"`,
not numeric order. Hand-typing this mapping is an easy way to get silently
wrong predictions (index 0 might not mean "10"). The script writes it for
you so this can't drift out of sync with the trained model.

If a `test/` folder is present, the script also prints a final held-out
test accuracy at the end, using the best (by validation accuracy)
checkpoint — never used for training or model selection, so this is the
number to trust/report, not the validation accuracy alone.

---

## 4. Verify it worked

```bash
cd ..   # back to the project root
uvicorn app.main:app --reload
```

Check the startup logs — you should see:
```
Denomination backends active: ['ml_classifier_onnx', 'heuristic_template_match']
```
(`ml_classifier_onnx` first now appearing means the trained model was
found and loaded.) Run an analysis via `/ui` or `/analyze` and confirm
`denomination.method` in the response now says `"ml_classifier_onnx"`
instead of `"fallback_unknown"` or `"heuristic_template_match"`.

---

## Tips for better accuracy

- **More images per class helps a lot** — aim for at least a few hundred
  per denomination if you can; a few dozen will train but won't generalize well.
- **Include varied lighting/angle/background** in your training images —
  if all your training photos look like clean scans, the model will
  struggle on a phone photo taken at an angle under normal room lighting.
- **Don't mirror-flip currency images** as an augmentation — the training
  script deliberately avoids this (text/serial numbers become unreadable
  when flipped, which would teach the model on physically impossible examples).
- If accuracy plateaus low, try more epochs, a lower learning rate, or
  unfreezing more of the backbone for fine-tuning (this script only trains
  the final classifier layer + lets BatchNorm adapt — full fine-tuning of
  all layers is a further improvement left for you to try if needed).
