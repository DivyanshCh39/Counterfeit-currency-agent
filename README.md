# Counterfeit Currency Identification Agent

A FastAPI backend that takes a photo of an Indian currency note and returns a verdict — likely genuine, suspicious, or unclear — along with a breakdown of the checks that led to it. Built as a hybrid system: a trained ONNX model handles denomination classification, OCR reads the serial number and printed text, and a rule-based scoring engine checks whether everything is internally consistent.

This is a prototype built for academic evaluation, not a certified currency-authentication tool. There is no legally available dataset of genuine-vs-counterfeit note images, so "counterfeit detection" here means checking a note's own printed details against each other, not learning what a forged note looks like.

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Architecture](#architecture)
3. [ML vs. Rule-Based Components](#ml-vs-rule-based-components)
4. [Dataset & Its Limitations](#dataset-its-limitations)
5. [Features](#features)
6. [API Overview](#api-overview)
7. [Setup & Run](#setup-run)
8. [Future Work](#future-work)

---

## Problem Statement

Bank tellers and cashiers check notes by combining several small cues at once — serial number format, thread continuity, microprint sharpness, whether the printed denomination matches what the note claims to be. Doing this consistently at scale is hard, and low-cost fake-detector hardware usually does little beyond a UV check.

Training a supervised genuine-vs-fake classifier isn't really an option here — there's no legal way to get a labeled dataset of real counterfeit notes. So this project takes a different approach: use a real trained model for the one thing that can be trained (denomination), and use explainable rules for everything else, cross-checking a note's own details for internal consistency rather than trying to spot forgery visually.

## Architecture

```
upload image
    │
    ▼
preprocess (validate format, check blur/brightness)
    │
    ▼
detect note + align (perspective correction — heuristic contour detector,
                       optional trained segmentation model as a swap-in)
    │
    ▼
classify denomination (ONNX MobileNetV2 classifier, trained;
                         falls back to a heuristic color match if no weights)
    │
    ▼
extract ROIs (serial number, microprint, security thread, printed
               denomination numeral, promise-clause text)
    │
    ▼
OCR + validate serial number        ─┐
cross-check printed numeral          │
verify promise-clause text           ├─▶  weighted decision engine  ─▶  verdict
score microprint clarity             │
score security thread                ┘
    │
    ▼
annotate output image → JSON response / UI
```

Each stage is a separate module. Denomination classification and note alignment are both built as pluggable backends — a trained model is tried first, and the pipeline falls back to a heuristic automatically if no weights are present. This was mainly done so the system keeps working end to end even before any model is trained, and so a backend can be swapped later without touching the rest of the code.

## ML vs. Rule-Based Components

| Component | What it is |
|---|---|
| Denomination classification | Trained ML (MobileNetV2, exported to ONNX). Falls back to a simple color-histogram match if no weights are loaded. |
| Note boundary / alignment | Heuristic by default (OpenCV contour detection). A trained segmentation model can replace it — built, but not trained yet. |
| Serial number / numeral / promise-clause text extraction | Pretrained OCR (EasyOCR / Tesseract), not fine-tuned on currency fonts. |
| Serial format check, numeral cross-check, promise-clause keyword match | Rule-based logic on top of the OCR output — length/charset checks, string comparison, keyword matching. |
| Microprint clarity, security thread presence | Heuristic image-quality proxies (sharpness, edge density, contrast, line continuity). Not a trained model — there's no labeled data to train one on. |
| Final verdict / overall score | Rule-based weighted sum. One hard rule: if the OCR'd printed denomination number doesn't match the classified denomination, the verdict is forced to "suspicious" regardless of other scores. |
| UV / IR features | Not implemented — no UV imagery in the dataset and no UV hardware support in the code. |

So: one trained model decides what denomination the note is, OCR reads what's printed, and everything after that is plain rule-based logic that can be read and audited line by line.

## Dataset & Its Limitations

The denomination classifier was trained on a Roboflow currency-detection export (~1,350 labeled images across 7 INR denominations) plus a small set of additional reference photos for ₹500 and ₹2000 pulled from a second public dataset. Before training, the data was checked for cross-split near-duplicates using perceptual hashing (`training/dataset_audit.py`) — this caught several duplicate photos that had ended up in both the training and validation sets, which were removed.

The dataset only labels denomination. It does not contain:
- Genuine-vs-counterfeit labels
- Region-of-interest ground truth (serial number / thread / microprint locations)
- UV or IR imagery

This is the main reason counterfeit screening in this project is rule-based instead of a second trained classifier — there's simply nothing to train a counterfeit detector on.

**Current trained model:** 96.6% validation accuracy, 88.5% accuracy on a held-out test set never used during training or model selection.

## Features

- Note detection and perspective correction (heuristic, with an optional trained segmentation-model upgrade path)
- Trained ONNX denomination classifier with a full retraining pipeline (`training/`), including a dataset deduplication/leakage check
- Denomination-aware ROI extraction: serial number, microprint, security thread, printed denomination numeral, promise-clause text
- Serial number OCR + format validation (length, character set, spacing, pattern)
- Printed denomination-numeral OCR, cross-checked against the classified denomination
- Promise-clause text OCR and keyword verification
- Microprint clarity scoring and security thread verification (heuristic)
- Weighted decision engine with per-check explanations, not just a single score
- Annotated output image with ROI regions drawn and labeled
- Pluggable backend design — any ML-capable stage can be swapped for a trained model without touching the rest of the pipeline
- `POST /upload` + `POST /analyze/{file_id}`, or one-shot `POST /analyze`
- Built-in HTML/JS demo UI, plus an optional Streamlit UI
- `tools/roi_calibrator.html` for manually measuring ROI coordinates on reference note photos

## API Overview

| Endpoint | Purpose |
|---|---|
| `GET /health` | App status |
| `GET /ui` | Demo frontend |
| `POST /upload` | Store an image, get back a `file_id` |
| `POST /analyze` | Upload and analyze in one call |
| `POST /analyze/{file_id}` | Analyze a previously uploaded file |

Example response from `/analyze`:
```json
{
  "verdict": "unclear",
  "overall_score": 74.4,
  "feature_scores": {
    "image_quality": 82.0,
    "denomination_confidence": 71.0,
    "serial_quality": 94.0,
    "microprint_clarity": 38.0,
    "security_thread": 66.0,
    "numeral_consistency": 100.0,
    "promise_clause": 100.0
  },
  "explanations": [
    "Image quality is strong (82.0/100), supporting reliable analysis.",
    "Microprint region shows low clarity (38.0/100), consistent with blurred or low-quality reproduction.",
    "Composite weighted score: 74.4/100.",
    "Score falls between the suspicious and genuine thresholds — flagged 'unclear' for manual review."
  ],
  "score_overridden": false,
  "denomination": { "predicted_value": "500", "confidence": 0.71, "method": "ml_classifier_onnx" },
  "annotated_image_url": "/static/results/<uuid>.jpg"
}
```
Full field docs are in the Swagger UI at `/docs` once the server is running.

`checks.denomination_numeral` and `checks.promise_clause` only appear when the classified denomination's ROI template defines those regions (currently "100", "500", and the default template — see `app/config/roi_config.py`).

## Setup & Run

```bash
git clone <this-repo>
cd counterfeit-currency-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open `http://localhost:8000/ui` for the demo UI, or `http://localhost:8000/docs` for API docs.

```bash
pytest tests/ -v
```

To retrain the denomination classifier or the optional note-boundary segmenter, see `training/README.md`.

## Future Work

- Train the note-boundary segmentation model (pipeline is built, just needs a training run)
- Wire up the existing TFLite backend template for on-device mobile inference
- Fine-tune OCR on actual currency fonts instead of general-purpose EasyOCR/Tesseract
- If a labeled genuine/counterfeit dataset ever becomes available, replace the microprint and thread heuristics with trained scorers — currently blocked on data that doesn't exist, not on time
