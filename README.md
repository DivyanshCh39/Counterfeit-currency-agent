# Counterfeit Currency Identification Agent

A modular, FastAPI-based system that analyzes a photographed Indian
currency note and flags it as **likely genuine**, **suspicious**, or
**unclear**, with a full breakdown of *why* — not just a score. Built
around a hybrid pipeline: a real trained ML model for denomination
classification, pretrained OCR for text extraction, and an explainable,
rule-based multi-signal consistency engine that cross-checks everything
the note "says about itself" against what it should say.

> **Honesty note, upfront:** this project does **not** perform legal-grade
> currency authentication and does **not** claim to reliably detect
> sophisticated counterfeits. There is no labeled genuine/counterfeit
> dataset behind it — none exists in what was used to build this — so
> "counterfeit detection" here means *consistency screening*, not learned
> forgery detection. See [What This Is *Not*](#what-this-is-not) and
> [Known Limitations](#known-limitations) for the full, specific picture.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Architecture Summary](#architecture-summary)
4. [ML vs. Heuristic — What's Actually Learned](#ml-vs-heuristic-whats-actually-learned)
5. [Dataset & Its Limitations](#dataset-its-limitations)
6. [Features](#features)
7. [API Overview](#api-overview)
8. [Deployment Targets](#deployment-targets)
9. [Setup & Run](#setup-run)
10. [Future Work](#future-work)
11. [What This Is *Not*](#what-this-is-not)
12. [Resume-Ready Summary](#resume-ready-summary)

---

## Project Overview

Currency counterfeiting is usually caught by combining several small
signals at once — does the serial number look right, is the microprint
crisp, is the security thread continuous, does the printed denomination
number even match what the note claims to be? A human bank teller does
this instinctively; this project tries to make that same multi-signal
reasoning explicit, automatic, and explainable in software.

The system takes a photo of a note (mobile camera, POS scanner, or
counting-machine feed), runs it through a seven-stage pipeline, and
returns a verdict alongside a per-signal breakdown and human-readable
explanations — so the output is auditable, not a black-box score.

## Problem Statement

Manual currency inspection doesn't scale — bank tellers, retail cashiers,
and small-business owners can't apply the same trained scrutiny to every
note, and cheap fake-detector hardware often does little more than a UV
lamp check. At the same time, *properly labeled* counterfeit currency
image data is (for obvious legal and practical reasons) not something
freely available to train a supervised "real vs. fake" classifier on.

This project explores a practical middle ground: **can a system built
from a real trained denomination classifier plus explainable, rule-based
cross-checks (does the OCR'd data agree with itself and with the
classified denomination?) catch the more common and unsophisticated forms
of alteration/misprint/low-quality reproduction — while being fully
transparent that it is not, and does not claim to be, a substitute for
certified authentication hardware?**

## Architecture Summary

```
upload image
    │
    ▼
preprocess (validate format, load, check blur/brightness)
    │
    ▼
detect note + align (perspective correction — heuristic contour detector,
                       optional trained segmentation model as a swap-in upgrade)
    │
    ▼
classify denomination (ML: ONNX MobileNetV2 classifier, trained;
                         heuristic color-match fallback if no weights loaded)
    │
    ▼
extract ROIs (serial number / microprint / security thread / denomination
               numeral / promise-clause text — config-driven per denomination)
    │
    ▼
OCR + validate serial number        ─┐
cross-check printed numeral          │
verify promise-clause legal text     ├─▶  weighted decision engine  ─▶  verdict + explanations
score microprint clarity             │      (+ hard-trigger override on
score security thread                ┘       confirmed numeral mismatch)
    │
    ▼
annotate output image (highlighted ROIs) → JSON response / UI display
```

Every stage is its own module behind a small, consistent interface, and
every ML-capable stage (denomination classification, note-boundary
alignment) is built as a **pluggable backend chain**: a real trained
model is tried first, and the pipeline falls back to a heuristic
automatically if no trained weights are present — the system never
crashes or refuses to run just because a model hasn't been trained yet.
This is the same reason the project stayed sane through several rounds of
audits and fixes: any single stage can be replaced (swap a heuristic for
a trained model, or a training dataset for a cleaner one) without
touching the others.

## ML vs. Heuristic — What's Actually Learned

Being precise about this is the whole point of this section.

| Component | What it actually is |
|---|---|
| **Denomination classification** | **Real trained ML** — MobileNetV2 (transfer learning), exported to ONNX, trained on a cleaned/deduplicated denomination-labeled dataset. This is the one component with genuine learned weights behind it. Falls back to a crude heuristic color-histogram match if no weights are loaded — the pipeline never breaks, it just gets less accurate. |
| **Note boundary / perspective alignment** | Heuristic by default (OpenCV edge/contour detection). An **optional** trained lightweight segmentation model (MobileNetV2-encoder U-Net) can replace it, trained from the same dataset's polygon annotations — inactive until those weights are trained and placed; the heuristic keeps working either way. |
| **Serial number / numeral / promise-clause text extraction** | **Pretrained ML** (EasyOCR / Tesseract) — general-purpose OCR, not fine-tuned on currency fonts specifically. |
| **Serial format validation, numeral cross-check, promise-clause keyword match** | **Rule-based.** OCR gives raw text; simple, explainable logic (length/charset/pattern checks, keyword matching, string comparison) decides if it's consistent with what's expected. |
| **Microprint clarity, security thread presence** | **Heuristic only**, explicitly. Image-quality proxies (sharpness, edge density, frequency-domain detail, contrast, vertical-line continuity) — not a model trained to recognize genuine microprint/thread patterns, because no labeled dataset for that exists. |
| **Final verdict / overall score** | **Rule-based weighted sum**, by deliberate design — not ML. Combines every signal above into a 0–100 score and a verdict, with one hard-coded override: a *confirmed* mismatch between the OCR'd printed denomination number and the classified denomination caps the score and forces a "suspicious" verdict outright, since that's a strong enough specific signal that it shouldn't be averaged away by unrelated high scores elsewhere. |
| **UV / infrared / other physical security features** | **Not implemented.** No UV imagery exists in the dataset, and no UV-capable hardware integration exists in this codebase. If someone tells you a currency scanner does UV verification, it means it has a UV light and a camera under it — this project doesn't. |

**In one sentence:** one real trained model decides *what* the note is;
pretrained OCR reads what's printed on it; everything after that is
transparent, auditable rules — not a second hidden classifier pretending
to "detect fakes."

## Dataset & Its Limitations

Training data for the denomination classifier came from a public Roboflow
currency-detection export (~1,350 images across 7 INR denominations,
deduplicated down to ~583 unique source photos after removing
near-identical Roboflow augmentation copies and cross-split leakage — see
`training/dataset_audit.py`).

**This dataset labels denomination only.** It contains:
- ✅ Which denomination each note is (₹10 through ₹2000)
- ❌ No genuine-vs-counterfeit labels of any kind
- ❌ No region-of-interest (serial number / thread / microprint) ground truth
- ❌ No UV/IR imagery

This is precisely *why* counterfeit screening in this project is
rule-based rather than a second trained classifier: there is nothing to
train a "counterfeit detector" on, and manufacturing synthetic fake-note
training data raises its own obvious problems this project isn't going
to solve by pretending otherwise. `training/dataset_audit.py` and
`training/prepare_roboflow_data.py --dedupe` exist specifically because
an early pass through this same dataset found real cross-split
near-duplicate leakage — worth knowing about if you retrain on it
yourself.

## Features

- Currency note detection + perspective correction (heuristic, with an
  optional trained segmentation-model upgrade path)
- **Trained ONNX denomination classifier** (MobileNetV2), with a
  ready-to-run retraining pipeline (`training/`) including dataset
  deduplication/leakage auditing
- Config-driven, denomination-aware ROI extraction: serial number,
  microprint, security thread, printed denomination numeral, and
  promise-clause text regions
- Serial number OCR (EasyOCR/Tesseract, pluggable) + 4-point format
  validation (length, character set, spacing, pattern)
- Printed denomination-numeral OCR, cross-checked against the classified
  denomination — a confirmed mismatch hard-triggers a "suspicious" verdict
- Promise-clause / legal-text OCR + expected-keyword verification
- Microprint clarity scoring (4 heuristics: sharpness, edge density,
  frequency-domain detail, patch texture)
- Security thread verification (3 heuristics: region contrast, vertical
  feature strength, band continuity)
- Explainable weighted decision engine — every score comes with a
  plain-English reason, not just a number
- Annotated output image with all ROI regions drawn and labeled
- Pluggable backend architecture throughout — every ML-capable stage can
  swap in a trained model without touching any other file
- Decoupled `POST /upload` + `POST /analyze/{file_id}`, plus one-shot
  `POST /analyze`
- Built-in HTML/JS demo UI + optional Streamlit UI for raw JSON inspection
- `tools/roi_calibrator.html` — offline visual tool for measuring ROI
  coordinates from your own reference note photos

## API Overview

| Endpoint | Purpose |
|---|---|
| `GET /health` | App status/version |
| `GET /ui` | Built-in demo frontend |
| `POST /upload` | Store a note image, get back a `file_id` (no analysis yet) |
| `POST /analyze` | Upload **and** analyze in one call |
| `POST /analyze/{file_id}` | Analyze a previously-uploaded file |

**Response shape** (both `/analyze` endpoints):
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
    "Score falls between the suspicious and genuine thresholds [45, 75) — flagged 'unclear' for manual review."
  ],
  "score_overridden": false,
  "denomination": { "predicted_value": "500", "confidence": 0.71, "method": "ml_classifier_onnx" },
  "checks": { "serial_number": { "...": "..." }, "microprint": { "...": "..." }, "security_thread": { "...": "..." } },
  "annotated_image_url": "/static/results/<uuid>.jpg",
  "notes": ["PROTOTYPE DISCLAIMER: ..."]
}
```
Full field-level docs and a complete example live in the interactive
Swagger UI at `/docs` once the server is running, and in the inline
docstrings across `app/schemas/response_schemas.py`.

## Deployment Targets

Designed with three real-world form factors in mind, though only the
first is actually demonstrated end-to-end today:

- **Backend API / web demo** — ✅ fully working today (FastAPI + built-in
  UI), this is what you can run and click through right now
- **Mobile devices** — designed for: the ONNX runtime is
  mobile-friendly, and a TFLite backend template already exists
  (`app/models/denomination_classifier/tflite_backend.py`, not yet wired
  in) for on-device inference
- **Bank counting machines / POS terminals** — designed for: the
  stateless `POST /analyze` API and lightweight model footprint fit an
  embedded/edge deployment model, but no embedded integration exists yet
  — this is an architectural target, not a shipped integration

## Setup & Run

```bash
git clone <this-repo>
cd counterfeit-currency-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Then open **http://localhost:8000/ui** for the demo UI, or
**http://localhost:8000/docs** for interactive API docs.

```bash
pytest tests/ -v   # 70+ tests, mirrors the app/ module structure
```

To retrain the denomination classifier or the optional note-boundary
segmenter, see `training/README.md`.

## Future Work

- Train and ship a real note-boundary segmentation model (pipeline
  already built, just needs the training run — see `training/`)
- Wire up the TFLite backend for genuine on-device mobile inference
- Custom-train the OCR backend on actual note fonts instead of
  general-purpose pretrained EasyOCR/Tesseract
- Build an actual embedded/POS integration, not just an API that would
  fit one
- If a labeled genuine/counterfeit dataset ever becomes ethically and
  legally available: train real learned scorers for microprint and
  security-thread authenticity, replacing today's heuristic proxies —
  this is explicitly gated on data that doesn't currently exist, not a
  near-term roadmap item
- UV/IR feature support, if paired with UV-capable capture hardware —
  currently fully out of scope

## What This Is *Not*

To be unambiguous, since this matters for how this project should be
represented anywhere it's shown:

- ❌ Not a certified or legal-grade currency authentication device
- ❌ Not a trained "genuine vs. counterfeit" classifier — no such model
  exists anywhere in this codebase, because no labeled dataset for one
  exists
- ❌ Not a guarantee of catching sophisticated counterfeits — it screens
  for internal inconsistency (numbers/text that don't agree with each
  other, low print/thread quality) using heuristics, not learned forgery
  detection
- ❌ Not a UV/IR verification system
- ✅ **Is** a working demonstration of a hybrid CV + rule-based
  explainable screening pipeline, with one genuinely trained ML model at
  its core and honest, visible boundaries everywhere else

## Known Limitations

| Component | Status |
|---|---|
| Note detection / alignment | Heuristic, ML upgrade path built but not trained |
| ROI coordinates | Manual, config-driven — calibrated per denomination against real photos, not learned |
| Denomination classification | Real trained ML, with heuristic fallback |
| Serial/numeral/promise-clause OCR | Pretrained general-purpose OCR, not fine-tuned |
| Microprint / security thread | Heuristic image-quality proxies only |
| Final verdict | Rule-based weighted sum + one hard-trigger override — not ML |
| UV/IR verification | Not implemented |

---

## Resume-Ready Summary

*Designed and built a full-stack, hybrid computer-vision system for
screening Indian currency notes, combining a trained ONNX denomination
classifier with an explainable, rule-based multi-signal consistency
engine (OCR-based serial/numeral/legal-text verification, microprint and
security-thread heuristics) behind a pluggable-backend FastAPI
architecture — engineered specifically to be honest about what is
learned vs. rule-based, given the absence of any labeled
counterfeit-currency training data.*

