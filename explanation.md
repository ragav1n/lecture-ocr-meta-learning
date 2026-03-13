# Project Explanation: German Lecture Slide OCR with Meta-Learning

## What Is This Project?

University lecture slides often contain handwritten annotations — a professor writes equations on a whiteboard, adds notes to a slide, or sketches diagrams. This project builds a system that:

1. **Detects** regions of handwritten text and math in a slide image
2. **Recognizes** the handwritten content (German text or LaTeX math)
3. **Replaces** the handwriting with clean typeset equivalents

The key innovation is **meta-learning**: instead of retraining for each professor, the system can adapt to a new professor's handwriting style using only 5–50 examples. This is important for a research paper — we want to show that our system works for *any* professor, not just one.

**Publication target**: ICDAR 2026 or CVPR 2026.

---

## The Big Picture: Three Sub-Problems

```
Input: Lecture slide image with handwriting
         │
         ▼
┌─────────────────────┐
│  1. DETECTION       │  Where is the handwriting?
│  YOLOv8 / DLAFormer │  → Bounding boxes around text/math regions
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  2. RECOGNITION     │  What does it say?
│  TrOCR (text)       │  → "Die Ableitung von f(x) ist..."
│  TAMER (math)       │  → "\frac{d}{dx} f(x) = ..."
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  3. REPLACEMENT     │  Render it cleanly
│  PIL / LaTeX render │  → Typeset text/equation overlaid on slide
└─────────────────────┘

Output: Clean slide with typeset content replacing handwriting
```

---

## Hardware & Environment

- **GPU**: NVIDIA RTX 4060 Ti (16GB VRAM)
- **Python**: 3.12
- **PyTorch**: 2.10.0
- **Key libraries**: transformers 5.3, ultralytics (YOLOv8), albumentations, spacy, loguru
- **Virtual environment**: `venv/` in project root

---

## Repository Structure

```
ocr_slide/
├── baseline/               # Phase 1: Working pipeline with off-the-shelf models
│   ├── baseline_pipeline.py    # YOLOv8 + TrOCR + Pix2Tex assembled together
│   └── test_baseline.py        # Test harness for single images and batches
│
├── models/                 # Phase 2: Better model implementations
│   ├── dlaformer_adapter.py    # Improved document layout detector
│   ├── math_ocr_tamer.py       # Better math OCR (uses pre-trained TAMER)
│   └── meta_learning_ocr.py    # MAML wrapper for professor adaptation
│
├── training/               # Training scripts
│   ├── train_detector_baseline.py  # Train YOLOv8 on document data
│   ├── finetune_german_ocr.py      # Fine-tune TrOCR on German handwriting
│   └── train_meta_learning.py      # MAML meta-training for professor adaptation
│
├── evaluate/               # Evaluation scripts
│   ├── eval_detection.py       # Detection quality (mAP)
│   ├── eval_german_ocr.py      # Text OCR quality (CER, WER)
│   ├── eval_math_ocr.py        # Math OCR quality (BLEU)
│   └── eval_pipeline.py        # End-to-end evaluation
│
├── scripts/                # Data preparation
│   ├── prepare_iam_german.py       # Extract German handwriting from IAM dataset
│   ├── prepare_doclaynet.py        # Extract document layout data from DocLayNet
│   ├── prepare_crohme.py           # Prepare math expression data (CROHME)
│   └── build_lecture_dataset.py    # Build our own benchmark dataset (Phase 4)
│
├── utils/                  # Shared utilities
│   ├── metrics.py              # CER, WER, mAP, BLEU implementations
│   ├── image_utils.py          # Image loading, resizing, augmentation
│   └── german_postprocessing.py # Fix common OCR errors in German text
│
├── configs/                # Configuration files
│   ├── handwriting_detection.yaml  # YOLOv8 training config
│   ├── meta_training.yaml          # MAML hyperparameters
│   └── final_pipeline.yaml         # Full system config
│
├── data/                   # Datasets (images gitignored, manifests committed)
│   ├── iam_downloads/          # IAM handwriting dataset archives
│   ├── DocLayNet/              # DocLayNet document layout dataset (28GB zip)
│   ├── CROHME/                 # Math expression dataset (⚠️ bad download)
│   ├── Dr_Judith_Jakob_Slides/ # Professor slides (⛔ do not process yet)
│   └── processed/              # Prepared data ready for training
│       ├── german_text/        # IAM German: JSON manifests + line images
│       ├── detection/          # DocLayNet: YOLO-format labels + images
│       └── math/               # CROHME: empty (data missing)
│
├── TAMER/                  # External math OCR library (git submodule)
├── CLAUDE.md               # Full development log
├── COMPLETE_IMPLEMENTATION_PLAN.md  # 12-week roadmap
└── requirements_project.txt         # All Python dependencies
```

---

## The Data

### 1. IAM German Handwriting Dataset
**Purpose**: Train and evaluate the German text OCR model.

The IAM dataset is a large collection of handwritten English text. However, a subset of writers are native German or Swiss German speakers — their handwriting samples contain German words and letter patterns (umlauts like ä, ö, ü, etc.).

**What we did**:
- Parsed `writers.xml` to find writers with `NativeLanguage="German"` or `"Swiss German"`
- Found **181 German-native writers**, **487 forms**
- Extracted **4,286 line-level images** with their text transcriptions
- Split into: **3,000 train / 642 val / 644 test**
- Stored as JSON manifests: `data/processed/german_text/german_text_*.json`

Each entry in the manifest looks like:
```json
{
  "image": "data/processed/german_text/images/a01-000u-00.png",
  "text": "Die Universität ist bekannt für ihre Forschung",
  "writer_id": "000"
}
```

### 2. DocLayNet Document Layout Dataset
**Purpose**: Train the region detector to find text/math boxes in documents.

DocLayNet is a large dataset of document page images with bounding box annotations for different content types (text, figures, tables, formulas, etc.).

**What we did**:
- Streamed directly from the 28GB zip (no full extraction — disk space constraint)
- Extracted **14,723 images** (5,000 train / 5,000 val / 4,723 test)
- Mapped annotation classes: `Formula → math (class 1)`, all text types → `text (class 0)`
- Converted to YOLO format (normalized bounding boxes in `.txt` files)
- **178,417 total bounding boxes** across all splits
- Dataset config: `data/processed/detection/dataset.yaml`

### 3. CROHME 2019 Math Expressions
**Purpose**: Train/evaluate math OCR on handwritten math expressions.

**Status**: ⚠️ The downloaded zip file is actually an HTML error page (the download failed silently). We handle this gracefully — math manifests exist but are empty.

**Workaround**: TAMER (our math OCR model) already has pre-trained checkpoints from CROHME training, so we don't need to train from scratch.

---

## The Models

### Detection: YOLOv8x
**What it does**: Takes a slide image, returns bounding boxes around each handwritten region, labeled as `text` or `math`.

YOLOv8 is a state-of-the-art real-time object detector. We use the `x` (extra-large) variant for maximum accuracy. It's trained on our DocLayNet subset.

**Training**: `python training/train_detector_baseline.py`
- Input: document page images + YOLO format labels
- Output: `checkpoints/yolov8_detector/best.pt`
- Target metric: **mAP@0.5 ≥ 88%**

**Current status**: Training in progress (`runs/detect/runs/detect/baseline_v1_r2/`). At epoch ~24/100, mAP50=82.2% and climbing.

### Text OCR: TrOCR
**What it does**: Takes a cropped image of handwritten text, outputs the Unicode string.

TrOCR (from Microsoft) is a transformer model that combines a Vision Transformer (ViT) image encoder with a language model decoder. We start from `microsoft/trocr-large-handwritten` (pre-trained on general handwriting) and fine-tune on German text.

**Fine-tuning**: `python training/finetune_german_ocr.py`
- Input: 3,000 German handwriting images + transcriptions
- Output: `checkpoints/trocr_german/best/`
- Techniques: AMP (mixed precision), gradient accumulation ×4, cosine LR schedule, beam search (n=4)
- Target metric: **CER ≤ 5%** (Character Error Rate — % of characters wrong)
- Baseline (no fine-tuning): ~15-20% CER

**German post-processing** (`utils/german_postprocessing.py`):
After OCR, we apply rule-based corrections specific to German:
- ASCII substitutions: `oe→ö`, `ae→ä`, `ue→ü`, `ss→ß`
- Common mistranscriptions: `fuer→für`, `nicht→nicht`, `Universitat→Universität`
- Spell-check using PySpellChecker with German dictionary

### Math OCR: TAMER
**What it does**: Takes a cropped image of a handwritten math expression, outputs LaTeX.

TAMER is a Tree-Aware Math Expression Recognizer — it understands the hierarchical tree structure of math expressions (fractions, superscripts, subscripts, etc.) rather than treating them as flat sequences.

The TAMER repository is included as a **git submodule** at `TAMER/`. It has pre-trained checkpoints:
- `TAMER/lightning_logs/version_3/` — trained on HME100K **with** tree-structure fusion → **ExpRate 69.5%** (best)
- `TAMER/lightning_logs/version_1/` — trained on HME100K without fusion → 68.5%
- `TAMER/lightning_logs/version_0/` — trained on CROHME → 61.1%

Our wrapper (`models/math_ocr_tamer.py`) loads version_3 by default. If TAMER fails to load, it falls back to Pix2Tex.

### Meta-Learning: MAML (Model-Agnostic Meta-Learning)
**What it does**: Allows the system to adapt to a new professor's handwriting using only 5–50 examples.

**The problem MAML solves**: Standard fine-tuning needs hundreds of examples and takes hours. For professor adaptation, we want to give the system 5-10 handwriting samples and have it adapt in seconds.

**How MAML works**:
1. **Meta-training** (offline, done once): Train the model on many different "writers" (people) from the IAM dataset. For each writer, simulate the adaptation: take 10 support samples, do a few gradient steps, measure performance on 10 query samples. Optimize the *starting weights* to be a good initialization for fast adaptation.
2. **Meta-testing** (online, per professor): Give 5-50 samples from the new professor, do a few gradient steps from the MAML-optimized starting point → adapted model.

**Implementation** (`models/meta_learning_ocr.py`):
- Uses the `learn2learn` library (MAML implementation)
- Falls back to manual MAML if `learn2learn` is not installed
- Wraps the fine-tuned TrOCR as the base model
- Writer tasks built from IAM German data (each writer = one task)

**Training**: `python training/train_meta_learning.py`
- Input: fine-tuned TrOCR + IAM German writer tasks
- Output: `checkpoints/meta_learning/meta_checkpoint_best.pt`
- Target: **CER ≤ 3%** after 5-shot adaptation (vs 5-8% without adaptation)

---

## The Metrics

| Metric | Used For | What It Measures | Target |
|--------|----------|-----------------|--------|
| **mAP@0.5** | Detection | % of bounding boxes correctly found (IoU > 0.5) | ≥ 88% |
| **CER** | Text OCR | % of characters wrong in the output | ≤ 5% (≤ 3% adapted) |
| **WER** | Text OCR | % of words with any error | ≤ 15% |
| **BLEU** | Math OCR | N-gram overlap between predicted and true LaTeX | ≥ 85 |
| **ExpRate** | Math OCR | % of expressions exactly correct | ≥ 70% |

**CER example**:
- Reference: `"die Universität"`
- Hypothesis: `"die Universitat"`
- CER = 2 errors (missing umlaut) / 16 chars = 12.5%

---

## Implementation Phases

### Phase 1 — Baseline ✅ Complete
Build a working end-to-end system using existing pre-trained models, no fine-tuning.

- YOLOv8x (pre-trained COCO weights, then trained on DocLayNet)
- TrOCR large handwritten (no fine-tuning)
- Pix2Tex for math (pre-trained)
- All data preparation scripts
- All evaluation scripts

### Phase 2 — SOTA Models 🟡 In Progress
Replace baseline components with better ones, fine-tune on domain data.

- **DLAFormer** for detection (transformer-based document layout model, more accurate than YOLOv8 on documents)
- **Fine-tuned TrOCR** on German IAM data (target: CER 5-8%)
- **TAMER** for math OCR (target: ExpRate ≥ 70%)
- Scripts written, training queue: YOLOv8 → TrOCR fine-tune → evaluation

### Phase 3 — Meta-Learning 🟡 Scripts Ready
Add professor-specific adaptation.

- MAML wrapper around fine-tuned TrOCR
- Meta-train on IAM German writer tasks
- Evaluate 5-shot adaptation on held-out writers
- Target: CER ≤ 3% after adaptation

### Phase 4 — LectureSlideOCR-500-DE Dataset 🔲 Planned
Build our own benchmark dataset for evaluation and publication.

- 500 professor lecture slides with ground-truth annotations
- Uses `scripts/build_lecture_dataset.py`
- Source: `data/Dr_Judith_Jakob_Slides/` (⚠️ requires user confirmation before processing)
- Bootstrap annotations using our trained pipeline, then manually verify

### Phase 5 — Final Evaluation & Paper 🔲 Planned
- Run full pipeline on LectureSlideOCR-500-DE
- Ablation studies (with/without MAML, with/without post-processing)
- Compare against baselines
- Write paper for ICDAR/CVPR 2026

---

## Current Status (2026-03-13)

**What's happening right now**: YOLOv8 detector training in background.
- Run: `runs/detect/runs/detect/baseline_v1_r2/`
- Check progress: `tail -f runs/detect/runs/detect/baseline_v1_r2/results.csv`
- At epoch ~24/100, mAP50 = 82.2% (target: 88%+)

**What's done**:
- All data prepared (IAM German, DocLayNet)
- All code written for Phase 1, 2, 3, and 4
- Repository pushed to GitHub

**What's next** (in order):
1. Wait for YOLOv8 training to finish
2. Evaluate detector (`evaluate/eval_detection.py`)
3. Evaluate TrOCR baseline CER (before fine-tuning)
4. Fine-tune TrOCR on German data (`training/finetune_german_ocr.py`)
5. MAML meta-training (`training/train_meta_learning.py`)
6. Build LectureSlideOCR-500-DE dataset (ask user first)
7. Final evaluation and paper writing

---

## Key Design Decisions

**Why stream DocLayNet instead of extracting it?**
The zip is 28GB and the disk has limited free space. We implemented a streaming reader that pulls only the images we need directly from the zip, writing ~5GB instead of 59GB+.

**Why use IAM German writers instead of a purpose-built German dataset?**
Purpose-built German handwriting OCR datasets don't exist at the scale we need. IAM has 181 German-native speakers — enough for training and meta-learning tasks.

**Why MAML instead of just fine-tuning per professor?**
Fine-tuning needs hundreds of labeled examples and hours of compute per professor. MAML learns a weight initialization that adapts well in 5-50 examples with seconds of compute. This is the core research contribution.

**Why TAMER over Pix2Tex for math?**
TAMER explicitly models the tree structure of math expressions (fractions contain numerator/denominator, superscripts attach to bases, etc.). This structural awareness gives better ExpRate than sequence-to-sequence models like Pix2Tex. Pre-trained checkpoints are already available.

**Why YOLOv8x (largest variant)?**
Accuracy matters more than speed for an offline slide processing tool. The RTX 4060 Ti handles yolov8x inference at real-time speeds anyway.

---

## GitHub Repository

**URL**: https://github.com/ragav1n/lecture-ocr-meta-learning

**What's in the repo**: All code, configs, JSON manifests (data splits), CLAUDE.md development log.

**What's NOT in the repo** (gitignored — too large or generated):
- `data/processed/detection/images/` — 14,723 document images (~5GB)
- `data/processed/german_text/images/` — 4,286 handwriting line images
- `runs/` — YOLOv8 training runs and checkpoints
- `checkpoints/` — Model weights
- `venv/` — Python virtual environment

To restore on a new machine, re-run the data preparation scripts from the original zip files.
