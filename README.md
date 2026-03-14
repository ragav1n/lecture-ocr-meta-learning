# German Lecture Slide OCR with Meta-Learning

A system that detects handwritten text and math in lecture slides, recognizes the content, and replaces it with clean typeset equivalents. Designed for professor-specific adaptation via meta-learning — a new professor's handwriting style can be learned from as few as 5 examples.

**Publication target**: ICDAR 2026 / CVPR 2026

---

## Pipeline

```
Lecture slide image
        │
        ▼
┌─────────────────────┐
│  Detection          │  YOLOv8x / DLAFormer
│                     │  → bounding boxes (text / math)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Recognition        │  TrOCR (German text)
│                     │  TAMER (math → LaTeX)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Replacement        │  Typeset overlay on slide
└─────────────────────┘
```

---

## Results

| Phase | Model | Metric | Score |
|-------|-------|--------|-------|
| Detection | YOLOv8x | mAP50 | **91.5%** |
| Text OCR (baseline) | TrOCR large | CER | 3.35% |
| Text OCR (fine-tuned) | TrOCR + IAM German | Val CER | **1.29%** |
| Meta-learning | Reptile (5-shot) | Test CER | **0.53%** |

Training data for meta-learning: IAM handwriting database (German writers subset, 4,286 samples across 95 writers).

---

## Repository Structure

```
├── baseline/           # Phase 1: off-the-shelf pipeline (YOLOv8 + TrOCR + Pix2Tex)
├── models/             # Phase 2: model wrappers (DLAFormer, TAMER, MAML)
├── training/           # Training scripts (detector, TrOCR fine-tuning, meta-learning)
├── evaluate/           # Evaluation scripts (mAP, CER/WER, BLEU, end-to-end)
├── scripts/            # Data preparation (IAM, DocLayNet, CROHME)
├── utils/              # Metrics, image utilities, German post-processing
├── configs/            # YAML configs (detection, meta-training, pipeline)
├── TAMER/              # Math OCR submodule (pre-trained on CROHME/HME100K)
└── outputs/            # Evaluation results (JSON)
```

---

## Setup

```bash
git clone git@github.com:ragav1n/lecture-ocr-meta-learning.git ocr_slide
cd ocr_slide
git submodule update --init --recursive   # restores TAMER/

python3 -m venv venv
source venv/bin/activate
pip install -r requirements_project.txt
pip install ultralytics
```

---

## Training Data

| Dataset | Samples | Use |
|---------|---------|-----|
| IAM (German writers) | 4,286 lines | TrOCR fine-tuning + meta-learning |
| DocLayNet | 14,723 images, 178K boxes | Detection training |
| CROHME 2019 | — | Math OCR (TAMER pre-trained) |

---

## Key Design Decisions

- **Reptile over MAML**: First-order meta-learning avoids the need to differentiate through the inner loop. Gradients on `deepcopy` don't flow back to the meta model — Reptile sidesteps this entirely with a direct parameter interpolation update.
- **ManualMAML fallback**: `learn2learn` is not installable on Python 3.12 (Cython extension uses removed headers). The Reptile update is implemented from scratch in `models/meta_learning_ocr.py`.
- **Streaming DocLayNet extraction**: The full dataset is 28GB. Only selected images are written to disk using streaming zip reads.

---

## Hardware

- GPU: NVIDIA RTX 4060 Ti 16GB
- Python 3.12, PyTorch 2.10.0, transformers 5.3
