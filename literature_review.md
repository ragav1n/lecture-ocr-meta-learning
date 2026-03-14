# Literature Review: Handwritten Text and Math Recognition in Lecture Slides with Meta-Learning

## 1. Introduction

The digitization of handwritten content in academic lecture slides presents a multi-faceted challenge at the intersection of document layout analysis, handwritten text recognition (HTR), mathematical expression recognition (HMER), and few-shot learning. This review surveys recent work (2025--2026) across these domains, identifies research gaps, and positions our approach -- a meta-learning-based pipeline for professor-specific handwriting adaptation in German lecture slides.

---

## 2. Document Layout Analysis and Handwriting Detection

Detecting handwritten regions in mixed-content documents (printed slides with handwritten annotations) requires robust layout analysis. Recent work has advanced both transformer-based and YOLO-based detectors.

### 2.1 Transformer-Based Layout Analysis

**HybriDLA** (Moured et al., AAAI 2026) unifies diffusion and autoregressive decoding within a single transformer layer. The diffusion component iteratively refines bounding-box hypotheses while the autoregressive component injects semantic and contextual awareness, simulating a coarse-to-fine reading strategy. It dynamically adjusts the number of object queries to handle documents with varying numbers of elements, achieving **83.5% mAP** on DocLayNet -- the best result for a vision-only layout analysis model, nearly matching DLAFormer's 83.8% mAP which requires multi-modal (text + visual) input [1].

**DLAFormer** (2024, widely cited in 2025--2026 work) remains the benchmark for multi-modal document layout analysis. It uses a DETR-inspired end-to-end transformer with a unified relation prediction module that infers all layout relationships concurrently. Its requirement for OCR/text features as input adds a preprocessing step, making it more computationally expensive than vision-only alternatives [2].

**DocSemi** (Shehzadi et al., ICCV 2025 Workshop) introduces a semi-supervised framework with guided queries and the TILT mechanism for simultaneous learning of textual, visual, and layout features. With only **10% labeled data**, it achieves **96.2% mAP** on PubLayNet, demonstrating the potential of semi-supervised methods when labeled handwriting data is scarce [3].

**SCAN** (arXiv 2025) proposes a VLM-friendly approach to semantic document layout analysis that identifies components at appropriate granularity for RAG pipelines, improving end-to-end textual RAG by up to **+9.4 points** and visual RAG by **+10.4 points** [4].

### 2.2 YOLO-Based Handwriting Detection

**YOLO-Handwritten** (Zhang et al., ICCGV 2025) integrates deformable convolution (DCNv3) and enhanced aggregate feature fusion (iAFF) into YOLOv8 to handle inconsistent handwriting strokes. It adds Biformer attention to the Neck and uses VFLoss for positive/negative sample imbalance, improving precision by **+4.3%**, recall by **+3%**, and **mAP@0.5 by +4.6%** over baseline YOLOv8 on examination papers [5].

**BGE-YOLO** (ICIC 2025) proposes GAM (Global Attention Mechanism), BiFPN for bidirectional cross-scale fusion, and EMA modules for Chinese handwritten text detection, achieving **+2.8% mAP50** and **+3.9% precision** over baseline YOLOv8 [6].

### 2.3 Relevance to Our Work

Our pipeline uses YOLOv8x trained on DocLayNet, achieving **91.5% mAP50** for handwriting detection. The YOLO-Handwritten and BGE-YOLO results suggest that architectural enhancements (deformable convolutions, attention mechanisms, BiFPN) could push detection performance further. However, our current detector already exceeds our 88% target, and the primary bottleneck lies in recognition rather than detection. The vision-only HybriDLA approach is particularly relevant as a potential upgrade path that avoids the multi-modal preprocessing overhead of DLAFormer.

---

## 3. Handwritten Text Recognition (HTR)

Transformer-based encoder-decoder models, particularly TrOCR, have become the dominant paradigm for HTR.

### 3.1 TrOCR and Vision-Encoder-Decoder Models

**TrOCR** (Li et al., originally 2021, extensively built upon in 2025) uses a ViT encoder and GPT-2 decoder for end-to-end handwritten text recognition. It remains the most widely used backbone for HTR adaptation in recent work.

**DLoRA-TrOCR** (Huttner et al., WACV 2025 Workshop) systematically evaluates parameter-efficient fine-tuning (PEFT) methods -- LoRA and DoRA -- versus full fine-tuning on TrOCR. PEFT methods achieve competitive performance with significantly fewer parameters: **CER 4.02 on IAM with only 0.7% trainable parameters**. This demonstrates that full fine-tuning may be unnecessary for domain adaptation [7].

**HTR-VT** (Li et al., Pattern Recognition 2025) proposes a data-efficient ViT-based model using only the encoder component with a CNN feature extractor, Sharpness-Aware Minimization (SAM) optimizer, and span mask regularization. It surpasses SOTA **without pre-training or additional data** on multiple benchmarks, establishing new results on the LAM dataset [8].

**AdapterTrOCR** (OpenReview 2025) integrates two adapter modules into TrOCR -- one for language adaptation and one for handwriting style adaptation -- targeting historical Latin manuscripts. It reduces **CER by 13.33--35.65%** and **WER by 8.56--27.7%** versus base TrOCR, demonstrating effective modular adaptation from English to Latin HTR [9].

**TrOCR-URRNN-MLD** (ETASR 2025) extends TrOCR with Unified Residual Recurrent Neural Networks for multilingual documents, surpassing existing models on IAM and Kannada datasets [10].

**DDViT** (Springer 2025) proposes a Vision Transformer with dual downsampling layers generating two predictions, achieving **CER 3.34% on IAM** -- a 1.5 percentage point improvement over prior works on the IAM-A partition [11].

### 3.2 German-Specific HTR

Community models on Hugging Face include `fhswf/TrOCR_german_handwritten` for modern German handwriting and `dh-unibe/trocr-kurrent-XVI-XVII` for historical German Kurrent script (trained on ~1.58M words from 16th--18th century manuscripts). These provide direct baselines for German handwriting OCR but lack rigorous benchmark documentation [12].

### 3.3 Data Augmentation for HTR

**Quo Vadis HTG** (Pippi et al., ICCV 2025 Workshop) systematically compares GAN-based, diffusion-based, and autoregressive handwritten text generation models for their impact on HTR fine-tuning, providing practical guidelines on which generative paradigm works best for augmentation [13].

A comprehensive survey on data augmentation for offline HTR (arXiv, July 2025) documents the progression from traditional augmentation to GANs, diffusion models, and transformer-based generation, identifying best practices by dataset size [14].

### 3.4 HTR Generalization

**On the Generalization of HTR Models** (Garrido-Munoz et al., CVPR 2025) provides a comprehensive study of 336 OOD cases across 8 SOTA HTR models, 7 datasets, and 5 languages. Key finding: **textual divergence is the most significant factor for generalization failure**, followed by visual divergence. This directly motivates our meta-learning approach -- adapting to professor-specific vocabulary and writing style [15].

### 3.5 Surveys

A comprehensive HTR survey (arXiv, February 2025) traces the field from heuristic methods to modern deep learning, identifying remaining challenges: writer adaptation, low-resource languages, and document-level recognition [16].

**Benchmarking LLMs for HTR** (arXiv, March 2025) evaluates proprietary and open-source LLMs against Transkribus on modern and historical datasets in English, French, German, and Italian. Claude 3.5 Sonnet outperforms open-source alternatives in zero-shot settings, but inference costs remain orders of magnitude higher than specialized models [17].

### 3.6 Relevance to Our Work

Our pipeline uses TrOCR-large-handwritten as the base model, fine-tuned on IAM German writers (val CER=1.29%). The LoRA/DoRA results from WACV 2025 suggest parameter-efficient tuning is viable, but our meta-learning approach goes further by learning an initialization that adapts in 5 shots. The CVPR 2025 generalization study validates our core hypothesis: writer-specific adaptation is critical because textual and visual divergence are the primary failure modes for HTR models.

---

## 4. Few-Shot and Meta-Learning for HTR

Meta-learning for writer-specific adaptation is the core innovation of our system. Recent work has validated this direction at top venues.

### 4.1 Meta-Learning Approaches

**MetaWriter** (Gu et al., CVPR 2025) is the most directly relevant work. It formulates writer personalization as prompt tuning within a pre-trained HTR model, using MAML-style meta-learning to learn optimal prompt initialization. A self-supervised MAE auxiliary task guides prompt adaptation using unlabeled test-time samples, updating **less than 1% of parameters**. Results: **CER 2.19% on RIMES, ~6% WER reduction on IAM**, with 8x speedup and 20x fewer adapted parameters compared to full fine-tuning [18].

**DocTTT** (Gu et al., WACV 2025) combines meta-learning with self-supervised MAE for test-time training. During inference, visual encoder parameters are adapted per-input using an MAE loss, with meta-learning during training ensuring effective initialization. It significantly outperforms SOTA by adapting at test time to each document's specific characteristics [19].

### 4.2 Other Adaptation Methods

**Optimal Transport for HTR** (arXiv, September 2025) casts HTR as cross-modal matching using Optimal Transport to align visual and semantic distributions with minimal supervision. An iterative bootstrapping framework generates pseudo-labels from high-confidence alignments, requiring only a word list and no annotated images. However, the closed-vocabulary assumption limits its applicability to open-vocabulary lecture content [20].

**LoRA vs. Fine-Tuning** (Huttner et al., WACV 2025 Workshop) shows that LoRA achieves competitive accuracy while updating far fewer parameters. However, each new writer/domain requires a separate LoRA training run, unlike meta-learning which provides a shared initialization for rapid adaptation [7].

### 4.3 Relevance to Our Work

Our Reptile-based meta-learning approach (val CER=0.85%, 5-shot test CER=0.53%) shares the same motivation as MetaWriter -- learning an initialization that enables rapid writer adaptation. Key differences:

- **MetaWriter** uses prompt tuning (<1% parameters), while we adapt the full model through Reptile's parameter interpolation update. This may allow deeper adaptation at the cost of more parameters.
- **MetaWriter** uses self-supervised MAE for unlabeled adaptation, while our approach requires labeled support examples. MetaWriter's approach is more practical when labels are unavailable.
- **DocTTT** adapts at test time per-input, while our approach adapts once per-professor and serves all subsequent inputs. Our approach has lower inference cost after the initial adaptation step.
- Neither MetaWriter nor DocTTT addresses German-specific text or lecture slide domains. Our work targets a specific application domain (German lecture slides) with professor-specific adaptation.

Our approach is complementary: Reptile provides the meta-learned initialization, and techniques like MetaWriter's prompt tuning or DocTTT's test-time MAE could be integrated as future enhancements.

---

## 5. Handwritten Mathematical Expression Recognition (HMER)

Recognizing handwritten math and converting it to LaTeX is critical for lecture slides containing equations.

### 5.1 Specialized HMER Models

**TAMER** (Qingzhenduyu et al., AAAI 2025) introduces a Tree-aware Module into the Transformer decoder that jointly optimizes LaTeX sequence prediction and tree structure modeling. The Fusion Module variant further boosts performance. Results: **ExpRate 61.23%/60.26%/61.97% on CROHME 2014/2016/2019**, surpassing CoMER by ~3% on each benchmark, with a 5.31% improvement on complex expressions (structural complexity >= 5) [21].

**TST** (Xie & Mouchère, ICDAR 2025) predicts the Symbol Layout Tree (SLT) using an encoder-decoder Transformer that predicts tree structure components (node labels, edge labels, parent indices). It supports both online (graph-based encoder) and offline (CNN encoder) HMER, producing interpretable tree-structured outputs [22].

**The Return of Structural HMER** (Seitz et al., ICDAR 2025) proposes a modular structural recognition system with independent segmentation, classification, and spatial relation prediction. It introduces an automatic annotation pipeline that generates symbol-level annotations, producing two new datasets: **CROHME+ and MathWriting+** with trace-level ground truth covering 374,000 expressions [23].

### 5.2 VLM-Based HMER

**Uni-MuMER** (Li et al., NeurIPS 2025 Spotlight) fine-tunes Qwen2.5-VL-3B for HMER with three auxiliary tasks: Tree-Aware Chain-of-Thought, Error-Driven Learning, and Symbol Counting. It achieves **new SOTA on CROHME and HME100K**, surpassing the best specialized model (SSAN) by 16.31% and the best VLM (Gemini-2.5-flash) by 24.42% in zero-shot settings [24].

**Boosting HMER with vLLMs** (ICDAR 2025 Workshop) benchmarks various SOTA vision-language models on CROHME, finding that dedicated specialized models (TAMER, SSAN) still outperform general-purpose vLLMs, though contextual prompting improves vLLM performance [25].

### 5.3 Datasets

**MathWriting** (Gervais et al., Google, 2025) introduces the largest online HMER dataset: **230K human-written plus 400K synthetic samples** with ink-level trace data and LaTeX ground truth, enabling more robust model training [26].

### 5.4 Relevance to Our Work

Our pipeline uses TAMER (version_3, Fusion Module, ExpRate 69.5% on HME100K) for math OCR. TAMER's AAAI 2025 publication validates our choice. Uni-MuMER's NeurIPS 2025 Spotlight results suggest that VLM-based approaches are beginning to surpass specialized models, representing a potential future upgrade path. However, Uni-MuMER's 3B-parameter model is significantly heavier than TAMER for deployment. The MathWriting dataset and CROHME+ could be used to further improve our math recognition component.

---

## 6. End-to-End Document OCR Pipelines

Recent work has increasingly moved toward unified models that jointly handle detection and recognition.

### 6.1 Unified VLM-Based Pipelines

**OCRVerse** (Zhong et al., arXiv January 2026) is the first holistic OCR method unifying text-centric and vision-centric OCR end-to-end. Using two-stage SFT-RL training with domain-customized rewards, the 4B-parameter model matches or surpasses GPT-5-level performance on specific tasks [27].

**dots.ocr** (RedNote HiLab, arXiv December 2025) is a 1.7B-parameter VLM that jointly learns layout detection and content recognition across **100+ languages**. It achieves SOTA on OmniDocBench (87.5 EN, 84.0 CH) and surpasses previous best on XDocParse (126 languages) by **+7.4 points** [28].

**VISTA-OCR** (Hamdi et al., arXiv April 2025) proposes a lightweight 150M-parameter encoder-decoder that unifies text detection and recognition. The VISTA_omni variant handles both handwritten and printed documents via prompting [29].

**LightOnOCR** (arXiv January 2026) is a 1B-parameter VLM achieving SOTA on OlmOCR-Bench while being **9x smaller and substantially faster** than prior best models. However, **handwritten text transcription remains inconsistent** -- a critical limitation for our use case [30].

**DODO** (Man et al., arXiv February 2026) introduces discrete diffusion for OCR, achieving near-SOTA accuracy with up to **3x faster inference** through block-causal masking [31].

**GutenOCR** (Heidenreich et al., arXiv January 2026) fine-tunes Qwen2.5-VL for grounded OCR with bounding box localization, doubling the composite grounded OCR score from 0.40 to 0.82 [32].

### 6.2 Pipeline-Based Approaches

**PreP-OCR** (Guan et al., ACL 2025) combines document image restoration with ByT5 post-OCR correction, reducing CER by **63.9--70.3%** on degraded historical documents [33].

**DeepSeek-OCR** (Wei et al., arXiv October 2025) compresses long contexts via optical 2D mapping, achieving 97% OCR precision at <10x compression and processing **200K+ pages/day** on a single A100 [34].

### 6.3 Relevance to Our Work

Our modular pipeline (YOLOv8 detection + TrOCR recognition + TAMER math) contrasts with the trend toward unified VLM-based OCR. While unified models like dots.ocr and OCRVerse achieve impressive results on printed text, LightOnOCR's authors explicitly note that **handwritten text transcription remains inconsistent** in these models. Our modular design allows each component to be optimized independently for the handwriting domain. Furthermore, none of the unified approaches incorporate writer adaptation or meta-learning -- our key contribution. The modular architecture also enables targeted upgrades (e.g., replacing TAMER with Uni-MuMER) without retraining the entire system.

---

## 7. Lecture Slide Understanding

### 7.1 Slide-Specific Work

**SynSlideGen** (arXiv June 2025) proposes an LLM-guided synthetic lecture slide generation pipeline producing the SynSlide dataset and a manually annotated RealSlide benchmark of 1,050 real slides. Pre-training on synthetic slides before fine-tuning with just **50 real images achieves a 9.7% mAP boost on YOLOv9**, with significant improvements in low-resource classes: code snippets (+32.5% mAP), natural images (+20.2% mAP) [35].

**PPTAgent** (arXiv January 2025) conceptualizes presentation generation as a two-stage editing task and introduces PPTEval, an evaluation framework for presentation quality [36].

**SlideTailor** (arXiv December 2025) addresses personalized presentation slide generation from scientific papers [37].

### 7.2 Relevance to Our Work

SynSlideGen is directly relevant -- its synthetic data augmentation approach for slide element detection with minimal real data mirrors our meta-learning philosophy of achieving good performance with few examples. However, existing lecture slide work focuses on slide generation or element detection, **not on recognizing and replacing handwritten annotations**. This represents a clear gap that our work addresses. Our proposed LectureSlideOCR-500-DE dataset would be the first benchmark specifically targeting handwritten annotation recognition in lecture slides.

---

## 8. Datasets and Benchmarks

### 8.1 OCR Benchmarks

**OCRBench v2** (NeurIPS 2025) provides 10,000 QA pairs across 31 scenarios and 23 sub-tasks. Notably, **36 out of 38 SOTA models scored below 50/100**, revealing significant gaps in text localization, fine-grained perception, and complex element parsing [38].

**CC-OCR** (Yang et al., ICCV 2025) introduces four OCR-centric tracks with 39 subsets and 7,058 annotated images, though it does not include a dedicated handwriting track [39].

### 8.2 Datasets Used in Our Work

| Dataset | Size | Use | Notes |
|---------|------|-----|-------|
| IAM Handwriting (German subset) | 4,286 lines | TrOCR fine-tuning + meta-learning | 95 German/Swiss-German writers |
| DocLayNet | 14,723 images, 178K boxes | Detection training | Streaming extraction from 28GB zip |
| CROHME 2019 | -- | Math OCR (via TAMER pretrained) | TAMER pretrained checkpoints used |
| HME100K | -- | Math OCR (via TAMER pretrained) | TAMER Fusion Module variant |

---

## 9. Identified Research Gaps

Based on this review, we identify the following gaps that our work addresses:

### Gap 1: No Meta-Learning for German Handwriting OCR
While MetaWriter (CVPR 2025) and DocTTT (WACV 2025) demonstrate meta-learning for writer adaptation, neither addresses German-specific text or lecture slide domains. German handwriting presents unique challenges: umlauts (ä, ö, ü), sharp-s (ß), and compound words that require domain-specific post-processing.

### Gap 2: No Integrated Lecture Slide Handwriting Pipeline
Existing work treats detection, text recognition, and math recognition as separate problems. No system combines all three with writer-specific adaptation for the lecture slide domain. SynSlideGen addresses slide element detection but not handwritten content recognition.

### Gap 3: No Lecture Slide Handwriting Benchmark
Despite OCRBench v2 and CC-OCR providing comprehensive evaluation, no benchmark exists for handwritten annotations in lecture slides. Our proposed LectureSlideOCR-500-DE dataset would fill this gap.

### Gap 4: Unified VLMs Struggle with Handwriting
As noted by LightOnOCR's authors, handwritten text transcription remains inconsistent in unified VLM-based OCR systems. Our modular pipeline with specialized components (TrOCR for text, TAMER for math) addresses this limitation directly.

### Gap 5: Few-Shot Adaptation vs. Parameter-Efficient Methods
LoRA/DoRA approaches (WACV 2025) demonstrate parameter efficiency but require per-writer training runs. Meta-learning provides a shared initialization enabling rapid adaptation from as few as 5 examples, which is more practical for real-world professor adaptation scenarios.

---

## 10. Our Approach and Justification

### 10.1 System Overview

Our pipeline consists of three stages:
1. **Detection**: YOLOv8x trained on DocLayNet (mAP50=91.5%)
2. **Recognition**: TrOCR-large fine-tuned on IAM German + Reptile meta-learning (val CER=0.85%, 5-shot test CER=0.53%)
3. **Math OCR**: TAMER with Fusion Module (ExpRate 69.5% on HME100K)

### 10.2 Why Reptile Over MAML

We chose Reptile (Nichol & Schulman, 2018) over MAML (Finn et al., 2017) for meta-learning because:
- **Memory efficiency**: MAML requires differentiating through the inner loop (second-order gradients) or storing computation graphs for all inner steps. On our RTX 4060 Ti (16GB), MAML caused OOM with even one model clone + inner loop gradients. Reptile only needs one model clone at a time with no backward pass through the inner loop.
- **Implementation correctness**: We discovered that naively applying MAML with `deepcopy` in PyTorch results in gradients that don't flow back to the meta-model, making the outer-loop optimizer a no-op. Reptile avoids this entirely with direct parameter interpolation: `p_meta += outer_lr * (p_adapted - p_meta)`.
- **Comparable performance**: First-order meta-learning methods (Reptile, first-order MAML) have been shown to achieve comparable performance to full MAML in many settings, and our results (0.85% val CER) exceed the target of 2--3%.

### 10.3 Why Modular Over Unified

Despite the trend toward unified VLM-based OCR (OCRVerse, dots.ocr), we chose a modular pipeline because:
- Unified models struggle with handwriting (noted by LightOnOCR authors)
- Modularity enables targeted meta-learning on the recognition component without affecting detection
- Each component can be evaluated and upgraded independently
- Smaller individual models fit our hardware constraints (16GB VRAM)

### 10.4 Why TrOCR as Base Model

TrOCR is the most widely adopted backbone for HTR adaptation (used in AdapterTrOCR, DLoRA-TrOCR, MetaWriter). Its encoder-decoder architecture with pre-trained ViT encoder provides strong visual features, and the GPT-2 decoder enables autoregressive generation of text sequences. Our fine-tuned model achieves 1.29% val CER on IAM German, and the Reptile meta-learned initialization further improves this to 0.85%.

---

## 11. Summary of Key Referenced Works

| Paper | Venue | Domain | Key Result |
|-------|-------|--------|------------|
| HybriDLA [1] | AAAI 2026 | Layout Analysis | 83.5% mAP (vision-only SOTA) |
| DLAFormer [2] | 2024 (baseline) | Layout Analysis | 83.8% mAP (multi-modal) |
| DocSemi [3] | ICCV 2025W | Layout Analysis | 96.2% mAP (10% labels) |
| YOLO-Handwritten [5] | ICCGV 2025 | Detection | +4.6% mAP50 over YOLOv8 |
| DLoRA-TrOCR [7] | WACV 2025W | HTR | CER 4.02, 0.7% params |
| HTR-VT [8] | Pattern Recog. 2025 | HTR | SOTA without pretraining |
| AdapterTrOCR [9] | 2025 | HTR | -35.65% CER reduction |
| DDViT [11] | Springer 2025 | HTR | CER 3.34% on IAM |
| HTR Generalization [15] | CVPR 2025 | HTR | Textual divergence = top factor |
| MetaWriter [18] | CVPR 2025 | Meta-HTR | CER 2.19% RIMES, <1% params |
| DocTTT [19] | WACV 2025 | Meta-HTR | SOTA via test-time training |
| TAMER [21] | AAAI 2025 | Math OCR | ExpRate 61.97% CROHME 2019 |
| Uni-MuMER [24] | NeurIPS 2025 | Math OCR | New CROHME/HME100K SOTA |
| dots.ocr [28] | arXiv 2025 | End-to-end OCR | SOTA 126 languages |
| SynSlideGen [35] | arXiv 2025 | Lecture slides | +9.7% mAP with 50 real images |
| OCRBench v2 [38] | NeurIPS 2025 | Benchmark | 36/38 models below 50/100 |

---

## References

[1] O. Moured et al., "HybriDLA: Hybrid Generation for Document Layout Analysis," AAAI 2026. arXiv:2511.19919

[2] "DLAFormer: An End-to-End Transformer for Document Layout Analysis," arXiv:2405.11757

[3] T. Shehzadi et al., "DocSemi: Efficient Document Layout Analysis with Guided Queries," ICCV 2025 Workshop (VisionDocs)

[4] "SCAN: Semantic Document Layout Analysis for Textual and Visual RAG," arXiv:2505.14381

[5] C. Zhang et al., "YOLO-Handwritten: Improved YOLOv8 for Handwritten Text Detection," ICCGV 2025, SPIE Proceedings

[6] "BGE-YOLO: An Improved YOLOv8 for Chinese Handwritten Text Detection," ICIC 2025

[7] Huttner et al., "Low-Rank Adaptation vs. Fine-Tuning for Handwritten Text Recognition," WACV 2025 Workshop (VisionDocs)

[8] Y. Li et al., "HTR-VT: Handwritten Text Recognition with Vision Transformer," Pattern Recognition, 2025. arXiv:2409.08573

[9] "Handwritten Text Recognition Adaptation for Low-Resource Languages: A Case Study on Historical Latin Manuscripts," OpenReview, 2025

[10] "TrOCR-URRNN-MLD: Transformer with Unified Residual Recurrent Networks for Multilingual HTR," ETASR, 2025

[11] "DDViT: Dual Downsample Vision Transformer for HTR," Springer, 2025

[12] fhswf/TrOCR_german_handwritten, dh-unibe/trocr-kurrent-XVI-XVII, Hugging Face, 2025

[13] V. Pippi et al., "Quo Vadis Handwritten Text Generation for Handwritten Text Recognition?," ICCV 2025 Workshop (VisionDocs). arXiv:2508.09936

[14] "Advancing Offline HTR: A Systematic Review of Data Augmentation and Generation Techniques," arXiv:2507.06275

[15] C. Garrido-Munoz et al., "On the Generalization of Handwritten Text Recognition Models," CVPR 2025

[16] "Handwritten Text Recognition: A Survey," arXiv:2502.08417

[17] "Benchmarking Large Language Models for Handwritten Text Recognition," arXiv:2503.15195

[18] C. Gu et al., "MetaWriter: Personalized Handwritten Text Recognition Using Meta-Learned Prompt Tuning," CVPR 2025. arXiv:2505.20513

[19] C. Gu et al., "DocTTT: Test-Time Training for Handwritten Document Recognition Using Meta-Auxiliary Learning," WACV 2025. arXiv:2501.12898

[20] "Optimal Transport for Handwritten Text Recognition in a Low-Resource Regime," arXiv:2509.16977

[21] Qingzhenduyu et al., "TAMER: Tree-Aware Transformer for Handwritten Mathematical Expression Recognition," AAAI 2025

[22] Xie, Mouchère, "TST: Tree Structured Transformer for Handwritten Mathematical Expression Recognition," ICDAR 2025

[23] Seitz, Lengfeld, Timofte, "The Return of Structural Handwritten Mathematical Expression Recognition," ICDAR 2025

[24] Li et al., "Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition," NeurIPS 2025 (Spotlight). arXiv:2505.23566

[25] "Boosting Handwritten Mathematical Expression Recognition Through Contextual Reasoning with Vision Large Language Models," ICDAR 2025 Workshop

[26] Gervais, Fadeeva et al., "MathWriting: A Dataset for Handwritten Mathematical Expression Recognition," 2025. arXiv:2404.10690

[27] Y. Zhong et al., "OCRVerse: Towards Holistic OCR in End-to-End Vision-Language Models," arXiv:2601.21639

[28] RedNote HiLab, "dots.ocr: Multilingual Document Layout Parsing in a Single Vision-Language Model," arXiv:2512.02498

[29] L. Hamdi et al., "VISTA-OCR: Towards Generative and Interactive End-to-End OCR Models," arXiv:2504.03621

[30] LightOn AI, "LightOnOCR: A 1B End-to-End Multilingual Vision-Language Model for SOTA OCR," arXiv:2601.14251

[31] S. Man et al., "DODO: Discrete OCR Diffusion Models," arXiv:2602.16872

[32] H. Heidenreich et al., "GutenOCR: A Grounded Vision-Language Front-End for Documents," arXiv:2601.14490

[33] S. Guan et al., "PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy," ACL 2025. arXiv:2505.20429

[34] H. Wei et al., "DeepSeek-OCR: Contexts Optical Compression," arXiv:2510.18234

[35] "SynSlideGen: AI-Generated Lecture Slides for Improving Slide Element Detection and Retrieval," arXiv:2506.23605

[36] "PPTAgent: Generating and Evaluating Presentations Beyond Text-to-Slides," arXiv:2501.03936

[37] "SlideTailor: Personalized Presentation Slide Generation for Scientific Papers," arXiv:2512.20292

[38] "OCRBench v2: Evaluating Large Multimodal Models on Visual Text," NeurIPS 2025. arXiv:2501.00321

[39] Yang et al., "CC-OCR: A Comprehensive and Challenging OCR Benchmark," ICCV 2025. arXiv:2412.02210

[40] "A Comparative Analysis of OCR Models on Diverse Datasets," WACV 2025 Workshop (VisionDocs)

[41] "Digitization of Document and Information Extraction using OCR," arXiv:2506.11156

[42] "Evaluating LLMs for Historical Document OCR," arXiv:2510.06743
