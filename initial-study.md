# Optimal ML architectures for real-time drum rudiment classification on iPhone

**HTS-AT or knowledge-distilled EfficientAT models represent the best path for SOUSA's requirements**—offering near-SOTA accuracy with dramatically better on-device performance than MIT's AST. The field has evolved significantly since AST's 2021 release, with newer architectures achieving higher accuracy at 35-65% of the computational cost. Critically, drum rudiment classification is an **unexplored research niche** with only one published paper directly addressing the task, positioning SOUSA's 100k+ sample dataset as potentially field-defining.

---

## The current SOTA landscape favors efficient transformers over vanilla AST

The Audio Spectrogram Transformer pioneered purely attention-based audio classification in 2021, but **three architectures now surpass it** on standard benchmarks while offering better deployment characteristics:

| Model | Parameters | AudioSet mAP | ESC-50 | On-device viability |
|-------|------------|--------------|--------|---------------------|
| **BEATs** (Microsoft, 2023) | 90M | **0.506** | **98.1%** | Challenging |
| **HTS-AT** (ICASSP 2022) | **31M** | 0.471 | 97.0% | **Good** |
| **PaSST** (Interspeech 2022) | 87M | 0.496 | 96.8% | Moderate |
| AST (MIT, 2021) | 87M | 0.485 | 95.6% | Challenging |
| **EfficientAT** (ICASSP 2023) | **1-10M** | 0.48+ | ~95% | **Excellent** |

**HTS-AT deserves primary consideration** for SOUSA. Its hierarchical Swin Transformer architecture uses window attention (linear complexity) instead of AST's global attention (quadratic), enabling **128 samples/batch versus AST's 12** on equivalent hardware. The **65% parameter reduction** translates directly to faster inference and smaller model footprints. Its token-semantic module also enables event localization—potentially useful for identifying rudiment boundaries within longer audio segments.

**BEATs achieves the highest accuracy** through iterative self-supervised pre-training with semantic acoustic tokenizers. If maximum classification accuracy is paramount and deployment constraints can be relaxed (server-side inference or high-end devices), BEATs' approach of distilling semantics into discrete audio tokens could help distinguish subtle rudiment differences like flam timing variations.

---

## On-device deployment strongly favors MobileNet-based architectures

Real-time iPhone inference requires models optimized for Apple's Neural Engine (ANE), which delivers **15.8 TFlops (FP16)** on A15+ chips but has specific architectural requirements. The most deployment-ready approach uses **EfficientAT**—MobileNetV3 variants trained via knowledge distillation from PaSST transformer ensembles.

**EfficientAT achieves remarkable efficiency**:
- **mn10** variant: ~1M parameters, ~2.7ms inference, 0.48+ mAP
- Depthwise separable convolutions execute efficiently on ANE
- Knowledge distillation preserves transformer-level accuracy in CNN architecture
- Pre-trained weights available with AudioSet embeddings

**Apple Neural Engine optimization principles** (from Apple's ane-transformers research):
- Use 4D channels-first format (B, C, 1, S) for all tensors
- Replace `nn.Linear` with `nn.Conv2d` for ANE compatibility  
- Minimize reshapes and transposes (memory copy operations)
- Chunk multi-head attention into single-head operations for L2 cache efficiency

**Quantization delivers substantial gains**: INT8 quantization typically causes 0.1-2% accuracy degradation while halving memory and accelerating inference. Core ML's iOS 18+ features include 4-bit block-wise quantization specifically optimized for on-device transformers. MLX (Apple's ML framework) supports 4-bit and 8-bit quantization with 40% model size reduction while retaining 95% audio fidelity.

For SOUSA specifically, the recommended pipeline is: **Train HTS-AT or BEATs → Knowledge distillation to EfficientAT student → INT8/FP16 quantization → Core ML conversion**. This yields transformer-level accuracy in a model that runs comfortably under 50ms per inference window on iPhone.

---

## Rudiment classification is an unexplored research frontier

**Only one paper directly addresses drum technique classification**: Wu & Lerch's ISMIR 2016 study "On Drum Playing Technique Detection in Polyphonic Mixtures" achieved **64.6% accuracy on isolated samples** and just **40.4% with background music** for four techniques (Strike, Buzz Roll, Flam, Drag). Their key finding: traditional spectral features overfit to specific drum timbres, while NMF activation-derived features provided timbre-invariant representations.

This research gap has profound implications for SOUSA:

**SOUSA would be the first large-scale rudiment dataset**. Existing drum datasets focus on onset detection with 3-21 instrument classes, not technique classification with 40 rudiments. The closest analog—E-GMD's 444 hours—provides velocity annotations but no rudiment labels. SOUSA's **100k+ samples across 40 PAS rudiments** would be **5-10x larger** than typical evaluation datasets and uniquely comprehensive in vocabulary.

**Critical features for distinguishing rudiments** identified in existing research:
- **Peak amplitude ratios** (α) between grace notes and primary strokes—Flams use ~30-60ms offsets
- **Inter-onset intervals** (IOI) for roll density and paradiddle sticking patterns
- **DTW template distances** comparing activation shapes to canonical rudiment profiles
- **Velocity/accent patterns** distinguishing paradiddle variants (RLRR vs RLRL)
- **Distribution features** (skewness, crest factor) of activation functions

**Hierarchical classification may improve accuracy**. Given the 40-class vocabulary, a two-stage approach—first classifying rudiment family (roll, paradiddle, flam, drag, diddle), then specific rudiment—could reduce confusion between acoustically similar techniques.

---

## Recommended architecture combines temporal modeling with efficient inference

For SOUSA's requirements (maximum accuracy, real-time, on-device), the optimal approach synthesizes insights from ADT research with modern efficient architectures:

**Primary recommendation: Fine-tuned HTS-AT with deployment via knowledge distillation**

```
Training Pipeline:
1. Start with AudioSet-pretrained HTS-AT (31M params)
2. Fine-tune on SOUSA with LoRA/Convpass adapters (0.29% trainable params)
3. Apply SpecAugment + Mixup augmentation during training
4. Distill to EfficientAT student (1-10M params) for deployment
5. Quantize to INT8 and convert to Core ML
```

**Parameter-efficient fine-tuning** (PETL) research from IEEE MLSP 2024 shows **Convpass adapters with Conformer convolution modules achieve full fine-tuning accuracy while updating only 0.29% of parameters**. This dramatically reduces training time and prevents catastrophic forgetting of AudioSet pre-training.

**Architecture considerations for rudiments specifically**:
- **Temporal modeling is critical**: Paradiddles, rolls, and flam patterns require sequence understanding. HTS-AT's hierarchical structure captures multi-scale temporal patterns better than flat attention.
- **Short input windows**: Rudiments typically span 0.5-2 seconds versus AST's default 10.24 seconds. Configure for 128-256 time frames (~1.3-2.6 seconds) to reduce computation while capturing full rudiment duration.
- **Multi-task potential**: Joint onset detection + technique classification could improve accuracy by explicitly modeling rudiment timing structure.

**Alternative if pure accuracy is paramount**: BEATs fine-tuned on SOUSA, with aggressive distillation to a smaller student. BEATs' semantic tokenization might excel at distinguishing subtle technique differences, though deployment complexity increases significantly.

---

## Real-time processing requires careful latency management

Recent ADT research demonstrates **42-60ms detection delay is achievable** for real-time drum transcription using streaming architectures. Key strategies:

**Audio buffering**: Use 1-2 second sliding windows with 50% overlap. This balances latency against capturing complete rudiment patterns. For rudiments averaging 1 second duration, a 1.5-second window with 750ms hop provides adequate context while maintaining sub-second response.

**Processing pipeline for iPhone**:
```
Microphone → Ring Buffer (1.5 sec) → Mel Spectrogram (128 bins)
→ Model Inference (ANE, <30ms target) → Post-processing → UI Update
```

**SoundAnalysis framework** integration: Apple's native audio classification framework handles microphone capture, buffering, and hardware acceleration automatically. Custom Core ML models integrate directly, with the framework managing window sizing and confidence aggregation.

**Latency budget** for real-time feel:
- Audio capture + buffering: ~20-50ms
- Spectrogram computation: ~5-10ms  
- Model inference (ANE): ~5-30ms (model-dependent)
- Post-processing: ~5ms
- **Total: 35-95ms achievable**

---

## SOUSA dataset positions strongly for novel contributions

Given the research landscape, SOUSA could establish **the definitive benchmark for drum rudiment classification**:

**Dataset advantages over existing resources**:
- **Scale**: 100k+ samples vs. typical 8-22k onsets in ADT datasets
- **Vocabulary depth**: 40 rudiments vs. 3-21 instrument classes
- **Task specificity**: Technique classification vs. onset detection
- **Skill modeling**: Player skill annotations enable difficulty-aware training

**Recommended evaluation protocol** following ADT conventions:
- F-measure with ±50ms onset tolerance for timing accuracy
- Per-rudiment precision/recall to identify challenging classes
- Cross-performer evaluation to test generalization
- Confusion matrices highlighting rudiment family boundaries

**Potential baseline results** based on analogous tasks: Indian tabla stroke classification (98%+ accuracy on 13 classes) and Wu & Lerch's work suggest **85-95% accuracy may be achievable on clean, isolated rudiment samples**. Polyphonic scenarios and real-world recording conditions would likely reduce accuracy by 20-40% based on ADT literature.

---

## Conclusion: HTS-AT offers the best accuracy-deployment tradeoff

For SOUSA's goal of maximum-accuracy real-time rudiment classification on iPhone, **HTS-AT fine-tuned with parameter-efficient adapters, then distilled to EfficientAT for deployment** provides the optimal path. This approach leverages:

- **Proven accuracy**: HTS-AT matches or exceeds AST while using 65% fewer parameters
- **Efficient training**: LoRA/Convpass adapters enable fine-tuning with minimal compute
- **Practical deployment**: Knowledge distillation to MobileNet architecture runs efficiently on Neural Engine
- **Research novelty**: No existing work addresses comprehensive rudiment classification—SOUSA would be pioneering
