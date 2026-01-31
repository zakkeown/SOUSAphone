# SOUSA Training Pipeline Design

**Date:** 2026-01-29
**Project:** SOUSAphone - SOTA Rudiment Classification & iPhone Deployment
**Dataset:** SOUSA (100K samples, 40 PAS rudiments)

## Overview

Complete research-to-deployment pipeline for state-of-the-art drum rudiment classification. Establishes benchmark results on the first large-scale rudiment dataset, then deploys optimized model to iPhone for real-time feedback.

### Goals

1. **Research Publication:** Benchmark 4 SOTA audio models on 40-class rudiment classification
2. **iPhone Deployment:** Knowledge distillation → quantization → Core ML for <30ms inference

### Timeline

- Implementation: ~10 days
- Training: 24-48 hours on HF Jobs
- Total cost: ~$120-150 for cloud compute

---

## Architecture

Three-phase pipeline:

### Phase 1: Research & Benchmarking

Train and evaluate four SOTA architectures on rudiment classification:
- **HTS-AT** - Hierarchical Swin Transformer (31M params, best efficiency)
- **BEATs** - Microsoft acoustic tokenizers (90M params, best accuracy)
- **EfficientAT** - MobileNetV3 distilled model (10M params, deployment target)
- **AST** - Audio Spectrogram Transformer (87M params, 2021 baseline)

**Experiment Matrix:** 4 models × 3 training strategies = 12 experiments
- PEFT (LoRA): Fast iteration, 0.29% trainable params
- Full fine-tuning: Best performance, all params trainable
- From scratch: Ablation to show pretraining value

### Phase 2: Deployment Pipeline

Take best teacher model (HTS-AT or BEATs) and distill to EfficientAT:
- Knowledge distillation with temperature scaling
- Student retains 95-98% of teacher accuracy
- 3-9× parameter reduction
- 5-10× inference speedup

### Phase 3: On-Device Integration

Optimize for Apple Neural Engine:
- INT8 or FP16 quantization
- Core ML conversion
- SoundAnalysis framework integration
- Real-time inference with sliding windows

---

## Repository Structure

```
SOUSAphone/
├── models/              # Model architectures
│   ├── hts_at.py       # Hierarchical Swin Transformer
│   ├── beats.py        # Microsoft BEATs
│   ├── efficient_at.py # EfficientAT (MobileNetV3)
│   ├── ast.py          # Audio Spectrogram Transformer
│   └── heads.py        # Classification heads
├── training/            # Training infrastructure
│   ├── trainer.py      # Main training loop
│   ├── config.py       # Experiment configs
│   ├── data.py         # SOUSA dataloader
│   └── metrics.py      # Evaluation metrics
├── distillation/        # Knowledge distillation
│   ├── distill.py      # Teacher-student training
│   └── temperature.py  # Temperature scaling
├── deployment/          # Core ML conversion
│   ├── quantize.py     # INT8/FP16 quantization
│   ├── coreml.py       # Core ML export
│   └── ane_optimize.py # Apple Neural Engine opts
├── experiments/         # Training scripts for HF Jobs
│   ├── train_hts_at.py
│   ├── train_beats.py
│   ├── train_efficient_at.py
│   └── train_ast.py
├── scripts/            # Utilities
│   ├── evaluate.py     # Evaluation on test set
│   ├── analyze.py      # Confusion matrices, stats
│   └── submit_all_experiments.sh
└── docs/
    └── plans/          # Design documents
```

---

## Data Pipeline

### Loading from Hugging Face Hub

Dataset: `zkeown/sousa` (100K samples, 97GB with audio)

```python
from datasets import load_dataset

class SOUSADataset:
    def __init__(self, split="train", sample_rate=16000,
                 max_duration=5.0, streaming=False):
        self.dataset = load_dataset(
            "zkeown/sousa",
            split=split,
            streaming=streaming
        )
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Load audio (FLAC from Hub)
        waveform = sample['audio']['array']

        # Standardize length (pad/crop to max_duration)
        waveform = self._process_audio(waveform)

        # Get label (rudiment_slug -> class_id mapping)
        rudiment_id = self.rudiment_to_id[sample['rudiment_slug']]

        return {
            'waveform': waveform,
            'rudiment_id': rudiment_id,
            'skill_tier': sample['skill_tier'],
            'overall_score': sample['overall_score'],
            'metadata': sample['sample_id']
        }
```

### Key Features

- **Streaming mode** for memory efficiency
- **On-the-fly augmentation** (SpecAugment, mixup)
- **Stratified splits** respect profile-based train/val/test
- **Multi-resolution support** for different model architectures
- **Caching** for processed spectrograms

### Rudiment Mapping

Canonical ordering of 40 PAS rudiments (alphabetical by slug):
- Saved as `rudiments.json` for reproducibility
- Consistent class IDs across all experiments

---

## Model Implementations

### 1. HTS-AT (Primary Candidate)

**Architecture:** Hierarchical Token-Semantic Audio Transformer
**Parameters:** 31M
**Performance:** 471 mAP on AudioSet, 97.0% ESC-50

**Implementation:**
- Use official repo: [RetroCirce/HTS-Audio-Transformer](https://github.com/RetroCirce/HTS-Audio-Transformer)
- Load AudioSet-pretrained checkpoint
- Replace 527-class head with 40-class rudiment head
- Configure for shorter windows (1.3-2.6s vs 10.24s default)
- Support LoRA adapters and full fine-tuning

**Why HTS-AT:**
- Hierarchical Swin Transformer with window attention (linear complexity)
- 65% fewer parameters than AST
- 128 samples/batch vs AST's 12 on same hardware
- Token-semantic module for event localization

### 2. BEATs (Max Accuracy Baseline)

**Architecture:** Iterative self-supervised with acoustic tokenizers
**Parameters:** 90M
**Performance:** 506 mAP on AudioSet (best), 98.1% ESC-50

**Implementation:**
- Use official repo: [microsoft/unilm/beats](https://github.com/microsoft/unilm/tree/master/beats)
- Load BEATs_iter3+ checkpoint
- Fine-tune with LoRA to manage parameter count
- Semantic tokenization may excel at subtle technique differences

**Why BEATs:**
- Highest accuracy among all models
- Semantic acoustic tokenizers for fine-grained distinctions
- May better distinguish flam timing variations, paradiddle patterns

### 3. EfficientAT (Deployment Target)

**Architecture:** MobileNetV3 distilled from PaSST transformers
**Parameters:** 1-10M (using mn10 variant)
**Performance:** 480+ mAP on AudioSet, ~95% ESC-50

**Implementation:**
- Use official repo: [fschmid56/EfficientAT](https://github.com/fschmid56/EfficientAT)
- Load mn10 pretrained on AudioSet
- Dual purpose: standalone baseline AND distillation target
- Depthwise separable convolutions for mobile efficiency

**Why EfficientAT:**
- Already optimized for mobile (MobileNetV3 architecture)
- Knowledge distillation preserves transformer accuracy in CNN
- Runs efficiently on Apple Neural Engine
- Target for deployment pipeline

### 4. AST (2021 Baseline)

**Architecture:** Vision Transformer on mel-spectrograms
**Parameters:** 87M
**Performance:** 485 mAP on AudioSet, 95.6% ESC-50

**Implementation:**
- Use HuggingFace: `ASTForAudioClassification`
- Load `MIT/ast-finetuned-audioset-10-10-0.4593`
- Simplest implementation (native transformers support)
- Serves as "older SOTA" comparison

**Why AST:**
- Established baseline from 2021
- Shows progress of field (newer models should beat it)
- Easy integration via transformers library

---

## Training Configuration

### Shared Hyperparameters

All models use identical training config for fair comparison:

```python
config = {
    # Optimization
    "learning_rate": 5e-5,  # PEFT: 1e-4, Full FT: 1e-5
    "batch_size": 16,  # Local M4: 4-8, HF Jobs: 16-32
    "num_epochs": 50,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "optimizer": "adamw",
    "scheduler": "cosine",

    # Regularization
    "dropout": 0.1,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.2,
    "specaugment": True,

    # Training
    "gradient_accumulation": 2,  # Effective batch = 32
    "mixed_precision": "fp16",
    "max_grad_norm": 1.0,
    "early_stopping_patience": 10,

    # Evaluation
    "eval_every_n_epochs": 1,
    "save_best_only": True,
    "metric": "balanced_accuracy",
}
```

### Augmentation Pipeline

- **SpecAugment:** 2 freq masks (width=20), 2 time masks (width=30)
- **Mixup:** α=0.2 for label smoothing
- **Time stretching:** ±10% tempo variation
- **No pitch shifting** (rudiments are rhythm-based)

### Training Strategies

**1. PEFT (LoRA) - Fast Iteration**
- Freeze backbone, add LoRA adapters
- Only 0.29% trainable parameters
- Training time: 2-4 hours on single GPU
- Use for: hyperparameter search, ablations

**2. Full Fine-tuning - Best Performance**
- Unfreeze all parameters after PEFT warmup
- Gradual unfreezing: heads → top layers → full
- Training time: 8-12 hours
- Use for: final benchmarks, paper results

**3. From Scratch - Ablation**
- Random initialization, no pretraining
- Shows value of AudioSet transfer learning
- Expected lower performance but scientifically valuable

---

## Evaluation Metrics

### Primary Metrics

**Overall Performance:**
- **Balanced Accuracy** - Primary metric (handles class imbalance)
- **Top-1 Accuracy** - Standard classification accuracy
- **Top-5 Accuracy** - Near-miss analysis
- **Macro F1** - Averaged across all rudiments

### Per-Rudiment Analysis

- **Precision, Recall, F1** for each of 40 rudiments
- Identify hardest rudiments
- Per-family confusion (rolls vs diddles vs flams vs drags)

### Confusion Matrix

- 40×40 heatmap showing prediction patterns
- Hierarchical grouping by rudiment family
- Systematic error identification

### Skill Tier Generalization

- Accuracy breakdown by skill tier (beginner/intermediate/advanced/expert)
- Cross-tier evaluation
- Does model work equally well across skill levels?

### Statistical Analysis

- **McNemar's test** for pairwise model comparison
- **Confidence intervals** via bootstrapping
- **Ablation studies** - pretraining, augmentation, model size impact

### Experiment Outputs

```
experiments/hts_at_lora_20250129/
├── checkpoints/
│   └── best.pt
├── logs/
│   ├── tensorboard/
│   └── training.log
├── results/
│   ├── test_metrics.json
│   ├── confusion_matrix.png
│   ├── per_rudiment_metrics.csv
│   └── skill_tier_breakdown.json
└── config.json
```

---

## Knowledge Distillation

### Teacher → Student Pipeline

After Phase 1, distill best model to EfficientAT for deployment.

**Teacher Selection:**
- Use best-performing model (likely HTS-AT or BEATs)
- Can ensemble multiple teachers for stronger supervision

**Student Model:**
- EfficientAT mn10 (10M params)
- Can compress further to mn04 (4M) or mn01 (1M) if needed

**Distillation Loss:**

```python
def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=4.0, alpha=0.7):
    # Soft target loss (knowledge from teacher)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)

    # Hard target loss (ground truth)
    hard_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

**Training Process:**
1. Freeze teacher (inference only)
2. Train student to match teacher outputs + true labels
3. Temperature=4.0 smooths distributions
4. Alpha=0.7 weights teacher knowledge higher
5. Train 30-40 epochs

**Expected Results:**
- Student retains 95-98% of teacher accuracy
- 3-9× fewer parameters
- 5-10× faster inference

---

## Deployment Pipeline

### Quantization

**INT8 Post-Training Quantization:**
- Convert FP32 → INT8
- 4× smaller, 2-4× faster
- Typical accuracy drop: 0.1-2%

**FP16 (Half Precision):**
- Alternative if INT8 degrades too much
- 2× smaller, 1.5-2× faster
- Negligible accuracy loss

**Calibration:**
- Use 1000-2000 samples from validation set
- Coverage across all 40 rudiments and skill tiers

### Core ML Conversion

```python
import coremltools as ct

traced_model = torch.jit.trace(model, example_input)

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="audio", shape=(1, 80000))],
    outputs=[ct.TensorType(name="logits", shape=(1, 40))],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS17,
)

mlmodel_quantized = ct.models.neural_network.quantization_utils.quantize_weights(
    mlmodel, nbits=8
)

mlmodel_quantized.save("RudimentClassifier.mlpackage")
```

### Apple Neural Engine Optimization

**Principles:**
- 4D channels-first tensor format (B, C, 1, T)
- Replace `nn.Linear` with `nn.Conv2d` for ANE compatibility
- Minimize reshapes and transposes
- Batch size 1 for real-time inference

### Real-Time Inference

**iOS SoundAnalysis Framework:**
```python
let audioAnalyzer = SNAudioStreamAnalyzer(format: audioFormat)
let request = try SNClassifySoundRequest(mlModel: rudimentModel)

request.windowDuration = CMTime(seconds: 1.5, preferredTimescale: 16000)
request.overlapFactor = 0.5

try audioAnalyzer.add(request, withObserver: resultObserver)
```

**Performance Target (iPhone A15+):**
- Inference latency: <30ms per window
- Total latency: 35-95ms (includes audio buffering)
- Real-time responsive feedback

---

## Experiment Tracking

### Weights & Biases

**Project:** `sousa-rudiment-classification`

**Features:**
- Real-time metrics, system stats, gradients
- Hyperparameter sweeps
- Artifact storage for checkpoints
- Comparison dashboard across all runs

### Experiment Naming

```
{model}_{strategy}_{date}

Examples:
- hts_at_lora_20250129
- beats_full_finetune_20250130
- efficient_at_scratch_20250131
```

### Configuration Management

Every experiment saves complete config:

```python
experiment_config = {
    "model": "hts_at",
    "strategy": "lora",
    "dataset": {
        "name": "zkeown/sousa",
        "version": "1.0.0",
        "splits": {"train": 80000, "val": 10000, "test": 10000},
    },
    "training": {...},
    "augmentation": {...},
    "environment": {
        "platform": "huggingface_jobs",
        "gpu": "A100-40GB",
        "pytorch_version": "2.2.0",
        "seed": 42,
    },
    "git_commit": "abc123...",
}
```

### Reproducibility Checklist

- ✅ Fixed random seed (42)
- ✅ Deterministic CUDA operations
- ✅ Versioned dataset on HF Hub
- ✅ Git commit hash in config
- ✅ Complete dependency specs
- ✅ Model architecture as code + config

---

## Hugging Face Jobs Integration

### Training Script Structure

UV format with PEP 723 inline dependencies:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.2.0",
#     "torchaudio>=2.2.0",
#     "transformers>=4.36.0",
#     "datasets>=2.16.0",
#     "wandb>=0.16.0",
#     "peft>=0.8.0",
#     "accelerate>=0.26.0",
# ]
# ///

def main():
    dataset = load_dataset("zkeown/sousa", streaming=True)
    wandb.init(project="sousa-rudiment-classification")

    model = HTSATForAudioClassification.from_pretrained(
        "hts-at-audioset",
        num_labels=40,
        use_lora=True,
    )

    train_model(model, dataset, config)
```

### Job Submission

```bash
hf jobs submit \
  --name "sousa-hts-at-lora" \
  --hardware "gpu-a100-small" \
  --script experiments/train_hts_at.py \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  --env HF_TOKEN=$HF_TOKEN \
  --timeout 86400
```

### Hardware & Cost Estimates

| Model | Strategy | Hardware | Time | Cost |
|-------|----------|----------|------|------|
| HTS-AT | LoRA | gpu-a10g (24GB) | 3-4h | $3-4 |
| HTS-AT | Full FT | gpu-a100-small (40GB) | 8-10h | $16-20 |
| BEATs | LoRA | gpu-a100-small (40GB) | 4-5h | $8-10 |
| BEATs | Full FT | gpu-a100-small (40GB) | 10-12h | $20-24 |
| EfficientAT | Any | gpu-a10g (24GB) | 2-3h | $2-3 |
| AST | Any | gpu-a10g (24GB) | 4-5h | $4-5 |

**Total for 12 experiments:** ~$120-150

### Result Persistence

- Checkpoints → HF Hub: `zkeown/sousa-models`
- Metrics → Weights & Biases cloud
- Best models tagged for retrieval
- Configs saved alongside checkpoints

### Local Development Workflow

1. Test on M4 Max with `--preset small` (1.2K samples, 2-3 min)
2. Validate training loop
3. Submit to HF Jobs for full 100K training

---

## Implementation Timeline

### Day 1: Setup & Data Pipeline (4-5 hours)

- [x] Repository structure and dependencies
- [ ] SOUSA dataloader implementation
- [ ] Rudiment mapping (40 classes)
- [ ] Augmentation pipeline
- [ ] Test on small dataset locally

### Day 2-3: Model Implementations (8-12 hours)

- [ ] Integrate HTS-AT from official repo
- [ ] Integrate BEATs from Microsoft repo
- [ ] Integrate EfficientAT from official repo
- [ ] AST from HuggingFace transformers
- [ ] Test forward passes for all models

### Day 3-4: Training Infrastructure (6-8 hours)

- [ ] Trainer class with LoRA support
- [ ] Config management system
- [ ] Experiment scripts for all 12 runs
- [ ] Metrics and evaluation code
- [ ] Wandb integration

### Day 4: Local Validation (2 hours)

- [ ] Test training on M4 Max with small dataset
- [ ] Verify wandb logging
- [ ] Validate checkpointing
- [ ] Confirm metrics computation

### Day 5: Submit HF Jobs

- [ ] Submit all 12 experiments
- [ ] Monitor training progress
- [ ] Debug any failures

### Day 6-7: Results Analysis (after training)

- [ ] Evaluate all models on test set
- [ ] Generate confusion matrices
- [ ] Per-rudiment performance analysis
- [ ] Statistical comparisons
- [ ] Identify best teacher model

### Day 8-9: Distillation (4-6 hours)

- [ ] Implement distillation training loop
- [ ] Train EfficientAT student from best teacher
- [ ] Evaluate student performance
- [ ] Compare to standalone EfficientAT baseline

### Day 10: Deployment (4-6 hours)

- [ ] Quantization (INT8/FP16)
- [ ] Core ML conversion
- [ ] ANE optimization validation
- [ ] Test on iPhone device
- [ ] Benchmark inference latency

---

## Deliverables

### Research Outputs

1. **12 Trained Models**
   - 4 architectures × 3 strategies
   - Comprehensive benchmark on rudiment classification

2. **Experimental Results**
   - Test metrics for all models
   - Confusion matrices
   - Per-rudiment analysis
   - Skill tier generalization study

3. **Analysis Report**
   - What works best for drum techniques?
   - Impact of pretraining on this task
   - Comparison to existing audio classification benchmarks

4. **Reproducible Codebase**
   - All code on GitHub
   - Dataset on HF Hub (already done)
   - Trained models on HF Hub
   - Complete experiment configs

### Deployment Outputs

1. **Optimized Model**
   - Distilled EfficientAT (10M → potentially 4M or 1M)
   - 95-98% of teacher accuracy retained
   - 5-10× faster inference

2. **Core ML Package**
   - `RudimentClassifier.mlpackage`
   - INT8 or FP16 quantized
   - Optimized for Apple Neural Engine
   - <30ms inference latency

3. **Integration Ready**
   - SoundAnalysis framework compatible
   - Real-time sliding window inference
   - Complete iOS deployment guide

---

## Success Criteria

### Research Success

- ✅ Achieve >85% balanced accuracy on 40-class rudiment classification
- ✅ Establish first benchmark on large-scale rudiment dataset
- ✅ Demonstrate value of modern architectures over 2021 baseline
- ✅ Publish reproducible results with open dataset and code

### Deployment Success

- ✅ Distilled model retains >95% of teacher accuracy
- ✅ Inference latency <30ms on iPhone A15+
- ✅ Total latency (buffering + inference) <100ms
- ✅ Core ML model <50MB for app distribution

---

## Future Extensions

### Research

- Hierarchical classification (family → specific rudiment)
- Multi-task learning (rudiment + skill + score)
- Cross-dataset evaluation (generalization to real recordings)
- Temporal modeling (onset detection + technique classification)

### Deployment

- On-device fine-tuning for user adaptation
- Federated learning across users
- Real-time technique correction feedback
- Practice session analytics

---

## References

### Key Papers

- **HTS-AT:** Ke Chen et al. "HTS-AT: A Hierarchical Token-Semantic Audio Transformer" ICASSP 2022
- **BEATs:** Sanyuan Chen et al. "BEATs: Audio Pre-Training with Acoustic Tokenizers" ICML 2023
- **EfficientAT:** Florian Schmid et al. "Efficient Large-Scale Audio Tagging via Transformer-to-CNN Knowledge Distillation" ICASSP 2023
- **AST:** Yuan Gong et al. "AST: Audio Spectrogram Transformer" Interspeech 2021

### Repositories

- HTS-AT: https://github.com/RetroCirce/HTS-Audio-Transformer
- BEATs: https://github.com/microsoft/unilm/tree/master/beats
- EfficientAT: https://github.com/fschmid56/EfficientAT
- SOUSA Dataset: https://huggingface.co/datasets/zkeown/sousa

### Domain Research

- Wu & Lerch "On Drum Playing Technique Detection in Polyphonic Mixtures" ISMIR 2016
- Apple Neural Engine optimization: ane-transformers
- Core ML quantization: Apple ML documentation
