# SOUSA Training Pipeline Implementation Design

**Date:** 2026-01-31
**Project:** SOUSAphone - Training Infrastructure (Research Phase)
**Status:** Approved for implementation

## Overview

Implementation design for the SOUSA training pipeline, optimized for local M4 Max (36GB) development with cloud portability to HuggingFace Jobs. This covers the research phase only: training 4 SOTA audio models with multiple PEFT strategies to establish benchmark results on 40-class drum rudiment classification.

**Scope:** Data pipeline, training infrastructure, model integration, evaluation metrics, experiment tracking.

**Out of scope:** Knowledge distillation, quantization, Core ML deployment (Phase 2, after identifying best teacher model).

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────┐
│   Experiment Runner (Hydra CLI)         │
│   - Compose configs: model + strategy    │
│   - Launch training jobs                 │
│   - Manage experiment directories        │
└─────────────┬───────────────────────────┘
              │
┌─────────────▼───────────────────────────┐
│   Lightning Module (SOUSAClassifier)    │
│   - Wraps any model architecture         │
│   - Handles PEFT injection              │
│   - Training/val/test steps             │
│   - Metrics computation                  │
└─────────────┬───────────────────────────┘
              │
       ┌──────┴──────┐
       │             │
┌──────▼─────┐  ┌───▼──────────┐
│   Models   │  │  Data Module │
│  Adapters  │  │   (SOUSA)    │
└────────────┘  └──────────────┘
```

### Key Architectural Decisions

1. **PyTorch Lightning** as training framework
   - Clean separation between model, data, and training orchestration
   - Automatic device management (MPS for M4, CUDA for cloud)
   - Built-in checkpointing, early stopping, distributed training

2. **Hydra configuration system**
   - Composable experiments: `python train.py model=hts_at strategy=lora`
   - All 12+ experiments are config variations, not code branches
   - Automatic experiment directory management

3. **Model adapter pattern**
   - Each model (AST, HTS-AT, BEATs, EfficientAT) wrapped in common interface
   - Preserves original implementations while enabling unified training
   - PEFT injection happens at training level, not in model code

4. **Local-first with cloud portability**
   - Same code runs on M4 Max (MPS) and HF Jobs (CUDA)
   - Device detection automatic
   - Only config overrides needed for cloud (batch size, paths)

---

## Repository Structure

```
SOUSAphone/
├── configs/                    # Hydra configuration
│   ├── config.yaml            # Base config
│   ├── model/                 # Model configs
│   │   ├── ast.yaml
│   │   ├── hts_at.yaml
│   │   ├── beats.yaml
│   │   └── efficient_at.yaml
│   ├── strategy/              # Training strategies
│   │   ├── lora.yaml
│   │   ├── adalora.yaml
│   │   ├── ia3.yaml
│   │   ├── full_finetune.yaml
│   │   └── from_scratch.yaml
│   ├── data/                  # Data configs
│   │   ├── full.yaml          # 100K samples
│   │   ├── medium.yaml        # 30K samples
│   │   ├── small.yaml         # 10K samples
│   │   └── tiny.yaml          # 1K for testing
│   ├── platform/              # Platform overrides
│   │   └── hf_jobs.yaml       # Cloud-specific settings
│   └── experiment/            # Preset experiments
│       └── quick_test.yaml    # Fast validation run
│
├── sousa/                     # Main package
│   ├── data/
│   │   ├── dataset.py         # SOUSADataset class
│   │   ├── transforms.py      # Audio augmentation
│   │   └── datamodule.py      # Lightning DataModule
│   ├── models/
│   │   ├── base.py            # AudioClassificationModel interface
│   │   ├── ast.py             # HF AST wrapper
│   │   ├── hts_at.py          # HTS-AT integration
│   │   ├── beats.py           # BEATs integration
│   │   └── efficient_at.py    # EfficientAT integration
│   ├── training/
│   │   ├── module.py          # SOUSAClassifier (LightningModule)
│   │   ├── callbacks.py       # Custom callbacks
│   │   └── metrics.py         # Evaluation metrics
│   └── utils/
│       ├── audio.py           # Audio processing utilities
│       └── rudiments.py       # 40-class mapping
│
├── scripts/
│   └── submit_all_experiments.sh  # Batch HF Jobs submission
│
├── train.py                   # Main training script
├── evaluate.py                # Standalone evaluation
├── pyproject.toml            # Dependencies (uv/pip)
│
└── experiments/              # Training outputs (gitignored)
    └── {date}/{time}/{model}_{strategy}/
        ├── checkpoints/
        │   ├── best.ckpt
        │   └── last.ckpt
        ├── logs/
        └── config.yaml       # Resolved config snapshot
```

**Organizational principles:**
- Hydra configs are source of truth (no hardcoded hyperparameters)
- Models are plugins conforming to common interface
- Lightning DataModule handles all data concerns
- Experiments directory auto-generated and gitignored

---

## Data Pipeline

### Dataset Loading

```python
class SOUSADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,  # ~/Code/SOUSA/output/dataset
        split: str = "train",
        sample_rate: int = 16000,
        max_duration: float = 5.0,
        transform: Optional[Callable] = None,
    ):
        # Load from local parquet/CSV metadata
        self.metadata = self._load_split(dataset_path, split)

        # Build rudiment_slug -> class_id mapping (0-39)
        self.rudiment_mapping = self._load_rudiment_mapping()

        self.transform = transform
```

### Audio Processing

**Loading strategy:**
- Load FLAC files on-demand (avoid 97GB in RAM)
- Standardize to 16kHz sample rate
- Pad or crop to fixed length (~5 seconds configurable)
- Convert to mel-spectrogram or raw waveform (model-dependent)

**Augmentation pipeline (training only):**
- **SpecAugment:** Frequency and time masking on spectrograms
- **Mixup:** Blend two samples for label smoothing (α=0.2)
- **Time stretching:** ±10% tempo variation
- **No pitch shifting** (rudiments are rhythm/technique-based)

### Lightning DataModule

```python
class SOUSADataModule(pl.LightningDataModule):
    def setup(self, stage):
        # Create train/val/test datasets
        # Apply augmentation only to train
        # Validate dataset structure and completeness

    def train_dataloader(self):
        # Batch size from config
        # num_workers optimized for M4 Max

    def val_dataloader(self):
        # No augmentation for validation
```

### Memory Optimization

With 36GB RAM:
- **Cache spectrograms** for small/medium datasets (< 15K samples)
- **On-demand loading** for full 100K dataset
- **Speeds up epochs 2+ by 3-5x** when cached

---

## Model Integration

### Common Interface

```python
class AudioClassificationModel(nn.Module):
    """Common interface for all audio models"""

    @abstractmethod
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Raw waveform or spectrogram (model-dependent)
        Returns:
            logits: (batch_size, num_classes)
        """
        pass

    @abstractmethod
    def get_feature_extractor(self):
        """Returns preprocessing config (sample rate, spec params, etc.)"""
        pass

    @property
    @abstractmethod
    def expected_input_type(self) -> str:
        """Either 'waveform' or 'spectrogram'"""
        pass
```

### Model Adapters

**1. AST (HuggingFace native):**
```python
class ASTModel(AudioClassificationModel):
    def __init__(self, num_classes=40, pretrained=True):
        self.model = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    expected_input_type = "spectrogram"  # 128 mel bins
```

**2. HTS-AT (official repo integration):**
```python
from sousa.models.hts_at_official import HTSAT_Swin_Transformer

class HTSATModel(AudioClassificationModel):
    def __init__(self, num_classes=40, pretrained=True):
        self.model = HTSAT_Swin_Transformer(
            spec_size=256,  # Shorter than default 1024
            patch_size=4,
            num_classes=num_classes,
        )
        if pretrained:
            self._load_audioset_weights()

    expected_input_type = "spectrogram"
```

**3. BEATs and EfficientAT:** Similar adapter pattern

**Approach:**
- Use HuggingFace for AST (already integrated)
- Directly integrate official implementations with minimal modification
- Create thin adapter layer for training pipeline compatibility
- Don't over-abstract until patterns emerge

---

## Training Module

### Lightning Module Structure

```python
class SOUSAClassifier(pl.LightningModule):
    def __init__(self, model: AudioClassificationModel, config: DictConfig):
        super().__init__()
        self.model = model
        self.config = config

        # Apply PEFT if strategy requires it
        if config.strategy.type in ["lora", "adalora", "ia3"]:
            self.model = self._apply_peft(model, config.strategy)

        # Initialize metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=40)
        self.val_acc = Accuracy(task="multiclass", num_classes=40)
        self.val_balanced_acc = MulticlassBalancedAccuracy(num_classes=40)
        # ... more metrics
```

### PEFT Injection

```python
def _apply_peft(self, model, strategy_config):
    """Inject PEFT adapters using HF peft library"""
    from peft import get_peft_model, LoraConfig, AdaLoraConfig, IA3Config

    if strategy_config.type == "lora":
        peft_config = LoraConfig(
            r=strategy_config.rank,
            lora_alpha=strategy_config.alpha,
            target_modules=strategy_config.target_modules,
            lora_dropout=strategy_config.dropout,
        )
    elif strategy_config.type == "adalora":
        peft_config = AdaLoraConfig(...)
    elif strategy_config.type == "ia3":
        peft_config = IA3Config(...)

    return get_peft_model(model, peft_config)
```

### Strategy Handling

- **from_scratch:** Don't load pretrained weights
- **lora/adalora/ia3:** Apply PEFT via `peft` library
- **full_finetune:** Load pretrained, unfreeze all parameters

### Training Step

```python
def training_step(self, batch, batch_idx):
    audio, labels = batch['audio'], batch['label']
    logits = self(audio)

    # Loss with label smoothing
    loss = F.cross_entropy(
        logits, labels,
        label_smoothing=self.config.training.label_smoothing
    )

    # Log metrics
    self.log('train/loss', loss)
    self.train_acc(logits, labels)
    self.log('train/acc', self.train_acc, on_step=False, on_epoch=True)

    return loss
```

---

## Configuration System

### Base Config Structure

```yaml
# configs/config.yaml
defaults:
  - model: ast
  - strategy: lora
  - data: small
  - _self_

training:
  max_epochs: 50
  learning_rate: 5e-5
  batch_size: 16
  gradient_accumulation_steps: 2
  warmup_ratio: 0.1
  weight_decay: 0.01
  label_smoothing: 0.1
  early_stopping_patience: 10

augmentation:
  specaugment: true
  mixup_alpha: 0.2
  time_stretch_range: [0.9, 1.1]

seed: 42
precision: "16-mixed"
accelerator: "auto"
num_workers: 4

wandb:
  project: "sousa-rudiment-classification"
  tags: ["${model.name}", "${strategy.type}"]
```

### Model Config Example

```yaml
# configs/model/hts_at.yaml
name: hts_at
class_path: sousa.models.hts_at.HTSATModel
num_classes: 40
pretrained: true

spec_size: 256
patch_size: 4
window_size: 8

input_type: spectrogram
sample_rate: 16000
n_mels: 128
hop_length: 160
```

### Strategy Config Example

```yaml
# configs/strategy/lora.yaml
type: lora
load_pretrained: true

rank: 8
alpha: 16
dropout: 0.1
target_modules:
  - "attention.query"
  - "attention.key"
  - "attention.value"

learning_rate: 1e-4  # Higher for PEFT
```

### Running Experiments

```bash
# Quick test
python train.py data=tiny

# HTS-AT with LoRA
python train.py model=hts_at strategy=lora

# BEATs full fine-tuning on full dataset
python train.py model=beats strategy=full_finetune data=full

# Override any parameter
python train.py model=hts_at strategy=adalora training.learning_rate=2e-4
```

### Automatic Experiment Organization

Hydra creates directories:
```
experiments/2026-01-31/14-23-45/hts_at_lora/
├── checkpoints/
├── logs/
└── config.yaml  # Resolved config with all overrides
```

---

## Evaluation & Metrics

### Metrics Collection

```python
class SOUSAMetrics:
    def __init__(self, num_classes=40):
        # Primary metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.balanced_accuracy = MulticlassBalancedAccuracy(num_classes=num_classes)
        self.top5_accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)

        # Per-class analysis
        self.precision = Precision(task="multiclass", num_classes=num_classes, average=None)
        self.recall = Recall(task="multiclass", num_classes=num_classes, average=None)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # Confusion matrix
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
```

### Logged to W&B

**During training:**
- Loss (train/val)
- Accuracy metrics (top-1, top-5, balanced)
- Learning rate (with scheduler)
- GPU/MPS memory usage
- Gradients (histograms, norms)
- Model parameter counts (total, trainable)

**End of epoch:**
- Confusion matrix heatmap (40×40, grouped by rudiment family)
- Per-rudiment precision/recall/F1 table
- Sample predictions with audio
- Training speed (samples/sec, epoch time)

### Saved Artifacts

- Best checkpoint (by validation balanced accuracy)
- Last checkpoint (for resuming)
- Complete Hydra config (reproducibility)
- Rudiment mapping JSON (class_id → rudiment_slug)

### Experiment Comparison

W&B parallel coordinates plot comparing:
- Model architecture
- PEFT strategy
- Final metrics (balanced accuracy, F1)
- Training time and memory usage

---

## M4 Max Optimization

### MPS Acceleration

PyTorch 2.0+ native Apple Silicon support:

```python
trainer = pl.Trainer(
    accelerator="auto",  # Detects MPS on M4
    precision="16-mixed",  # 2x faster with mixed precision
    devices=1,
)
```

### Expected Performance

| Model | Batch Size | Memory | Samples/sec | Epoch (10K) |
|-------|-----------|--------|-------------|-------------|
| **EfficientAT** | 32 | ~4GB | ~800 | ~2 min |
| **AST** | 12 | ~18GB | ~180 | ~9 min |
| **HTS-AT** | 16-24 | ~12-16GB | ~240 | ~7 min |
| **BEATs** | 8-12 | ~20-24GB | ~120 | ~14 min |

### Memory Optimization

1. **Gradient accumulation** instead of large batches
2. **Gradient checkpointing** for large models (20-30% slower, 40% less memory)
3. **Dataset caching** for < 15K samples (3-5x faster epochs)

### Local Capabilities

What runs comfortably on M4 Max:
- ✅ **EfficientAT:** All strategies (LoRA, Full FT, From Scratch)
- ✅ **HTS-AT:** LoRA and lightweight PEFT
- ⚠️ **HTS-AT:** Full fine-tune (batch_size=8, tight on memory)
- ⚠️ **AST/BEATs:** LoRA only (full fine-tune likely OOM)

### Recommended Local Workflow

1. **Development** (minutes): `data=tiny` (1K samples) - Fast iteration
2. **Validation** (1-2 hours): `data=small` (10K samples) - Verify training
3. **Pre-cloud test** (4-6 hours): `data=medium` (30K) with EfficientAT
4. **Production** (HF Jobs): `data=full` (100K) - All models, all strategies

---

## Cloud Portability (HF Jobs)

### Training Script (PEP 723)

```python
# train.py
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.2.0",
#     "torchaudio>=2.2.0",
#     "pytorch-lightning>=2.1.0",
#     "transformers>=4.36.0",
#     "wandb>=0.16.0",
#     "peft>=0.8.0",
#     "hydra-core>=1.3.0",
# ]
# ///

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    trainer = pl.Trainer(accelerator="auto", ...)
    trainer.fit(model, datamodule)
```

### Submission

```bash
# Single experiment
hf jobs submit \
  --name "sousa-hts-at-lora" \
  --hardware "gpu-a10g" \
  --script train.py \
  --args "model=hts_at strategy=lora data=full" \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  --timeout 28800

# Batch submit all 12 experiments
bash scripts/submit_all_experiments.sh
```

### Cloud Config Overrides

```yaml
# configs/platform/hf_jobs.yaml
training:
  batch_size: 32  # Bigger batches on A100
  num_workers: 8

data:
  dataset_path: "/data/sousa"
  cache_dir: "/tmp/cache"
```

### Cost Estimates

| Model | Strategy | Hardware | Time | Cost |
|-------|----------|----------|------|------|
| HTS-AT | LoRA | gpu-a10g | 3-4h | $3-4 |
| HTS-AT | Full FT | gpu-a100-small | 8-10h | $16-20 |
| BEATs | LoRA | gpu-a100-small | 4-5h | $8-10 |
| BEATs | Full FT | gpu-a100-small | 10-12h | $20-24 |
| EfficientAT | Any | gpu-a10g | 2-3h | $2-3 |
| AST | Any | gpu-a10g | 4-5h | $4-5 |

**Total for 12 experiments:** ~$120-150

### Result Persistence

- Checkpoints → HF Hub: `zkeown/sousa-models/{experiment_name}`
- Metrics → W&B cloud
- Download best checkpoint locally after completion

---

## Error Handling & Robustness

### Automatic Checkpointing

- Save best checkpoint (by validation balanced accuracy)
- Save last checkpoint every N epochs
- Auto-resume from last checkpoint on crash

### Data Validation

```python
def setup(self, stage):
    # Validate dataset exists
    assert self.dataset_path.exists()

    # Verify all 40 rudiments present
    assert len(set(self.train_dataset.rudiments)) == 40

    # Check for corrupted audio files
    self._validate_audio_files()
```

### Training Safety

- **Gradient anomaly detection:** Catch NaN/Inf gradients
- **Gradient clipping:** Prevent exploding gradients (max_norm=1.0)
- **Memory management:** Auto-detect OOM, clear cache between epochs
- **Sanity check:** Run 1 batch before full epoch

### Experiment Safety

- Hydra prevents config overwrites (unique directories)
- Git commit hash logged for reproducibility
- Config snapshot saved to experiment directory

### Graceful Degradation

- W&B offline mode if network unavailable
- Fall back to CPU if MPS/CUDA fails
- Skip augmentation on audio processing errors

---

## Implementation Timeline

### Phase 1: Foundation (Day 1-2, ~8-10 hours)

- Set up repository structure and dependencies
- Implement `SOUSADataset` (load from `~/Code/SOUSA/output/dataset`)
- Create rudiment mapping (40 classes)
- Build augmentation pipeline (SpecAugment, mixup, time stretch)
- Test data loading with tiny subset
- Implement Lightning `SOUSADataModule`

### Phase 2: Minimal Training Loop (Day 2-3, ~6-8 hours)

- Create base Hydra configs
- Implement AST model adapter (pure HuggingFace)
- Build `SOUSAClassifier` Lightning module
- Add basic metrics (accuracy, balanced accuracy)
- Run first end-to-end training: `data=tiny model=ast strategy=lora`
- Verify W&B logging

### Phase 3: Expand Models (Day 3-5, ~10-12 hours)

- Integrate HTS-AT (clone official repo, create adapter)
- Integrate BEATs (Microsoft repo, create adapter)
- Integrate EfficientAT (official repo, create adapter)
- Test each model forward pass
- Run validation: `data=small` with each model

### Phase 4: PEFT Strategies (Day 5-6, ~4-6 hours)

- Implement LoRA injection via `peft` library
- Add AdaLoRA and IA3 strategies
- Create `full_finetune` and `from_scratch` configs
- Test strategy switching
- Verify trainable parameter counts

### Phase 5: Full Metrics & Evaluation (Day 6-7, ~4-6 hours)

- Add confusion matrix logging
- Per-rudiment precision/recall/F1
- Top-5 accuracy
- Skill tier breakdown
- Create standalone `evaluate.py` script

### Phase 6: Local Validation (Day 7-8, ~6-8 hours)

- Run complete experiment matrix on `data=small`
- Test: 4 models × 3 strategies = 12 runs
- Verify all experiments complete successfully
- Validate W&B comparison dashboard
- Tune batch sizes for M4 Max

### Phase 7: Cloud Preparation (Day 8-9, ~4-6 hours)

- Create HF Jobs submission scripts
- Test single cloud job with `data=medium`
- Verify checkpoint syncing to HF Hub
- Create batch submission script
- Cost estimation and resource allocation

### Phase 8: Production Training (Day 9-10)

- Submit all 12 experiments to HF Jobs
- Monitor training progress
- Download checkpoints and analyze results

**Total:** ~8-10 days implementation + 24-48 hours cloud training

---

## Success Criteria

### Implementation Complete

- ✅ All 4 models run successfully on `data=small` locally
- ✅ All 5 PEFT strategies (LoRA, AdaLoRA, IA3, Full FT, Scratch) work
- ✅ Hydra config composition enables easy experiment switching
- ✅ W&B logging captures all metrics and artifacts
- ✅ Single experiment runs successfully on HF Jobs

### Research Phase Complete

- ✅ 12+ experiments complete on full 100K dataset
- ✅ Balanced accuracy >85% on test set (estimated baseline)
- ✅ Comprehensive evaluation: confusion matrices, per-rudiment analysis
- ✅ Best teacher model identified for distillation (Phase 2)
- ✅ Results reproducible with saved configs and checkpoints

---

## Next Steps

After design approval:

1. **Set up implementation workspace** (git worktree or branch)
2. **Create detailed implementation plan** (task breakdown)
3. **Begin Phase 1:** Foundation and data pipeline
4. **Iterate rapidly** on M4 Max with `data=tiny` for fast validation
5. **Scale to cloud** once local experiments validate approach

---

## Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Training framework | PyTorch Lightning | Clean abstraction, device management, checkpointing |
| Config system | Hydra | Composable experiments, automatic organization |
| Model integration | Pragmatic adapters | HF for AST, direct integration for others, thin adapter layer |
| PEFT library | HuggingFace `peft` | Multiple methods (LoRA, AdaLoRA, IA3), mature library |
| Implementation order | Data → Training → Models | Bottom-up: prove data works first |
| Deployment scope | Research phase only | Distillation/Core ML after identifying best teacher |
| Development target | M4 Max local-first | Minimize cloud costs, fast iteration |
| Cloud platform | HuggingFace Jobs | PEP 723 scripts, cost-effective, checkpoint syncing |
| Experiment tracking | Weights & Biases | Industry standard, comprehensive metrics, comparison tools |
| Dataset strategy | Local files | Already downloaded at `~/Code/SOUSA/output/dataset` |

---

**Status:** Ready for implementation
