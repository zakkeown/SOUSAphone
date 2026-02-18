# SOUSAphone

[![CI](https://github.com/zakkeown/SOUSAphone/actions/workflows/ci.yml/badge.svg)](https://github.com/zakkeown/SOUSAphone/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/zkeown/sousaphone)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-blue)](https://huggingface.co/datasets/zkeown/sousa)

Training infrastructure for 40-class drum rudiment classification using PyTorch Lightning, Hydra, and state-of-the-art audio models.

## Overview

SOUSAphone classifies all 40 Percussive Arts Society (PAS) International Drum Rudiments from audio recordings. It provides a modular training pipeline with five model architectures, configurable training strategies, and a Hydra-based configuration system that makes it easy to run experiments across different model/data/strategy combinations.

## Models

| Model | Input Type | Architecture | Pretrained On |
|---|---|---|---|
| **AST** | Spectrogram | Audio Spectrogram Transformer | AudioSet |
| **BEATs** | Waveform | WavLM (self-supervised audio transformer) | Audio data |
| **HTS-AT** | Spectrogram | Hierarchical Token-Semantic Audio Transformer (via CLAP) | LAION audio-text |
| **EfficientAT** | Spectrogram | MobileNetV2 with attention pooling | ImageNet |
| **OnsetTransformer** | Stroke features | Custom transformer encoder (~111K params) | None (trains from scratch) |

## Training Strategies

- **LoRA** — Parameter-efficient fine-tuning via PEFT. Injects low-rank adapters into attention layers while keeping the pretrained backbone frozen.
- **Full fine-tune** — Standard end-to-end training of all parameters.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Train AST with LoRA on a tiny dataset (smoke test)
python train.py model=ast strategy=lora data=tiny

# Train OnsetTransformer on the full dataset
python train.py model=onset_transformer data=full

# Evaluate a checkpoint
python evaluate.py model=ast strategy=lora ckpt_path=/path/to/checkpoint.ckpt

# Analyze confusion matrix
python analyze_confusion.py model=ast strategy=lora ckpt_path=/path/to/checkpoint.ckpt
```

## Configuration

SOUSAphone uses [Hydra](https://hydra.cc/) for configuration. The config hierarchy is:

```
configs/
├── config.yaml          # Master config (training hyperparams, augmentation, W&B)
├── model/               # Model-specific configs
│   ├── ast.yaml
│   ├── beats.yaml
│   ├── htsat.yaml
│   ├── efficientat.yaml
│   └── onset_transformer.yaml
├── strategy/            # Training strategy configs
│   ├── lora.yaml
│   └── full_finetune.yaml
└── data/                # Dataset size configs
    ├── tiny.yaml        # 1K samples
    ├── mini.yaml        # 3K samples
    ├── small.yaml       # 10K samples
    ├── medium.yaml      # 25K samples
    └── full.yaml        # Entire dataset (~100K samples)
```

Override any parameter from the command line:

```bash
# Use a different model and data size
python train.py model=htsat data=medium

# Override training hyperparameters
python train.py training.max_epochs=100 training.batch_size=8

# Disable W&B logging
python train.py wandb.mode=offline
```

## Project Structure

```
SOUSAphone/
├── train.py                    # Training entry point
├── evaluate.py                 # Checkpoint evaluation
├── analyze_confusion.py        # Confusion matrix analysis
├── configs/                    # Hydra configuration hierarchy
├── sousa/                      # Main package
│   ├── models/                 # Model adapters (AST, BEATs, HTS-AT, EfficientAT, OnsetTransformer)
│   ├── data/                   # Dataset, DataModule, augmentations, transforms
│   ├── training/               # PyTorch Lightning training module
│   └── utils/                  # Audio loading, rudiment mapping
├── tests/                      # Test suite
├── pyproject.toml              # Dependencies and project metadata
└── LICENSE
```

## Tech Stack

- **PyTorch Lightning** — Training loop, callbacks, checkpointing
- **Hydra** — Configuration management
- **HuggingFace Transformers** — Pretrained audio models
- **PEFT** — Parameter-efficient fine-tuning (LoRA)
- **Weights & Biases** — Experiment tracking
- **torchmetrics** — Accuracy, F1, confusion matrix

## The 40 PAS Rudiments

SOUSAphone classifies all 40 PAS International Drum Rudiments:

Rolls: single-stroke-roll, single-stroke-four, single-stroke-seven, multiple-bounce-roll, double-stroke-open-roll, five-stroke-roll, six-stroke-roll, seven-stroke-roll, nine-stroke-roll, ten-stroke-roll, eleven-stroke-roll, thirteen-stroke-roll, fifteen-stroke-roll, seventeen-stroke-roll, triple-stroke-roll

Paradiddles: single-paradiddle, double-paradiddle, triple-paradiddle, single-paradiddle-diddle

Flams: flam, flam-accent, flam-tap, flamacue, flam-paradiddle, single-flammed-mill, flam-paradiddle-diddle, pataflafla, swiss-army-triplet, inverted-flam-tap, flam-drag

Drags: drag, single-drag-tap, double-drag-tap, lesson-25, single-dragadiddle, drag-paradiddle-1, drag-paradiddle-2, single-ratamacue, double-ratamacue, triple-ratamacue

## License

MIT — see [LICENSE](LICENSE).
