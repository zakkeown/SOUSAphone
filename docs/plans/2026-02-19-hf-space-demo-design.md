# SOUSAphone HF Space Demo — Design Document

**Date**: 2026-02-19
**Status**: Proposed

## Problem

The SOUSAphone OnsetTransformer (shipped at `zkeown/sousaphone`) classifies drum rudiments from 12-dimensional per-stroke onset features. These features include stroke types, sticking, grace notes, and articulation data that were extracted from MIDI ground truth during dataset generation. Raw audio onset detection only provides onset times and strengths — a 9-feature gap.

To build a public demo where visitors upload audio and get rudiment predictions, we need to bridge this gap.

## Solution: Two-Model Pipeline

Train a **Feature Inference Model** that translates raw onset detection output into the 12-dimensional feature space the OnsetTransformer expects. Chain both models in a Gradio demo on HF Spaces.

```
Audio → Onset Detection → Beat Tracking → Feature Inference → OnsetTransformer → Prediction
         (librosa)        (librosa)        Model (NEW)         (shipped, 120K)
```

## Architecture

### Pipeline Stages

1. **Audio Input**: Gradio `gr.Audio` — file upload or microphone recording
2. **Onset Detection**: librosa `onset_detect` + `onset_strength` (CPU-only, ~100ms for 30s audio)
3. **Beat Tracking**: librosa `beat_track` → tempo estimation (auto, no user input)
4. **Feature Inference Model**: (onset_time_ms, onset_strength, tempo_bpm) → 12-dim features per stroke
5. **OnsetTransformer**: 12-dim features → 40-class rudiment prediction (existing shipped model)

### Feature Inference Model

**Input**: per-stroke `(onset_time_ms, onset_strength)` + scalar `tempo_bpm`

**Output**: per-stroke 12-dimensional feature vector matching the OnsetTransformer's expected input:

| Idx | Feature | Type |
|-----|---------|------|
| 0 | `norm_ioi` | continuous |
| 1 | `norm_velocity` | continuous |
| 2 | `is_grace` | binary |
| 3 | `is_accent` | binary |
| 4 | `is_tap` | binary |
| 5 | `is_diddle` | binary |
| 6 | `hand_R` | binary |
| 7 | `diddle_pos` | continuous (0, 0.5, 1) |
| 8 | `norm_flam_spacing` | continuous |
| 9 | `position_in_beat` | continuous |
| 10 | `is_buzz` | binary |
| 11 | `norm_buzz_count` | continuous |

**Architecture**: Small Transformer encoder (matching OnsetTransformer family):
- Input projection: Linear(3 → d_model)
- Positional encoding: learnable
- Encoder: TransformerEncoderLayers
- Output projection: Linear(d_model → 12)
- Estimated ~50-100K parameters

**Loss function**: Combined loss for heterogeneous outputs:
- BCE for binary features (is_grace, is_accent, is_tap, is_diddle, hand_R, is_buzz)
- MSE for continuous features (norm_ioi, norm_velocity, diddle_pos, norm_flam_spacing, position_in_beat, norm_buzz_count)
- Weighted to balance binary vs continuous losses

**Training data**: SOUSA dataset (`zkeown/sousa`), 100K samples:
- Source: `strokes.parquet` ground truth features
- Input simulation: take ground truth onset times + velocities, add realistic noise:
  - Timing jitter: Gaussian ±5-15ms
  - Strength noise: ±10-20% of velocity
  - Occasional missed onsets (dropout)
  - Occasional spurious onsets (false positives)
- Target: full 12-dim feature vector as computed by `OnsetDataset._encode_strokes()`

**Known limitations**:
- `hand_R` (sticking) is hardest to infer — velocity asymmetry between hands may be subtle. Model may learn canonical sticking patterns per rudiment.
- Perfect accuracy unlikely on noisy real audio. This is documented honestly.

### UI Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  SOUSAphone — Drum Rudiment Classifier                           │
├──────────────────────────────────────────────────────────────────┤
│  [Audio Input: Upload or Record]                                 │
│  [ Classify ]                                                    │
├──────────────────────────┬───────────────────────────────────────┤
│  PREDICTION              │  RUDIMENT NOTATION                    │
│  Top-5 confidence bars   │  Name + sticking pattern (RLRR LRLL) │
│                          │  Stroke type annotations              │
├──────────────────────────┴───────────────────────────────────────┤
│  ONSET TIMELINE                                                  │
│  Waveform + onset markers, color-coded by predicted stroke type  │
├──────────────────────────────────────────────────────────────────┤
│  FEATURE HEATMAP                                                 │
│  12 features × N strokes, showing predicted pattern structure    │
└──────────────────────────────────────────────────────────────────┘
```

Four visualization panels:
1. **Prediction**: confidence bar chart, top-5 rudiments
2. **Rudiment notation**: standard sticking pattern for predicted rudiment
3. **Onset timeline**: waveform with color-coded onset markers
4. **Feature heatmap**: 12×N heatmap of predicted features

## Deployment

**Platform**: HF Spaces, Gradio SDK, Free CPU Basic (2 vCPU, 16 GB RAM)

**Both models are tiny** (~120K + ~100K params) — CPU inference is fast.

**Repository structure** (inside SOUSAphone):
```
SOUSAphone/
├── sousa/
│   ├── models/
│   │   ├── onset_transformer.py      # existing
│   │   └── feature_inference.py      # NEW: Feature Inference Model
│   ├── data/
│   │   ├── onset_dataset.py          # existing
│   │   └── feature_inference_dataset.py  # NEW: training dataset
│   └── inference/
│       └── pipeline.py               # NEW: full audio→prediction pipeline
├── configs/
│   └── model/
│       └── feature_inference.yaml    # NEW: Hydra config
├── space/
│   ├── README.md                     # HF Space config YAML
│   ├── app.py                        # Gradio app
│   ├── requirements.txt              # torch, librosa, gradio
│   └── visualizations.py             # onset timeline, heatmap, notation
├── train.py                          # existing (trains both models via Hydra)
└── hf_upload/                        # existing OnsetTransformer artifacts
```

**HF repos**:
- `zkeown/sousaphone` — OnsetTransformer (existing)
- `zkeown/sousaphone-feature-model` — Feature Inference Model (new)
- `zkeown/sousaphone-demo` — HF Space (new, code synced from `space/`)

## Local Development

Write onset detection code against the standard **librosa API**. Test locally via MetalMom's compatibility layer (`from metalmom.compat import librosa`). Same code runs with plain `import librosa` on the HF Space (Linux).

## Dependencies (Space)

```
torch>=2.0
librosa>=0.10.0
gradio>=5.0
numpy>=1.24
soundfile>=0.12
matplotlib>=3.7    # for visualizations
```

## Success Criteria

1. Feature Inference Model trains successfully on SOUSA data
2. End-to-end pipeline classifies known rudiment recordings correctly
3. Gradio demo is deployed and accessible on HF Spaces
4. All four visualizations render correctly
5. Inference completes in <5 seconds on free CPU tier
