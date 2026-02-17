# Multi-Soundfont Dataset Regeneration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Regenerate the SOUSA dataset with ~35 soundfonts (vs current 1) to break the 70% accuracy plateau.

**Architecture:** Download additional SF2 soundfonts, generate a new dataset using all SF2 + Frankensnare SFZ programs, then retrain AST+LoRA with identical config for controlled comparison.

**Tech Stack:** SOUSA generator (Python), FluidSynth (SF2), SFZ renderer, SOUSAphone training pipeline (PyTorch Lightning)

---

### Task 1: Download Additional Soundfonts

**Files:**
- Run: `/Users/zakkeown/Code/SOUSA/scripts/setup_soundfonts.py`
- Verify: `/Users/zakkeown/Code/SOUSA/data/soundfonts/`

**Step 1: Download GeneralUser_GS and Douglas_Natural_Studio**

```bash
cd /Users/zakkeown/Code/SOUSA
python scripts/setup_soundfonts.py --name GeneralUser_GS --name Douglas_Natural_Studio
```

**Step 2: Verify all SF2 soundfonts are installed**

```bash
ls -lh /Users/zakkeown/Code/SOUSA/data/soundfonts/*.sf2
```

Expected: 5 SF2 files (FluidR3_GM_GS, MT_PowerDrumKit, Marching_Snare, GeneralUser_GS, Douglas_Natural_Studio)

---

### Task 2: Verify Frankensnare SFZ Note Mapping

**Files:**
- Check: `/Users/zakkeown/Code/SOUSA/data/sfz/Frankensnare/Programs/*.sfz`

**Step 1: Confirm note mapping**

SOUSA MIDI uses note 38 (GM Acoustic Snare). Frankensnare uses key=37. Required remap: `38:37`.

**Step 2: Quick synthesis test**

```bash
cd /Users/zakkeown/Code/SOUSA
python -c "
from dataset_gen.audio_synth.sfz_synthesizer import SfzSynthesizer
synth = SfzSynthesizer()
synth.load_sfz('data/sfz/Frankensnare/Programs/03-10x6ash.sfz', note_remap={38: 37})
# Render a single test note
events = [(0.0, 38, 100)]  # time, note, velocity
audio = synth.render('03-10x6ash', events, duration=1.0, sample_rate=44100)
print(f'Audio shape: {audio.shape}, max: {audio.max():.3f}')
assert audio.max() > 0.01, 'No audio produced - check note remap'
print('SFZ synthesis OK')
"
```

Expected: Audio shape with nonzero values confirming the note remap works.

---

### Task 3: Generate Multi-Soundfont Dataset

**Files:**
- Run: `/Users/zakkeown/Code/SOUSA/scripts/generate_dataset.py`
- Output: `/Users/zakkeown/Code/SOUSA/output/dataset_multisf/`

**Step 1: Calculate sizing**

With 5 SF2 + 30 SFZ = 35 soundfonts, and 5 room presets, we want ~100K samples.
- 20 profiles × 40 rudiments × 5 tempos × 25 augmentations = 100,000 samples
- Each augmentation cycles through a different soundfont (25 of 35 per base sample)
- Room presets also cycle across augmentations
- ~75GB estimated disk usage

**Step 2: Generate dataset**

```bash
cd /Users/zakkeown/Code/SOUSA
nohup python scripts/generate_dataset.py \
  --profiles 20 \
  --tempos 5 \
  --augmentations 25 \
  --with-audio \
  --soundfont data/soundfonts \
  --sfz data/sfz/Frankensnare/Programs \
  --note-remap 38:37 \
  --fixed-duration 5.0 \
  --seed 42 \
  --workers 4 \
  -o output/dataset_multisf \
  > /Users/zakkeown/Code/SOUSA/generate_multisf.log 2>&1 &
```

**Step 3: Monitor generation**

```bash
tail -f /Users/zakkeown/Code/SOUSA/generate_multisf.log
```

**Step 4: Verify dataset**

```bash
python -c "
import pandas as pd
df = pd.read_csv('/Users/zakkeown/Code/SOUSA/output/dataset_multisf/metadata.csv')
print(f'Total samples: {len(df)}')
print(f'Unique soundfonts: {df[\"soundfont\"].nunique()}')
print(f'Soundfonts: {sorted(df[\"soundfont\"].unique())}')
print(f'Samples per split: {df[\"split\"].value_counts().to_dict()}')
print(f'Rudiments: {df[\"rudiment_slug\"].nunique()}')
"
```

Expected: ~100K samples across ~35 soundfonts with train/val/test splits.

**Step 5: Commit checkpoint (nothing to commit in SOUSAphone yet)**

---

### Task 4: Update SOUSAphone Dataset Path and Retrain

**Files:**
- Modify: `/Users/zakkeown/Code/SOUSAphone/configs/data/full.yaml` (or equivalent)
- Run: `/Users/zakkeown/Code/SOUSAphone/train.py`

**Step 1: Point SOUSAphone to new dataset**

Update the data config to use the new multi-soundfont dataset path:
`/Users/zakkeown/Code/SOUSA/output/dataset_multisf`

**Step 2: Launch training**

```bash
cd /Users/zakkeown/Code/SOUSAphone
nohup ~/.pyenv/versions/3.11.14/bin/python train.py data=full \
  > train_multisf_nohup.log 2>&1 &
```

Uses same AST+LoRA config (hop=80, rank=8, alpha=16, LR=1e-3).

**Step 3: Monitor training**

```bash
tail -f train_multisf_nohup.log
```

Expected: Training starts, loss decreases over epochs.

---

### Task 5: Evaluate and Compare

**Files:**
- Run: `/Users/zakkeown/Code/SOUSAphone/analyze_confusion.py`

**Step 1: Run confusion analysis on best checkpoint**

When training converges (early stopping or epoch 15+), run:

```bash
cd /Users/zakkeown/Code/SOUSAphone
python analyze_confusion.py --checkpoint <best_checkpoint_path>
```

**Step 2: Compare against baselines**

| Metric | hop=160 (1 SF) | hop=80 (1 SF) | hop=80 (35 SF) |
|--------|---------------|---------------|----------------|
| Val accuracy | 70.3% | 70.1% | ??? |
| double-drag→single-drag | 89% confused | ??? | ??? |
| swiss-army→flam-accent | 72% confused | ??? | ??? |

**Step 3: Decide next steps based on results**

- If accuracy improves significantly: data diversity was the bottleneck. Consider even more soundfonts or real recordings.
- If accuracy doesn't improve: model architecture is the bottleneck. Try HTS-AT or EfficientAT next.
