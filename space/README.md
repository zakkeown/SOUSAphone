---
title: SOUSAphone
emoji: "\U0001F941"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.12.0"
python_version: "3.11"
app_file: app.py
pinned: false
models:
  - zkeown/sousaphone
  - zkeown/sousaphone-feature-model
datasets:
  - zkeown/sousa
tags:
  - audio-classification
  - drum-rudiments
  - music
  - percussion
short_description: "Classify all 40 PAS drum rudiments from audio"
---

# SOUSAphone Demo

Upload a drum rudiment performance and SOUSAphone identifies which of the 40 PAS International Drum Rudiments it is.

**Pipeline:** Audio → Onset Detection → Feature Inference → OnsetTransformer → Classification

See [zkeown/sousaphone](https://huggingface.co/zkeown/sousaphone) for model details.
