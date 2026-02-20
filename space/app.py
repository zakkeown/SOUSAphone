"""SOUSAphone Gradio demo — Drum Rudiment Classifier."""

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download

# Add parent to path so we can import sousa modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from sousa.inference.pipeline import RudimentPipeline
from space.visualizations import (
    plot_onset_timeline,
    plot_feature_heatmap,
    format_rudiment_notation,
)


def load_pipeline() -> RudimentPipeline:
    """Download models from HF Hub and initialize pipeline."""
    feature_model_path = hf_hub_download(
        repo_id="zkeown/sousaphone", filename="feature_inference_model.bin"
    )
    classifier_model_path = hf_hub_download(
        repo_id="zkeown/sousaphone", filename="pytorch_model.bin"
    )
    return RudimentPipeline(
        feature_model_path=feature_model_path,
        classifier_model_path=classifier_model_path,
    )


pipeline = load_pipeline()


def classify(audio_input):
    """Main classification function called by Gradio."""
    if audio_input is None:
        return None, None, None, None

    sr, audio = audio_input

    # Convert to float32 mono
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Normalize
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()

    result = pipeline.predict(audio, sr=sr)

    if "error" in result:
        return result["error"], None, None, None

    # Confidence chart (dict for gr.Label)
    confidences = {r["rudiment"]: r["confidence"] for r in result["top5"]}

    # Rudiment notation
    notation = format_rudiment_notation(result["predicted_rudiment"])

    # Onset timeline plot
    timeline_fig = plot_onset_timeline(
        audio, sr,
        result["onset_times"],
        result["predicted_features"],
        result["attention_mask"],
    )

    # Feature heatmap
    heatmap_fig = plot_feature_heatmap(
        result["predicted_features"],
        result["attention_mask"],
    )

    return confidences, notation, timeline_fig, heatmap_fig


demo = gr.Interface(
    fn=classify,
    inputs=gr.Audio(label="Upload or record a drum rudiment"),
    outputs=[
        gr.Label(num_top_classes=5, label="Prediction"),
        gr.Markdown(label="Rudiment Notation"),
        gr.Plot(label="Onset Timeline"),
        gr.Plot(label="Feature Heatmap"),
    ],
    title="SOUSAphone — Drum Rudiment Classifier",
    description=(
        "Upload a recording of a drum rudiment and SOUSAphone will identify which "
        "of the 40 PAS International Drum Rudiments it is. The model detects onsets, "
        "infers stroke-level features, and classifies using a lightweight Transformer."
    ),
    examples=[],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
