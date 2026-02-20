"""Visualization functions for SOUSAphone Gradio demo."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Stroke type colors
STROKE_COLORS = {
    "tap": "#4A90D9",
    "accent": "#E74C3C",
    "grace/flam": "#F39C12",
    "diddle": "#2ECC71",
    "buzz": "#9B59B6",
}

# Rudiment notation patterns (canonical sticking for each rudiment)
RUDIMENT_STICKING = {
    "single-stroke-roll": "R L R L R L R L",
    "single-stroke-four": "R L R L",
    "single-stroke-seven": "R L R L R L R",
    "double-stroke-open-roll": "R R L L R R L L",
    "five-stroke-roll": "R R L L R",
    "seven-stroke-roll": "R R L L R R L",
    "nine-stroke-roll": "R R L L R R L L R",
    "single-paradiddle": "R L R R L R L L",
    "double-paradiddle": "R L R L R R L R L R L L",
    "triple-paradiddle": "R L R L R L R R L R L R L R L L",
    "single-paradiddle-diddle": "R L R R L L",
    "flam": "lR rL",
    "flam-accent": "lR R L rL L R",
    "flam-tap": "lR R rL L",
    "flamacue": "lR L R L rL",
    "flam-paradiddle": "lR L R R rL R L L",
    "drag": "llR rrL",
    "single-drag-tap": "llR L rrL R",
    "double-drag-tap": "llR llR L rrL rrL R",
    "lesson-25": "llR L R L llR L R L",
    "single-ratamacue": "llR L R L",
    "double-ratamacue": "llR llR L R L",
    "triple-ratamacue": "llR llR llR L R L",
    "multiple-bounce-roll": "z z z z z z z z",
}


def plot_onset_timeline(
    audio: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    predicted_features: np.ndarray,
    attention_mask: np.ndarray,
) -> plt.Figure:
    """Plot waveform with color-coded onset markers."""
    fig, ax = plt.subplots(figsize=(12, 3))

    # Plot waveform
    t = np.arange(len(audio)) / sr
    ax.plot(t, audio, color="#cccccc", linewidth=0.5, alpha=0.7)

    # Plot onset markers colored by predicted stroke type
    n_real = int(attention_mask.sum())
    for i in range(min(n_real, len(onset_times))):
        feat = predicted_features[i]
        # Determine stroke type from binary features
        color = STROKE_COLORS["tap"]  # default
        if feat[2] > 0.5:  # is_grace
            color = STROKE_COLORS["grace/flam"]
        elif feat[5] > 0.5:  # is_diddle
            color = STROKE_COLORS["diddle"]
        elif feat[10] > 0.5:  # is_buzz
            color = STROKE_COLORS["buzz"]
        elif feat[3] > 0.5:  # is_accent
            color = STROKE_COLORS["accent"]

        ax.axvline(onset_times[i], color=color, alpha=0.8, linewidth=1.5)

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in STROKE_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Detected Onsets")
    fig.tight_layout()
    return fig


def plot_feature_heatmap(
    predicted_features: np.ndarray,
    attention_mask: np.ndarray,
) -> plt.Figure:
    """Plot 12×N heatmap of predicted features."""
    n_real = int(attention_mask.sum())
    features = predicted_features[:n_real].T  # (12, n_real)

    feature_names = [
        "IOI", "velocity", "grace", "accent", "tap",
        "diddle", "hand_R", "diddle_pos", "flam_sp",
        "beat_pos", "buzz", "buzz_ct",
    ]

    fig, ax = plt.subplots(figsize=(max(8, n_real * 0.3), 4))
    im = ax.imshow(features, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(12))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel("Stroke #")
    ax.set_title("Predicted Features")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def format_rudiment_notation(rudiment_name: str) -> str:
    """Get notation string for a rudiment."""
    sticking = RUDIMENT_STICKING.get(rudiment_name)
    display_name = rudiment_name.replace("-", " ").title()
    if sticking:
        return f"**{display_name}**\n\nSticking: `{sticking}`"
    return f"**{display_name}**\n\nSee PAS International Drum Rudiments chart for sticking."
