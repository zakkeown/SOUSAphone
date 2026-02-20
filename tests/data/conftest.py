"""Shared fixtures for data tests."""

import pandas as pd
import pytest


@pytest.fixture
def mock_fi_dataset(tmp_path):
    """Create minimal mock dataset for feature inference training."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "labels").mkdir()

    # Create mock metadata CSV
    metadata = dataset_dir / "metadata.csv"
    metadata.write_text(
        "sample_id,rudiment_slug,split,audio_path,duration,tempo_bpm,soundfont,augmentation_preset\n"
        "s001,flam,train,audio/flam/s001.flac,2.5,120,piano,clean\n"
        "s002,single-stroke-roll,train,audio/single-stroke-roll/s002.flac,3.0,100,piano,clean\n"
        "s003,flam,val,audio/flam/s003.flac,2.2,120,piano,clean\n"
    )

    # Create mock strokes parquet
    strokes_data = []
    # Sample s001: 8 strokes of a flam
    for i in range(8):
        strokes_data.append({
            "sample_id": "s001",
            "actual_time_ms": i * 250.0,
            "actual_velocity": 80 + (i % 2) * 30,
            "is_grace_note": i % 2 == 0,
            "is_accent": i % 2 == 1,
            "stroke_type": "flam" if i % 2 == 0 else "tap",
            "hand": "R" if i % 2 == 0 else "L",
            "diddle_position": float("nan"),
            "flam_spacing_ms": 30.0 if i % 2 == 0 else float("nan"),
            "buzz_count": float("nan"),
        })
    # Sample s002: 16 strokes
    for i in range(16):
        strokes_data.append({
            "sample_id": "s002",
            "actual_time_ms": i * 125.0,
            "actual_velocity": 90,
            "is_grace_note": False,
            "is_accent": i % 4 == 0,
            "stroke_type": "tap",
            "hand": "R" if i % 2 == 0 else "L",
            "diddle_position": float("nan"),
            "flam_spacing_ms": float("nan"),
            "buzz_count": float("nan"),
        })
    # Sample s003 (val)
    for i in range(8):
        strokes_data.append({
            "sample_id": "s003",
            "actual_time_ms": i * 250.0,
            "actual_velocity": 85,
            "is_grace_note": False,
            "is_accent": False,
            "stroke_type": "tap",
            "hand": "R" if i % 2 == 0 else "L",
            "diddle_position": float("nan"),
            "flam_spacing_ms": float("nan"),
            "buzz_count": float("nan"),
        })

    strokes_df = pd.DataFrame(strokes_data)
    strokes_df.to_parquet(dataset_dir / "labels" / "strokes.parquet")

    return dataset_dir
