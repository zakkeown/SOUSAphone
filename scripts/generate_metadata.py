#!/usr/bin/env python3
"""Generate metadata.csv from SOUSA dataset index.json and splits.json."""

import json
import pandas as pd
from pathlib import Path


def parse_sample_id(sample_id: str) -> dict:
    """
    Parse sample ID to extract rudiment and profile.

    Format: {profile_id}_{rudiment_parts}_{tempo}bpm_{soundfont}_{room}
    Examples:
        adv000_single_ratamacue_70bpm_fluidr3_cleanstudio
        adv000_drag_paradiddle_2_70bpm_fluidr3_cleanstudio
        adv000_five_stroke_roll_70bpm_fluidr3_practiceroom
    """
    parts = sample_id.split('_')

    # Profile is first part (e.g., "adv000", "int001", "pro005")
    profile_id = parts[0]

    # Find where tempo starts (ends with "bpm")
    tempo_idx = None
    for i, part in enumerate(parts):
        if part.endswith("bpm"):
            tempo_idx = i
            break

    if tempo_idx is None:
        raise ValueError(f"Could not find tempo in sample_id: {sample_id}")

    # Rudiment is everything between profile and tempo
    # Join all parts between index 1 and tempo_idx
    rudiment_parts = parts[1:tempo_idx]
    rudiment_slug = '_'.join(rudiment_parts) if rudiment_parts else 'unknown'

    # Extract tempo (remove "bpm" suffix)
    tempo = parts[tempo_idx].replace("bpm", "")

    # Soundfont and room are after tempo
    soundfont = parts[tempo_idx + 1] if tempo_idx + 1 < len(parts) else None
    room = parts[tempo_idx + 2] if tempo_idx + 2 < len(parts) else None

    # For now, don't try to parse variation - just use the full rudiment slug
    return {
        'profile_id': profile_id,
        'variation': '',  # Not parsed for now
        'rudiment_slug': rudiment_slug,
        'tempo': int(tempo),
        'soundfont': soundfont,
        'room': room,
    }


def generate_metadata(dataset_path: Path, output_path: Path):
    """Generate metadata.csv from index.json and splits.json."""

    # Load index
    with open(dataset_path / "index.json") as f:
        index = json.load(f)

    # Load splits
    with open(dataset_path / "splits.json") as f:
        splits = json.load(f)

    # Create profile ID to split mapping
    profile_to_split = {}
    for profile_id in splits['train_profile_ids']:
        profile_to_split[profile_id] = 'train'
    for profile_id in splits['val_profile_ids']:
        profile_to_split[profile_id] = 'val'
    for profile_id in splits['test_profile_ids']:
        profile_to_split[profile_id] = 'test'

    # Process all samples
    rows = []
    for sample_id in index['sample_ids']:
        # Parse sample ID
        parsed = parse_sample_id(sample_id)

        # Determine split based on profile ID
        # Profile IDs in the sample_id are short codes like "adv000"
        # We need to map these to the full UUIDs in splits.json
        # For now, we'll assign based on the profile short code
        # This is a simplification - we may need to load additional metadata

        # Create audio path
        audio_path = f"audio/{sample_id}.flac"

        row = {
            'sample_id': sample_id,
            'rudiment_slug': parsed['rudiment_slug'],
            'profile_id': parsed['profile_id'],
            'variation': parsed['variation'],
            'tempo': parsed['tempo'],
            'soundfont': parsed['soundfont'],
            'room': parsed['room'],
            'audio_path': audio_path,
            'split': 'train',  # Default to train, will be updated
        }
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # For now, use a simple heuristic for splits based on sample index
    # This ensures balanced splits across rudiments
    # Group by rudiment and assign splits
    df_grouped = []
    for rudiment, group in df.groupby('rudiment_slug'):
        n = len(group)
        # 80% train, 10% val, 10% test
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        group = group.reset_index(drop=True)
        group.loc[:train_end, 'split'] = 'train'
        group.loc[train_end:val_end, 'split'] = 'val'
        group.loc[val_end:, 'split'] = 'test'

        df_grouped.append(group)

    df = pd.concat(df_grouped, ignore_index=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"Generated metadata.csv with {len(df)} samples")
    print(f"\nSplit distribution:")
    print(df['split'].value_counts().sort_index())
    print(f"\nRudiment distribution:")
    print(df['rudiment_slug'].value_counts().sort_index())


if __name__ == "__main__":
    dataset_path = Path("~/Code/SOUSA/output/dataset").expanduser()
    output_path = dataset_path / "metadata.csv"

    generate_metadata(dataset_path, output_path)
