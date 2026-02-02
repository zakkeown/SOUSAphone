"""Rudiment mapping utilities for SOUSA training pipeline.

This module provides canonical mapping of the 40 PAS (Percussive Arts Society)
International Drum Rudiments to class IDs for reproducible training.

The rudiments are ordered alphabetically by slug to ensure consistent
class ID assignment across all training runs.
"""

from typing import Dict

# All 40 PAS International Drum Rudiments (alphabetically sorted by slug)
RUDIMENT_NAMES = [
    "double-drag-tap",
    "double-paradiddle",
    "double-ratamacue",
    "double-stroke-open-roll",
    "drag",
    "drag-paradiddle-1",
    "drag-paradiddle-2",
    "eleven-stroke-roll",
    "fifteen-stroke-roll",
    "five-stroke-roll",
    "flam",
    "flam-accent",
    "flam-drag",
    "flam-paradiddle",
    "flam-paradiddle-diddle",
    "flam-tap",
    "flamacue",
    "inverted-flam-tap",
    "lesson-25",
    "multiple-bounce-roll",
    "nine-stroke-roll",
    "pataflafla",
    "seven-stroke-roll",
    "seventeen-stroke-roll",
    "single-drag-tap",
    "single-dragadiddle",
    "single-flammed-mill",
    "single-paradiddle",
    "single-paradiddle-diddle",
    "single-ratamacue",
    "single-stroke-four",
    "single-stroke-roll",
    "single-stroke-seven",
    "six-stroke-roll",
    "swiss-army-triplet",
    "ten-stroke-roll",
    "thirteen-stroke-roll",
    "triple-paradiddle",
    "triple-ratamacue",
    "triple-stroke-roll",
]


def get_rudiment_mapping() -> Dict[str, int]:
    """Get mapping from rudiment slug to class ID.

    Returns:
        Dictionary mapping rudiment names (slugs) to class IDs (0-39)

    Example:
        >>> mapping = get_rudiment_mapping()
        >>> mapping["flam"]
        10
    """
    return {name: idx for idx, name in enumerate(RUDIMENT_NAMES)}


def get_inverse_mapping() -> Dict[int, str]:
    """Get mapping from class ID to rudiment slug.

    Returns:
        Dictionary mapping class IDs (0-39) to rudiment names (slugs)

    Example:
        >>> inv_mapping = get_inverse_mapping()
        >>> inv_mapping[10]
        'flam'
    """
    return {idx: name for idx, name in enumerate(RUDIMENT_NAMES)}


def get_num_classes() -> int:
    """Get total number of rudiment classes.

    Returns:
        Number of rudiment classes (should be 40 for PAS rudiments)

    Example:
        >>> get_num_classes()
        40
    """
    return len(RUDIMENT_NAMES)
