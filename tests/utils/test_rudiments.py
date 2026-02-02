from sousa.utils.rudiments import (
    get_rudiment_mapping,
    get_inverse_mapping,
    get_num_classes,
    RUDIMENT_NAMES,
)

def test_rudiment_mapping_has_40_classes():
    """Verify we have exactly 40 PAS rudiments"""
    mapping = get_rudiment_mapping()
    assert len(mapping) == 40

def test_rudiment_mapping_starts_at_zero():
    """Class IDs should be 0-39"""
    mapping = get_rudiment_mapping()
    ids = list(mapping.values())
    assert min(ids) == 0
    assert max(ids) == 39

def test_rudiment_mapping_is_unique():
    """No duplicate class IDs"""
    mapping = get_rudiment_mapping()
    ids = list(mapping.values())
    assert len(ids) == len(set(ids))

def test_rudiment_names_list():
    """RUDIMENT_NAMES should have all 40 rudiments in order"""
    assert len(RUDIMENT_NAMES) == 40
    # Should be alphabetically sorted by slug
    sorted_names = sorted(RUDIMENT_NAMES)
    assert RUDIMENT_NAMES == sorted_names


def test_inverse_mapping():
    """Verify inverse mapping works correctly"""
    inv_mapping = get_inverse_mapping()
    assert len(inv_mapping) == 40
    assert inv_mapping[10] == "flam"  # Verify one entry


def test_round_trip_mapping():
    """Verify mapping and inverse are truly inverse"""
    mapping = get_rudiment_mapping()
    inv_mapping = get_inverse_mapping()
    for name, idx in mapping.items():
        assert inv_mapping[idx] == name


def test_get_num_classes():
    """Verify get_num_classes returns 40"""
    assert get_num_classes() == 40
