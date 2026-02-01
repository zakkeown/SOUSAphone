"""Test script to verify small dataset configuration loading."""
from hydra import compose, initialize
from omegaconf import OmegaConf


def test_small_config():
    """Load and print small dataset configuration."""
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config", overrides=["data=small"])

        print("=" * 80)
        print("Small dataset configuration loaded successfully!")
        print("=" * 80)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)

        # Verify key sections exist
        assert cfg.data.subset == "small"
        print(f"\nVerification successful! Dataset subset: {cfg.data.subset}")


if __name__ == "__main__":
    test_small_config()
