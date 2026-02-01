"""Test script to verify Hydra configuration loading without CLI."""
from hydra import compose, initialize
from omegaconf import OmegaConf


def test_config():
    """Load and print configuration."""
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config")

        print("=" * 80)
        print("Configuration loaded successfully!")
        print("=" * 80)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)

        # Verify key sections exist
        assert "training" in cfg
        assert "augmentation" in cfg
        assert "model" in cfg
        assert "wandb" in cfg
        assert "paths" in cfg
        assert "data" in cfg

        print("\nVerification successful!")
        print(f"- Dataset subset: {cfg.data.subset}")
        print(f"- Dataset path: {cfg.data.dataset_path}")
        print(f"- Batch size: {cfg.training.batch_size}")
        print(f"- Learning rate: {cfg.training.learning_rate}")
        print(f"- W&B project: {cfg.wandb.project}")


if __name__ == "__main__":
    test_config()
