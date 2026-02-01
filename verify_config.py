"""Verify configuration files are correct."""
from hydra import compose, initialize
from omegaconf import OmegaConf

def verify_config():
    """Load and verify configuration."""
    try:
        with initialize(version_base=None, config_path="configs"):
            cfg = compose(config_name="config")

            print("Configuration loaded successfully!")
            print("=" * 80)
            print(OmegaConf.to_yaml(cfg))
            print("=" * 80)

            # Verify the exact values from the plan
            assert cfg.data.name == "tiny"
            assert cfg.training.batch_size == 16
            assert cfg.training.max_epochs == 50
            assert cfg.training.learning_rate == 5e-5
            assert cfg.wandb.project == "sousa-rudiment-classification"
            assert cfg.dataset_path == "~/Code/SOUSA/output/dataset"

            print(f"\nDataset: {cfg.data.name}")
            print(f"Batch size: {cfg.training.batch_size}")
            print("\nAll verifications passed!")

    except Exception as e:
        print(f"Error loading config: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_config()
