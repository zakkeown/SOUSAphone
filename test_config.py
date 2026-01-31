import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg)
    print(f"Dataset: {cfg.data.name}")
    print(f"Batch size: {cfg.training.batch_size}")

if __name__ == "__main__":
    main()
