# train.py
"""Training script for SOUSA rudiment classification."""

import importlib
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

from sousa.data.datamodule import SOUSADataModule
from sousa.training.module import SOUSAClassifier


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    # Initialize W&B logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
    )

    # Expand ~ in dataset path
    dataset_path = Path(cfg.dataset_path).expanduser()

    # Dynamically load model class from config
    module_path, class_name = cfg.model.class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    # Instantiate model with config parameters
    model = model_class(
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    )

    # Create data module with appropriate input type
    model_needs_spectrogram = (cfg.model.input_type == "spectrogram")
    datamodule = SOUSADataModule(
        dataset_path=str(dataset_path),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.num_workers,
        use_spectrogram=model_needs_spectrogram,
    )

    # Create Lightning module
    classifier = SOUSAClassifier(model, cfg)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/acc',
        mode='max',
        save_top_k=1,
        filename='best-{epoch}-{val_acc:.2f}',
    )

    early_stop_callback = EarlyStopping(
        monitor='val/acc',
        patience=cfg.training.early_stopping_patience,
        mode='max',
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
    )

    # Train
    trainer.fit(classifier, datamodule)

    # Test
    trainer.test(classifier, datamodule)


if __name__ == "__main__":
    main()
