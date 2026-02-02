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

    # Prepare SpecAugment parameters if enabled
    specaugment_params = None
    use_specaugment = False
    if hasattr(cfg, 'augmentation') and cfg.augmentation.specaugment and model_needs_spectrogram:
        use_specaugment = True
        specaugment_params = {
            'freq_mask_param': cfg.augmentation.specaugment_freq_mask,
            'time_mask_param': cfg.augmentation.specaugment_time_mask,
            'n_freq_masks': cfg.augmentation.specaugment_n_freq_masks,
            'n_time_masks': cfg.augmentation.specaugment_n_time_masks,
        }

    # Get model-specific audio parameters (if using spectrograms)
    audio_params = {}
    if model_needs_spectrogram and hasattr(cfg.model, 'n_mels'):
        audio_params = {
            'n_mels': cfg.model.n_mels,
            'n_fft': cfg.model.n_fft,
            'hop_length': cfg.model.hop_length,
            'max_length': cfg.model.max_length,
        }

    # Determine if using tiny dataset
    use_tiny = (cfg.data.name == "tiny")

    datamodule = SOUSADataModule(
        dataset_path=str(dataset_path),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.num_workers,
        use_spectrogram=model_needs_spectrogram,
        use_specaugment=use_specaugment,
        specaugment_params=specaugment_params,
        use_tiny=use_tiny,
        **audio_params,
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
