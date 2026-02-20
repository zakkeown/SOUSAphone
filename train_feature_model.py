"""Training script for Feature Inference Model."""

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

from sousa.models.feature_inference import FeatureInferenceModel
from sousa.data.feature_inference_datamodule import FeatureInferenceDataModule
from sousa.training.feature_module import FeatureInferenceModule


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Train the Feature Inference Model."""

    pl.seed_everything(cfg.seed)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        tags=["feature-inference"],
    )

    dataset_path = Path(cfg.dataset_path).expanduser()

    model = FeatureInferenceModel(
        input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
    )

    datamodule = FeatureInferenceDataModule(
        dataset_path=str(dataset_path),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.num_workers,
        max_seq_len=cfg.model.max_seq_len,
    )

    module = FeatureInferenceModule(model, cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{val_loss:.4f}",
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=cfg.training.early_stopping_patience,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
    )

    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
