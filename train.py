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
from sousa.data.onset_datamodule import OnsetDataModule
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
    model_kwargs = {
        "num_classes": cfg.model.num_classes,
        "pretrained": cfg.model.pretrained,
    }
    # Add optional model-specific parameters (only if model accepts them)
    import inspect
    model_sig = inspect.signature(model_class.__init__)
    model_params = set(model_sig.parameters.keys())

    if hasattr(cfg.model, 'model_name') and 'model_name' in model_params:
        model_kwargs['model_name'] = cfg.model.model_name
    if hasattr(cfg.model, 'sample_rate') and 'sample_rate' in model_params:
        model_kwargs['sample_rate'] = cfg.model.sample_rate

    # Onset transformer model-specific kwargs
    for key in ('feature_dim', 'd_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout', 'max_seq_len'):
        if hasattr(cfg.model, key) and key in model_params:
            model_kwargs[key] = getattr(cfg.model, key)

    model = model_class(**model_kwargs)

    # Get max samples from data config (None = use all)
    max_samples = getattr(cfg.data, 'num_samples', None)

    # Get curriculum learning filters from data config
    curriculum_filters = {}
    if hasattr(cfg.data, 'soundfonts') and cfg.data.soundfonts is not None:
        curriculum_filters['soundfonts'] = list(cfg.data.soundfonts)
    if hasattr(cfg.data, 'augmentation_presets') and cfg.data.augmentation_presets is not None:
        curriculum_filters['augmentation_presets'] = list(cfg.data.augmentation_presets)
    if hasattr(cfg.data, 'tempo_range') and cfg.data.tempo_range is not None:
        curriculum_filters['tempo_range'] = tuple(cfg.data.tempo_range)

    if cfg.model.input_type == "onset":
        # Onset (stroke-level) datamodule — no audio processing
        datamodule = OnsetDataModule(
            dataset_path=str(dataset_path),
            batch_size=cfg.training.batch_size,
            num_workers=cfg.num_workers,
            max_seq_len=cfg.model.max_seq_len,
            max_samples=max_samples,
            **curriculum_filters,
        )
    else:
        # Audio datamodule (spectrogram or waveform)
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

        # Get model-specific audio parameters
        audio_params = {}
        if hasattr(cfg.model, 'sample_rate'):
            audio_params['sample_rate'] = cfg.model.sample_rate
        if model_needs_spectrogram and hasattr(cfg.model, 'n_mels'):
            audio_params.update({
                'n_mels': cfg.model.n_mels,
                'n_fft': cfg.model.n_fft,
                'hop_length': cfg.model.hop_length,
                'max_length': cfg.model.max_length,
            })
            if hasattr(cfg.model, 'normalize_spec'):
                audio_params['normalize_spec'] = cfg.model.normalize_spec
            if hasattr(cfg.model, 'norm_mean'):
                audio_params['norm_mean'] = cfg.model.norm_mean
            if hasattr(cfg.model, 'norm_std'):
                audio_params['norm_std'] = cfg.model.norm_std

        # Reference tempo (audio-only, for tempo normalization)
        if hasattr(cfg.data, 'reference_tempo') and cfg.data.reference_tempo is not None:
            curriculum_filters['reference_tempo'] = float(cfg.data.reference_tempo)

        # Time-stretch augmentation
        time_stretch_params = {}
        if hasattr(cfg, 'augmentation') and hasattr(cfg.augmentation, 'time_stretch') and cfg.augmentation.time_stretch:
            time_stretch_params['use_time_stretch'] = True
            if hasattr(cfg.augmentation, 'time_stretch_min'):
                time_stretch_params['time_stretch_min'] = cfg.augmentation.time_stretch_min
            if hasattr(cfg.augmentation, 'time_stretch_max'):
                time_stretch_params['time_stretch_max'] = cfg.augmentation.time_stretch_max

        datamodule = SOUSADataModule(
            dataset_path=str(dataset_path),
            batch_size=cfg.training.batch_size,
            num_workers=cfg.num_workers,
            use_spectrogram=model_needs_spectrogram,
            use_specaugment=use_specaugment,
            specaugment_params=specaugment_params,
            max_samples=max_samples,
            **audio_params,
            **curriculum_filters,
            **time_stretch_params,
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
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
    )

    # Resume from checkpoint if provided
    ckpt_path = getattr(cfg, 'ckpt_path', None)

    # Train
    trainer.fit(
        classifier, datamodule,
        ckpt_path=ckpt_path,
        weights_only=False if ckpt_path else True,
    )

    # Test
    trainer.test(classifier, datamodule)


if __name__ == "__main__":
    main()
