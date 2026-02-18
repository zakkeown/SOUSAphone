"""Evaluate a checkpoint on the test set."""

import sys
import importlib
import inspect
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pathlib import Path

from sousa.data.datamodule import SOUSADataModule
from sousa.data.onset_datamodule import OnsetDataModule
from sousa.training.module import SOUSAClassifier


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    ckpt_path = getattr(cfg, 'ckpt_path', None)
    if not ckpt_path:
        print("ERROR: must provide ckpt_path=<path>")
        sys.exit(1)

    dataset_path = Path(cfg.dataset_path).expanduser()

    # Build model
    module_path, class_name = cfg.model.class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    model_kwargs = {"num_classes": cfg.model.num_classes, "pretrained": cfg.model.pretrained}
    model_sig = inspect.signature(model_class.__init__)
    model_params = set(model_sig.parameters.keys())
    if hasattr(cfg.model, 'model_name') and 'model_name' in model_params:
        model_kwargs['model_name'] = cfg.model.model_name
    if hasattr(cfg.model, 'sample_rate') and 'sample_rate' in model_params:
        model_kwargs['sample_rate'] = cfg.model.sample_rate
    for key in ('feature_dim', 'd_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout', 'max_seq_len'):
        if hasattr(cfg.model, key) and key in model_params:
            model_kwargs[key] = getattr(cfg.model, key)

    model = model_class(**model_kwargs)

    # Build datamodule
    max_samples = getattr(cfg.data, 'num_samples', None)

    if cfg.model.input_type == "onset":
        datamodule = OnsetDataModule(
            dataset_path=str(dataset_path),
            batch_size=cfg.training.batch_size,
            num_workers=cfg.num_workers,
            max_seq_len=cfg.model.max_seq_len,
            max_samples=max_samples,
        )
    else:
        model_needs_spectrogram = (cfg.model.input_type == "spectrogram")
        audio_params = {}
        if hasattr(cfg.model, 'sample_rate'):
            audio_params['sample_rate'] = cfg.model.sample_rate
        if model_needs_spectrogram and hasattr(cfg.model, 'n_mels'):
            audio_params.update({
                'n_mels': cfg.model.n_mels, 'n_fft': cfg.model.n_fft,
                'hop_length': cfg.model.hop_length, 'max_length': cfg.model.max_length,
            })
            if hasattr(cfg.model, 'normalize_spec'):
                audio_params['normalize_spec'] = cfg.model.normalize_spec
            if hasattr(cfg.model, 'norm_mean'):
                audio_params['norm_mean'] = cfg.model.norm_mean
            if hasattr(cfg.model, 'norm_std'):
                audio_params['norm_std'] = cfg.model.norm_std

        extra_params = {}
        if hasattr(cfg.data, 'reference_tempo') and cfg.data.reference_tempo is not None:
            extra_params['reference_tempo'] = float(cfg.data.reference_tempo)

        datamodule = SOUSADataModule(
            dataset_path=str(dataset_path),
            batch_size=cfg.training.batch_size,
            num_workers=cfg.num_workers,
            use_spectrogram=model_needs_spectrogram,
            max_samples=max_samples,
            **audio_params,
            **extra_params,
        )

    # Build classifier and load checkpoint
    classifier = SOUSAClassifier.load_from_checkpoint(
        ckpt_path, model=model, config=cfg, weights_only=False,
    )

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        logger=False,
    )

    trainer.test(classifier, datamodule)


if __name__ == "__main__":
    main()
