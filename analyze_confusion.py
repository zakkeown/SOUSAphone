"""Analyze confusion matrix from a checkpoint."""

import sys
import importlib
import inspect
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from sousa.data.datamodule import SOUSADataModule
from sousa.training.module import SOUSAClassifier
from sousa.utils.rudiments import RUDIMENT_NAMES


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

    model = model_class(**model_kwargs)

    # Build datamodule
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

    max_samples = getattr(cfg.data, 'num_samples', None)

    datamodule = SOUSADataModule(
        dataset_path=str(dataset_path),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.num_workers,
        use_spectrogram=model_needs_spectrogram,
        max_samples=max_samples,
        **audio_params,
    )
    datamodule.setup("test")

    # Load checkpoint correctly via Lightning
    classifier = SOUSAClassifier.load_from_checkpoint(
        ckpt_path, model=model, config=cfg, weights_only=False,
    )

    # Move to device and set eval mode
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    classifier = classifier.to(device)
    classifier.eval()

    # Collect predictions manually (can't use torchmetrics confusion—it gets reset in on_test_epoch_end)
    print("Running inference on test set...")
    all_preds = []
    all_labels = []
    n_batches = len(datamodule.test_dataloader())
    with torch.no_grad():
        for i, batch in enumerate(datamodule.test_dataloader()):
            audio = batch['audio'].to(device)
            labels = batch['label']
            logits = classifier(audio)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{n_batches} batches processed")

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    overall_acc = (all_preds == all_labels).mean()
    print(f"\nOverall accuracy: {overall_acc:.1%} ({(all_preds == all_labels).sum()}/{len(all_labels)})")

    # Build confusion matrix
    n_classes = len(RUDIMENT_NAMES)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    # Print per-class accuracy and top confusions
    print("\n" + "=" * 90)
    print(f"{'CLASS':<30} {'ACC':>6} {'N':>5}  TOP CONFUSIONS")
    print("=" * 90)

    class_results = []
    for i in range(n_classes):
        name = RUDIMENT_NAMES[i]
        total = int(cm[i].sum())
        correct = int(cm[i][i])
        acc = correct / total if total > 0 else 0

        # Find top confusions (excluding correct)
        confusions = []
        for j in range(n_classes):
            if j != i and cm[i][j] > 0:
                confusions.append((RUDIMENT_NAMES[j], int(cm[i][j]), cm[i][j] / total))
        confusions.sort(key=lambda x: -x[1])

        class_results.append((name, acc, total, confusions))

    # Sort by accuracy (worst first)
    class_results.sort(key=lambda x: x[1])

    for name, acc, total, confusions in class_results:
        conf_str = ", ".join(f"{c[0]}({c[2]:.0%})" for c in confusions[:3])
        print(f"{name:<30} {acc:>5.1%} {total:>5}  {conf_str}")

    # Print family-level analysis
    print("\n" + "=" * 90)
    print("FAMILY-LEVEL CONFUSION ANALYSIS")
    print("=" * 90)

    families = defaultdict(list)
    for i, name in enumerate(RUDIMENT_NAMES):
        if "paradiddle" in name:
            families["paradiddles"].append(i)
        elif "roll" in name:
            families["rolls"].append(i)
        elif "flam" in name:
            families["flams"].append(i)
        elif "drag" in name or "dragadiddle" in name:
            families["drags"].append(i)
        elif "ratamacue" in name:
            families["ratamacues"].append(i)
        else:
            families["other"].append(i)

    for family, indices in sorted(families.items()):
        members = [RUDIMENT_NAMES[i] for i in indices]
        # Within-family confusion
        within_confused = 0
        within_total = 0
        for i in indices:
            for j in indices:
                if i != j:
                    within_confused += int(cm[i][j])
            within_total += int(cm[i].sum())

        # Outside-family confusion
        outside_confused = 0
        for i in indices:
            for j in range(n_classes):
                if j not in indices and j != i:
                    outside_confused += int(cm[i][j])

        total_errors = within_confused + outside_confused
        if total_errors > 0:
            within_pct = within_confused / total_errors
        else:
            within_pct = 0

        correct = sum(int(cm[i][i]) for i in indices)
        acc = correct / within_total if within_total > 0 else 0

        print(f"\n{family.upper()} (acc={acc:.1%}, {len(members)} classes)")
        print(f"  Members: {', '.join(members)}")
        print(f"  Within-family confusion: {within_confused}/{total_errors} errors ({within_pct:.0%})")
        print(f"  Cross-family confusion: {outside_confused}/{total_errors} errors ({1-within_pct:.0%})")

    # Print the most confused pairs
    print("\n" + "=" * 90)
    print("TOP 20 MOST CONFUSED PAIRS")
    print("=" * 90)

    pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i][j] > 0:
                total_i = int(cm[i].sum())
                pairs.append((RUDIMENT_NAMES[i], RUDIMENT_NAMES[j], int(cm[i][j]), cm[i][j] / total_i))
    pairs.sort(key=lambda x: -x[2])

    print(f"{'TRUE':<30} {'PREDICTED AS':<30} {'COUNT':>5} {'RATE':>6}")
    print("-" * 75)
    for true, pred, count, rate in pairs[:20]:
        print(f"{true:<30} {pred:<30} {count:>5} {rate:>5.0%}")


if __name__ == "__main__":
    main()
