#!/usr/bin/env python3
"""
Systematic model comparison sweep script.

Runs all combinations of models, PEFT strategies, and augmentation settings
to identify best performers and catch issues early.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback

# Configuration matrix
MODELS = ["ast", "htsat", "beats", "efficientat"]
STRATEGIES = ["lora", "adalora", "ia3"]
AUGMENTATION = [True, False]

# Default paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "sweep"
ERRORS_DIR = RESULTS_DIR / "errors"


def setup_logging(log_file: Optional[Path] = None):
    """Set up logging to console and optionally to file."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def generate_experiments(
    models: List[str],
    strategies: List[str],
    data: str,
    max_epochs: int,
    augmentation: Optional[List[bool]] = None,
) -> List[Dict]:
    """
    Generate all experiment configurations.

    Args:
        models: List of model names to test
        strategies: List of PEFT strategies to test
        data: Dataset size (tiny, small, full)
        max_epochs: Maximum training epochs
        augmentation: List of augmentation settings (default: [True, False])

    Returns:
        List of experiment configuration dictionaries
    """
    if augmentation is None:
        augmentation = AUGMENTATION

    experiments = []
    for model in models:
        for strategy in strategies:
            for use_aug in augmentation:
                exp = {
                    "model": model,
                    "strategy": strategy,
                    "augmentation": use_aug,
                    "data": data,
                    "max_epochs": max_epochs,
                }
                experiments.append(exp)

    return experiments


def get_experiment_id(config: Dict) -> str:
    """Generate unique experiment ID from configuration."""
    aug_str = "aug" if config["augmentation"] else "noaug"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config['model']}_{config['strategy']}_{aug_str}_{config['data']}_{timestamp}"


def get_result_path(exp_id: str) -> Path:
    """Get path for experiment result JSON."""
    return RESULTS_DIR / f"{exp_id}.json"


def check_experiment_exists(config: Dict) -> Optional[Path]:
    """
    Check if experiment with this configuration already exists.

    Returns path to existing result file, or None if not found.
    """
    # Look for any result file matching this config (ignoring timestamp)
    aug_str = "aug" if config["augmentation"] else "noaug"
    pattern = f"{config['model']}_{config['strategy']}_{aug_str}_{config['data']}_*.json"

    existing = list(RESULTS_DIR.glob(pattern))
    if existing:
        # Return most recent
        existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return existing[0]

    return None


def load_result(result_path: Path) -> Optional[Dict]:
    """Load experiment result from JSON file."""
    try:
        with open(result_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load result from {result_path}: {e}")
        return None


def save_result(exp_id: str, result: Dict):
    """Save experiment result to JSON file."""
    result_path = get_result_path(exp_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    logging.info(f"Saved results to {result_path}")


def run_experiment(config: Dict, exp_id: str, timeout: int = 7200) -> Dict:
    """
    Run a single experiment.

    Args:
        config: Experiment configuration
        exp_id: Experiment ID
        timeout: Maximum time in seconds (default: 2 hours)

    Returns:
        Result dictionary with metrics and status
    """
    logging.info(f"Starting experiment: {exp_id}")
    logging.info(f"Config: {config}")

    # Initialize result
    result = {
        "config": config,
        "metrics": {},
        "efficiency": {},
        "status": "running",
        "error": None,
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "experiment_id": exp_id,
    }

    # Save initial status
    save_result(exp_id, result)

    start_time = time.time()

    try:
        # Build command
        aug_str = "true" if config["augmentation"] else "false"

        # Create override arguments for Hydra
        overrides = [
            f"model={config['model']}",
            f"strategy={config['strategy']}",
            f"data={config['data']}",
            f"training.max_epochs={config['max_epochs']}",
            f"augmentation.specaugment={aug_str}",
            f"augmentation.mixup={aug_str}",
            f"wandb.mode=offline",  # Use offline mode for sweep
            f"wandb.tags=[{config['model']},{config['strategy']},sweep]",
        ]

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
        ] + overrides

        logging.info(f"Running command: {' '.join(cmd)}")

        # Set up logging for this experiment
        log_file = ERRORS_DIR / f"{exp_id}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Run training
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT_ROOT,
            )

            # Stream output to log file and console
            for line in process.stdout:
                f.write(line)
                f.flush()
                # Only log important lines to console
                if any(keyword in line.lower() for keyword in ['epoch', 'error', 'loss', 'acc', 'f1']):
                    logging.debug(line.strip())

            process.wait(timeout=timeout)

        if process.returncode != 0:
            raise RuntimeError(f"Training failed with return code {process.returncode}")

        # Parse metrics from log file
        metrics = parse_metrics_from_log(log_file)
        result["metrics"] = metrics

        # Calculate efficiency metrics
        result["efficiency"] = {
            "training_time_seconds": time.time() - start_time,
            "avg_epoch_time": (time.time() - start_time) / config["max_epochs"],
        }

        # TODO: Extract parameter counts from log
        # For now, leave empty - will be populated when we parse the actual output

        result["status"] = "completed"
        logging.info(f"Experiment {exp_id} completed successfully")

    except subprocess.TimeoutExpired:
        result["status"] = "failed"
        result["error"] = f"Training exceeded timeout of {timeout} seconds"
        logging.error(f"Experiment {exp_id} timed out")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

        # Save full error to log
        error_log = ERRORS_DIR / f"{exp_id}_error.log"
        with open(error_log, 'w') as f:
            f.write(f"Error in experiment {exp_id}\n")
            f.write(f"Config: {json.dumps(config, indent=2)}\n\n")
            f.write(f"Error: {e}\n\n")
            f.write(traceback.format_exc())

        logging.error(f"Experiment {exp_id} failed: {e}")
        logging.error(f"Full traceback saved to {error_log}")

    finally:
        # Always save final result
        result["efficiency"]["training_time_seconds"] = time.time() - start_time
        save_result(exp_id, result)

    return result


def parse_metrics_from_log(log_file: Path) -> Dict:
    """
    Parse metrics from training log file.

    Looks for PyTorch Lightning progress bar outputs and final test metrics.
    """
    metrics = {}

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Look for validation and test metrics
        # This is a simple parser - may need refinement based on actual log format
        for line in lines:
            # Example: "val/acc=0.85" or "test/f1_macro: 0.83"
            if 'val/acc' in line.lower():
                try:
                    # Extract number after 'val/acc'
                    parts = line.split('val/acc')
                    if len(parts) > 1:
                        val_str = parts[1].split()[0].strip('=:')
                        metrics['val_acc'] = float(val_str)
                except (ValueError, IndexError):
                    pass

            if 'test/acc' in line.lower():
                try:
                    parts = line.split('test/acc')
                    if len(parts) > 1:
                        val_str = parts[1].split()[0].strip('=:')
                        metrics['test_acc'] = float(val_str)
                except (ValueError, IndexError):
                    pass

            if 'val/f1_macro' in line.lower():
                try:
                    parts = line.split('val/f1_macro')
                    if len(parts) > 1:
                        val_str = parts[1].split()[0].strip('=:')
                        metrics['val_f1_macro'] = float(val_str)
                except (ValueError, IndexError):
                    pass

            if 'test/f1_macro' in line.lower():
                try:
                    parts = line.split('test/f1_macro')
                    if len(parts) > 1:
                        val_str = parts[1].split()[0].strip('=:')
                        metrics['test_f1_macro'] = float(val_str)
                except (ValueError, IndexError):
                    pass

            # Parse parameter counts
            if 'trainable params' in line.lower():
                try:
                    # Example: "LoRA applied: 86,219,560 -> 442,368 trainable params"
                    if '->' in line:
                        parts = line.split('->')
                        trainable_str = parts[1].split('trainable')[0].strip().replace(',', '')
                        metrics['trainable_params'] = int(trainable_str)

                        # Get total params from before ->
                        total_str = parts[0].split(':')[-1].strip().replace(',', '')
                        metrics['total_params'] = int(total_str)
                except (ValueError, IndexError):
                    pass

    except Exception as e:
        logging.warning(f"Failed to parse metrics from {log_file}: {e}")

    return metrics


def run_sweep(
    models: List[str],
    strategies: List[str],
    data: str,
    max_epochs: int,
    augmentation: Optional[List[bool]] = None,
    resume: bool = False,
    force: bool = False,
    timeout: int = 7200,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run systematic sweep of all experiment configurations.

    Args:
        models: List of models to test
        strategies: List of PEFT strategies to test
        data: Dataset size
        max_epochs: Maximum training epochs
        augmentation: List of augmentation settings
        resume: Skip already-completed experiments
        force: Re-run all experiments even if they exist
        timeout: Per-experiment timeout in seconds

    Returns:
        Tuple of (completed_experiments, failed_experiments)
    """
    # Generate all experiments
    experiments = generate_experiments(models, strategies, data, max_epochs, augmentation)

    total = len(experiments)
    logging.info(f"Generated {total} experiments")
    logging.info(f"Models: {models}")
    logging.info(f"Strategies: {strategies}")
    logging.info(f"Augmentation: {augmentation}")
    logging.info(f"Data: {data}, Max epochs: {max_epochs}")

    completed = []
    failed = []
    skipped = 0

    for i, config in enumerate(experiments, 1):
        exp_id = get_experiment_id(config)

        logging.info(f"\n{'='*80}")
        logging.info(f"Experiment {i}/{total}: {exp_id}")
        logging.info(f"{'='*80}")

        # Check if experiment already exists
        if resume and not force:
            existing = check_experiment_exists(config)
            if existing:
                result = load_result(existing)
                if result and result.get("status") == "completed":
                    logging.info(f"Skipping - already completed: {existing}")
                    completed.append(result)
                    skipped += 1
                    continue

        # Run experiment
        start = time.time()
        result = run_experiment(config, exp_id, timeout)
        duration = time.time() - start

        if result["status"] == "completed":
            completed.append(result)
            logging.info(f"Completed in {duration:.1f}s")
        else:
            failed.append(result)
            logging.error(f"Failed after {duration:.1f}s: {result.get('error', 'Unknown error')}")

        # Progress summary
        remaining = total - i
        avg_time = duration  # Simple estimate
        eta_seconds = remaining * avg_time
        eta_mins = eta_seconds / 60

        logging.info(f"\nProgress: {i}/{total} ({100*i/total:.1f}%)")
        logging.info(f"Completed: {len(completed)}, Failed: {len(failed)}, Skipped: {skipped}")
        logging.info(f"Estimated time remaining: {eta_mins:.1f} minutes")

    # Final summary
    logging.info(f"\n{'='*80}")
    logging.info("SWEEP COMPLETE")
    logging.info(f"{'='*80}")
    logging.info(f"Total experiments: {total}")
    logging.info(f"Completed: {len(completed)}")
    logging.info(f"Failed: {len(failed)}")
    logging.info(f"Skipped: {skipped}")
    logging.info(f"Results saved to: {RESULTS_DIR}")

    if failed:
        logging.warning("\nFailed experiments:")
        for result in failed:
            logging.warning(f"  - {result['experiment_id']}: {result.get('error', 'Unknown error')}")

    return completed, failed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run systematic model comparison sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run tiny sweep (smoke test)
  python scripts/run_sweep.py --data tiny --max-epochs 5

  # Run small sweep (real comparison)
  python scripts/run_sweep.py --data small --max-epochs 20

  # Resume interrupted sweep
  python scripts/run_sweep.py --data tiny --resume

  # Run specific subset
  python scripts/run_sweep.py --models ast htsat --strategies lora

  # Force re-run all experiments
  python scripts/run_sweep.py --data tiny --force
        """
    )

    parser.add_argument(
        "--data",
        type=str,
        default="tiny",
        choices=["tiny", "small", "full"],
        help="Dataset size to use (default: tiny)"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Maximum training epochs (default: 5)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help=f"Models to test (default: all)"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=STRATEGIES,
        choices=STRATEGIES,
        help=f"PEFT strategies to test (default: all)"
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        nargs="+",
        choices=["true", "false"],
        default=None,
        help="Augmentation settings to test (default: both true and false)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed experiments"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all experiments even if they exist"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Per-experiment timeout in seconds (default: 7200 = 2 hours)"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path"
    )

    args = parser.parse_args()

    # Parse augmentation argument
    aug_list = None
    if args.augmentation:
        aug_list = [a.lower() == "true" for a in args.augmentation]

    # Set up logging
    setup_logging(args.log_file)

    # Run sweep
    completed, failed = run_sweep(
        models=args.models,
        strategies=args.strategies,
        data=args.data,
        max_epochs=args.max_epochs,
        augmentation=aug_list,
        resume=args.resume,
        force=args.force,
        timeout=args.timeout,
    )

    # Exit with error code if any experiments failed
    sys.exit(len(failed))


if __name__ == "__main__":
    main()
