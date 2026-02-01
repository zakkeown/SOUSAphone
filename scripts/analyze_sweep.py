#!/usr/bin/env python3
"""
Analyze sweep results and generate comparison reports.

Loads all experiment results, generates tables and plots,
identifies top performers, and creates summary reports.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Default paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "sweep"
PLOTS_DIR = RESULTS_DIR / "plots"


def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def load_all_results(results_dir: Path) -> List[Dict]:
    """
    Load all experiment results from JSON files.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        List of result dictionaries
    """
    results = []

    json_files = list(results_dir.glob("*.json"))
    logging.info(f"Found {len(json_files)} result files in {results_dir}")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            logging.warning(f"Failed to load {json_file}: {e}")

    return results


def filter_results(
    results: List[Dict],
    models: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
    augmentation: Optional[bool] = None,
    status: Optional[str] = None,
) -> List[Dict]:
    """
    Filter results based on configuration parameters.

    Args:
        results: List of all results
        models: Filter by model names
        strategies: Filter by PEFT strategies
        augmentation: Filter by augmentation setting
        status: Filter by status (completed, failed, running)

    Returns:
        Filtered list of results
    """
    filtered = results

    if models:
        filtered = [r for r in filtered if r['config']['model'] in models]

    if strategies:
        filtered = [r for r in filtered if r['config']['strategy'] in strategies]

    if augmentation is not None:
        filtered = [r for r in filtered if r['config']['augmentation'] == augmentation]

    if status:
        filtered = [r for r in filtered if r.get('status') == status]

    return filtered


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """
    Convert results list to pandas DataFrame.

    Args:
        results: List of result dictionaries

    Returns:
        DataFrame with flattened results
    """
    rows = []

    for result in results:
        row = {
            'experiment_id': result.get('experiment_id', 'unknown'),
            'model': result['config']['model'],
            'strategy': result['config']['strategy'],
            'augmentation': result['config']['augmentation'],
            'data': result['config']['data'],
            'max_epochs': result['config']['max_epochs'],
            'status': result.get('status', 'unknown'),
            'timestamp': result.get('timestamp', ''),
        }

        # Add metrics
        metrics = result.get('metrics', {})
        row.update({
            'val_acc': metrics.get('val_acc'),
            'val_f1_macro': metrics.get('val_f1_macro'),
            'test_acc': metrics.get('test_acc'),
            'test_f1_macro': metrics.get('test_f1_macro'),
            'best_epoch': metrics.get('best_epoch'),
            'final_train_loss': metrics.get('final_train_loss'),
            'final_val_loss': metrics.get('final_val_loss'),
        })

        # Add efficiency metrics
        efficiency = result.get('efficiency', {})
        row.update({
            'total_params': efficiency.get('total_params'),
            'trainable_params': efficiency.get('trainable_params'),
            'param_efficiency': efficiency.get('param_efficiency'),
            'training_time_seconds': efficiency.get('training_time_seconds'),
            'avg_epoch_time': efficiency.get('avg_epoch_time'),
        })

        # Add error if failed
        if result.get('status') == 'failed':
            row['error'] = result.get('error', 'Unknown error')

        rows.append(row)

    df = pd.DataFrame(rows)

    # Calculate param efficiency if we have the numbers
    if 'trainable_params' in df.columns and 'total_params' in df.columns:
        df['param_efficiency'] = df['trainable_params'] / df['total_params'].replace(0, np.nan)

    return df


def save_comparison_table(df: pd.DataFrame, output_path: Path):
    """
    Save comparison table to CSV.

    Args:
        df: Results DataFrame
        output_path: Output CSV path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved comparison table to {output_path}")


def plot_accuracy_comparison(df: pd.DataFrame, output_path: Path):
    """
    Create bar chart comparing accuracies across experiments.

    Args:
        df: Results DataFrame
        output_path: Output PNG path
    """
    # Filter completed experiments with test_acc
    plot_df = df[df['status'] == 'completed'].copy()
    plot_df = plot_df.dropna(subset=['test_acc'])

    if len(plot_df) == 0:
        logging.warning("No completed experiments with test_acc to plot")
        return

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create experiment labels
    plot_df['experiment'] = (
        plot_df['model'] + '\n' +
        plot_df['strategy'] + '\n' +
        plot_df['augmentation'].map({True: 'aug', False: 'no-aug'})
    )

    # Sort by test_acc
    plot_df = plot_df.sort_values('test_acc', ascending=False)

    # Plot
    colors = sns.color_palette("husl", len(plot_df))
    bars = ax.bar(range(len(plot_df)), plot_df['test_acc'], color=colors)

    # Customize
    ax.set_xlabel('Experiment Configuration')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Model Comparison: Test Accuracy')
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df['experiment'], rotation=45, ha='right', fontsize=8)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, plot_df['test_acc'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved accuracy comparison to {output_path}")


def plot_efficiency_scatter(df: pd.DataFrame, output_path: Path):
    """
    Create scatter plot of parameter efficiency vs accuracy.

    Args:
        df: Results DataFrame
        output_path: Output PNG path
    """
    # Filter completed experiments with required metrics
    plot_df = df[df['status'] == 'completed'].copy()
    plot_df = plot_df.dropna(subset=['test_acc', 'trainable_params'])

    if len(plot_df) == 0:
        logging.warning("No completed experiments with efficiency metrics to plot")
        return

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by model, marker by strategy
    models = plot_df['model'].unique()
    strategies = plot_df['strategy'].unique()

    model_colors = dict(zip(models, sns.color_palette("Set1", len(models))))
    strategy_markers = dict(zip(strategies, ['o', 's', '^', 'D', 'v']))

    for model in models:
        for strategy in strategies:
            subset = plot_df[(plot_df['model'] == model) & (plot_df['strategy'] == strategy)]
            if len(subset) > 0:
                ax.scatter(
                    subset['trainable_params'],
                    subset['test_acc'],
                    c=[model_colors[model]],
                    marker=strategy_markers.get(strategy, 'o'),
                    s=100,
                    alpha=0.7,
                    label=f'{model} + {strategy}',
                    edgecolors='black',
                    linewidths=0.5
                )

    ax.set_xlabel('Trainable Parameters')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Efficiency vs Accuracy: Trainable Parameters vs Test Accuracy')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved efficiency scatter to {output_path}")


def plot_training_time(df: pd.DataFrame, output_path: Path):
    """
    Create bar chart comparing training times.

    Args:
        df: Results DataFrame
        output_path: Output PNG path
    """
    # Filter completed experiments with timing data
    plot_df = df[df['status'] == 'completed'].copy()
    plot_df = plot_df.dropna(subset=['training_time_seconds'])

    if len(plot_df) == 0:
        logging.warning("No completed experiments with timing data to plot")
        return

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create experiment labels
    plot_df['experiment'] = (
        plot_df['model'] + '\n' +
        plot_df['strategy'] + '\n' +
        plot_df['augmentation'].map({True: 'aug', False: 'no-aug'})
    )

    # Convert to minutes
    plot_df['training_time_minutes'] = plot_df['training_time_seconds'] / 60

    # Sort by time
    plot_df = plot_df.sort_values('training_time_minutes')

    # Plot
    colors = sns.color_palette("viridis", len(plot_df))
    bars = ax.bar(range(len(plot_df)), plot_df['training_time_minutes'], color=colors)

    # Customize
    ax.set_xlabel('Experiment Configuration')
    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('Model Comparison: Training Time')
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df['experiment'], rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, plot_df['training_time_minutes']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}m',
                ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved training time plot to {output_path}")


def generate_summary_report(df: pd.DataFrame, output_path: Path):
    """
    Generate human-readable markdown summary report.

    Args:
        df: Results DataFrame
        output_path: Output markdown file path
    """
    report_lines = [
        "# Sweep Results Summary",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "---\n",
        "## Overview\n",
        f"- Total experiments: {len(df)}",
        f"- Completed: {len(df[df['status'] == 'completed'])}",
        f"- Failed: {len(df[df['status'] == 'failed'])}",
        f"- Running: {len(df[df['status'] == 'running'])}",
        "",
    ]

    # Get completed experiments
    completed_df = df[df['status'] == 'completed'].copy()

    if len(completed_df) > 0:
        report_lines.extend([
            "## Top Performers\n",
            "### By Test Accuracy\n",
        ])

        # Top 5 by test accuracy
        top_acc = completed_df.nlargest(5, 'test_acc')
        for i, row in enumerate(top_acc.itertuples(), 1):
            aug_str = "with aug" if row.augmentation else "no aug"
            acc = row.test_acc if pd.notna(row.test_acc) else 0
            f1 = row.test_f1_macro if pd.notna(row.test_f1_macro) else 0
            report_lines.append(
                f"{i}. **{row.model} + {row.strategy}** ({aug_str}): "
                f"Acc={acc:.4f}, F1={f1:.4f}"
            )

        report_lines.append("")

        # Top 5 by F1 score
        if 'test_f1_macro' in completed_df.columns:
            top_f1 = completed_df.nlargest(5, 'test_f1_macro')
            report_lines.extend([
                "### By Test F1 (Macro)\n",
            ])
            for i, row in enumerate(top_f1.itertuples(), 1):
                aug_str = "with aug" if row.augmentation else "no aug"
                acc = row.test_acc if pd.notna(row.test_acc) else 0
                f1 = row.test_f1_macro if pd.notna(row.test_f1_macro) else 0
                report_lines.append(
                    f"{i}. **{row.model} + {row.strategy}** ({aug_str}): "
                    f"F1={f1:.4f}, Acc={acc:.4f}"
                )

        report_lines.append("")

        # Most efficient (fewest trainable params with good accuracy)
        if 'trainable_params' in completed_df.columns:
            # Filter for reasonable accuracy (> 0.5) and sort by params
            efficient_df = completed_df[completed_df['test_acc'] > 0.5].copy()
            if len(efficient_df) > 0:
                top_efficient = efficient_df.nsmallest(5, 'trainable_params')
                report_lines.extend([
                    "### Most Parameter Efficient\n",
                    "(Minimum 50% accuracy)\n",
                ])
                for i, row in enumerate(top_efficient.itertuples(), 1):
                    aug_str = "with aug" if row.augmentation else "no aug"
                    params = row.trainable_params if pd.notna(row.trainable_params) else 0
                    acc = row.test_acc if pd.notna(row.test_acc) else 0
                    report_lines.append(
                        f"{i}. **{row.model} + {row.strategy}** ({aug_str}): "
                        f"{params:,} params, Acc={acc:.4f}"
                    )

        report_lines.append("")

        # Fastest training
        if 'training_time_seconds' in completed_df.columns:
            top_fast = completed_df.nsmallest(5, 'training_time_seconds')
            report_lines.extend([
                "### Fastest Training\n",
            ])
            for i, row in enumerate(top_fast.itertuples(), 1):
                aug_str = "with aug" if row.augmentation else "no aug"
                time_min = row.training_time_seconds / 60 if pd.notna(row.training_time_seconds) else 0
                acc = row.test_acc if pd.notna(row.test_acc) else 0
                report_lines.append(
                    f"{i}. **{row.model} + {row.strategy}** ({aug_str}): "
                    f"{time_min:.1f}m, Acc={acc:.4f}"
                )

        report_lines.append("")

        # Statistics by model
        report_lines.extend([
            "## Statistics by Model\n",
        ])

        for model in sorted(completed_df['model'].unique()):
            model_df = completed_df[completed_df['model'] == model]
            avg_acc = model_df['test_acc'].mean()
            std_acc = model_df['test_acc'].std()
            best_acc = model_df['test_acc'].max()

            report_lines.append(
                f"- **{model}**: Avg Acc={avg_acc:.4f} ± {std_acc:.4f}, "
                f"Best={best_acc:.4f} ({len(model_df)} runs)"
            )

        report_lines.append("")

        # Statistics by strategy
        report_lines.extend([
            "## Statistics by PEFT Strategy\n",
        ])

        for strategy in sorted(completed_df['strategy'].unique()):
            strategy_df = completed_df[completed_df['strategy'] == strategy]
            avg_acc = strategy_df['test_acc'].mean()
            std_acc = strategy_df['test_acc'].std()
            best_acc = strategy_df['test_acc'].max()

            report_lines.append(
                f"- **{strategy}**: Avg Acc={avg_acc:.4f} ± {std_acc:.4f}, "
                f"Best={best_acc:.4f} ({len(strategy_df)} runs)"
            )

        report_lines.append("")

        # Augmentation impact
        if len(completed_df['augmentation'].unique()) > 1:
            report_lines.extend([
                "## Augmentation Impact\n",
            ])

            aug_df = completed_df[completed_df['augmentation'] == True]
            no_aug_df = completed_df[completed_df['augmentation'] == False]

            if len(aug_df) > 0:
                avg_aug = aug_df['test_acc'].mean()
                report_lines.append(f"- **With augmentation**: Avg Acc={avg_aug:.4f} ({len(aug_df)} runs)")

            if len(no_aug_df) > 0:
                avg_no_aug = no_aug_df['test_acc'].mean()
                report_lines.append(f"- **Without augmentation**: Avg Acc={avg_no_aug:.4f} ({len(no_aug_df)} runs)")

            report_lines.append("")

    # Failed experiments
    failed_df = df[df['status'] == 'failed']
    if len(failed_df) > 0:
        report_lines.extend([
            "## Failed Experiments\n",
        ])
        for row in failed_df.itertuples():
            aug_str = "with aug" if row.augmentation else "no aug"
            error = row.error if hasattr(row, 'error') and pd.notna(row.error) else "Unknown error"
            report_lines.append(
                f"- **{row.model} + {row.strategy}** ({aug_str}): {error}"
            )

        report_lines.append("")

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logging.info(f"Saved summary report to {output_path}")

    # Also print to console
    print("\n" + '\n'.join(report_lines))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze sweep results and generate reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all results
  python scripts/analyze_sweep.py

  # Compare specific runs
  python scripts/analyze_sweep.py --filter-models ast htsat

  # Generate report only (no plots)
  python scripts/analyze_sweep.py --no-plots

  # Specify custom results directory
  python scripts/analyze_sweep.py --results-dir /path/to/results
        """
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory containing result JSON files (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--filter-models",
        type=str,
        nargs="+",
        default=None,
        help="Filter by model names"
    )
    parser.add_argument(
        "--filter-strategies",
        type=str,
        nargs="+",
        default=None,
        help="Filter by PEFT strategies"
    )
    parser.add_argument(
        "--filter-augmentation",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Filter by augmentation setting"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots and reports (default: same as results-dir)"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Determine output directory
    output_dir = args.output_dir if args.output_dir else args.results_dir

    # Load results
    logging.info(f"Loading results from {args.results_dir}")
    results = load_all_results(args.results_dir)

    if len(results) == 0:
        logging.error("No results found!")
        sys.exit(1)

    logging.info(f"Loaded {len(results)} results")

    # Apply filters
    aug_filter = None
    if args.filter_augmentation:
        aug_filter = args.filter_augmentation.lower() == "true"

    filtered_results = filter_results(
        results,
        models=args.filter_models,
        strategies=args.filter_strategies,
        augmentation=aug_filter,
    )

    logging.info(f"After filtering: {len(filtered_results)} results")

    if len(filtered_results) == 0:
        logging.error("No results after filtering!")
        sys.exit(1)

    # Convert to DataFrame
    df = results_to_dataframe(filtered_results)

    # Save comparison table
    table_path = output_dir / "comparison_table.csv"
    save_comparison_table(df, table_path)

    # Generate plots
    if not args.no_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_accuracy_comparison(df, plots_dir / "accuracy_comparison.png")
        plot_efficiency_scatter(df, plots_dir / "efficiency_scatter.png")
        plot_training_time(df, plots_dir / "training_time.png")

    # Generate summary report
    report_path = output_dir / "summary_report.md"
    generate_summary_report(df, report_path)

    logging.info("\nAnalysis complete!")
    logging.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
