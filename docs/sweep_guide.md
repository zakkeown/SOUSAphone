# Systematic Model Comparison Sweep Guide

This guide explains how to run systematic model comparison sweeps to identify the best-performing model and PEFT strategy combinations for SOUSA rudiment classification.

## Overview

The sweep system runs all combinations of:
- **Models**: AST, HTS-AT, BEaTS, EfficientAT (4 models)
- **PEFT Strategies**: LoRA, AdaLoRA, IA3 (3 strategies)
- **Augmentation**: With and without augmentation (2 settings)

This produces **24 total experiments** (4 × 3 × 2).

## Quick Start

### 1. Run a Tiny Sweep (Smoke Test)

Test that everything works with a small dataset and few epochs:

```bash
python scripts/run_sweep.py --data tiny --max-epochs 5
```

This completes in ~30-60 minutes and helps catch configuration issues early.

### 2. Run a Small Sweep (Real Comparison)

Run a more meaningful comparison with more data:

```bash
python scripts/run_sweep.py --data small --max-epochs 20
```

This may take several hours (2-8 hours depending on hardware).

### 3. Analyze Results

Generate comparison tables, plots, and summary report:

```bash
python scripts/analyze_sweep.py
```

This creates:
- `results/sweep/comparison_table.csv` - All metrics in table format
- `results/sweep/plots/accuracy_comparison.png` - Accuracy bar chart
- `results/sweep/plots/efficiency_scatter.png` - Parameters vs accuracy
- `results/sweep/plots/training_time.png` - Training time comparison
- `results/sweep/summary_report.md` - Human-readable summary

## Usage Examples

### Run Specific Subset

Test only certain models:
```bash
python scripts/run_sweep.py --models ast htsat --data tiny --max-epochs 5
```

Test only certain strategies:
```bash
python scripts/run_sweep.py --strategies lora adalora --data tiny --max-epochs 5
```

Test only with augmentation:
```bash
python scripts/run_sweep.py --augmentation true --data tiny --max-epochs 5
```

### Resume Interrupted Sweep

If a sweep is interrupted (power loss, timeout, etc.), resume it:

```bash
python scripts/run_sweep.py --data tiny --resume
```

This will skip already-completed experiments and continue from where it left off.

### Force Re-run All Experiments

To re-run all experiments even if they already exist:

```bash
python scripts/run_sweep.py --data tiny --force
```

### Increase Per-Experiment Timeout

If experiments are timing out (default: 2 hours), increase the timeout:

```bash
python scripts/run_sweep.py --data small --timeout 14400  # 4 hours
```

## Result Format

Each experiment saves a JSON file with the following structure:

```json
{
  "experiment_id": "ast_lora_aug_tiny_20260201_103000",
  "config": {
    "model": "ast",
    "strategy": "lora",
    "augmentation": true,
    "data": "tiny",
    "max_epochs": 5
  },
  "metrics": {
    "val_acc": 0.85,
    "val_f1_macro": 0.83,
    "test_acc": 0.82,
    "test_f1_macro": 0.80,
    "best_epoch": 3,
    "final_train_loss": 0.45,
    "final_val_loss": 0.52
  },
  "efficiency": {
    "total_params": 86219560,
    "trainable_params": 442368,
    "param_efficiency": 0.0051,
    "training_time_seconds": 1234.5,
    "avg_epoch_time": 246.9
  },
  "status": "completed",
  "error": null,
  "timestamp": "2026-02-01T10:30:00",
  "git_commit": "abc123def"
}
```

## Analyzing Results

### Basic Analysis

Run the analyzer on all results:

```bash
python scripts/analyze_sweep.py
```

### Filter Results

Compare only specific models:
```bash
python scripts/analyze_sweep.py --filter-models ast htsat
```

Compare only specific strategies:
```bash
python scripts/analyze_sweep.py --filter-strategies lora ia3
```

Filter by augmentation:
```bash
python scripts/analyze_sweep.py --filter-augmentation true
```

### Skip Plots

Generate only the summary report (faster):
```bash
python scripts/analyze_sweep.py --no-plots
```

### Custom Output Directory

Save analysis to a different location:
```bash
python scripts/analyze_sweep.py --output-dir /path/to/custom/output
```

## Interpreting Results

### Top Performers by Accuracy

The summary report lists the top 5 experiments by test accuracy:

```markdown
### By Test Accuracy

1. **ast + lora** (with aug): Acc=0.8542, F1=0.8401
2. **htsat + adalora** (with aug): Acc=0.8501, F1=0.8389
3. **ast + ia3** (no aug): Acc=0.8456, F1=0.8312
...
```

This tells you which combination achieved the highest accuracy.

### Parameter Efficiency

The "Most Parameter Efficient" section shows which models achieve good accuracy with the fewest trainable parameters:

```markdown
### Most Parameter Efficient

1. **efficientat + ia3** (with aug): 125,000 params, Acc=0.8234
2. **ast + ia3** (no aug): 342,000 params, Acc=0.8456
...
```

Lower parameter count means:
- Faster training
- Less memory usage
- Easier to deploy

### Training Time

The "Fastest Training" section shows which experiments completed quickest:

```markdown
### Fastest Training

1. **efficientat + lora** (with aug): 45.3m, Acc=0.8123
2. **ast + ia3** (no aug): 52.7m, Acc=0.8456
...
```

This helps identify efficient options for rapid iteration.

### Statistics by Model/Strategy

The report includes aggregate statistics:

```markdown
## Statistics by Model

- **ast**: Avg Acc=0.8401 ± 0.0123, Best=0.8542 (6 runs)
- **htsat**: Avg Acc=0.8356 ± 0.0098, Best=0.8501 (6 runs)
- **beats**: Avg Acc=0.8289 ± 0.0156, Best=0.8445 (6 runs)
- **efficientat**: Avg Acc=0.8201 ± 0.0087, Best=0.8345 (6 runs)
```

This shows:
- Which models are most consistent (lower std dev)
- Which have highest potential (best score)
- Average performance across strategies

### Augmentation Impact

The report compares augmentation vs. no augmentation:

```markdown
## Augmentation Impact

- **With augmentation**: Avg Acc=0.8412 (12 runs)
- **Without augmentation**: Avg Acc=0.8289 (12 runs)
```

This quantifies how much augmentation helps (or hurts).

## Best Practices

### 1. Start Small

Always start with a tiny sweep to catch issues:
- Configuration errors
- Import problems
- GPU memory issues
- Data loading bugs

### 2. Use Resume

Long sweeps can be interrupted. Always use `--resume` to avoid wasting computation:

```bash
python scripts/run_sweep.py --data small --max-epochs 20 --resume
```

### 3. Monitor Progress

The sweep script logs progress:
```
Experiment 5/24: htsat_lora_aug_tiny_20260201_103000
Progress: 5/24 (20.8%)
Completed: 4, Failed: 1, Skipped: 0
Estimated time remaining: 45.2 minutes
```

Watch for failed experiments and investigate errors early.

### 4. Check Error Logs

Failed experiments write detailed logs to `results/sweep/errors/`:

```bash
cat results/sweep/errors/ast_lora_aug_tiny_20260201_103000_error.log
```

Common issues:
- Out of memory (reduce batch size)
- Missing dependencies
- Data path issues
- Timeout (increase with `--timeout`)

### 5. Analyze Incrementally

You don't need to wait for the entire sweep to finish. Analyze partial results:

```bash
# After 10/24 experiments complete
python scripts/analyze_sweep.py
```

This helps identify issues or promising configurations early.

### 6. Use W&B Offline Mode

The sweep runs with W&B in offline mode to avoid overwhelming the dashboard. To sync results later:

```bash
wandb sync wandb/offline-*
```

## Troubleshooting

### Out of Memory Errors

If experiments fail with OOM errors:

1. Reduce batch size in `configs/config.yaml`:
   ```yaml
   training:
     batch_size: 8  # Reduce from 16
   ```

2. Reduce max epochs for memory-intensive models:
   ```bash
   python scripts/run_sweep.py --data tiny --max-epochs 3
   ```

3. Enable gradient checkpointing (edit model configs)

### Experiments Timing Out

If experiments exceed the 2-hour timeout:

1. Increase timeout:
   ```bash
   python scripts/run_sweep.py --timeout 14400  # 4 hours
   ```

2. Reduce epochs:
   ```bash
   python scripts/run_sweep.py --max-epochs 10
   ```

3. Use smaller dataset:
   ```bash
   python scripts/run_sweep.py --data tiny
   ```

### Missing Metrics in Results

If some metrics are missing from the summary:

1. Check the experiment log:
   ```bash
   cat results/sweep/errors/experiment_id.log
   ```

2. Verify the metric parsing in `run_sweep.py:parse_metrics_from_log()`

3. The training may have completed but metric parsing failed (non-critical)

### No Plots Generated

If `analyze_sweep.py` doesn't generate plots:

1. Check that you have matplotlib and seaborn installed:
   ```bash
   pip install matplotlib seaborn
   ```

2. Verify completed experiments exist:
   ```bash
   python scripts/analyze_sweep.py --filter-status completed
   ```

3. Run without plots to still get the report:
   ```bash
   python scripts/analyze_sweep.py --no-plots
   ```

### Different Results on Re-run

If re-running the same configuration gives different results:

1. Check that the seed is set consistently (it should be)
2. Some variation is normal due to:
   - GPU non-determinism
   - Data loading order
   - Random augmentation

3. To quantify variation, run multiple trials:
   ```bash
   # Run same config 3 times manually and compare
   ```

## Advanced Usage

### Custom Experiment Matrix

Edit `scripts/run_sweep.py` to define custom experiment combinations:

```python
# In run_sweep.py
MODELS = ["ast", "htsat"]  # Remove models you don't want
STRATEGIES = ["lora"]  # Test only LoRA
AUGMENTATION = [True]  # Only with augmentation
```

### Extract Specific Metrics

Query the comparison table for specific information:

```python
import pandas as pd

df = pd.read_csv('results/sweep/comparison_table.csv')

# Get best by F1 score
best_f1 = df.loc[df['test_f1_macro'].idxmax()]
print(f"Best F1: {best_f1['model']} + {best_f1['strategy']}")

# Get most efficient
efficient = df[df['test_acc'] > 0.8].nsmallest(5, 'trainable_params')
print(efficient[['model', 'strategy', 'trainable_params', 'test_acc']])
```

### Parallel Execution

The current implementation runs experiments sequentially. To run in parallel (advanced):

1. Split experiments into chunks
2. Run multiple sweep processes with different subsets
3. Combine results with `analyze_sweep.py`

Example:
```bash
# Terminal 1
python scripts/run_sweep.py --models ast htsat --data tiny

# Terminal 2
python scripts/run_sweep.py --models beats efficientat --data tiny

# After both complete
python scripts/analyze_sweep.py
```

## Next Steps

After completing a sweep:

1. **Identify top performers** - Use summary report to find best configs
2. **Run longer training** - Take top 3-5 configs and train on full dataset
3. **Fine-tune hyperparameters** - Adjust learning rate, rank, etc. for top configs
4. **Cross-validate** - Run multiple seeds to verify stability
5. **Deploy best model** - Package and serve the winner

## References

- [Train.py Documentation](../README.md)
- [Model Configuration Guide](model_configs.md)
- [PEFT Strategy Guide](peft_strategies.md)
- [Augmentation Guide](augmentation.md)
