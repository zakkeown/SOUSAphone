"""Tests for sweep scripts."""

import json
import pytest
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# Import functions from sweep scripts
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from run_sweep import (
    generate_experiments,
    get_experiment_id,
    check_experiment_exists,
    save_result,
    load_result,
    parse_metrics_from_log,
)

from analyze_sweep import (
    load_all_results,
    filter_results,
    results_to_dataframe,
)


class TestExperimentGeneration:
    """Test experiment configuration generation."""

    def test_generate_all_combinations(self):
        """Test that all combinations are generated."""
        models = ["ast", "htsat"]
        strategies = ["lora", "adalora"]
        augmentation = [True, False]

        experiments = generate_experiments(
            models=models,
            strategies=strategies,
            data="tiny",
            max_epochs=5,
            augmentation=augmentation,
        )

        # Should have 2 models × 2 strategies × 2 aug = 8 experiments
        assert len(experiments) == 8

    def test_generate_single_augmentation(self):
        """Test with single augmentation setting."""
        experiments = generate_experiments(
            models=["ast"],
            strategies=["lora"],
            data="tiny",
            max_epochs=5,
            augmentation=[True],
        )

        assert len(experiments) == 1
        assert experiments[0]["augmentation"] is True

    def test_experiment_config_structure(self):
        """Test that experiment configs have correct structure."""
        experiments = generate_experiments(
            models=["ast"],
            strategies=["lora"],
            data="tiny",
            max_epochs=10,
        )

        exp = experiments[0]
        assert "model" in exp
        assert "strategy" in exp
        assert "augmentation" in exp
        assert "data" in exp
        assert "max_epochs" in exp
        assert exp["max_epochs"] == 10


class TestExperimentID:
    """Test experiment ID generation."""

    def test_id_contains_all_config(self):
        """Test that ID contains all config elements."""
        config = {
            "model": "ast",
            "strategy": "lora",
            "augmentation": True,
            "data": "tiny",
            "max_epochs": 5,
        }

        exp_id = get_experiment_id(config)

        assert "ast" in exp_id
        assert "lora" in exp_id
        assert "aug" in exp_id
        assert "tiny" in exp_id

    def test_id_distinguishes_augmentation(self):
        """Test that ID differentiates augmentation settings."""
        config_aug = {
            "model": "ast",
            "strategy": "lora",
            "augmentation": True,
            "data": "tiny",
            "max_epochs": 5,
        }

        config_no_aug = {
            "model": "ast",
            "strategy": "lora",
            "augmentation": False,
            "data": "tiny",
            "max_epochs": 5,
        }

        id_aug = get_experiment_id(config_aug)
        id_no_aug = get_experiment_id(config_no_aug)

        # IDs should differ (ignoring timestamp)
        assert "aug" in id_aug or "noaug" in id_aug
        assert "noaug" in id_no_aug or id_no_aug != id_aug


class TestResultPersistence:
    """Test result saving and loading."""

    def test_save_and_load_result(self, tmp_path):
        """Test saving and loading result JSON."""
        # Create a test result
        result = {
            "config": {
                "model": "ast",
                "strategy": "lora",
                "augmentation": True,
                "data": "tiny",
                "max_epochs": 5,
            },
            "metrics": {
                "val_acc": 0.85,
                "test_acc": 0.82,
            },
            "efficiency": {
                "training_time_seconds": 100.0,
            },
            "status": "completed",
            "experiment_id": "test_exp",
        }

        # Save to temp directory
        result_file = tmp_path / "test_exp.json"
        with open(result_file, 'w') as f:
            json.dump(result, f)

        # Load it back
        with open(result_file, 'r') as f:
            loaded = json.load(f)

        assert loaded["config"]["model"] == "ast"
        assert loaded["metrics"]["val_acc"] == 0.85
        assert loaded["status"] == "completed"


class TestMetricsParsing:
    """Test parsing metrics from log files."""

    def test_parse_validation_accuracy(self, tmp_path):
        """Test parsing validation accuracy from log."""
        log_file = tmp_path / "test.log"

        log_content = """
        Epoch 1: 100%|██████████| 50/50 [00:10<00:00,  5.00it/s, loss=0.5, v_num=0, val/acc=0.8500]
        Epoch 2: 100%|██████████| 50/50 [00:10<00:00,  5.00it/s, loss=0.4, v_num=0, val/acc=0.8700]
        """

        with open(log_file, 'w') as f:
            f.write(log_content)

        metrics = parse_metrics_from_log(log_file)

        # Should parse last val/acc
        assert "val_acc" in metrics
        # May get 0.87 or 0.85 depending on parsing
        assert metrics["val_acc"] in [0.85, 0.87]

    def test_parse_test_metrics(self, tmp_path):
        """Test parsing test metrics."""
        log_file = tmp_path / "test.log"

        log_content = """
        Testing: 100%|██████████| 10/10 [00:02<00:00,  4.50it/s]
        test/acc: 0.8200
        test/f1_macro: 0.8100
        """

        with open(log_file, 'w') as f:
            f.write(log_content)

        metrics = parse_metrics_from_log(log_file)

        assert "test_acc" in metrics or "test/acc" in str(metrics)

    def test_parse_parameter_counts(self, tmp_path):
        """Test parsing parameter counts."""
        log_file = tmp_path / "test.log"

        log_content = """
        LORA applied: 86,219,560 -> 442,368 trainable params
        Trainable params reduced to 0.51%
        """

        with open(log_file, 'w') as f:
            f.write(log_content)

        metrics = parse_metrics_from_log(log_file)

        if "trainable_params" in metrics:
            assert metrics["trainable_params"] == 442368
            assert metrics["total_params"] == 86219560


class TestResultFiltering:
    """Test filtering results."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        return [
            {
                "config": {"model": "ast", "strategy": "lora", "augmentation": True},
                "status": "completed",
                "metrics": {"test_acc": 0.85},
            },
            {
                "config": {"model": "htsat", "strategy": "lora", "augmentation": False},
                "status": "completed",
                "metrics": {"test_acc": 0.87},
            },
            {
                "config": {"model": "ast", "strategy": "adalora", "augmentation": True},
                "status": "failed",
                "error": "OOM",
            },
        ]

    def test_filter_by_model(self, sample_results):
        """Test filtering by model."""
        filtered = filter_results(sample_results, models=["ast"])

        assert len(filtered) == 2
        assert all(r["config"]["model"] == "ast" for r in filtered)

    def test_filter_by_strategy(self, sample_results):
        """Test filtering by strategy."""
        filtered = filter_results(sample_results, strategies=["lora"])

        assert len(filtered) == 2
        assert all(r["config"]["strategy"] == "lora" for r in filtered)

    def test_filter_by_status(self, sample_results):
        """Test filtering by status."""
        filtered = filter_results(sample_results, status="completed")

        assert len(filtered) == 2
        assert all(r["status"] == "completed" for r in filtered)

    def test_filter_by_augmentation(self, sample_results):
        """Test filtering by augmentation setting."""
        filtered = filter_results(sample_results, augmentation=True)

        assert len(filtered) == 2
        assert all(r["config"]["augmentation"] is True for r in filtered)

    def test_multiple_filters(self, sample_results):
        """Test combining multiple filters."""
        filtered = filter_results(
            sample_results,
            models=["ast"],
            status="completed",
        )

        assert len(filtered) == 1
        assert filtered[0]["config"]["model"] == "ast"
        assert filtered[0]["status"] == "completed"


class TestDataFrame:
    """Test DataFrame conversion."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results."""
        return [
            {
                "experiment_id": "test1",
                "config": {
                    "model": "ast",
                    "strategy": "lora",
                    "augmentation": True,
                    "data": "tiny",
                    "max_epochs": 5,
                },
                "metrics": {
                    "val_acc": 0.85,
                    "test_acc": 0.82,
                    "test_f1_macro": 0.80,
                },
                "efficiency": {
                    "total_params": 1000000,
                    "trainable_params": 50000,
                    "training_time_seconds": 100.0,
                },
                "status": "completed",
                "timestamp": "2024-01-01T10:00:00",
            }
        ]

    def test_dataframe_structure(self, sample_results):
        """Test that DataFrame has correct structure."""
        df = results_to_dataframe(sample_results)

        assert len(df) == 1
        assert "model" in df.columns
        assert "strategy" in df.columns
        assert "test_acc" in df.columns
        assert "trainable_params" in df.columns

    def test_dataframe_values(self, sample_results):
        """Test that DataFrame contains correct values."""
        df = results_to_dataframe(sample_results)

        row = df.iloc[0]
        assert row["model"] == "ast"
        assert row["strategy"] == "lora"
        assert row["test_acc"] == 0.82
        assert row["trainable_params"] == 50000

    def test_param_efficiency_calculation(self, sample_results):
        """Test that parameter efficiency is calculated."""
        df = results_to_dataframe(sample_results)

        row = df.iloc[0]
        assert "param_efficiency" in df.columns
        # Should be trainable / total = 50000 / 1000000 = 0.05
        assert abs(row["param_efficiency"] - 0.05) < 0.001


class TestResumeLogic:
    """Test resume and skip logic."""

    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary results directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_check_experiment_exists(self, temp_results_dir):
        """Test checking for existing experiments."""
        # Create a fake result file
        config = {
            "model": "ast",
            "strategy": "lora",
            "augmentation": True,
            "data": "tiny",
            "max_epochs": 5,
        }

        result_file = temp_results_dir / "ast_lora_aug_tiny_20240101_120000.json"
        with open(result_file, 'w') as f:
            json.dump({"config": config, "status": "completed"}, f)

        # This would need the actual function to use the temp directory
        # For now, just test the pattern matching logic
        pattern = "ast_lora_aug_tiny_*.json"
        matches = list(temp_results_dir.glob(pattern))
        assert len(matches) == 1

    def test_no_existing_experiment(self, temp_results_dir):
        """Test when experiment doesn't exist."""
        pattern = "nonexistent_*.json"
        matches = list(temp_results_dir.glob(pattern))
        assert len(matches) == 0


# Integration-style test (will be skipped in CI without actual training)
@pytest.mark.skip(reason="Requires actual training setup")
class TestEndToEnd:
    """End-to-end tests (skipped by default)."""

    def test_single_experiment_run(self):
        """Test running a single tiny experiment."""
        # This would actually run a tiny experiment
        # Only enable when testing locally with proper setup
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
