#  Copyright 2025 mama-sia-eval contributors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Tests for the training visualisation module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mama_sia_eval.training_visualization import TrainingVisualizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def viz_dir(tmp_path):
    """Provide a temporary output directory for visualisations."""
    return tmp_path / "viz_output"


@pytest.fixture
def binary_data():
    """Generate synthetic binary classification data."""
    rng = np.random.RandomState(42)
    n = 100
    y_true = rng.randint(0, 2, size=n).astype(np.int64)
    y_score = rng.rand(n).astype(np.float64)
    # Make scores somewhat correlated with truth
    y_score = np.clip(y_score + 0.3 * y_true, 0, 1)
    y_pred = (y_score >= 0.5).astype(np.int64)
    return y_true, y_pred, y_score


@pytest.fixture
def trained_rf():
    """Create a trained RandomForest for feature importance tests."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(42)
    X = rng.randn(50, 10)
    y = (X[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Tests: TrainingVisualizer
# ---------------------------------------------------------------------------


class TestTrainingVisualizer:
    def test_init_creates_directory(self, viz_dir):
        viz = TrainingVisualizer(output_dir=viz_dir)
        assert viz_dir.exists()

    def test_confusion_matrix(self, viz_dir, binary_data):
        y_true, y_pred, _ = binary_data
        viz = TrainingVisualizer(output_dir=viz_dir)
        paths = viz.confusion_matrix(y_true, y_pred, task="tnbc")

        # Should produce at least: raw CM png, normalised CM png, JSON
        assert len(paths) >= 3
        assert any(p.suffix == ".png" for p in paths)
        assert any(p.suffix == ".json" for p in paths)

        # Check PNG exists
        cm_png = viz_dir / "confusion_matrix_tnbc.png"
        assert cm_png.exists()

        # Check JSON content
        cm_json_path = viz_dir / "confusion_matrix_tnbc.json"
        assert cm_json_path.exists()
        with open(cm_json_path) as f:
            cm_data = json.load(f)
        assert "matrix" in cm_data
        assert "tp" in cm_data
        assert "tn" in cm_data
        assert "fp" in cm_data
        assert "fn" in cm_data
        assert cm_data["task"] == "tnbc"

    def test_confusion_matrix_class_names(self, viz_dir, binary_data):
        y_true, y_pred, _ = binary_data
        viz = TrainingVisualizer(output_dir=viz_dir)
        paths = viz.confusion_matrix(
            y_true, y_pred, task="custom",
            class_names=["ClassA", "ClassB"],
        )
        assert len(paths) >= 1

    def test_roc_curve(self, viz_dir, binary_data):
        y_true, _, y_score = binary_data
        viz = TrainingVisualizer(output_dir=viz_dir)
        paths = viz.roc_curve(y_true, y_score, task="luminal")

        assert len(paths) == 1
        roc_path = viz_dir / "roc_curve_luminal.png"
        assert roc_path.exists()

    def test_roc_curve_single_class(self, viz_dir):
        """ROC should be skipped when only one class is present."""
        y_true = np.zeros(20, dtype=np.int64)
        y_score = np.random.rand(20)
        viz = TrainingVisualizer(output_dir=viz_dir)
        paths = viz.roc_curve(y_true, y_score, task="tnbc")
        assert len(paths) == 0

    def test_precision_recall_curve(self, viz_dir, binary_data):
        y_true, _, y_score = binary_data
        viz = TrainingVisualizer(output_dir=viz_dir)
        paths = viz.precision_recall_curve(y_true, y_score, task="tnbc")

        assert len(paths) == 1
        pr_path = viz_dir / "pr_curve_tnbc.png"
        assert pr_path.exists()

    def test_feature_importance(self, viz_dir, trained_rf):
        viz = TrainingVisualizer(output_dir=viz_dir)
        paths = viz.feature_importance(
            trained_rf, task="tnbc",
            feature_names=[f"feat_{i}" for i in range(10)],
            top_k=5,
        )
        assert len(paths) == 1
        assert (viz_dir / "feature_importance_tnbc.png").exists()

    def test_feature_importance_no_attr(self, viz_dir):
        """Models without feature_importances_ should be handled."""
        from unittest.mock import MagicMock
        model = MagicMock(spec=[])
        del model.feature_importances_
        viz = TrainingVisualizer(output_dir=viz_dir)
        paths = viz.feature_importance(model, task="tnbc")
        assert len(paths) == 0

    def test_classification_report(self, viz_dir, binary_data):
        y_true, y_pred, y_score = binary_data
        viz = TrainingVisualizer(output_dir=viz_dir)
        paths = viz.classification_report(y_true, y_pred, y_score, task="tnbc")

        assert len(paths) == 2  # txt + json
        txt_path = viz_dir / "classification_report_tnbc.txt"
        json_path = viz_dir / "classification_report_tnbc.json"
        assert txt_path.exists()
        assert json_path.exists()

        # Validate text report content
        text = txt_path.read_text()
        assert "AUROC" in text
        assert "Balanced Accuracy" in text
        assert "TNBC" in text

        # Validate JSON
        with open(json_path) as f:
            report = json.load(f)
        assert "auroc" in report
        assert "balanced_accuracy" in report
        assert "mcc" in report

    def test_generate_dashboard(self, viz_dir, binary_data, trained_rf):
        y_true, y_pred, y_score = binary_data
        viz = TrainingVisualizer(output_dir=viz_dir)
        paths = viz.generate_dashboard(
            y_true, y_pred, y_score,
            model=trained_rf, task="tnbc",
            feature_names=[f"feat_{i}" for i in range(10)],
            dataset_label="Test",
        )
        assert len(paths) == 1
        dashboard_path = viz_dir / "dashboard_tnbc.png"
        assert dashboard_path.exists()

    def test_generate_dashboard_no_model(self, viz_dir, binary_data):
        """Dashboard should still work without a model (metrics summary)."""
        y_true, y_pred, y_score = binary_data
        viz = TrainingVisualizer(output_dir=viz_dir)
        paths = viz.generate_dashboard(
            y_true, y_pred, y_score,
            model=None, task="luminal",
        )
        assert len(paths) == 1

    def test_generate_all(self, viz_dir, binary_data, trained_rf):
        y_true, y_pred, y_score = binary_data
        viz = TrainingVisualizer(output_dir=viz_dir)
        paths = viz.generate_all(
            y_true, y_pred, y_score,
            model=trained_rf, task="tnbc",
            dataset_label="Validation",
        )
        # Should produce multiple files
        assert len(paths) >= 5
        # Check key files
        expected_files = [
            "confusion_matrix_tnbc.png",
            "roc_curve_tnbc.png",
            "pr_curve_tnbc.png",
            "feature_importance_tnbc.png",
            "dashboard_tnbc.png",
            "classification_report_tnbc.txt",
            "classification_report_tnbc.json",
            "confusion_matrix_tnbc.json",
        ]
        created_names = {p.name for p in paths}
        for fname in expected_files:
            assert fname in created_names, f"Missing: {fname}"

    def test_task_labels(self, viz_dir, binary_data):
        """Check that TNBC and luminal tasks get proper labels."""
        y_true, y_pred, y_score = binary_data
        viz = TrainingVisualizer(output_dir=viz_dir)

        # TNBC task
        paths = viz.confusion_matrix(y_true, y_pred, task="tnbc")
        json_path = viz_dir / "confusion_matrix_tnbc.json"
        with open(json_path) as f:
            data = json.load(f)
        assert data["class_names"] == ["non-TNBC", "TNBC"]


# ---------------------------------------------------------------------------
# Tests: split_train_test_patients (from train_classifier)
# ---------------------------------------------------------------------------


class TestSplitTrainTestPatients:
    def test_with_split_column(self):
        """Test splitting when a split column exists."""
        import pandas as pd
        from mama_sia_eval.train_classifier import split_train_test_patients

        df = pd.DataFrame({
            "patient_id": ["P001", "P002", "P003", "P004", "P005"],
            "dataset_split": ["train", "train", "test", "train", "test"],
        })

        train_ids, test_ids = split_train_test_patients(df)
        assert set(train_ids) == {"P001", "P002", "P004"}
        assert set(test_ids) == {"P003", "P005"}

    def test_no_split_column(self):
        """All patients should be training when no split column exists."""
        import pandas as pd
        from mama_sia_eval.train_classifier import split_train_test_patients

        df = pd.DataFrame({
            "patient_id": ["P001", "P002", "P003"],
            "age": [45, 52, 61],
        })

        train_ids, test_ids = split_train_test_patients(df)
        assert len(train_ids) == 3
        assert len(test_ids) == 0

    def test_explicit_split_column(self):
        """Test with an explicitly specified column name."""
        import pandas as pd
        from mama_sia_eval.train_classifier import split_train_test_patients

        df = pd.DataFrame({
            "patient_id": ["P001", "P002", "P003"],
            "my_split": ["train", "test", "train"],
        })

        train_ids, test_ids = split_train_test_patients(
            df, split_column="my_split"
        )
        assert set(train_ids) == {"P001", "P003"}
        assert set(test_ids) == {"P002"}


class TestDetectSplitColumn:
    def test_detects_dataset_split(self):
        import pandas as pd
        from mama_sia_eval.train_classifier import detect_split_column

        df = pd.DataFrame({
            "patient_id": ["P001", "P002"],
            "dataset_split": ["train", "test"],
        })
        assert detect_split_column(df) == "dataset_split"

    def test_detects_split(self):
        import pandas as pd
        from mama_sia_eval.train_classifier import detect_split_column

        df = pd.DataFrame({
            "patient_id": ["P001", "P002"],
            "split": ["train", "test"],
        })
        assert detect_split_column(df) == "split"

    def test_no_split_returns_none(self):
        import pandas as pd
        from mama_sia_eval.train_classifier import detect_split_column

        df = pd.DataFrame({
            "patient_id": ["P001", "P002"],
            "age": [45, 52],
        })
        assert detect_split_column(df) is None


# ---------------------------------------------------------------------------
# Tests: New parse_args flags
# ---------------------------------------------------------------------------


class TestNewParseArgs:
    def test_slice_mode_flag(self):
        from mama_sia_eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--slice-mode", "max_tumor",
        ])
        assert args.slice_mode == "max_tumor"

    def test_slice_mode_default_none(self):
        from mama_sia_eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
        ])
        assert args.slice_mode is None

    def test_n_slices_flag(self):
        from mama_sia_eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--slice-mode", "multi_slice",
            "--n-slices", "10",
        ])
        assert args.n_slices == 10

    def test_evaluate_test_set_flag(self):
        from mama_sia_eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--evaluate-test-set",
        ])
        assert args.evaluate_test_set is True

    def test_no_viz_flag(self):
        from mama_sia_eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--no-viz",
        ])
        assert args.no_viz is True

    def test_split_column_flag(self):
        from mama_sia_eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--split-column", "my_split",
        ])
        assert args.split_column == "my_split"

    def test_invalid_slice_mode_rejected(self):
        from mama_sia_eval.train_classifier import parse_args

        with pytest.raises(SystemExit):
            parse_args([
                "--data-dir", "/tmp/data",
                "--output-dir", "/tmp/out",
                "--slice-mode", "invalid_mode",
            ])
