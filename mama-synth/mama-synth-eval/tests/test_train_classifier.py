#  Copyright 2025 mama-synth-eval contributors
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
Tests for the MAMA-MIA classifier training pipeline.

These tests use synthetic data to verify the training module logic
without requiring the actual MAMA-MIA dataset.
"""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from eval.train_classifier import (
    ALL_SUBTYPES,
    CLINICAL_EXCEL_FILENAME,
    CLINICAL_SHEET_NAME,
    DEFAULT_PHASE,
    DEFAULT_SEED,
    IMAGES_SUBDIR,
    LUMINAL_SUBTYPES,
    SEGMENTATIONS_SUBDIR,
    TNBC_SUBTYPES,
    _get_image_path,
    _get_model_configs,
    _get_segmentation_path,
    create_labels,
    evaluate_model,
    load_clinical_data,
    parse_args,
    save_model,
    save_training_report,
    train_single_model,
    train_with_cross_validation,
    train_with_model_selection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_clinical_df():
    """Create a sample clinical DataFrame resembling MAMA-MIA data."""
    import pandas as pd

    data = {
        "patient_id": [
            "DUKE_001", "DUKE_002", "DUKE_003", "DUKE_004", "DUKE_005",
            "DUKE_006", "DUKE_007", "DUKE_008", "DUKE_009", "DUKE_010",
            "ISPY1_001", "ISPY1_002", "ISPY1_003", "ISPY1_004", "ISPY1_005",
            "ISPY1_006", "ISPY1_007", "ISPY1_008", "ISPY1_009", "ISPY1_010",
            "ISPY2_001", "ISPY2_002", "ISPY2_003", "ISPY2_004", "ISPY2_005",
            "NACT_001", "NACT_002", "NACT_003", "NACT_004", "NACT_005",
        ],
        "dataset": (
            ["DUKE"] * 10 + ["ISPY1"] * 10 + ["ISPY2"] * 5 + ["NACT"] * 5
        ),
        "tumor_subtype": [
            "triple_negative", "luminal_a", "luminal_b", "her2_pure", "luminal",
            "her2_enriched", "triple_negative", "luminal_a", np.nan, "luminal_b",
            "triple_negative", "luminal", "her2_enriched", "triple_negative", "luminal_a",
            "her2_pure", "luminal_b", "triple_negative", np.nan, "luminal",
            "her2_enriched", "luminal_a", "triple_negative", "her2_pure", np.nan,
            "luminal", "triple_negative", "her2_enriched", "luminal_b", "luminal_a",
        ],
        "hr": [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        "er": [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        "pr": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "her2": [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        "age": [45, 52, 61, 48, 55, 43, 67, 50, 59, 62, 44, 53, 49, 56, 47, 64, 51, 60, 42, 58, 46, 54, 63, 57, 41, 50, 66, 48, 55, 52],
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_feature_matrix():
    """Create a sample feature matrix for training."""
    rng = np.random.RandomState(42)
    n_samples = 27  # 30 patients - 3 NaN subtypes
    n_features = 93  # Typical pyradiomics feature count
    return rng.randn(n_samples, n_features).astype(np.float64)


@pytest.fixture
def sample_labels_tnbc():
    """Binary labels for TNBC task (matching 27 valid patients from fixture)."""
    # Based on the tumor_subtype values above (excluding NaN entries: indices 8, 18, 24)
    return np.array([
        1, 0, 0, 0, 0, 0, 1, 0, 0,  # DUKE (minus DUKE_009=NaN)
        1, 0, 0, 1, 0, 0, 0, 1,      # ISPY1 (minus ISPY1_009=NaN)
        0, 0, 1, 0,                    # ISPY2 (minus ISPY2_005=NaN)
        0, 1, 0, 0, 0,                 # NACT
    ], dtype=np.int64)


@pytest.fixture
def sample_labels_luminal():
    """Binary labels for luminal task (matching 27 valid patients from fixture)."""
    return np.array([
        0, 1, 1, 0, 1, 0, 0, 1, 1,  # DUKE (minus DUKE_009=NaN)
        0, 1, 0, 0, 1, 0, 1, 0,      # ISPY1 (minus ISPY1_009=NaN)
        0, 1, 0, 0,                    # ISPY2 (minus ISPY2_005=NaN)
        1, 0, 0, 1, 1,                 # NACT
    ], dtype=np.int64)


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create a temporary output directory."""
    return tmp_path / "output"


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Test that module constants are correctly defined."""

    def test_tnbc_subtypes(self):
        assert "triple_negative" in TNBC_SUBTYPES
        assert len(TNBC_SUBTYPES) == 1

    def test_luminal_subtypes(self):
        assert "luminal" in LUMINAL_SUBTYPES
        assert "luminal_a" in LUMINAL_SUBTYPES
        assert "luminal_b" in LUMINAL_SUBTYPES
        assert len(LUMINAL_SUBTYPES) == 3

    def test_all_subtypes_covers_tnbc_and_luminal(self):
        assert TNBC_SUBTYPES.issubset(ALL_SUBTYPES)
        assert LUMINAL_SUBTYPES.issubset(ALL_SUBTYPES)

    def test_default_phase(self):
        assert DEFAULT_PHASE == 1

    def test_clinical_filename(self):
        assert CLINICAL_EXCEL_FILENAME == "clinical_and_imaging_info.xlsx"


# ---------------------------------------------------------------------------
# Test label creation
# ---------------------------------------------------------------------------

class TestCreateLabels:
    """Tests for create_labels function."""

    def test_tnbc_labels(self, sample_clinical_df):
        pids, labels = create_labels(sample_clinical_df, "tnbc")
        # Should have 27 valid patients (30 - 3 NaN)
        assert len(pids) == 27
        assert len(labels) == 27
        # At least some positive and negative labels
        assert labels.sum() > 0
        assert (1 - labels).sum() > 0
        # All labels are 0 or 1
        assert set(np.unique(labels)).issubset({0, 1})

    def test_luminal_labels(self, sample_clinical_df):
        pids, labels = create_labels(sample_clinical_df, "luminal")
        assert len(pids) == 27
        assert len(labels) == 27
        assert labels.sum() > 0
        assert (1 - labels).sum() > 0

    def test_tnbc_labels_correct_mapping(self, sample_clinical_df):
        pids, labels = create_labels(sample_clinical_df, "tnbc")
        # DUKE_001 is triple_negative → should be 1
        duke_001_idx = pids.index("DUKE_001")
        assert labels[duke_001_idx] == 1
        # DUKE_002 is luminal_a → should be 0
        duke_002_idx = pids.index("DUKE_002")
        assert labels[duke_002_idx] == 0

    def test_luminal_labels_correct_mapping(self, sample_clinical_df):
        pids, labels = create_labels(sample_clinical_df, "luminal")
        # DUKE_001 is triple_negative → should be 0
        duke_001_idx = pids.index("DUKE_001")
        assert labels[duke_001_idx] == 0
        # DUKE_002 is luminal_a → should be 1
        duke_002_idx = pids.index("DUKE_002")
        assert labels[duke_002_idx] == 1
        # DUKE_005 is luminal → should be 1
        duke_005_idx = pids.index("DUKE_005")
        assert labels[duke_005_idx] == 1

    def test_nan_subtypes_excluded(self, sample_clinical_df):
        pids, labels = create_labels(sample_clinical_df, "tnbc")
        # DUKE_009, ISPY1_009, ISPY2_005 have NaN subtypes
        assert "DUKE_009" not in pids
        assert "ISPY1_009" not in pids
        assert "ISPY2_005" not in pids

    def test_invalid_task_raises(self, sample_clinical_df):
        with pytest.raises(ValueError, match="task must be one of"):
            create_labels(sample_clinical_df, "invalid")

    def test_empty_clinical_data_raises(self):
        import pandas as pd
        empty_df = pd.DataFrame(
            {"patient_id": [], "tumor_subtype": []}
        )
        with pytest.raises(ValueError, match="No patients"):
            create_labels(empty_df, "tnbc")


# ---------------------------------------------------------------------------
# Test path construction
# ---------------------------------------------------------------------------

class TestPathConstruction:
    """Tests for image/segmentation path construction."""

    def test_get_image_path(self):
        path = _get_image_path(Path("/data/images"), "DUKE_001", 1)
        assert path == Path("/data/images/DUKE_001/DUKE_001_0001.nii.gz")

    def test_get_image_path_phase_zero(self):
        path = _get_image_path(Path("/data/images"), "ISPY1_005", 0)
        assert path == Path("/data/images/ISPY1_005/ISPY1_005_0000.nii.gz")

    def test_get_segmentation_path(self):
        path = _get_segmentation_path(Path("/data/segs"), "DUKE_001")
        assert path == Path("/data/segs/DUKE_001.nii.gz")


# ---------------------------------------------------------------------------
# Test model evaluation
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    """Tests for the evaluate_model function."""

    def test_perfect_predictions(self):
        model = MagicMock()
        X = np.random.randn(20, 5)
        y = np.array([0]*10 + [1]*10, dtype=np.int64)
        # Perfect predictions
        proba = np.zeros((20, 2))
        proba[:10, 0] = 1.0
        proba[10:, 1] = 1.0
        model.predict_proba.return_value = proba

        metrics = evaluate_model(model, X, y)
        assert metrics["auroc"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0

    def test_random_predictions(self):
        model = MagicMock()
        X = np.random.randn(100, 5)
        y = np.array([0]*50 + [1]*50, dtype=np.int64)
        rng = np.random.RandomState(42)
        proba = rng.rand(100, 2)
        proba = proba / proba.sum(axis=1, keepdims=True)
        model.predict_proba.return_value = proba

        metrics = evaluate_model(model, X, y)
        assert 0.0 <= metrics["auroc"] <= 1.0
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0

    def test_single_class(self):
        model = MagicMock()
        X = np.random.randn(10, 5)
        y = np.zeros(10, dtype=np.int64)
        proba = np.zeros((10, 2))
        proba[:, 0] = 0.8
        proba[:, 1] = 0.2
        model.predict_proba.return_value = proba

        metrics = evaluate_model(model, X, y)
        # AUROC undefined for single class — should be NaN
        assert np.isnan(metrics["auroc"])


# ---------------------------------------------------------------------------
# Test model training
# ---------------------------------------------------------------------------

class TestTrainSingleModel:
    """Tests for train_single_model function."""

    def test_train_random_forest(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 10)
        y_train = (rng.randn(50) > 0).astype(np.int64)
        X_val = rng.randn(20, 10)
        y_val = (rng.randn(20) > 0).astype(np.int64)

        config = {
            "name": "TestRF",
            "create_fn": lambda: RandomForestClassifier(
                n_estimators=10, random_state=42
            ),
        }

        model, train_m, val_m = train_single_model(
            config, X_train, y_train, X_val, y_val
        )

        assert model is not None
        assert hasattr(model, "predict_proba")
        assert "auroc" in train_m
        assert "balanced_accuracy" in train_m
        assert "auroc" in val_m
        assert "balanced_accuracy" in val_m


class TestTrainWithModelSelection:
    """Tests for train_with_model_selection function."""

    def test_selects_best_model(self):
        rng = np.random.RandomState(42)
        n_train, n_val, n_feat = 80, 20, 15

        # Create separable data for reliable training
        X_train = np.vstack([
            rng.randn(n_train // 2, n_feat) - 1,
            rng.randn(n_train // 2, n_feat) + 1,
        ])
        y_train = np.array([0] * (n_train // 2) + [1] * (n_train // 2), dtype=np.int64)
        X_val = np.vstack([
            rng.randn(n_val // 2, n_feat) - 1,
            rng.randn(n_val // 2, n_feat) + 1,
        ])
        y_val = np.array([0] * (n_val // 2) + [1] * (n_val // 2), dtype=np.int64)

        best_model, best_name, best_metrics, _ = train_with_model_selection(
            X_train, y_train, X_val, y_val, task="tnbc"
        )

        assert best_model is not None
        assert isinstance(best_name, str)
        assert len(best_name) > 0
        assert "auroc" in best_metrics
        assert best_metrics["auroc"] > 0.5  # Should do better than random


class TestTrainWithCrossValidation:
    """Tests for train_with_cross_validation function."""

    def test_cv_training(self):
        rng = np.random.RandomState(42)
        n_samples = 100
        n_feat = 10

        X = np.vstack([
            rng.randn(n_samples // 2, n_feat) - 1,
            rng.randn(n_samples // 2, n_feat) + 1,
        ])
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.int64)

        model, name, metrics, _ = train_with_cross_validation(
            X, y, task="luminal", n_folds=3, seed=42
        )

        assert model is not None
        assert isinstance(name, str)
        assert "cv_auroc" in metrics
        assert metrics["cv_auroc"] > 0.5


# ---------------------------------------------------------------------------
# Test model saving
# ---------------------------------------------------------------------------

class TestSaveModel:
    """Tests for model saving and compatibility with inference code."""

    def test_save_model_creates_pkl(self, tmp_output_dir):
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(np.random.randn(20, 5), np.array([0]*10 + [1]*10))

        path = save_model(model, "tnbc", tmp_output_dir)
        assert path.exists()
        assert path.name == "tnbc_classifier.pkl"

    def test_saved_model_loadable_by_inference_code(self, tmp_output_dir):
        """Verify the saved model can be loaded by RadiomicsClassifier._load_model."""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.randn(20, 5)
        y = np.array([0]*10 + [1]*10)
        model.fit(X, y)

        path = save_model(model, "luminal", tmp_output_dir)

        # Load exactly as the inference code does
        with open(path, "rb") as f:
            loaded_model = pickle.load(f)

        assert hasattr(loaded_model, "predict_proba")
        proba = loaded_model.predict_proba(X)
        assert proba.shape == (20, 2)

    def test_saved_model_works_with_radiomics_classifier(self, tmp_output_dir):
        """End-to-end: save model → load with RadiomicsClassifier → predict."""
        from eval.classification import RadiomicsClassifier

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.randn(20, 5)
        y = np.array([0]*10 + [1]*10)
        model.fit(X, y)

        save_model(model, "tnbc", tmp_output_dir)

        # Load with RadiomicsClassifier (as evaluation.py does)
        clf = RadiomicsClassifier(
            task="tnbc",
            model_path=tmp_output_dir / "tnbc_classifier.pkl",
        )
        proba = clf.predict_proba(X)
        assert proba.shape == (20,)
        assert np.all(proba >= 0) and np.all(proba <= 1)


class TestSaveTrainingReport:
    """Tests for training report saving."""

    def test_save_report(self, tmp_output_dir):
        report = {
            "data_dir": "/data",
            "tasks": {"tnbc": {"val_metrics": {"auroc": 0.85}}},
        }
        path = save_training_report(report, tmp_output_dir)
        assert path.exists()

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["tasks"]["tnbc"]["val_metrics"]["auroc"] == 0.85


# ---------------------------------------------------------------------------
# Test model configs
# ---------------------------------------------------------------------------

class TestModelConfigs:
    """Tests for _get_model_configs function."""

    def test_configs_nonempty(self):
        configs = _get_model_configs()
        assert len(configs) > 0

    def test_configs_have_required_keys(self):
        configs = _get_model_configs()
        for config in configs:
            assert "name" in config
            assert "create_fn" in config
            assert callable(config["create_fn"])

    def test_configs_create_valid_models(self):
        configs = _get_model_configs()
        for config in configs:
            model = config["create_fn"]()
            assert hasattr(model, "fit")
            assert hasattr(model, "predict_proba")


# ---------------------------------------------------------------------------
# Test CLI parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    """Tests for command-line argument parsing."""

    def test_minimal_args(self):
        args = parse_args([
            "--data-dir", "/data",
            "--output-dir", "/output",
        ])
        assert args.data_dir == Path("/data")
        assert args.output_dir == Path("/output")
        assert args.tasks == ["tnbc", "luminal"]
        assert args.phase == DEFAULT_PHASE
        assert args.val_ratio == 0.2
        assert args.seed == DEFAULT_SEED

    def test_single_task(self):
        args = parse_args([
            "--data-dir", "/data",
            "--output-dir", "/output",
            "--tasks", "tnbc",
        ])
        assert args.tasks == ["tnbc"]

    def test_custom_phase(self):
        args = parse_args([
            "--data-dir", "/data",
            "--output-dir", "/output",
            "--phase", "0",
        ])
        assert args.phase == 0

    def test_cv_folds(self):
        args = parse_args([
            "--data-dir", "/data",
            "--output-dir", "/output",
            "--cv-folds", "5",
        ])
        assert args.cv_folds == 5

    def test_custom_paths(self):
        args = parse_args([
            "--data-dir", "/data",
            "--output-dir", "/output",
            "--clinical-data", "/custom/clinical.xlsx",
            "--images-dir", "/custom/images",
            "--segmentations-dir", "/custom/segs",
        ])
        assert args.clinical_data == Path("/custom/clinical.xlsx")
        assert args.images_dir == Path("/custom/images")
        assert args.segmentations_dir == Path("/custom/segs")

    def test_verbose_flag(self):
        args = parse_args([
            "--data-dir", "/data",
            "--output-dir", "/output",
            "-v",
        ])
        assert args.verbose is True


# ---------------------------------------------------------------------------
# Test clinical data loading
# ---------------------------------------------------------------------------

class TestLoadClinicalData:
    """Tests for load_clinical_data function."""

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Clinical data file not found"):
            load_clinical_data(tmp_path)

    def test_load_from_excel(self, tmp_path, sample_clinical_df):
        """Test loading from a real Excel file."""
        import pandas as pd

        excel_path = tmp_path / CLINICAL_EXCEL_FILENAME
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            sample_clinical_df.to_excel(
                writer, sheet_name=CLINICAL_SHEET_NAME, index=False
            )

        df = load_clinical_data(tmp_path)
        assert len(df) == 30
        assert "patient_id" in df.columns
        assert "tumor_subtype" in df.columns

    def test_load_with_custom_path(self, tmp_path, sample_clinical_df):
        """Test loading from a custom path."""
        import pandas as pd

        custom_path = tmp_path / "custom_clinical.xlsx"
        with pd.ExcelWriter(custom_path, engine="openpyxl") as writer:
            sample_clinical_df.to_excel(
                writer, sheet_name=CLINICAL_SHEET_NAME, index=False
            )

        df = load_clinical_data(tmp_path, clinical_data_path=custom_path)
        assert len(df) == 30


# ---------------------------------------------------------------------------
# Test feature extraction (mocked)
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    """Tests for feature extraction with mocked NIfTI loading."""

    @patch("eval.train_classifier._load_mask_as_array")
    @patch("eval.train_classifier._load_nifti_as_array")
    @patch("eval.frd.extract_radiomic_features")
    def test_extract_features_basic(
        self, mock_extract, mock_load_img, mock_load_mask
    ):
        """Test basic feature extraction with mocked IO."""
        from eval.train_classifier import extract_features_for_patients

        # Mock return values
        mock_load_img.return_value = np.random.randn(10, 64, 64)
        mock_load_mask.return_value = np.ones((10, 64, 64), dtype=bool)
        mock_extract.return_value = np.random.randn(93)

        # Create temp directory structure
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            images_dir = data_dir / IMAGES_SUBDIR
            segs_dir = data_dir / SEGMENTATIONS_SUBDIR

            # Create dummy directories and files
            for pid in ["P001", "P002", "P003"]:
                (images_dir / pid).mkdir(parents=True, exist_ok=True)
                (images_dir / pid / f"{pid}_0001.nii.gz").touch()
                segs_dir.mkdir(parents=True, exist_ok=True)
                (segs_dir / f"{pid}.nii.gz").touch()

            feat_matrix, valid_pids, valid_idx = extract_features_for_patients(
                patient_ids=["P001", "P002", "P003"],
                data_dir=data_dir,
            )

        assert feat_matrix.shape == (3, 93)
        assert len(valid_pids) == 3
        assert valid_idx == [0, 1, 2]

    @patch("eval.frd.extract_radiomic_features")
    @patch("eval.train_classifier._load_mask_as_array")
    @patch("eval.train_classifier._load_nifti_as_array")
    def test_extract_features_missing_image(self, mock_load_img, mock_load_mask, mock_extract):
        """Test that missing images are gracefully skipped (partial failure)."""
        from eval.train_classifier import extract_features_for_patients

        # First patient fails, second and third succeed
        mock_load_img.side_effect = [
            FileNotFoundError("not found"),
            np.random.randn(10, 64, 64),
            np.random.randn(10, 64, 64),
        ]
        mock_load_mask.return_value = np.ones((10, 64, 64), dtype=bool)
        mock_extract.return_value = np.random.randn(93)

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)

            feat_matrix, valid_pids, valid_idx = extract_features_for_patients(
                patient_ids=["P001", "P002", "P003"],
                data_dir=data_dir,
            )

        # P001 failed, P002 and P003 succeeded
        assert len(valid_pids) == 2
        assert "P001" not in valid_pids
        assert valid_idx == [1, 2]

    @patch("eval.train_classifier._load_nifti_as_array")
    def test_all_images_missing_raises(self, mock_load_img):
        """Test that RuntimeError is raised when no features can be extracted."""
        from eval.train_classifier import extract_features_for_patients

        mock_load_img.side_effect = FileNotFoundError("not found")

        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(RuntimeError, match="No features could be extracted"):
                extract_features_for_patients(
                    patient_ids=["P001", "P002"],
                    data_dir=Path(tmp_dir),
                )

    @patch("eval.train_classifier._load_mask_as_array")
    @patch("eval.train_classifier._load_nifti_as_array")
    @patch("eval.frd.extract_radiomic_features")
    def test_feature_caching(
        self, mock_extract, mock_load_img, mock_load_mask
    ):
        """Test that features are cached and loaded from cache."""
        from eval.train_classifier import extract_features_for_patients

        expected_feat = np.random.randn(93)
        mock_load_img.return_value = np.random.randn(10, 64, 64)
        mock_load_mask.return_value = np.ones((10, 64, 64), dtype=bool)
        mock_extract.return_value = expected_feat

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            cache_dir = Path(tmp_dir) / "cache"
            images_dir = data_dir / IMAGES_SUBDIR
            segs_dir = data_dir / SEGMENTATIONS_SUBDIR

            for pid in ["P001"]:
                (images_dir / pid).mkdir(parents=True, exist_ok=True)
                (images_dir / pid / f"{pid}_0001.nii.gz").touch()
                segs_dir.mkdir(parents=True, exist_ok=True)
                (segs_dir / f"{pid}.nii.gz").touch()

            # First run: extract and cache
            feat1, _, _ = extract_features_for_patients(
                patient_ids=["P001"],
                data_dir=data_dir,
                cache_dir=cache_dir,
            )

            # Reset mock
            mock_extract.reset_mock()

            # Second run: should load from cache
            feat2, _, _ = extract_features_for_patients(
                patient_ids=["P001"],
                data_dir=data_dir,
                cache_dir=cache_dir,
            )

            # Feature extraction should NOT have been called again
            mock_extract.assert_not_called()
            np.testing.assert_array_equal(feat1, feat2)


# ---------------------------------------------------------------------------
# Integration-style test (no real data needed)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Integration tests using synthetic data."""

    def test_full_training_pipeline_with_synthetic_data(self, tmp_output_dir):
        """Test the full training flow with synthetic features."""
        rng = np.random.RandomState(42)
        n_train, n_val = 60, 15
        n_feat = 20

        # Create linearly separable data
        X_train = np.vstack([
            rng.randn(n_train // 2, n_feat) - 1,
            rng.randn(n_train // 2, n_feat) + 1,
        ])
        y_train = np.array([0] * (n_train // 2) + [1] * (n_train // 2), dtype=np.int64)
        X_val = np.vstack([
            rng.randn(n_val // 2, n_feat) - 1,
            rng.randn(n_val - n_val // 2, n_feat) + 1,
        ])
        y_val = np.array(
            [0] * (n_val // 2) + [1] * (n_val - n_val // 2), dtype=np.int64
        )

        # Train
        best_model, best_name, best_metrics, _ = train_with_model_selection(
            X_train, y_train, X_val, y_val, task="tnbc"
        )

        # Save
        model_path = save_model(best_model, "tnbc", tmp_output_dir)

        # Verify
        assert model_path.exists()
        assert model_path.name == "tnbc_classifier.pkl"
        assert best_metrics["auroc"] > 0.7  # Should work well on separable data

        # Load with pickle (as inference does)
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)
        proba = loaded.predict_proba(X_val)
        assert proba.shape[0] == n_val

    def test_both_tasks_save_separate_models(self, tmp_output_dir):
        """Test that TNBC and luminal models are saved as separate files."""
        rng = np.random.RandomState(42)
        n = 50
        X = rng.randn(n, 10)

        for task in ["tnbc", "luminal"]:
            y = rng.randint(0, 2, n).astype(np.int64)
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            model.fit(X, y)
            save_model(model, task, tmp_output_dir)

        assert (tmp_output_dir / "tnbc_classifier.pkl").exists()
        assert (tmp_output_dir / "luminal_classifier.pkl").exists()
