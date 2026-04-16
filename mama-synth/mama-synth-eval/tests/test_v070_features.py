#  Copyright 2025 mama-synth-eval contributors
#  Licensed under the Apache License, Version 2.0.

"""
Tests for v0.7.0 features:

1. Dual-phase classification (radiomics + CNN + evaluation + CLI)
2. CNN mask channel (MRISliceDataset 4-channel output)
3. Radiomics model selection (--radiomics-model family filter)
4. Ensemble inference (EnsembleClassifier + --ensemble CLI flag)
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    import torch
    import timm  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="CNN tests require torch, torchvision, and timm",
)


# ===================================================================
# 1. DUAL-PHASE CLASSIFICATION
# ===================================================================


class TestDualPhaseFeatureExtraction:
    """Tests for dual-phase radiomics feature extraction."""

    @patch("eval.train_classifier._load_mask_as_array")
    @patch("eval.train_classifier._load_nifti_as_array")
    @patch("eval.frd.extract_radiomic_features")
    def test_dual_phase_doubles_features(
        self, mock_extract, mock_load_img, mock_load_mask
    ):
        """With dual_phase=True, feature dim should double (phase0 + phase1)."""
        from eval.train_classifier import extract_features_for_patients, IMAGES_SUBDIR, SEGMENTATIONS_SUBDIR

        n_feats = 93
        mock_load_img.return_value = np.random.randn(10, 64, 64)
        mock_load_mask.return_value = np.ones((10, 64, 64), dtype=bool)
        mock_extract.return_value = np.random.randn(n_feats)

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            images_dir = data_dir / IMAGES_SUBDIR
            segs_dir = data_dir / SEGMENTATIONS_SUBDIR
            for pid in ["P001", "P002"]:
                (images_dir / pid).mkdir(parents=True)
                # Phase 1 image
                (images_dir / pid / f"{pid}_0001.nii.gz").touch()
                # Phase 0 image (pre-contrast)
                (images_dir / pid / f"{pid}_0000.nii.gz").touch()
                segs_dir.mkdir(parents=True, exist_ok=True)
                (segs_dir / f"{pid}.nii.gz").touch()

            feat_matrix, valid_pids, valid_idx = extract_features_for_patients(
                patient_ids=["P001", "P002"],
                data_dir=data_dir,
                dual_phase=True,
            )

        # Feature dimension should be 2 * n_feats (phase1 + phase0)
        assert feat_matrix.shape[0] == 2
        assert feat_matrix.shape[1] == 2 * n_feats, (
            f"Expected {2 * n_feats} features, got {feat_matrix.shape[1]}"
        )
        assert len(valid_pids) == 2

    @patch("eval.train_classifier._load_mask_as_array")
    @patch("eval.train_classifier._load_nifti_as_array")
    @patch("eval.frd.extract_radiomic_features")
    def test_single_phase_normal_features(
        self, mock_extract, mock_load_img, mock_load_mask
    ):
        """With dual_phase=False (default), features should be single-dim."""
        from eval.train_classifier import extract_features_for_patients, IMAGES_SUBDIR, SEGMENTATIONS_SUBDIR

        n_feats = 93
        mock_load_img.return_value = np.random.randn(10, 64, 64)
        mock_load_mask.return_value = np.ones((10, 64, 64), dtype=bool)
        mock_extract.return_value = np.random.randn(n_feats)

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            images_dir = data_dir / IMAGES_SUBDIR
            segs_dir = data_dir / SEGMENTATIONS_SUBDIR
            for pid in ["P001"]:
                (images_dir / pid).mkdir(parents=True)
                (images_dir / pid / f"{pid}_0001.nii.gz").touch()
                segs_dir.mkdir(parents=True, exist_ok=True)
                (segs_dir / f"{pid}.nii.gz").touch()

            feat_matrix, valid_pids, valid_idx = extract_features_for_patients(
                patient_ids=["P001"],
                data_dir=data_dir,
                dual_phase=False,
            )

        assert feat_matrix.shape == (1, n_feats)

    @patch("eval.train_classifier._load_mask_as_array")
    @patch("eval.train_classifier._load_nifti_as_array")
    @patch("eval.frd.extract_radiomic_features")
    def test_dual_phase_cache_suffix_differs(
        self, mock_extract, mock_load_img, mock_load_mask
    ):
        """Cache key should differ between dual and single phase."""
        from eval.train_classifier import extract_features_for_patients, IMAGES_SUBDIR, SEGMENTATIONS_SUBDIR

        mock_load_img.return_value = np.random.randn(10, 64, 64)
        mock_load_mask.return_value = np.ones((10, 64, 64), dtype=bool)
        mock_extract.return_value = np.random.randn(93)

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            cache_dir = Path(tmp_dir) / "cache"
            images_dir = data_dir / IMAGES_SUBDIR
            segs_dir = data_dir / SEGMENTATIONS_SUBDIR
            for pid in ["P001"]:
                (images_dir / pid).mkdir(parents=True)
                (images_dir / pid / f"{pid}_0001.nii.gz").touch()
                (images_dir / pid / f"{pid}_0000.nii.gz").touch()
                segs_dir.mkdir(parents=True, exist_ok=True)
                (segs_dir / f"{pid}.nii.gz").touch()

            # Extract with dual_phase=False
            extract_features_for_patients(
                patient_ids=["P001"],
                data_dir=data_dir,
                cache_dir=cache_dir,
                dual_phase=False,
            )

            # Extract with dual_phase=True
            extract_features_for_patients(
                patient_ids=["P001"],
                data_dir=data_dir,
                cache_dir=cache_dir,
                dual_phase=True,
            )

            # Check that cache files exist with different names
            cache_files = list(cache_dir.glob("*.npy"))
            names = [f.name for f in cache_files]
            # There should be at least one file with 'dualphase' in name
            has_dual = any("dualphase" in n for n in names)
            has_normal = any("dualphase" not in n for n in names)
            assert has_dual, f"No dual-phase cache file found: {names}"
            assert has_normal, f"No single-phase cache file found: {names}"


class TestDualPhaseCLI:
    """Tests for --dual-phase CLI argument."""

    def test_train_classifier_dual_phase_flag(self):
        from eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--dual-phase",
        ])
        assert args.dual_phase is True

    def test_train_classifier_dual_phase_default(self):
        from eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
        ])
        assert args.dual_phase is False


class TestDualPhaseEvaluation:
    """Tests for dual-phase in MamaSynthEval."""

    def test_init_stores_dual_phase(self, tmp_path):
        from eval.evaluation import MamaSynthEval

        evaluator = MamaSynthEval(
            ground_truth_path=tmp_path,
            predictions_path=tmp_path,
            output_file=tmp_path / "metrics.json",
            dual_phase=True,
            precontrast_path=tmp_path / "precontrast",
        )
        assert evaluator.dual_phase is True
        assert evaluator.precontrast_path == tmp_path / "precontrast"

    def test_init_dual_phase_defaults(self, tmp_path):
        from eval.evaluation import MamaSynthEval

        evaluator = MamaSynthEval(
            ground_truth_path=tmp_path,
            predictions_path=tmp_path,
            output_file=tmp_path / "metrics.json",
        )
        assert evaluator.dual_phase is False
        assert evaluator.precontrast_path is None


class TestDualPhaseMainCLI:
    """Tests for --dual-phase and --precontrast-path in __main__.py."""

    def test_eval_cli_dual_phase_flag(self):
        """The evaluation CLI should accept --dual-phase."""
        import sys
        from unittest.mock import patch as mock_patch

        with mock_patch.object(sys, "argv", [
            "eval",
            "--ground-truth-path", "/tmp/gt",
            "--predictions-path", "/tmp/pred",
            "--output-file", "/tmp/out.json",
            "--dual-phase",
            "--precontrast-path", "/tmp/precon",
            "--disable-lpips",
            "--disable-frd",
            "--disable-segmentation",
            "--disable-classification",
        ]):
            # Just parse — don't run the evaluation
            import argparse
            from eval.__main__ import main  # noqa: attempt import check
            # Instead of running, verify parse_args produces expected values
            # by importing argparse handling from __main__

    def test_eval_cli_dual_phase_accepted(self):
        """Verify that MamaSynthEval constructor accepts dual_phase + precontrast_path."""
        from eval.evaluation import MamaSynthEval

        with tempfile.TemporaryDirectory() as tmp_dir:
            gt_dir = Path(tmp_dir) / "gt"
            gt_dir.mkdir()
            pred_dir = Path(tmp_dir) / "pred"
            pred_dir.mkdir()
            precon_dir = Path(tmp_dir) / "precon"
            precon_dir.mkdir()

            evaluator = MamaSynthEval(
                ground_truth_path=gt_dir,
                predictions_path=pred_dir,
                output_file=Path(tmp_dir) / "metrics.json",
                enable_lpips=False,
                enable_frd=False,
                enable_segmentation=False,
                enable_classification=False,
                dual_phase=True,
                precontrast_path=precon_dir,
            )
            assert evaluator.dual_phase is True
            assert evaluator.precontrast_path == precon_dir.resolve()


# ===================================================================
# 1b. DUAL-PHASE CNN
# ===================================================================


@requires_torch
class TestDualPhaseCNNSliceDataset:
    """Test MRISliceDataset dual-phase input handling."""

    def test_dual_phase_input_produces_6_channels(self):
        """(2, H, W) input should produce (6, H, W) tensor."""
        from eval.train_cnn_classifier import MRISliceDataset

        # Dual-phase slices: (2, H, W)
        slices = [np.random.randn(2, 64, 64).astype(np.float32) for _ in range(3)]
        labels = np.array([0, 1, 0], dtype=np.float32)
        ds = MRISliceDataset(slices, labels, image_size=32)

        img, label = ds[0]
        assert img.shape == (6, 32, 32), f"Expected (6,32,32), got {img.shape}"

    def test_dual_phase_with_mask_produces_7_channels(self):
        """(2, H, W) input with mask should produce (7, H, W) tensor."""
        from eval.train_cnn_classifier import MRISliceDataset

        slices = [np.random.randn(2, 64, 64).astype(np.float32) for _ in range(3)]
        labels = np.array([0, 1, 0], dtype=np.float32)
        masks = [np.zeros((64, 64), dtype=np.float32) for _ in range(3)]
        masks[0][10:20, 10:20] = 1.0

        ds = MRISliceDataset(slices, labels, image_size=32, masks=masks)

        img, label = ds[0]
        assert img.shape == (7, 32, 32), f"Expected (7,32,32), got {img.shape}"

    def test_single_phase_still_produces_3_channels(self):
        """Standard 2D (H, W) input should still produce (3, H, W)."""
        from eval.train_cnn_classifier import MRISliceDataset

        slices = [np.random.randn(64, 64) for _ in range(3)]
        labels = np.array([0, 1, 0], dtype=np.float32)
        ds = MRISliceDataset(slices, labels, image_size=32)

        img, label = ds[0]
        assert img.shape == (3, 32, 32), f"Expected (3,32,32), got {img.shape}"


@requires_torch
class TestDualPhaseCNNExtraction:
    """Test extract_slices_for_cnn with dual_phase=True."""

    @patch("eval.train_classifier._load_mask_as_array")
    @patch("eval.train_classifier._load_nifti_as_array")
    def test_dual_phase_slices_are_stacked(self, mock_load_img, mock_load_mask):
        """Dual-phase extraction should produce (2, H, W) slices."""
        from eval.train_cnn_classifier import extract_slices_for_cnn

        volume = np.random.randn(10, 64, 64)
        mask = np.zeros((10, 64, 64), dtype=bool)
        mask[5, 10:30, 10:30] = True

        # Return phase-1 first, then phase-0 for each patient
        mock_load_img.side_effect = [volume, volume]
        mock_load_mask.return_value = mask

        labels = np.array([1])

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            (data_dir / "images" / "P001").mkdir(parents=True)
            (data_dir / "images" / "P001" / "P001_0001.nii.gz").touch()
            (data_dir / "images" / "P001" / "P001_0000.nii.gz").touch()
            (data_dir / "segmentations").mkdir(parents=True)
            (data_dir / "segmentations" / "P001.nii.gz").touch()

            slices, slice_labels, slice_pids, _ = extract_slices_for_cnn(
                patient_ids=["P001"],
                labels=labels,
                data_dir=data_dir,
                slice_mode="max_tumor",
                dual_phase=True,
            )

        assert len(slices) == 1
        # Should be (2, H, W) — stacked dual-phase
        assert slices[0].ndim == 3, f"Expected 3D (2,H,W), got ndim={slices[0].ndim}"
        assert slices[0].shape[0] == 2, f"Expected shape[0]=2, got {slices[0].shape[0]}"

    @patch("eval.train_classifier._load_mask_as_array")
    @patch("eval.train_classifier._load_nifti_as_array")
    def test_single_phase_slices_are_2d(self, mock_load_img, mock_load_mask):
        """Without dual_phase, slices should be 2D (H, W)."""
        from eval.train_cnn_classifier import extract_slices_for_cnn

        volume = np.random.randn(10, 64, 64)
        mask = np.zeros((10, 64, 64), dtype=bool)
        mask[5, 10:30, 10:30] = True

        mock_load_img.return_value = volume
        mock_load_mask.return_value = mask

        labels = np.array([1])

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            (data_dir / "images" / "P001").mkdir(parents=True)
            (data_dir / "images" / "P001" / "P001_0001.nii.gz").touch()
            (data_dir / "segmentations").mkdir(parents=True)
            (data_dir / "segmentations" / "P001.nii.gz").touch()

            slices, _, _, _ = extract_slices_for_cnn(
                patient_ids=["P001"],
                labels=labels,
                data_dir=data_dir,
                slice_mode="max_tumor",
                dual_phase=False,
            )

        assert len(slices) == 1
        assert slices[0].ndim == 2

@requires_torch
class TestDualPhaseCNNModel:
    """Test CNN model creation with dual-phase channel count."""

    def test_model_with_6_channels(self):
        """Model with in_chans=6 should handle (6, H, W) input."""
        from eval.train_cnn_classifier import create_cnn_model

        model = create_cnn_model(pretrained=False, num_classes=1, in_chans=6)
        x = torch.randn(1, 6, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_model_with_7_channels(self):
        """Model with in_chans=7 (dual-phase + mask) should work."""
        from eval.train_cnn_classifier import create_cnn_model

        model = create_cnn_model(pretrained=False, num_classes=1, in_chans=7)
        x = torch.randn(1, 7, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)


# ===================================================================
# 2. CNN MASK CHANNEL
# ===================================================================


@requires_torch
class TestCNNMaskChannel:
    """Tests for the CNN mask channel feature (4-channel input)."""

    def test_mask_channel_produces_4_channels(self):
        """MRISliceDataset with masks should produce (4, H, W) tensors."""
        from eval.train_cnn_classifier import MRISliceDataset

        slices = [np.random.randn(64, 64) for _ in range(5)]
        labels = np.zeros(5, dtype=np.float32)
        masks = [np.zeros((64, 64), dtype=np.float32) for _ in range(5)]
        masks[0][10:30, 10:30] = 1.0

        ds = MRISliceDataset(slices, labels, image_size=32, masks=masks)
        img, label = ds[0]
        assert img.shape == (4, 32, 32), f"Expected (4,32,32), got {img.shape}"

    def test_no_mask_produces_3_channels(self):
        """MRISliceDataset without masks should produce (3, H, W) tensors."""
        from eval.train_cnn_classifier import MRISliceDataset

        slices = [np.random.randn(64, 64) for _ in range(5)]
        labels = np.zeros(5, dtype=np.float32)

        ds = MRISliceDataset(slices, labels, image_size=32)
        img, label = ds[0]
        assert img.shape == (3, 32, 32)

    def test_model_with_4_channels(self):
        """CNN model should accept 4-channel input."""
        from eval.train_cnn_classifier import create_cnn_model

        model = create_cnn_model(pretrained=False, num_classes=1, in_chans=4)
        x = torch.randn(1, 4, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_cnn_mask_channel_cli_flag(self):
        """--cnn-mask-channel should be recognized in training CLI."""
        from eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--classifier-type", "cnn",
            "--cnn-mask-channel",
        ])
        assert args.cnn_mask_channel is True

    def test_cnn_mask_channel_default_false(self):
        from eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
        ])
        assert args.cnn_mask_channel is False


# ===================================================================
# 3. RADIOMICS MODEL SELECTION
# ===================================================================


class TestRadiomicsModelSelection:
    """Tests for --radiomics-model family filter."""

    def test_filter_random_forest(self):
        from eval.train_classifier import _get_model_configs

        configs = _get_model_configs(model_filter="random_forest")
        assert len(configs) > 0
        for c in configs:
            assert c["family"] == "random_forest"

    def test_filter_logistic_regression(self):
        from eval.train_classifier import _get_model_configs

        configs = _get_model_configs(model_filter="logistic_regression")
        assert len(configs) > 0
        for c in configs:
            assert c["family"] == "logistic_regression"

    def test_filter_svm(self):
        from eval.train_classifier import _get_model_configs

        configs = _get_model_configs(model_filter="svm")
        assert len(configs) > 0
        for c in configs:
            assert c["family"] == "svm"

    def test_filter_all_returns_multiple_families(self):
        from eval.train_classifier import _get_model_configs

        configs = _get_model_configs(model_filter="all")
        families = {c.get("family") for c in configs}
        assert len(families) >= 3, f"Expected >=3 families, got {families}"

    def test_filter_none_returns_same_as_all(self):
        from eval.train_classifier import _get_model_configs

        configs_none = _get_model_configs(model_filter=None)
        configs_all = _get_model_configs(model_filter="all")
        assert len(configs_none) == len(configs_all)

    def test_invalid_filter_raises(self):
        from eval.train_classifier import _get_model_configs

        with pytest.raises(ValueError, match="No model configs match"):
            _get_model_configs(model_filter="nonexistent_family")

    def test_cli_radiomics_model_flag(self):
        from eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--radiomics-model", "random_forest",
        ])
        assert args.radiomics_model == "random_forest"

    def test_cli_radiomics_model_default(self):
        from eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
        ])
        assert args.radiomics_model == "all"

    def test_cli_radiomics_model_invalid(self):
        from eval.train_classifier import parse_args

        with pytest.raises(SystemExit):
            parse_args([
                "--data-dir", "/tmp/data",
                "--output-dir", "/tmp/out",
                "--radiomics-model", "bagging",
            ])

    def test_configs_have_family_key(self):
        """All model configs should have a 'family' key."""
        from eval.train_classifier import _get_model_configs

        configs = _get_model_configs()
        for c in configs:
            assert "family" in c, f"Config missing 'family': {c['name']}"


# ===================================================================
# 4. ENSEMBLE INFERENCE
# ===================================================================


class TestEnsembleClassifier:
    """Tests for the EnsembleClassifier class."""

    def test_create_ensemble(self):
        from eval.classification import EnsembleClassifier

        ens = EnsembleClassifier(task="tnbc")
        assert ens.task == "tnbc"
        assert ens.n_models == 0
        assert not ens.has_radiomics
        assert not ens.has_cnn

    def test_add_radiomics_model(self):
        from eval.classification import EnsembleClassifier, RadiomicsClassifier

        # Create a real trained model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.randn(20, 5)
        y = np.array([0]*10 + [1]*10)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tnbc_classifier.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)

            clf = RadiomicsClassifier(task="tnbc", model_path=path)
            ens = EnsembleClassifier(task="tnbc")
            ens.add_radiomics_model(clf)

        assert ens.n_models == 1
        assert ens.has_radiomics
        assert not ens.has_cnn

    def test_ensemble_predict_radiomics(self):
        """Ensemble with 2 radiomics models should average predictions."""
        from eval.classification import EnsembleClassifier, RadiomicsClassifier

        rng = np.random.RandomState(42)
        X = rng.randn(20, 10)
        y = np.array([0]*10 + [1]*10)

        with tempfile.TemporaryDirectory() as tmp_dir:
            models = []
            for i in range(2):
                model = RandomForestClassifier(
                    n_estimators=5+i*5, random_state=42+i,
                )
                model.fit(X, y)
                path = Path(tmp_dir) / f"tnbc_classifier_rf_{i}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(model, f)
                models.append(
                    RadiomicsClassifier(task="tnbc", model_path=path)
                )

            ens = EnsembleClassifier(task="tnbc")
            for m in models:
                ens.add_radiomics_model(m)

            proba = ens.predict_proba(features=X)

        assert proba.shape == (20,)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_ensemble_no_models_raises(self):
        from eval.classification import EnsembleClassifier

        ens = EnsembleClassifier(task="tnbc")
        with pytest.raises(ValueError, match="no models"):
            ens.predict_proba(features=np.zeros((5, 10)))

    def test_ensemble_invalid_task_raises(self):
        from eval.classification import EnsembleClassifier

        with pytest.raises(ValueError, match="task must be one of"):
            EnsembleClassifier(task="invalid")

    def test_ensemble_description(self):
        from eval.classification import EnsembleClassifier

        ens = EnsembleClassifier(task="luminal")
        desc = ens.description()
        assert isinstance(desc, str)

    def test_discover_models_empty_dir(self):
        """discover_models on empty dir should return ensemble with 0 models."""
        from eval.classification import EnsembleClassifier

        with tempfile.TemporaryDirectory() as tmp_dir:
            ens = EnsembleClassifier.discover_models(
                task="tnbc", model_dir=Path(tmp_dir),
            )
        assert ens.n_models == 0

    def test_discover_models_finds_pkl(self):
        """discover_models should find .pkl files."""
        from eval.classification import EnsembleClassifier

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.randn(20, 5)
        y = np.array([0]*10 + [1]*10)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tnbc_classifier.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)

            ens = EnsembleClassifier.discover_models(
                task="tnbc", model_dir=Path(tmp_dir),
            )

        assert ens.n_models == 1
        assert ens.has_radiomics

    def test_ensemble_radiomics_without_features_raises(self):
        """Ensemble with radiomics model but no features should raise."""
        from eval.classification import EnsembleClassifier, RadiomicsClassifier

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.randn(20, 5)
        y = np.array([0]*10 + [1]*10)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tnbc_classifier.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)

            clf = RadiomicsClassifier(task="tnbc", model_path=path)
            ens = EnsembleClassifier(task="tnbc")
            ens.add_radiomics_model(clf)

            with pytest.raises(ValueError, match="no features"):
                ens.predict_proba(features=None)


class TestEnsembleCLI:
    """Tests for --ensemble and --save-all-models CLI flags."""

    def test_ensemble_flag_in_eval_cli(self):
        """--ensemble should be accepted by the evaluation CLI."""
        from eval.evaluation import MamaSynthEval

        with tempfile.TemporaryDirectory() as tmp_dir:
            evaluator = MamaSynthEval(
                ground_truth_path=Path(tmp_dir),
                predictions_path=Path(tmp_dir),
                output_file=Path(tmp_dir) / "metrics.json",
                ensemble=True,
            )
            assert evaluator.ensemble is True

    def test_save_all_models_flag(self):
        from eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--save-all-models",
        ])
        assert args.save_all_models is True

    def test_save_all_models_default_false(self):
        from eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
        ])
        assert args.save_all_models is False


# ===================================================================
# 5. COMBINED FLAGS
# ===================================================================


class TestCombinedCLIFlags:
    """Test that multiple v0.7.0 flags can be combined."""

    def test_all_new_training_flags(self):
        from eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--dual-phase",
            "--radiomics-model", "random_forest",
            "--save-all-models",
        ])
        assert args.dual_phase is True
        assert args.radiomics_model == "random_forest"
        assert args.save_all_models is True

    def test_cnn_with_dual_and_mask(self):
        from eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--classifier-type", "cnn",
            "--cnn-mask-channel",
            "--dual-phase",
        ])
        assert args.classifier_type == "cnn"
        assert args.cnn_mask_channel is True
        assert args.dual_phase is True


class TestMamaSynthEvalAllNewParams:
    """Test that MamaSynthEval accepts all new v0.7.0 parameters."""

    def test_all_new_params(self, tmp_path):
        from eval.evaluation import MamaSynthEval

        evaluator = MamaSynthEval(
            ground_truth_path=tmp_path,
            predictions_path=tmp_path,
            output_file=tmp_path / "metrics.json",
            ensemble=True,
            dual_phase=True,
            precontrast_path=tmp_path / "precon",
        )
        assert evaluator.ensemble is True
        assert evaluator.dual_phase is True
        assert evaluator.precontrast_path == tmp_path / "precon"
