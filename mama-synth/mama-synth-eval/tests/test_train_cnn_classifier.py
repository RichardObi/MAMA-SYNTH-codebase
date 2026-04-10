#  Copyright 2025 mama-synth-eval contributors
#  Licensed under the Apache License, Version 2.0.

"""Tests for the CNN classifier training module (train_cnn_classifier.py)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Check whether PyTorch + timm are available (tests are skipped otherwise)
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


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------


class TestDependencyCheck:
    """Tests for _check_cnn_dependencies()."""

    @requires_torch
    def test_no_error_when_available(self):
        from eval.train_cnn_classifier import _check_cnn_dependencies

        # Should not raise
        _check_cnn_dependencies()


# ---------------------------------------------------------------------------
# MRISliceDataset
# ---------------------------------------------------------------------------


@requires_torch
class TestMRISliceDataset:
    """Tests for the MRISliceDataset class."""

    def test_length(self):
        from eval.train_cnn_classifier import MRISliceDataset

        slices = [np.random.randn(64, 64) for _ in range(10)]
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        ds = MRISliceDataset(slices, labels, image_size=32)
        assert len(ds) == 10

    def test_item_shape(self):
        from eval.train_cnn_classifier import MRISliceDataset

        slices = [np.random.randn(64, 64) for _ in range(5)]
        labels = np.zeros(5, dtype=np.float32)
        ds = MRISliceDataset(slices, labels, image_size=32)
        img, label = ds[0]
        assert img.shape == (3, 32, 32), f"Expected (3,32,32), got {img.shape}"
        assert label.ndim == 0

    def test_normalisation_range(self):
        from eval.train_cnn_classifier import MRISliceDataset

        # Fixed image with known range
        slices = [np.ones((8, 8)) * 100.0]
        labels = np.array([1.0])
        ds = MRISliceDataset(slices, labels, image_size=8, augment=False)
        img, _ = ds[0]
        # Constant image → all zeros after min-max normalisation
        assert img.max() <= 1.0 + 1e-5

    def test_augmentation_runs(self):
        from eval.train_cnn_classifier import MRISliceDataset

        slices = [np.random.randn(64, 64) for _ in range(3)]
        labels = np.array([0, 1, 0], dtype=np.float32)
        ds = MRISliceDataset(slices, labels, image_size=32, augment=True)
        # Should not raise
        img, label = ds[0]
        assert img.shape == (3, 32, 32)

    def test_varying_sizes(self):
        """Slices with different spatial sizes should all produce the same output size."""
        from eval.train_cnn_classifier import MRISliceDataset

        slices = [
            np.random.randn(32, 32),
            np.random.randn(64, 128),
            np.random.randn(100, 50),
        ]
        labels = np.array([0, 1, 0], dtype=np.float32)
        ds = MRISliceDataset(slices, labels, image_size=48)
        for i in range(3):
            img, _ = ds[i]
            assert img.shape == (3, 48, 48)


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


@requires_torch
class TestModelCreation:
    """Tests for create_cnn_model()."""

    def test_create_default_model(self):
        from eval.train_cnn_classifier import create_cnn_model

        model = create_cnn_model(pretrained=False, num_classes=1)
        assert hasattr(model, "forward")
        # Check output shape with a dummy input
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_custom_model_name(self):
        from eval.train_cnn_classifier import create_cnn_model

        # efficientnet_b0 should work
        model = create_cnn_model(
            model_name="efficientnet_b0", pretrained=False, num_classes=1,
        )
        assert model is not None


# ---------------------------------------------------------------------------
# Slice extraction for CNN
# ---------------------------------------------------------------------------


@requires_torch
class TestExtractSlicesForCNN:
    """Tests for extract_slices_for_cnn()."""

    @patch("eval.train_classifier._load_mask_as_array")
    @patch("eval.train_classifier._load_nifti_as_array")
    def test_all_tumor_extraction(self, mock_load_img, mock_load_mask):
        from eval.train_cnn_classifier import extract_slices_for_cnn

        # Create a volume where 3 slices have mask voxels
        volume = np.random.randn(10, 64, 64)
        mask = np.zeros((10, 64, 64), dtype=bool)
        mask[3, 10:20, 10:20] = True
        mask[5, 15:25, 15:25] = True
        mask[7, 20:30, 20:30] = True

        mock_load_img.return_value = volume
        mock_load_mask.return_value = mask

        labels = np.array([1, 0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            (data_dir / "images" / "P001").mkdir(parents=True)
            (data_dir / "images" / "P001" / "P001_0001.nii.gz").touch()
            (data_dir / "images" / "P002").mkdir(parents=True)
            (data_dir / "images" / "P002" / "P002_0001.nii.gz").touch()
            (data_dir / "segmentations").mkdir(parents=True)
            (data_dir / "segmentations" / "P001.nii.gz").touch()
            (data_dir / "segmentations" / "P002.nii.gz").touch()

            slices, slice_labels, slice_pids = extract_slices_for_cnn(
                patient_ids=["P001", "P002"],
                labels=labels,
                data_dir=data_dir,
                slice_mode="all_tumor",
            )

        # 3 tumor slices × 2 patients = 6 slices
        assert len(slices) == 6
        assert len(slice_labels) == 6
        assert len(slice_pids) == 6
        assert all(s.ndim == 2 for s in slices)

    @patch("eval.train_classifier._load_mask_as_array")
    @patch("eval.train_classifier._load_nifti_as_array")
    def test_max_tumor_extraction(self, mock_load_img, mock_load_mask):
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

            slices, slice_labels, slice_pids = extract_slices_for_cnn(
                patient_ids=["P001"],
                labels=labels,
                data_dir=data_dir,
                slice_mode="max_tumor",
            )

        # Single-slice mode: 1 slice per patient
        assert len(slices) == 1
        assert slices[0].ndim == 2


# ---------------------------------------------------------------------------
# CNN training (minimal smoke test)
# ---------------------------------------------------------------------------


@requires_torch
class TestTrainCNN:
    """Smoke tests for the CNN training loop."""

    def test_train_minimal(self):
        """Training for 2 epochs with tiny data should not crash."""
        from eval.train_cnn_classifier import train_cnn

        # Tiny synthetic dataset
        n_train, n_val = 8, 4
        train_slices = [np.random.randn(32, 32) for _ in range(n_train)]
        val_slices = [np.random.randn(32, 32) for _ in range(n_val)]
        train_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        val_labels = np.array([0, 1, 0, 1], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model, metrics = train_cnn(
                train_slices=train_slices,
                train_labels=train_labels,
                val_slices=val_slices,
                val_labels=val_labels,
                task="tnbc",
                output_dir=Path(tmp_dir),
                model_name="efficientnet_b0",
                image_size=32,
                num_epochs=2,
                batch_size=4,
                learning_rate=1e-3,
                patience=5,
                seed=42,
            )

        assert model is not None
        assert "auroc" in metrics
        # Model file should exist
        assert (Path(tmp_dir) / "tnbc_classifier_cnn.pt").exists()

    def test_model_save_load_roundtrip(self):
        """Saved model can be loaded and used for inference."""
        from eval.train_cnn_classifier import (
            create_cnn_model,
            load_cnn_model,
            CNN_MODEL_SUFFIX,
        )

        model = create_cnn_model(pretrained=False, num_classes=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / f"tnbc{CNN_MODEL_SUFFIX}"
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": "efficientnet_b0",
                "num_classes": 1,
                "image_size": 224,
                "task": "tnbc",
                "best_epoch": 1,
                "best_val_auroc": 0.75,
            }
            torch.save(checkpoint, path)

            loaded_model, config = load_cnn_model(path)

        assert config["model_name"] == "efficientnet_b0"
        assert config["image_size"] == 224
        assert config["task"] == "tnbc"

        # Verify inference works
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = loaded_model(x)
        assert out.shape == (1, 1)


# ---------------------------------------------------------------------------
# CNN evaluation
# ---------------------------------------------------------------------------


@requires_torch
class TestEvaluateCNN:
    """Tests for evaluate_cnn()."""

    def test_evaluate_returns_metrics(self):
        from eval.train_cnn_classifier import create_cnn_model, evaluate_cnn

        model = create_cnn_model(pretrained=False, num_classes=1)
        slices = [np.random.randn(32, 32) for _ in range(8)]
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)

        metrics = evaluate_cnn(
            model, slices, labels, image_size=32, batch_size=4,
        )
        assert "auroc" in metrics
        assert "balanced_accuracy" in metrics
        assert "loss" in metrics


# ---------------------------------------------------------------------------
# CNNClassifier (classification.py integration)
# ---------------------------------------------------------------------------


@requires_torch
class TestCNNClassifier:
    """Tests for the CNNClassifier class in classification.py."""

    def test_predict_from_2d_images(self):
        from eval.classification import CNNClassifier
        from eval.train_cnn_classifier import create_cnn_model, CNN_MODEL_SUFFIX

        model = create_cnn_model(pretrained=False, num_classes=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / f"tnbc{CNN_MODEL_SUFFIX}"
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "efficientnet_b0",
                "num_classes": 1,
                "image_size": 32,
                "task": "tnbc",
                "best_epoch": 1,
                "best_val_auroc": 0.5,
            }, path)

            clf = CNNClassifier(task="tnbc", model_path=path)

        # 2D images
        images = [np.random.randn(64, 64) for _ in range(5)]
        probs = clf.predict_proba_from_images(images)
        assert probs.shape == (5,)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_predict_from_3d_images(self):
        """3D volumes should be auto-sliced (middle slice, no mask)."""
        from eval.classification import CNNClassifier
        from eval.train_cnn_classifier import create_cnn_model, CNN_MODEL_SUFFIX

        model = create_cnn_model(pretrained=False, num_classes=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / f"luminal{CNN_MODEL_SUFFIX}"
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "efficientnet_b0",
                "num_classes": 1,
                "image_size": 32,
                "task": "luminal",
                "best_epoch": 1,
                "best_val_auroc": 0.5,
            }, path)

            clf = CNNClassifier(task="luminal", model_path=path)

        # 3D volumes (no masks → middle slice fallback)
        images = [np.random.randn(10, 64, 64) for _ in range(3)]
        probs = clf.predict_proba_from_images(images)
        assert probs.shape == (3,)

    def test_predict_with_masks(self):
        """3D volumes with masks should use max_tumor slice."""
        from eval.classification import CNNClassifier
        from eval.train_cnn_classifier import create_cnn_model, CNN_MODEL_SUFFIX

        model = create_cnn_model(pretrained=False, num_classes=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / f"tnbc{CNN_MODEL_SUFFIX}"
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": "efficientnet_b0",
                "num_classes": 1,
                "image_size": 32,
                "task": "tnbc",
                "best_epoch": 1,
                "best_val_auroc": 0.5,
            }, path)

            clf = CNNClassifier(task="tnbc", model_path=path)

        images = [np.random.randn(10, 64, 64)]
        masks = [np.zeros((10, 64, 64), dtype=bool)]
        masks[0][5, 20:40, 20:40] = True

        probs = clf.predict_proba_from_images(images, masks)
        assert probs.shape == (1,)
        assert 0.0 <= probs[0] <= 1.0


# ---------------------------------------------------------------------------
# CLI argument: --classifier-type
# ---------------------------------------------------------------------------


class TestClassifierTypeCLI:
    """Tests for the --classifier-type argument in parse_args."""

    def test_default_is_radiomics(self):
        from eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
        ])
        assert args.classifier_type == "radiomics"

    def test_cnn_type(self):
        from eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--classifier-type", "cnn",
        ])
        assert args.classifier_type == "cnn"

    def test_cnn_specific_args(self):
        from eval.train_classifier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--classifier-type", "cnn",
            "--cnn-model", "efficientnet_b2",
            "--cnn-image-size", "128",
            "--cnn-epochs", "10",
            "--cnn-batch-size", "16",
            "--cnn-lr", "0.001",
            "--cnn-patience", "5",
        ])
        assert args.cnn_model == "efficientnet_b2"
        assert args.cnn_image_size == 128
        assert args.cnn_epochs == 10
        assert args.cnn_batch_size == 16
        assert args.cnn_lr == 0.001
        assert args.cnn_patience == 5

    def test_invalid_classifier_type_rejected(self):
        from eval.train_classifier import parse_args

        with pytest.raises(SystemExit):
            parse_args([
                "--data-dir", "/tmp/data",
                "--output-dir", "/tmp/out",
                "--classifier-type", "invalid",
            ])
