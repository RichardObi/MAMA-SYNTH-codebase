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

"""Integration tests for the tumor_roi classification pipeline.

Exercises the end-to-end flow: mirror_utils → radiomic feature extraction →
classifier training → prediction → evaluation → visualization, all using
synthetic breast data.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from eval.classification import RadiomicsClassifier, evaluate_classification
from eval.mirror_utils import create_mirrored_mask, detect_midline, mirror_mask

_has_radiomics = True
try:
    import radiomics  # noqa: F401
except ImportError:
    _has_radiomics = False

requires_radiomics = pytest.mark.skipif(
    not _has_radiomics, reason="pyradiomics not installed",
)


# ===================================================================
# Fixtures
# ===================================================================


def _make_breast_volume(
    slices: int = 12,
    rows: int = 64,
    cols: int = 128,
    midline: int = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthetic 3D breast volume with two bright lobes and dark midline."""
    if rng is None:
        rng = np.random.default_rng(42)
    vol = np.zeros((slices, rows, cols), dtype=np.float64)
    # Left breast
    vol[:, 5:rows - 5, 8 : midline - 4] = 200.0
    # Right breast
    vol[:, 5:rows - 5, midline + 4 : cols - 8] = 200.0
    # Add slight noise
    vol += rng.normal(0, 3, vol.shape).clip(0)
    return vol


def _make_tumor_mask(
    shape: tuple[int, ...],
    z_range: tuple[int, int] = (3, 8),
    row_range: tuple[int, int] = (20, 35),
    col_range: tuple[int, int] = (15, 30),
) -> np.ndarray:
    """Binary tumor mask on the left breast."""
    mask = np.zeros(shape, dtype=bool)
    mask[z_range[0]:z_range[1], row_range[0]:row_range[1],
         col_range[0]:col_range[1]] = True
    return mask


# ===================================================================
# 1. Mirror utils integration
# ===================================================================


class TestMirrorIntegration:
    """Test the full mirroring pipeline on 3D synthetic data."""

    def test_midline_detection_on_volume(self) -> None:
        vol = _make_breast_volume()
        mid = detect_midline(vol)
        assert abs(mid - 64) <= 6

    def test_mirror_mask_lands_on_opposite_breast(self) -> None:
        vol = _make_breast_volume()
        mask = _make_tumor_mask(vol.shape)
        mid = detect_midline(vol)
        mirrored = mirror_mask(mask, mid)
        # Original is on left (cols < 64), mirror should be on right
        orig_cols = np.argwhere(mask)[:, 2]
        mirr_cols = np.argwhere(mirrored)[:, 2]
        assert np.all(orig_cols < 64)
        assert np.all(mirr_cols > 64)

    def test_create_mirrored_mask_on_volume(self) -> None:
        vol = _make_breast_volume()
        mask = _make_tumor_mask(vol.shape)
        result = create_mirrored_mask(vol, mask)
        assert result is not None
        assert result.shape == mask.shape
        # No overlap
        assert not np.any(mask & result)

    def test_mirrored_mask_on_right_breast_tumor(self) -> None:
        """Tumor on right breast → mirror should land on left."""
        vol = _make_breast_volume(cols=128, midline=64)
        mask = _make_tumor_mask(
            vol.shape, col_range=(80, 100),
        )
        result = create_mirrored_mask(vol, mask)
        assert result is not None
        mirr_cols = np.argwhere(result)[:, 2]
        assert np.all(mirr_cols < 64)


# ===================================================================
# 2. Feature extraction on tumor ROI vs mirrored ROI
# ===================================================================


@requires_radiomics
class TestTumorROIFeatureExtraction:
    """Test radiomic feature extraction from tumor and mirrored ROI."""

    def test_extract_features_from_both_rois(self) -> None:
        from eval.frd import extract_radiomic_features

        vol = _make_breast_volume()
        mask = _make_tumor_mask(vol.shape)
        mirrored = create_mirrored_mask(vol, mask)
        assert mirrored is not None

        # Use just one slice for speed
        z = 5
        img_2d = vol[z]
        t_feat = extract_radiomic_features(img_2d, mask=mask[z])
        m_feat = extract_radiomic_features(img_2d, mask=mirrored[z])

        assert t_feat.ndim == 1
        assert m_feat.ndim == 1
        assert t_feat.shape == m_feat.shape
        assert t_feat.shape[0] > 10  # Should have many radiomic features
        # Not identical (different regions)
        assert not np.allclose(t_feat, m_feat)

    def test_feature_extraction_stable_across_runs(self) -> None:
        """Radiomic extraction should be deterministic."""
        from eval.frd import extract_radiomic_features

        vol = _make_breast_volume()
        mask = _make_tumor_mask(vol.shape)
        z = 5
        f1 = extract_radiomic_features(vol[z], mask=mask[z])
        f2 = extract_radiomic_features(vol[z], mask=mask[z])
        np.testing.assert_array_equal(f1, f2)


# ===================================================================
# 3. Classifier training and evaluation
# ===================================================================


class TestTumorROIClassifier:
    """Test RadiomicsClassifier for the tumor_roi task."""

    @pytest.fixture()
    def tumor_roi_data(self) -> tuple:
        """Generate a simple feature matrix for the tumor_roi task."""
        rng = np.random.default_rng(0)
        n = 40  # 20 tumor, 20 mirror
        n_feat = 93  # Typical pyradiomics feature count

        # Make tumor features slightly different from mirror features
        tumor_feats = rng.normal(loc=1.0, scale=0.5, size=(n // 2, n_feat))
        mirror_feats = rng.normal(loc=-1.0, scale=0.5, size=(n // 2, n_feat))
        features = np.vstack([tumor_feats, mirror_feats])
        labels = np.concatenate([
            np.ones(n // 2, dtype=np.int64),
            np.zeros(n // 2, dtype=np.int64),
        ])
        return features, labels

    def test_train_and_predict(self, tumor_roi_data) -> None:
        features, labels = tumor_roi_data
        clf = RadiomicsClassifier(task="tumor_roi")
        clf.train(features, labels)
        proba = clf.predict_proba(features)
        assert proba.shape == (40,)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_evaluate_classification_with_tumor_roi(
        self, tumor_roi_data
    ) -> None:
        features, labels = tumor_roi_data
        clf = RadiomicsClassifier(task="tumor_roi")
        clf.train(features, labels)
        proba = clf.predict_proba(features)
        result = evaluate_classification(labels, proba)
        assert "auroc" in result
        assert "balanced_accuracy" in result
        # With well-separated data, AUROC should be high
        assert result["auroc"] > 0.8
        assert result["balanced_accuracy"] > 0.7

    def test_save_and_reload(self, tumor_roi_data, tmp_path: Path) -> None:
        features, labels = tumor_roi_data
        clf = RadiomicsClassifier(task="tumor_roi")
        clf.train(features, labels)

        model_path = tmp_path / "tumor_roi_classifier.pkl"
        clf.save(model_path)
        assert model_path.exists()

        clf2 = RadiomicsClassifier(
            task="tumor_roi", model_path=model_path,
        )
        proba1 = clf.predict_proba(features)
        proba2 = clf2.predict_proba(features)
        np.testing.assert_array_almost_equal(proba1, proba2)

    def test_predict_labels(self, tumor_roi_data) -> None:
        features, labels = tumor_roi_data
        clf = RadiomicsClassifier(task="tumor_roi")
        clf.train(features, labels)
        preds = clf.predict(features, threshold=0.5)
        assert preds.shape == labels.shape
        assert set(np.unique(preds)).issubset({0, 1})


# ===================================================================
# 4. Visualization labels
# ===================================================================


class TestTumorROIVisualization:
    """Test that tumor_roi task produces correctly-labelled visualizations."""

    @pytest.fixture()
    def viz_data(self, tmp_path: Path):
        from eval.training_visualization import TrainingVisualizer

        rng = np.random.default_rng(99)
        y_true = np.array([0] * 20 + [1] * 20)
        y_pred = y_true.copy()  # Perfect predictions
        y_score = np.where(y_true == 1, 0.9, 0.1) + rng.normal(0, 0.02, 40)
        y_score = np.clip(y_score, 0, 1)

        viz = TrainingVisualizer(output_dir=tmp_path)
        return viz, y_true, y_pred, y_score

    def test_confusion_matrix_labels(self, viz_data) -> None:
        viz, y_true, y_pred, y_score = viz_data
        paths = viz.confusion_matrix(y_true, y_pred, task="tumor_roi")
        # Should produce files (may be empty list if matplotlib unavailable)
        if paths:
            assert any("confusion" in str(p).lower() for p in paths)

    def test_roc_curve(self, viz_data) -> None:
        viz, y_true, y_pred, y_score = viz_data
        paths = viz.roc_curve(y_true, y_score, task="tumor_roi")
        if paths:
            assert any("roc" in str(p).lower() for p in paths)

    def test_classification_report(self, viz_data) -> None:
        viz, y_true, y_pred, y_score = viz_data
        paths = viz.classification_report(
            y_true, y_pred, y_score, task="tumor_roi",
        )
        if paths:
            # Check the text report contains correct class names
            txt_files = [p for p in paths if p.suffix == ".txt"]
            if txt_files:
                text = txt_files[0].read_text()
                assert "Tumor ROI" in text or "Contralateral" in text

    def test_dashboard(self, viz_data) -> None:
        viz, y_true, y_pred, y_score = viz_data
        paths = viz.generate_dashboard(
            y_true, y_pred, y_score, task="tumor_roi",
        )
        if paths:
            assert any("dashboard" in str(p).lower() for p in paths)

    def test_task_label_string(self) -> None:
        from eval.training_visualization import _task_label

        assert "tumor" in _task_label("tumor_roi").lower()
        assert "contralateral" in _task_label("tumor_roi").lower()
        assert "contrast" in _task_label("contrast").lower()


# ===================================================================
# 5. Evaluation pipeline ─ _evaluate_tumor_roi
# ===================================================================


class TestEvaluateTumorROI:
    """Test the evaluation-time tumor_roi scoring path."""

    def test_metric_constant_exported(self) -> None:
        from eval import METRIC_AUROC_TUMOR_ROI

        assert METRIC_AUROC_TUMOR_ROI == "auroc_tumor_roi"

    def test_evaluate_tumor_roi_no_model_graceful(self, tmp_path: Path) -> None:
        """Without a trained model, _evaluate_tumor_roi should not crash."""
        from eval.evaluation import MamaSynthEval

        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        gt_dir.mkdir()
        pred_dir.mkdir()

        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=tmp_path / "metrics.json",
        )

        agg: dict = {}
        detail: dict = {}
        # No model dir → should add a note, not crash
        evaluator._evaluate_tumor_roi([], {}, agg, detail)
        assert "note_tumor_roi" in detail

    def test_evaluate_tumor_roi_no_masks_graceful(
        self, tmp_path: Path,
    ) -> None:
        """Without masks, _evaluate_tumor_roi should not crash."""
        from eval.evaluation import MamaSynthEval

        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        model_dir = tmp_path / "models"
        gt_dir.mkdir()
        pred_dir.mkdir()
        model_dir.mkdir()

        # Create a fake model file so the model-check passes
        (model_dir / "tumor_roi_classifier.pkl").touch()

        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=tmp_path / "metrics.json",
            clf_model_dir=model_dir,
        )

        agg: dict = {}
        detail: dict = {}
        evaluator._evaluate_tumor_roi([], {}, agg, detail)
        assert "note_tumor_roi" in detail


# ===================================================================
# 6. Caching round-trip
# ===================================================================


class TestTumorROICaching:
    """Test that features can be cached and reloaded correctly."""

    def test_cache_roundtrip_1d(self, tmp_path: Path) -> None:
        """Single-volume features cache as 1D arrays."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        feats = np.random.rand(93).astype(np.float64)
        np.save(cache_dir / "DUKE_001_phase1_tumor.npy", feats)
        loaded = np.load(cache_dir / "DUKE_001_phase1_tumor.npy")
        np.testing.assert_array_equal(feats, loaded)
        assert loaded.ndim == 1

    def test_cache_roundtrip_2d(self, tmp_path: Path) -> None:
        """All-tumor-slices features cache as 2D arrays."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        feats = np.random.rand(5, 93).astype(np.float64)
        np.save(cache_dir / "DUKE_001_phase1_all_tumor_tumor.npy", feats)
        loaded = np.load(
            cache_dir / "DUKE_001_phase1_all_tumor_tumor.npy",
        )
        np.testing.assert_array_equal(feats, loaded)
        assert loaded.ndim == 2
