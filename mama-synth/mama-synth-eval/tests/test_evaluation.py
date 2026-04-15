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

"""Unit tests for the MamaSynthEval evaluation class."""

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from eval.evaluation import (
    MamaSynthEval,
    DatasetNormalizer,
    normalize_intensity,
    METRIC_MSE_FULL,
    METRIC_SSIM_ROI,
    METRIC_DICE,
    METRIC_HD95,
)
from tests.conftest import save_sitk_image, IMG_SHAPE_2D


def create_test_image(path: Path, shape: tuple, value: float = 0.0) -> None:
    """Create a test medical image file."""
    data = np.full(shape, value, dtype=np.float32)
    image = sitk.GetImageFromArray(data)
    sitk.WriteImage(image, str(path))


class TestNormalizeIntensity:
    """Tests for intensity normalization."""

    def test_zero_mean(self) -> None:
        img = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = normalize_intensity(img)
        assert abs(np.mean(normed)) < 1e-10

    def test_unit_std(self) -> None:
        img = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = normalize_intensity(img)
        assert abs(np.std(normed) - 1.0) < 1e-10

    def test_constant_image(self) -> None:
        img = np.ones((10, 10), dtype=np.float64) * 5
        normed = normalize_intensity(img)
        assert np.allclose(normed, 0.0)


class TestMamaSynthEval:
    """Tests for the MamaSynthEval class."""

    def test_evaluate_identical_images(self, temp_dirs) -> None:
        """Evaluation should return perfect scores for identical images."""
        gt_dir, pred_dir, _, output_file = temp_dirs

        shape = (10, 32, 32)
        create_test_image(gt_dir / "case001.nii.gz", shape, value=100.0)
        create_test_image(pred_dir / "case001.nii.gz", shape, value=100.0)

        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=False,
        )
        results = evaluator.evaluate()

        # GC-compatible keys
        assert "aggregates" in results
        assert "results" in results
        # Legacy keys
        assert "aggregate" in results
        assert "cases" in results
        assert len(results["cases"]) == 1

        case = results["cases"][0]
        assert case["mae"] == 0.0
        assert case["mse"] == 0.0
        assert case["ssim"] == pytest.approx(1.0)
        assert case["ncc"] == pytest.approx(1.0)

        # GC aggregates should have the full-image MSE
        assert METRIC_MSE_FULL in results["aggregates"]

    def test_evaluate_multiple_cases(self, temp_dirs) -> None:
        gt_dir, pred_dir, _, output_file = temp_dirs

        shape = (5, 16, 16)
        for i in range(3):
            create_test_image(gt_dir / f"case{i:03d}.nii.gz", shape, value=50.0)
            create_test_image(pred_dir / f"case{i:03d}.nii.gz", shape, value=50.0)

        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=False,
        )
        results = evaluator.evaluate()

        assert len(results["cases"]) == 3
        assert results["aggregate"]["mae"]["n_samples"] == 3

    def test_missing_predictions_imputed(self, temp_dirs) -> None:
        """Missing predictions should be worst-score imputed, not silently skipped."""
        gt_dir, pred_dir, _, output_file = temp_dirs

        shape = (5, 16, 16)
        create_test_image(gt_dir / "case001.nii.gz", shape)
        create_test_image(gt_dir / "case002.nii.gz", shape)
        create_test_image(pred_dir / "case001.nii.gz", shape)

        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=False,
        )
        results = evaluator.evaluate()

        # Now we have the real case + the imputed one
        assert len(results["cases"]) == 2
        assert results["cases"][0]["case_id"] == "case001"
        # Second entry is the imputed missing case
        assert results["cases"][1]["case_id"] == "case002"
        assert results["cases"][1].get("_imputed") is True
        # missing_predictions key should be present
        assert "missing_predictions" in results
        assert "case002" in results["missing_predictions"]

    def test_output_file_created(self, temp_dirs) -> None:
        gt_dir, pred_dir, _, output_file = temp_dirs

        create_test_image(gt_dir / "test.nii.gz", (5, 16, 16))
        create_test_image(pred_dir / "test.nii.gz", (5, 16, 16))

        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=False,
        )
        evaluator.evaluate()

        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert "aggregates" in data
        assert "results" in data
        assert "aggregate" in data
        assert "cases" in data

    def test_nonexistent_gt_path_raises(self, tmp_path: Path) -> None:
        evaluator = MamaSynthEval(
            ground_truth_path=tmp_path / "nonexistent",
            predictions_path=tmp_path,
            output_file=tmp_path / "metrics.json",
        )
        with pytest.raises(FileNotFoundError, match="Ground truth path"):
            evaluator.evaluate()

    def test_nonexistent_pred_path_raises(self, temp_dirs) -> None:
        gt_dir, _, _, output_file = temp_dirs
        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=gt_dir.parent / "nonexistent",
            output_file=output_file,
        )
        with pytest.raises(FileNotFoundError, match="Predictions path"):
            evaluator.evaluate()

    def test_no_matching_pairs_raises(self, temp_dirs) -> None:
        gt_dir, pred_dir, _, output_file = temp_dirs

        create_test_image(gt_dir / "case001.nii.gz", (5, 16, 16))
        create_test_image(pred_dir / "other_case.nii.gz", (5, 16, 16))

        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
        )
        with pytest.raises(ValueError, match="No matching image pairs"):
            evaluator.evaluate()

    def test_different_file_formats(self, temp_dirs) -> None:
        gt_dir, pred_dir, _, output_file = temp_dirs

        shape = (5, 16, 16)
        create_test_image(gt_dir / "case001.mha", shape)
        create_test_image(pred_dir / "case001.mha", shape)

        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=False,
        )
        results = evaluator.evaluate()
        assert len(results["cases"]) == 1

    def test_full_image_metrics_in_results(self, populated_dirs) -> None:
        """Full-image MSE should be present in results and GC aggregates."""
        gt_dir, pred_dir, _, output_file = populated_dirs

        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=False,
        )
        results = evaluator.evaluate()

        assert "full_image" in results
        assert "mse" in results["full_image"]
        assert results["full_image"]["mse"]["mean"] >= 0

        # GC aggregates
        assert METRIC_MSE_FULL in results["aggregates"]
        assert results["aggregates"][METRIC_MSE_FULL]["mean"] >= 0

    def test_roi_metrics_with_masks(self, populated_dirs) -> None:
        """ROI metrics (SSIM) should be computed when masks are provided."""
        gt_dir, pred_dir, masks_dir, output_file = populated_dirs

        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            masks_path=masks_dir,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=False,
        )
        results = evaluator.evaluate()

        assert "roi" in results
        assert "ssim" in results["roi"]
        assert results["roi"]["ssim"]["mean"] >= 0

        # GC aggregates
        assert METRIC_SSIM_ROI in results["aggregates"]

    def test_segmentation_eval_with_masks(self, populated_dirs) -> None:
        """Segmentation metrics should be computed when masks and flag are set."""
        gt_dir, pred_dir, masks_dir, output_file = populated_dirs

        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            masks_path=masks_dir,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=True,
            enable_classification=False,
        )
        results = evaluator.evaluate()

        assert "segmentation" in results
        assert "dice" in results["segmentation"]
        assert "hd95" in results["segmentation"]


class TestGetStem:
    """Tests for the _get_stem static method."""

    def test_nii_gz(self) -> None:
        assert MamaSynthEval._get_stem(Path("/path/to/case001.nii.gz")) == "case001"

    def test_nii(self) -> None:
        assert MamaSynthEval._get_stem(Path("/path/to/case001.nii")) == "case001"

    def test_mha(self) -> None:
        assert MamaSynthEval._get_stem(Path("/path/to/case001.mha")) == "case001"

    def test_png(self) -> None:
        assert MamaSynthEval._get_stem(Path("/path/to/case001.png")) == "case001"


class TestDatasetNormalizer:
    """Tests for the DatasetNormalizer class."""

    def test_fit_and_transform(self) -> None:
        images = [np.ones((10, 10)) * v for v in [10.0, 20.0, 30.0]]
        normalizer = DatasetNormalizer()
        normalizer.fit(images)
        assert normalizer.mean is not None
        assert normalizer.std is not None
        transformed = normalizer.transform(images[0])
        assert transformed.shape == images[0].shape

    def test_fit_computes_correct_global_stats(self) -> None:
        a = np.zeros((4,)) + 2.0
        b = np.zeros((4,)) + 6.0
        normalizer = DatasetNormalizer()
        normalizer.fit([a, b])
        # Global mean of [2,2,2,2,6,6,6,6] = 4.0
        assert normalizer.mean == pytest.approx(4.0)
        # Global std of [2,2,2,2,6,6,6,6] = 2.0
        assert normalizer.std == pytest.approx(2.0)

    def test_transform_uses_dataset_stats(self) -> None:
        normalizer = DatasetNormalizer()
        normalizer.fit([np.array([2.0, 6.0])])
        result = normalizer.transform(np.array([2.0, 6.0]))
        assert result[0] == pytest.approx(-1.0)
        assert result[1] == pytest.approx(1.0)

    def test_transform_before_fit_raises(self) -> None:
        normalizer = DatasetNormalizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            normalizer.transform(np.ones((5, 5)))


class TestNormalizeIntensityWithParams:
    """Tests for normalize_intensity with explicit mean/std."""

    def test_with_dataset_stats(self) -> None:
        img = np.array([10.0, 20.0, 30.0])
        normed = normalize_intensity(img, mean=20.0, std=10.0)
        assert normed[0] == pytest.approx(-1.0)
        assert normed[1] == pytest.approx(0.0)
        assert normed[2] == pytest.approx(1.0)

    def test_with_zero_std(self) -> None:
        img = np.array([5.0, 5.0, 5.0])
        normed = normalize_intensity(img, mean=5.0, std=0.0)
        assert np.allclose(normed, 0.0)


class TestLoadLabelsCSV:
    """Tests for CSV label loading."""

    def test_csv_label_loading(self, temp_dirs) -> None:
        gt_dir, pred_dir, _, output_file = temp_dirs
        # Create a CSV labels file
        csv_path = gt_dir.parent / "labels.csv"
        csv_path.write_text("case_id,tnbc,luminal\ncase001,1,0\ncase002,0,1\n")

        create_test_image(gt_dir / "case001.nii.gz", (5, 16, 16))
        create_test_image(pred_dir / "case001.nii.gz", (5, 16, 16))

        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            labels_path=csv_path,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=True,  # requires labels
        )
        # _load_labels should work with CSV
        labels = evaluator._load_labels()
        assert labels["case001"]["tnbc"] == 1
        assert labels["case002"]["luminal"] == 1


# ---------------------------------------------------------------------------
# _resize_pred_to_gt
# ---------------------------------------------------------------------------


class TestResizePredToGt:
    """Tests for the evaluation-time resolution mismatch handler."""

    def test_noop_when_shapes_match(self):
        """No resize when pred and gt have the same shape."""
        gt = np.random.rand(64, 64).astype(np.float64)
        pred = np.random.rand(64, 64).astype(np.float64)
        result = MamaSynthEval._resize_pred_to_gt(pred, gt)
        np.testing.assert_array_equal(result, pred)

    def test_resizes_512_to_448(self):
        """Pred at 512×512 is resized to match gt at 448×448."""
        gt = np.random.rand(448, 448).astype(np.float64)
        pred = np.random.rand(512, 512).astype(np.float64)
        result = MamaSynthEval._resize_pred_to_gt(pred, gt)
        assert result.shape == (448, 448)
        assert result.dtype == pred.dtype

    def test_resizes_nonsquare(self):
        """Handles non-square dimensions correctly."""
        gt = np.random.rand(100, 200).astype(np.float64)
        pred = np.random.rand(50, 100).astype(np.float64)
        result = MamaSynthEval._resize_pred_to_gt(pred, gt)
        assert result.shape == (100, 200)

    def test_preserves_dtype(self):
        """Output dtype matches input pred dtype."""
        gt = np.zeros((32, 32), dtype=np.float64)
        pred = np.ones((64, 64), dtype=np.float32)
        result = MamaSynthEval._resize_pred_to_gt(pred, gt)
        assert result.dtype == np.float32


class TestEvaluationWithResizeMismatch:
    """Integration test: evaluation succeeds when pred/gt sizes differ."""

    def test_full_image_metrics_with_mismatched_sizes(self, tmp_path):
        """_evaluate_full_image works when pred is larger than gt."""
        from PIL import Image

        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        gt_dir.mkdir()
        pred_dir.mkdir()

        # GT at 48×48, pred at 64×64
        for name in ("case001.png", "case002.png"):
            gt_img = Image.fromarray(
                np.random.randint(0, 255, (48, 48), dtype=np.uint8)
            )
            gt_img.save(gt_dir / name)
            pred_img = Image.fromarray(
                np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            )
            pred_img.save(pred_dir / name)

        output_file = tmp_path / "metrics.json"
        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=False,
        )
        # Should not raise ValueError about shape mismatch
        results = evaluator.evaluate()
        assert "aggregates" in results


# ---------------------------------------------------------------------------
# _resize_array_to_target (general resize helper)
# ---------------------------------------------------------------------------


class TestResizeArrayToTarget:
    """Tests for the general-purpose _resize_array_to_target helper."""

    def test_noop_when_shapes_match(self):
        arr = np.random.rand(64, 64).astype(np.float64)
        result = MamaSynthEval._resize_array_to_target(arr, (64, 64))
        np.testing.assert_array_equal(result, arr)

    def test_resizes_float_with_bicubic(self):
        arr = np.random.rand(512, 512).astype(np.float64)
        result = MamaSynthEval._resize_array_to_target(arr, (448, 448))
        assert result.shape == (448, 448)
        assert result.dtype == np.float64

    def test_resizes_bool_mask_with_nearest(self):
        """Boolean masks use NEAREST interpolation to stay binary."""
        mask = np.zeros((64, 64), dtype=np.bool_)
        mask[20:40, 20:40] = True
        result = MamaSynthEval._resize_array_to_target(mask, (32, 32))
        assert result.shape == (32, 32)
        assert result.dtype == np.bool_
        # The resized mask should still contain True values
        assert np.any(result)

    def test_resizes_uint8_mask_with_nearest(self):
        """Integer arrays use NEAREST interpolation."""
        arr = np.zeros((64, 64), dtype=np.uint8)
        arr[10:30, 10:30] = 255
        result = MamaSynthEval._resize_array_to_target(arr, (32, 32))
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8
        assert np.any(result == 255)

    def test_preserves_float32_dtype(self):
        arr = np.ones((64, 64), dtype=np.float32)
        result = MamaSynthEval._resize_array_to_target(arr, (32, 32))
        assert result.dtype == np.float32

    def test_resize_pred_to_gt_delegates(self):
        """_resize_pred_to_gt still works as a thin wrapper."""
        gt = np.random.rand(48, 48).astype(np.float64)
        pred = np.random.rand(64, 64).astype(np.float64)
        result = MamaSynthEval._resize_pred_to_gt(pred, gt)
        assert result.shape == (48, 48)


class TestSegmentationWithResizeMismatch:
    """Integration: segmentation works when pred/gt mask sizes differ."""

    def test_segmentation_metrics_with_mismatched_sizes(self, tmp_path):
        """_evaluate_segmentation resizes pred to match GT mask before segmenting."""
        from PIL import Image

        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        masks_dir = tmp_path / "masks"
        gt_dir.mkdir()
        pred_dir.mkdir()
        masks_dir.mkdir()

        # GT and masks at 48×48, pred at 64×64
        rng = np.random.RandomState(42)
        for name in ("case001.png", "case002.png"):
            gt_img = Image.fromarray(
                rng.randint(0, 255, (48, 48), dtype=np.uint8)
            )
            gt_img.save(gt_dir / name)

            pred_img = Image.fromarray(
                rng.randint(0, 255, (64, 64), dtype=np.uint8)
            )
            pred_img.save(pred_dir / name)

            # Mask at GT resolution
            mask_data = np.zeros((48, 48), dtype=np.float32)
            mask_data[15:35, 15:35] = 1.0
            mask_img = Image.fromarray((mask_data * 255).astype(np.uint8))
            mask_img.save(masks_dir / name)

        output_file = tmp_path / "metrics.json"
        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            masks_path=masks_dir,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=True,
            enable_classification=False,
        )
        # Should not raise ValueError about shape mismatch
        results = evaluator.evaluate()
        assert "aggregates" in results

class TestFRDROIPassesMasks:
    """Verify that _evaluate_roi calls official frd_score.compute_frd with masks."""

    def test_frd_receives_file_paths_and_masks(self, tmp_path: Path) -> None:
        """frd_score.compute_frd should receive file-path lists + masks."""
        from PIL import Image
        from unittest.mock import patch

        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        masks_dir = tmp_path / "masks"
        gt_dir.mkdir()
        pred_dir.mkdir()
        masks_dir.mkdir()

        rng = np.random.RandomState(42)
        # Create 3 image pairs (FRD needs >= 2 samples)
        for i in range(3):
            name = f"case_{i:03d}.png"
            arr = rng.randint(0, 255, (48, 48), dtype=np.uint8)
            Image.fromarray(arr).save(gt_dir / name)
            Image.fromarray(rng.randint(0, 255, (48, 48), dtype=np.uint8)).save(
                pred_dir / name
            )
            # Mask with a small foreground region
            mask_data = np.zeros((48, 48), dtype=np.uint8)
            mask_data[15:35, 15:35] = 255
            Image.fromarray(mask_data).save(masks_dir / name)

        output_file = tmp_path / "metrics.json"
        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            masks_path=masks_dir,
            enable_lpips=False,
            enable_frd=True,
            enable_segmentation=False,
            enable_classification=False,
        )

        captured_args: list = []
        captured_kwargs: dict = {}

        def spy_frd(*args, **kwargs):
            captured_args.extend(args)
            captured_kwargs.update(kwargs)
            return 42.0

        with patch("frd_score.compute_frd", side_effect=spy_frd):
            results = evaluator.evaluate()

        # frd_score.compute_frd receives [gt_paths, pred_paths]
        assert len(captured_args) >= 1, "compute_frd must receive paths argument"
        paths_arg = captured_args[0]
        assert isinstance(paths_arg, list) and len(paths_arg) == 2
        gt_paths, pred_paths = paths_arg
        assert len(gt_paths) == 3
        assert len(pred_paths) == 3
        # Each should be a string file path
        for p in gt_paths + pred_paths:
            assert isinstance(p, str), f"Expected str path, got {type(p)}"

        # paths_masks should be [mask_paths, mask_paths]
        assert "paths_masks" in captured_kwargs, (
            "compute_frd must receive paths_masks kwarg"
        )
        mask_lists = captured_kwargs["paths_masks"]
        assert isinstance(mask_lists, list) and len(mask_lists) == 2
        assert len(mask_lists[0]) == 3
        assert len(mask_lists[1]) == 3
        # Both mask lists should reference the same files (same masks for GT & pred)
        assert mask_lists[0] == mask_lists[1]
        for p in mask_lists[0]:
            assert isinstance(p, str), f"Expected str mask path, got {type(p)}"

class TestContrastClassification:
    """Tests for the _evaluate_contrast method."""

    def test_contrast_called_when_model_dir_provided(self, tmp_path: Path) -> None:
        """Contrast evaluation should run when --clf-model-dir-contrast is set."""
        from PIL import Image
        from unittest.mock import patch, MagicMock

        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        precon_dir = tmp_path / "precontrast"
        clf_dir = tmp_path / "clf_contrast"
        gt_dir.mkdir(); pred_dir.mkdir(); precon_dir.mkdir(); clf_dir.mkdir()

        rng = np.random.RandomState(55)
        for i in range(4):
            name = f"case_{i:03d}.png"
            for d in (gt_dir, pred_dir, precon_dir):
                Image.fromarray(
                    rng.randint(0, 255, (32, 32), dtype=np.uint8)
                ).save(d / name)

        # Create a dummy pkl model
        import pickle
        from sklearn.ensemble import RandomForestClassifier
        dummy_model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = rng.rand(10, 93)
        y = np.array([0]*5 + [1]*5)
        dummy_model.fit(X, y)
        with open(clf_dir / "contrast_classifier.pkl", "wb") as f:
            pickle.dump(dummy_model, f)

        output_file = tmp_path / "metrics.json"
        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=True,
            clf_model_dir_contrast=clf_dir,
            precontrast_path=precon_dir,
        )

        # Mock extract_radiomic_features since pyradiomics may not be installed
        fake_feats = rng.rand(93).astype(np.float64)
        with patch(
            "eval.evaluation.extract_radiomic_features",
            return_value=fake_feats,
            create=True,
        ), patch(
            "eval.frd.extract_radiomic_features",
            return_value=fake_feats,
            create=True,
        ):
            results = evaluator.evaluate()

        agg = results.get("aggregates", {})
        # AUROC contrast should be present
        assert "auroc_contrast" in agg, (
            f"Expected auroc_contrast in aggregates, got keys: {list(agg.keys())}"
        )
        # Should be a float between 0 and 1
        assert 0.0 <= agg["auroc_contrast"] <= 1.0

    def test_contrast_skipped_without_precontrast_path(self, tmp_path: Path) -> None:
        """Contrast should produce a note when --precontrast-path is missing."""
        from PIL import Image

        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        clf_dir = tmp_path / "clf_contrast"
        gt_dir.mkdir(); pred_dir.mkdir(); clf_dir.mkdir()

        rng = np.random.RandomState(56)
        for i in range(2):
            name = f"case_{i:03d}.png"
            for d in (gt_dir, pred_dir):
                Image.fromarray(
                    rng.randint(0, 255, (32, 32), dtype=np.uint8)
                ).save(d / name)

        # Create dummy model
        import pickle
        from sklearn.ensemble import RandomForestClassifier
        dummy_model = RandomForestClassifier(n_estimators=2, random_state=42)
        dummy_model.fit(rng.rand(10, 5), np.array([0]*5 + [1]*5))
        with open(clf_dir / "contrast_classifier.pkl", "wb") as f:
            pickle.dump(dummy_model, f)

        output_file = tmp_path / "metrics.json"
        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=True,
            clf_model_dir_contrast=clf_dir,
            # No precontrast_path!
        )
        results = evaluator.evaluate()
        agg = results.get("aggregates", {})
        # Should NOT have contrast AUROC without pre-contrast images
        assert "auroc_contrast" not in agg

    def test_contrast_independent_of_labels_path(self, tmp_path: Path) -> None:
        """Contrast evaluation should NOT require --labels-path."""
        from PIL import Image
        from unittest.mock import patch

        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        precon_dir = tmp_path / "precontrast"
        clf_dir = tmp_path / "clf_contrast"
        gt_dir.mkdir(); pred_dir.mkdir(); precon_dir.mkdir(); clf_dir.mkdir()

        rng = np.random.RandomState(57)
        for i in range(4):
            name = f"case_{i:03d}.png"
            for d in (gt_dir, pred_dir, precon_dir):
                Image.fromarray(
                    rng.randint(0, 255, (32, 32), dtype=np.uint8)
                ).save(d / name)

        import pickle
        from sklearn.ensemble import RandomForestClassifier
        dummy_model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = rng.rand(10, 93)
        y = np.array([0]*5 + [1]*5)
        dummy_model.fit(X, y)
        with open(clf_dir / "contrast_classifier.pkl", "wb") as f:
            pickle.dump(dummy_model, f)

        output_file = tmp_path / "metrics.json"
        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=True,
            clf_model_dir_contrast=clf_dir,
            precontrast_path=precon_dir,
            labels_path=None,  # Explicitly no labels
        )

        # Mock extract_radiomic_features since pyradiomics may not be installed
        fake_feats = rng.rand(93).astype(np.float64)
        with patch(
            "eval.evaluation.extract_radiomic_features",
            return_value=fake_feats,
            create=True,
        ), patch(
            "eval.frd.extract_radiomic_features",
            return_value=fake_feats,
            create=True,
        ):
            results = evaluator.evaluate()

        agg = results.get("aggregates", {})
        assert "auroc_contrast" in agg, (
            "Contrast classification should work without labels_path"
        )


class TestCSVLabelParser:
    """Tests that CSV label parser preserves extra columns."""

    def test_csv_includes_extra_columns(self, tmp_path: Path) -> None:
        """Extra columns beyond tnbc/luminal should be preserved."""
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text(
            "case_id,tnbc,luminal,contrast\n"
            "case_001,0,1,1\n"
            "case_002,1,0,0\n"
        )
        evaluator = MamaSynthEval(
            ground_truth_path=tmp_path,
            predictions_path=tmp_path,
            output_file=tmp_path / "out.json",
            labels_path=csv_path,
        )
        labels = evaluator._load_labels()
        assert labels["case_001"]["contrast"] == 1
        assert labels["case_002"]["contrast"] == 0
        # Standard columns still work
        assert labels["case_001"]["tnbc"] == 0
        assert labels["case_001"]["luminal"] == 1


class TestSSIMROIUseMask:
    """Tests that _evaluate_roi passes mask to compute_ssim."""

    def test_ssim_receives_mask(self, tmp_path: Path) -> None:
        """compute_ssim in ROI eval should receive the roi_mask argument."""
        from PIL import Image
        from unittest.mock import patch

        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        masks_dir = tmp_path / "masks"
        gt_dir.mkdir(); pred_dir.mkdir(); masks_dir.mkdir()

        rng = np.random.RandomState(88)
        for i in range(2):
            name = f"case_{i:03d}.png"
            Image.fromarray(
                rng.randint(0, 255, (48, 48), dtype=np.uint8)
            ).save(gt_dir / name)
            Image.fromarray(
                rng.randint(0, 255, (48, 48), dtype=np.uint8)
            ).save(pred_dir / name)
            mask_data = np.zeros((48, 48), dtype=np.uint8)
            mask_data[15:35, 15:35] = 255
            Image.fromarray(mask_data).save(masks_dir / name)

        output_file = tmp_path / "metrics.json"
        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            masks_path=masks_dir,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=False,
        )

        ssim_calls = []
        original_ssim = __import__("eval.metrics", fromlist=["compute_ssim"]).compute_ssim

        def spy_ssim(*args, **kwargs):
            ssim_calls.append(kwargs)
            return original_ssim(*args, **kwargs)

        with patch("eval.evaluation.compute_ssim", side_effect=spy_ssim):
            evaluator.evaluate()

        # Filter to only calls that include 'mask' (ROI path).
        # Legacy pairwise path calls compute_ssim without mask.
        roi_calls = [kw for kw in ssim_calls if "mask" in kw]
        assert len(roi_calls) >= 2, (
            f"Expected >= 2 ROI SSIM calls with mask, got {len(roi_calls)} "
            f"out of {len(ssim_calls)} total calls"
        )
        for call_kwargs in roi_calls:
            assert call_kwargs["mask"] is not None


# ------------------------------------------------------------------
# FRD z-score standardisation
# ------------------------------------------------------------------

class TestFRDStandardization:
    """Verify that FRD with z-score standardisation returns reasonable values."""

    def test_frd_identical_distributions_near_zero(self) -> None:
        """Identical image sets should yield FRD ~ 0 (well below 1e6)."""
        pytest.importorskip("radiomics")
        from eval.frd import compute_frd

        rng = np.random.RandomState(42)
        imgs = [rng.rand(64, 64).astype(np.float64) * 1000 for _ in range(4)]
        frd_val = compute_frd(imgs, imgs)
        # With standardisation, identical distributions -> FRD ~ 0
        assert frd_val < 1.0, f"FRD for identical images should be ~0, got {frd_val}"

    def test_frd_different_distributions_moderate(self) -> None:
        """Different distributions should yield a moderate FRD (not 1e14)."""
        pytest.importorskip("radiomics")
        from eval.frd import compute_frd

        rng = np.random.RandomState(42)
        real = [rng.rand(64, 64).astype(np.float64) * 500 for _ in range(5)]
        synth = [(rng.rand(64, 64).astype(np.float64) * 500) + 200 for _ in range(5)]
        frd_val = compute_frd(real, synth)
        # After standardisation the FRD should be moderate, not astronomical
        assert frd_val < 1e6, (
            f"FRD between different distributions should be moderate, got {frd_val}"
        )

    def test_compute_frd_from_features_standardized(self) -> None:
        """compute_frd_from_features should also apply z-score standardization."""
        from eval.frd import compute_frd_from_features

        rng = np.random.RandomState(99)
        # Create features with wildly different scales (mimicking pyradiomics)
        real = rng.randn(20, 93).astype(np.float64)
        real[:, 0] *= 1e8  # Energy-scale feature
        real[:, -1] *= 1e-2  # Kurtosis-scale feature

        synth = rng.randn(20, 93).astype(np.float64)
        synth[:, 0] *= 1e8
        synth[:, -1] *= 1e-2

        frd_val = compute_frd_from_features(real, synth)
        assert frd_val < 1e6, f"Expected moderate FRD after standardization, got {frd_val}"


# ------------------------------------------------------------------
# Tumor ROI classification without labels_path
# ------------------------------------------------------------------

class TestTumorROIStandalone:
    """Tumor ROI classification must run without molecular-subtype labels."""

    def test_tumor_roi_called_without_labels(self, tmp_path: Path) -> None:
        """_evaluate_tumor_roi should be invoked even when labels_path is None."""
        from unittest.mock import patch
        from PIL import Image

        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        masks_dir = tmp_path / "masks"
        gt_dir.mkdir(); pred_dir.mkdir(); masks_dir.mkdir()

        rng = np.random.RandomState(77)
        for i in range(3):
            name = f"case_{i:03d}.png"
            Image.fromarray(
                rng.randint(0, 255, (48, 48), dtype=np.uint8)
            ).save(gt_dir / name)
            Image.fromarray(
                rng.randint(0, 255, (48, 48), dtype=np.uint8)
            ).save(pred_dir / name)
            mask_data = np.zeros((48, 48), dtype=np.uint8)
            mask_data[10:38, 10:38] = 255
            Image.fromarray(mask_data).save(masks_dir / name)

        output_file = tmp_path / "metrics.json"
        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            masks_path=masks_dir,
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=True,
            # No labels_path!
        )

        tumor_roi_called = []

        def spy(*args, **kwargs):
            tumor_roi_called.append(True)
            return {"aggregates": {}, "detail": {"note_tumor_roi": "spy"}}

        with patch.object(evaluator, "_evaluate_tumor_roi", side_effect=spy):
            evaluator.evaluate()

        assert len(tumor_roi_called) >= 1, (
            "_evaluate_tumor_roi should be called even without labels_path"
        )


# ------------------------------------------------------------------
# Radiomic feature caching
# ------------------------------------------------------------------

class TestFeatureCaching:
    """Verify that _extract_features_cached avoids duplicate extraction."""

    def test_cache_avoids_redundant_extraction(self) -> None:
        """Calling _extract_features_cached twice with same key should
        only invoke extract_radiomic_features once."""
        from unittest.mock import patch

        evaluator = MamaSynthEval.__new__(MamaSynthEval)
        evaluator._feature_cache = {}  # enable cache

        fake_feats = np.arange(93, dtype=np.float64)
        call_count = [0]

        def mock_extract(image, mask=None):
            call_count[0] += 1
            return fake_feats.copy()

        with patch("eval.frd.extract_radiomic_features", side_effect=mock_extract):
            img = np.random.rand(48, 48).astype(np.float64)
            f1 = evaluator._extract_features_cached(img, "key1")
            f2 = evaluator._extract_features_cached(img, "key1")

        assert call_count[0] == 1, (
            f"Expected 1 extraction call with caching, got {call_count[0]}"
        )
        np.testing.assert_array_equal(f1, f2)

    def test_different_masks_different_cache_keys(self) -> None:
        """Same image but different masks should cache separately."""
        from unittest.mock import patch

        evaluator = MamaSynthEval.__new__(MamaSynthEval)
        evaluator._feature_cache = {}

        call_count = [0]

        def mock_extract(image, mask=None):
            call_count[0] += 1
            if mask is not None and mask.any():
                return np.ones(93, dtype=np.float64)
            return np.zeros(93, dtype=np.float64)

        with patch("eval.frd.extract_radiomic_features", side_effect=mock_extract):
            img = np.random.rand(48, 48).astype(np.float64)
            mask_a = np.zeros((48, 48), dtype=bool)
            mask_a[10:30, 10:30] = True
            mask_b = np.zeros((48, 48), dtype=bool)
            mask_b[20:40, 20:40] = True

            f1 = evaluator._extract_features_cached(img, "img1", mask=mask_a)
            f2 = evaluator._extract_features_cached(img, "img1", mask=mask_b)
            f3 = evaluator._extract_features_cached(img, "img1", mask=mask_a)

        assert call_count[0] == 2, (
            f"Expected 2 extraction calls (2 different masks), got {call_count[0]}"
        )
        np.testing.assert_array_equal(f1, f3)


# ---------------------------------------------------------------------------
# PNG → MHA conversion for frd-score
# ---------------------------------------------------------------------------


class TestEnsureMha:
    """Tests for MamaSynthEval._ensure_mha PNG→MHA conversion."""

    def test_converts_png_to_mha(self, tmp_path: Path) -> None:
        from PIL import Image

        png_path = tmp_path / "test.png"
        Image.fromarray(
            np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        ).save(png_path)

        mha_dir = tmp_path / "mha_out"
        mha_dir.mkdir()

        result = MamaSynthEval._ensure_mha(
            [str(png_path)], str(mha_dir), "img"
        )

        assert len(result) == 1
        assert result[0].endswith(".mha")
        assert Path(result[0]).exists()

        # Verify the MHA is readable by SimpleITK
        import SimpleITK as sitk

        img = sitk.ReadImage(result[0])
        arr = sitk.GetArrayFromImage(img)
        assert arr.shape == (32, 32)

    def test_nifti_paths_unchanged(self, tmp_path: Path) -> None:
        nifti_path = str(tmp_path / "vol.nii.gz")

        result = MamaSynthEval._ensure_mha(
            [nifti_path], str(tmp_path), "img"
        )

        assert result == [nifti_path]

    def test_mixed_paths(self, tmp_path: Path) -> None:
        from PIL import Image

        png_path = tmp_path / "img.png"
        Image.fromarray(
            np.random.randint(0, 255, (16, 16), dtype=np.uint8)
        ).save(png_path)
        nifti_path = str(tmp_path / "vol.nii.gz")

        mha_dir = tmp_path / "mha_out"
        mha_dir.mkdir()

        result = MamaSynthEval._ensure_mha(
            [str(png_path), nifti_path], str(mha_dir), "mix"
        )

        assert len(result) == 2
        assert result[0].endswith(".mha")
        assert result[1] == nifti_path


class TestFRDROIConvertsToMha:
    """Verify that _evaluate_roi converts PNG to MHA before calling frd-score."""

    def test_frd_receives_mha_paths_when_input_is_png(
        self, tmp_path: Path
    ) -> None:
        from PIL import Image
        from unittest.mock import patch

        gt_dir = tmp_path / "gt"
        pred_dir = tmp_path / "pred"
        masks_dir = tmp_path / "masks"
        gt_dir.mkdir()
        pred_dir.mkdir()
        masks_dir.mkdir()

        rng = np.random.RandomState(99)
        for i in range(3):
            name = f"case_{i:03d}.png"
            Image.fromarray(
                rng.randint(0, 255, (48, 48), dtype=np.uint8)
            ).save(gt_dir / name)
            Image.fromarray(
                rng.randint(0, 255, (48, 48), dtype=np.uint8)
            ).save(pred_dir / name)
            mask_data = np.zeros((48, 48), dtype=np.uint8)
            mask_data[10:30, 10:30] = 255
            Image.fromarray(mask_data).save(masks_dir / name)

        output_file = tmp_path / "metrics.json"
        evaluator = MamaSynthEval(
            ground_truth_path=gt_dir,
            predictions_path=pred_dir,
            output_file=output_file,
            masks_path=masks_dir,
            enable_lpips=False,
            enable_frd=True,
            enable_segmentation=False,
            enable_classification=False,
        )

        captured_args: list = []

        def spy_frd(*args, **kwargs):
            captured_args.extend(args)
            return 42.0

        with patch("frd_score.compute_frd", side_effect=spy_frd):
            evaluator.evaluate()

        assert len(captured_args) >= 1
        gt_paths, pred_paths = captured_args[0]
        # All paths should be .mha (converted from PNG)
        for p in gt_paths + pred_paths:
            assert p.endswith(".mha"), f"Expected .mha path, got {p}"
