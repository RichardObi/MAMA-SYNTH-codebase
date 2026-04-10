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
