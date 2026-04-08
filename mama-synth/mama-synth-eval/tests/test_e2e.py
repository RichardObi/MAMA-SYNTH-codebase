#  Copyright 2025 mama-sia-eval contributors
#  Licensed under the Apache License, Version 2.0

"""End-to-end integration tests using artificial test data.

These tests exercise the full pipeline:
  data generation → evaluation → visualization → result validation.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mama_sia_eval.evaluation import MamaSiaEval, DatasetNormalizer
from mama_sia_eval.generate_test_data import generate_case, save_dataset
from mama_sia_eval.visualization import ResultVisualizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory) -> Path:
    """Generate a small artificial dataset (module-scoped for efficiency)."""
    root = tmp_path_factory.mktemp("e2e_data")
    save_dataset(root, n_cases=6, shape=(32, 32))
    return root


@pytest.fixture(scope="module")
def eval_results(synthetic_dataset: Path) -> dict:
    """Run the full evaluation on the synthetic dataset."""
    out = synthetic_dataset / "output" / "metrics.json"
    evaluator = MamaSiaEval(
        ground_truth_path=synthetic_dataset / "ground-truth",
        predictions_path=synthetic_dataset / "predictions",
        output_file=out,
        masks_path=synthetic_dataset / "masks",
        labels_path=synthetic_dataset / "labels.csv",
        enable_lpips=False,
        enable_frd=False,
        enable_segmentation=True,
        enable_classification=False,  # No classifier model available
    )
    return evaluator.evaluate()


# ---------------------------------------------------------------------------
# Data generation tests
# ---------------------------------------------------------------------------


class TestDataGeneration:
    """Tests for the artificial data generator."""

    def test_generate_single_case(self) -> None:
        case = generate_case("test001", shape=(32, 32), seed=0)
        assert case["precontrast"].shape == (32, 32)
        assert case["postcontrast_real"].shape == (32, 32)
        assert case["postcontrast_synth"].shape == (32, 32)
        assert case["tumor_mask"].shape == (32, 32)
        assert case["tumor_mask"].dtype == np.uint8
        assert "tnbc" in case["labels"]
        assert "luminal" in case["labels"]

    def test_synthetic_differs_from_real(self) -> None:
        case = generate_case("test002", shape=(32, 32), seed=1)
        assert not np.array_equal(
            case["postcontrast_real"], case["postcontrast_synth"]
        )

    def test_tumor_present_in_mask(self) -> None:
        case = generate_case("test003", shape=(64, 64), seed=2)
        assert case["tumor_mask"].sum() > 0

    def test_save_dataset_creates_files(self, tmp_path: Path) -> None:
        save_dataset(tmp_path, n_cases=3, shape=(16, 16))
        assert (tmp_path / "ground-truth").exists()
        assert (tmp_path / "predictions").exists()
        assert (tmp_path / "masks").exists()
        assert (tmp_path / "labels.csv").exists()
        assert (tmp_path / "labels.json").exists()
        gt_files = list((tmp_path / "ground-truth").iterdir())
        assert len(gt_files) == 3

    def test_labels_csv_valid(self, synthetic_dataset: Path) -> None:
        csv_path = synthetic_dataset / "labels.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 6
        assert "case_id" in rows[0]
        assert "tnbc" in rows[0]
        assert "luminal" in rows[0]

    def test_labels_json_valid(self, synthetic_dataset: Path) -> None:
        with open(synthetic_dataset / "labels.json") as f:
            labels = json.load(f)
        assert len(labels) == 6
        for case_id, data in labels.items():
            assert "tnbc" in data
            assert "luminal" in data

    def test_deterministic_generation(self) -> None:
        c1 = generate_case("x", shape=(16, 16), seed=99)
        c2 = generate_case("x", shape=(16, 16), seed=99)
        np.testing.assert_array_equal(c1["precontrast"], c2["precontrast"])
        np.testing.assert_array_equal(c1["postcontrast_real"], c2["postcontrast_real"])


# ---------------------------------------------------------------------------
# E2E evaluation tests
# ---------------------------------------------------------------------------


class TestE2EPipeline:
    """End-to-end evaluation pipeline tests."""

    def test_results_has_expected_keys(self, eval_results: dict) -> None:
        assert "aggregate" in eval_results
        assert "cases" in eval_results
        assert "full_image" in eval_results

    def test_all_cases_evaluated(self, eval_results: dict) -> None:
        assert len(eval_results["cases"]) == 6

    def test_aggregate_metrics_valid(self, eval_results: dict) -> None:
        agg = eval_results["aggregate"]
        for metric in ("mae", "mse", "psnr", "ssim", "ncc"):
            assert metric in agg
            assert "mean" in agg[metric]
            assert np.isfinite(agg[metric]["mean"])

    def test_full_image_mse(self, eval_results: dict) -> None:
        full = eval_results["full_image"]
        assert "mse" in full
        assert full["mse"]["mean"] >= 0

    def test_roi_metrics_present(self, eval_results: dict) -> None:
        assert "roi" in eval_results
        assert "mse" in eval_results["roi"]

    def test_segmentation_metrics_present(self, eval_results: dict) -> None:
        assert "segmentation" in eval_results
        seg = eval_results["segmentation"]
        assert "dice" in seg
        assert "hd95" in seg
        assert seg["dice"]["mean"] >= 0
        assert seg["dice"]["mean"] <= 1.0

    def test_output_json_valid(self, synthetic_dataset: Path) -> None:
        out = synthetic_dataset / "output" / "metrics.json"
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert "aggregate" in data

    def test_dataset_normalizer_consistency(self, synthetic_dataset: Path) -> None:
        """DatasetNormalizer should produce consistent results."""
        import SimpleITK as sitk

        gt_dir = synthetic_dataset / "ground-truth"
        images = []
        for f in sorted(gt_dir.glob("*.nii.gz")):
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(f)))
            images.append(arr.astype(np.float64))

        norm = DatasetNormalizer()
        norm.fit(images)
        t1 = norm.transform(images[0])
        t2 = norm.transform(images[0])
        np.testing.assert_array_equal(t1, t2)


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------


class TestVisualization:
    """Tests for the visualization module."""

    def test_summary_table(self, eval_results: dict, tmp_path: Path) -> None:
        viz = ResultVisualizer(eval_results, output_dir=tmp_path)
        paths = viz.summary_table()
        assert len(paths) == 2
        assert (tmp_path / "summary.txt").exists()
        assert (tmp_path / "summary.html").exists()

    def test_per_case_table(self, eval_results: dict, tmp_path: Path) -> None:
        viz = ResultVisualizer(eval_results, output_dir=tmp_path)
        paths = viz.per_case_table()
        assert len(paths) == 2
        assert (tmp_path / "per_case.csv").exists()
        assert (tmp_path / "per_case.html").exists()

    def test_bar_charts(self, eval_results: dict, tmp_path: Path) -> None:
        viz = ResultVisualizer(eval_results, output_dir=tmp_path)
        paths = viz.metric_bar_charts()
        # Should have at least some bar charts
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"

    def test_radar_plot(self, eval_results: dict, tmp_path: Path) -> None:
        viz = ResultVisualizer(eval_results, output_dir=tmp_path)
        paths = viz.radar_plot()
        # May be empty if <3 groups
        for p in paths:
            assert p.exists()

    def test_generate_all(self, eval_results: dict, tmp_path: Path) -> None:
        viz = ResultVisualizer(eval_results, output_dir=tmp_path)
        paths = viz.generate_all()
        assert len(paths) >= 4

    def test_from_json_file(self, synthetic_dataset: Path, tmp_path: Path) -> None:
        json_path = synthetic_dataset / "output" / "metrics.json"
        viz = ResultVisualizer(json_path, output_dir=tmp_path)
        paths = viz.summary_table()
        assert len(paths) == 2


# ---------------------------------------------------------------------------
# Missing prediction imputation E2E
# ---------------------------------------------------------------------------


class TestMissingPredictionE2E:
    """Test worst-score imputation end-to-end."""

    def test_missing_predictions_imputed(self, synthetic_dataset: Path) -> None:
        """Remove one prediction and verify imputation."""
        import shutil

        tmp = synthetic_dataset.parent / "missing_test"
        if tmp.exists():
            shutil.rmtree(tmp)

        # Copy the dataset
        shutil.copytree(synthetic_dataset, tmp)

        # Remove one prediction
        pred_dir = tmp / "predictions"
        preds = sorted(pred_dir.glob("*.nii.gz"))
        removed = preds[-1]
        removed_stem = removed.name.replace(".nii.gz", "")
        removed.unlink()

        out = tmp / "output2" / "metrics.json"
        evaluator = MamaSiaEval(
            ground_truth_path=tmp / "ground-truth",
            predictions_path=pred_dir,
            output_file=out,
            masks_path=tmp / "masks",
            enable_lpips=False,
            enable_frd=False,
            enable_segmentation=False,
            enable_classification=False,
        )
        results = evaluator.evaluate()

        assert "missing_predictions" in results
        assert removed_stem in results["missing_predictions"]
        # Should have n_cases entries (including imputed)
        assert len(results["cases"]) == 6
        imputed = [c for c in results["cases"] if c.get("_imputed")]
        assert len(imputed) == 1
        assert imputed[0]["case_id"] == removed_stem
