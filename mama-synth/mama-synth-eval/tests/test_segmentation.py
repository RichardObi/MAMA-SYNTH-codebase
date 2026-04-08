"""Unit tests for segmentation module."""

import numpy as np
import pytest

from mama_sia_eval.segmentation import (
    ThresholdSegmenter,
    evaluate_segmentation_pair,
    evaluate_segmentation,
)


class TestThresholdSegmenter:
    """Tests for the baseline threshold-based segmenter."""

    def test_default_segmentation(self) -> None:
        rng = np.random.RandomState(42)
        # Create image with a bright region
        img = rng.normal(100, 30, (64, 64)).astype(np.float32)
        img[20:40, 20:40] = 500  # bright tumor-like region
        seg = ThresholdSegmenter()
        mask = seg.predict(img)
        assert mask.shape == img.shape
        assert mask.dtype == bool or mask.dtype == np.uint8
        # The bright region should mostly be detected
        bright_region = mask[20:40, 20:40]
        assert bright_region.sum() > 0.5 * bright_region.size

    def test_custom_percentile(self) -> None:
        img = np.zeros((32, 32), dtype=np.float32)
        img[10:20, 10:20] = 100.0
        seg_low = ThresholdSegmenter(threshold_percentile=50)
        seg_high = ThresholdSegmenter(threshold_percentile=99)
        mask_low = seg_low.predict(img)
        mask_high = seg_high.predict(img)
        # Lower percentile → more pixels above threshold
        assert mask_low.sum() >= mask_high.sum()


class TestEvaluateSegmentationPair:
    """Tests for pairwise segmentation evaluation."""

    def test_perfect_match(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        result = evaluate_segmentation_pair(mask, mask)
        assert result["dice"] == pytest.approx(1.0)
        assert result["hd95"] == pytest.approx(0.0, abs=1.0)

    def test_no_overlap(self) -> None:
        pred = np.zeros((64, 64), dtype=np.uint8)
        pred[0:10, 0:10] = 1
        gt = np.zeros((64, 64), dtype=np.uint8)
        gt[50:60, 50:60] = 1
        result = evaluate_segmentation_pair(pred, gt)
        assert result["dice"] == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        gt = np.zeros((64, 64), dtype=np.uint8)
        gt[10:30, 10:30] = 1
        pred = np.zeros((64, 64), dtype=np.uint8)
        pred[20:40, 20:40] = 1
        result = evaluate_segmentation_pair(pred, gt)
        assert 0.0 < result["dice"] < 1.0

    def test_result_keys(self) -> None:
        mask = np.ones((16, 16), dtype=np.uint8)
        result = evaluate_segmentation_pair(mask, mask)
        assert "dice" in result
        assert "hd95" in result


class TestEvaluateSegmentation:
    """Tests for batch segmentation evaluation."""

    def test_with_precomputed_masks(self) -> None:
        gt_masks = [np.zeros((32, 32), dtype=np.uint8) for _ in range(3)]
        pred_masks = [np.zeros((32, 32), dtype=np.uint8) for _ in range(3)]
        dummy_images = [np.zeros((32, 32), dtype=np.float32) for _ in range(3)]
        for i in range(3):
            gt_masks[i][5:15, 5:15] = 1
            pred_masks[i][5:15, 5:15] = 1
        results = evaluate_segmentation(
            synthetic_images=dummy_images,
            gt_masks=gt_masks,
            pred_masks=pred_masks,
        )
        assert len(results["dice"]) == 3
        for d in results["dice"]:
            assert d == pytest.approx(1.0)

    def test_mixed_quality(self) -> None:
        gt = np.zeros((32, 32), dtype=np.uint8)
        gt[5:15, 5:15] = 1
        perfect_pred = gt.copy()
        bad_pred = np.zeros_like(gt)
        dummy_images = [np.zeros((32, 32), dtype=np.float32)] * 2
        results = evaluate_segmentation(
            synthetic_images=dummy_images,
            gt_masks=[gt, gt],
            pred_masks=[perfect_pred, bad_pred],
        )
        assert results["dice"][0] > results["dice"][1]
