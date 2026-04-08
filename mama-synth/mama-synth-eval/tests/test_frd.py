"""Unit tests for Fréchet Radiomics Distance (FRD)."""

import numpy as np
import pytest

from mama_sia_eval.frd import compute_frechet_distance, compute_frd_from_features


class TestFrechetDistance:
    """Tests for multivariate Gaussian Fréchet distance."""

    def test_identical_distributions(self) -> None:
        """FD should be 0 for identical distributions."""
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.eye(3)
        fd = compute_frechet_distance(mu, sigma, mu, sigma)
        assert fd == pytest.approx(0.0, abs=1e-6)

    def test_different_means(self) -> None:
        """FD should reflect mean difference."""
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([3.0, 4.0])
        sigma = np.eye(2)
        fd = compute_frechet_distance(mu1, sigma, mu2, sigma)
        # ||mu1 - mu2||^2 = 25; trace term = 0 for identical covariances
        assert fd == pytest.approx(25.0, abs=1e-3)

    def test_non_negative(self) -> None:
        """FD should always be non-negative."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            d = 5
            mu1 = rng.randn(d)
            mu2 = rng.randn(d)
            A = rng.randn(d, d)
            B = rng.randn(d, d)
            sigma1 = A @ A.T + 0.1 * np.eye(d)
            sigma2 = B @ B.T + 0.1 * np.eye(d)
            fd = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
            assert fd >= -1e-6

    def test_symmetry(self) -> None:
        """FD should be symmetric."""
        rng = np.random.RandomState(123)
        d = 4
        mu1, mu2 = rng.randn(d), rng.randn(d)
        A, B = rng.randn(d, d), rng.randn(d, d)
        sigma1 = A @ A.T + 0.1 * np.eye(d)
        sigma2 = B @ B.T + 0.1 * np.eye(d)
        fd12 = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
        fd21 = compute_frechet_distance(mu2, sigma2, mu1, sigma1)
        assert fd12 == pytest.approx(fd21, rel=1e-4)


class TestFRDFromFeatures:
    """Tests for FRD computation from pre-extracted feature matrices."""

    def test_identical_features(self) -> None:
        """FRD should be near 0 for identical feature sets."""
        rng = np.random.RandomState(42)
        features = rng.randn(20, 10)
        frd = compute_frd_from_features(features, features)
        assert frd == pytest.approx(0.0, abs=1e-3)

    def test_different_features(self) -> None:
        """FRD should be positive for different feature sets."""
        rng = np.random.RandomState(42)
        real = rng.randn(20, 10)
        synth = rng.randn(20, 10) + 5.0
        frd = compute_frd_from_features(real, synth)
        assert frd > 1.0

    def test_similar_features_lower_frd(self) -> None:
        """More similar features should yield lower FRD."""
        rng = np.random.RandomState(42)
        real = rng.randn(30, 10)
        synth_close = real + rng.randn(30, 10) * 0.1
        synth_far = real + rng.randn(30, 10) * 5.0
        frd_close = compute_frd_from_features(real, synth_close)
        frd_far = compute_frd_from_features(real, synth_far)
        assert frd_close < frd_far

    def test_too_few_samples_raises(self) -> None:
        """FRD should raise with fewer than 2 samples."""
        single = np.array([[1.0, 2.0]])
        multi = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="at least 2"):
            compute_frd_from_features(single, multi)

    def test_dimension_mismatch_raises(self) -> None:
        features1 = np.random.randn(10, 5)
        features2 = np.random.randn(10, 3)
        with pytest.raises(ValueError, match="dimension mismatch"):
            compute_frd_from_features(features1, features2)

    def test_handles_nan_features(self) -> None:
        """NaN features should be replaced with 0 and not crash."""
        rng = np.random.RandomState(42)
        real = rng.randn(10, 5)
        synth = rng.randn(10, 5)
        synth[0, 0] = np.nan
        synth[1, 2] = np.inf
        frd = compute_frd_from_features(real, synth)
        assert np.isfinite(frd)

    def test_realistic_radiomics_scale(self) -> None:
        """Test with features mimicking typical radiomics feature ranges."""
        rng = np.random.RandomState(42)
        # Simulate firstorder + texture features (varying scales)
        real = np.column_stack([
            rng.normal(500, 100, 20),     # Mean intensity
            rng.normal(50, 20, 20),       # Std intensity
            rng.normal(0.3, 0.1, 20),     # Entropy
            rng.normal(100, 30, 20),      # GLCM contrast
            rng.normal(0.8, 0.05, 20),    # GLCM homogeneity
        ])
        synth = np.column_stack([
            rng.normal(480, 110, 20),     # Slightly different
            rng.normal(55, 22, 20),
            rng.normal(0.32, 0.12, 20),
            rng.normal(105, 35, 20),
            rng.normal(0.78, 0.06, 20),
        ])
        frd = compute_frd_from_features(real, synth)
        assert frd >= 0
        assert np.isfinite(frd)
