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
Fréchet Radiomics Distance (FRD) for evaluating synthetic DCE-MRI images.

FRD measures distributional similarity between real and synthetic image sets
in a radiomics feature space. It uses IBSI-compliant pyradiomics feature sets
(firstorder, GLCM, GLRLM, GLDM, GLSZM, NGTDM) as described in:

    Konz et al., "Fréchet Radiomic Distance (FRD): A versatile metric
    for comparing medical imaging datasets", 2025.
    https://github.com/RichardObi/frd-score

Reference: [18] in the MAMA-SYNTH challenge description.
"""

import hashlib
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

logger = logging.getLogger(__name__)

# Pyradiomics feature classes used in FRD v1 (IBSI-compliant)
FRD_FEATURE_CLASSES = [
    "firstorder",
    "glcm",
    "glrlm",
    "gldm",
    "glszm",
    "ngtdm",
]

# Use default binWidth from frd-score library (25)
FRD_DEFAULT_BIN_WIDTH = 25


def _extract_single(
    image: NDArray[np.floating],
    mask: Optional[NDArray[np.bool_]],
    feature_classes: list[str],
    bin_width: int,
) -> NDArray[np.floating]:
    """Worker function for parallel feature extraction (must be top-level for pickle)."""
    return extract_radiomic_features(
        image, mask=mask, feature_classes=feature_classes, bin_width=bin_width
    )


# ---------------------------------------------------------------------------
# Cached pyradiomics extractor
# ---------------------------------------------------------------------------
# Creating a RadiomicsFeatureExtractor is expensive: it imports many
# sub-modules, builds filter kernels, etc.  We cache configured instances
# keyed by (feature_classes, bin_width, was_2d) so repeated calls with the
# same configuration reuse the same object.  The cache lives in the module
# and is therefore per-process (safe for ProcessPoolExecutor workers which
# each get their own copy via forking/spawning).

_EXTRACTOR_CACHE: dict[tuple, object] = {}


def _get_cached_extractor(
    feature_classes: tuple[str, ...],
    bin_width: int,
    was_2d: bool,
) -> object:
    """Return a cached RadiomicsFeatureExtractor, creating one if needed."""
    key = (feature_classes, bin_width, was_2d)
    extractor = _EXTRACTOR_CACHE.get(key)
    if extractor is not None:
        return extractor

    from radiomics import featureextractor

    settings = {
        "binWidth": bin_width,
        "resampledPixelSpacing": None,  # No resampling
        "interpolator": "sitkBSpline",
        "minimumROIDimensions": 1,
        "minimumROISize": 1,
        "force2D": was_2d,
        "force2Ddimension": 0 if was_2d else None,
    }
    ext = featureextractor.RadiomicsFeatureExtractor(**settings)
    ext.disableAllFeatures()
    for fc in feature_classes:
        ext.enableFeatureClassByName(fc)

    _EXTRACTOR_CACHE[key] = ext
    return ext


def extract_radiomic_features(
    image: NDArray[np.floating],
    mask: Optional[NDArray[np.bool_]] = None,
    feature_classes: Optional[list[str]] = None,
    bin_width: int = FRD_DEFAULT_BIN_WIDTH,
) -> NDArray[np.floating]:
    """Extract IBSI-compliant radiomic features from a 2D image.

    Uses pyradiomics to extract firstorder, GLCM, GLRLM, GLDM, GLSZM,
    and NGTDM feature sets as defined by FRD v1.

    Args:
        image: 2D or 3D image array (float).
        mask: Optional binary mask defining the region of interest.
              If None, uses the entire non-zero region.
        feature_classes: List of pyradiomics feature class names to extract.
                         Defaults to FRD_FEATURE_CLASSES.
        bin_width: Bin width for discretisation (default: 25, the frd-score default).

    Returns:
        1D array of radiomic feature values.

    Raises:
        ImportError: If radiomics (pyradiomics) is not installed.
    """
    try:
        import SimpleITK as sitk
        import radiomics
        from radiomics import featureextractor
    except ImportError as exc:
        raise ImportError(
            "FRD computation requires 'pyradiomics' package. "
            "Install it with: pip install pyradiomics"
        ) from exc

    if feature_classes is None:
        feature_classes = FRD_FEATURE_CLASSES

    # Suppress verbose pyradiomics logging (once per process)
    if not getattr(extract_radiomic_features, "_verbosity_set", False):
        radiomics.setVerbosity(60)
        extract_radiomic_features._verbosity_set = True

    # Ensure image is at least 3D for SimpleITK (add slice dimension if 2D)
    was_2d = image.ndim == 2
    if was_2d:
        image = image[np.newaxis, :, :]

    # Create mask if not provided
    if mask is None:
        mask_arr = np.ones(image.shape, dtype=np.uint8)
    else:
        if mask.ndim == 2 and was_2d:
            mask_arr = mask[np.newaxis, :, :].astype(np.uint8)
        else:
            mask_arr = mask.astype(np.uint8)

    # Ensure mask has at least one foreground voxel
    if np.sum(mask_arr) == 0:
        mask_arr = np.ones(image.shape, dtype=np.uint8)

    # Convert to SimpleITK
    sitk_image = sitk.GetImageFromArray(image.astype(np.float64))
    sitk_mask = sitk.GetImageFromArray(mask_arr)

    # Get or create a cached extractor for this configuration
    extractor = _get_cached_extractor(
        feature_classes=tuple(feature_classes),
        bin_width=bin_width,
        was_2d=was_2d,
    )

    try:
        result = extractor.execute(sitk_image, sitk_mask)
    except Exception as e:
        logger.warning(f"Pyradiomics extraction failed: {e}. Returning zeros.")
        return np.zeros(1, dtype=np.float64)

    # Collect feature values (skip diagnostics prefixed with "diagnostics_")
    features = []
    for key, value in sorted(result.items()):
        if key.startswith("diagnostics_"):
            continue
        try:
            features.append(float(value))
        except (TypeError, ValueError):
            features.append(0.0)

    return np.array(features, dtype=np.float64)


def extract_radiomic_features_batch(
    images: list[NDArray[np.floating]],
    masks: Optional[list[NDArray[np.bool_]]] = None,
    feature_classes: Optional[list[str]] = None,
    bin_width: int = FRD_DEFAULT_BIN_WIDTH,
    n_workers: int = 1,
    cache_dir: Optional[Path] = None,
) -> NDArray[np.floating]:
    """Extract radiomic features from a batch of images.

    Supports parallel extraction via ``ProcessPoolExecutor`` and on-disk
    caching of per-image feature vectors (keyed by a hash of the image data).

    Args:
        images: List of image arrays.
        masks: Optional list of masks (one per image).
        feature_classes: Pyradiomics feature classes.
        bin_width: Discretisation bin width.
        n_workers: Number of parallel workers.  1 = sequential.
        cache_dir: Directory for caching features as ``.npz`` files.
                   If ``None``, no caching is performed.

    Returns:
        Feature matrix of shape ``(len(images), n_features)``.
    """
    if feature_classes is None:
        feature_classes = FRD_FEATURE_CLASSES

    features_list: list[NDArray[np.floating]] = [np.array([]) for _ in range(len(images))]

    # Identify which images need extraction (vs cached)
    to_extract: list[int] = []
    for i, img in enumerate(images):
        cached = _load_from_cache(img, cache_dir)
        if cached is not None:
            features_list[i] = cached
        else:
            to_extract.append(i)

    if not to_extract:
        return np.stack(features_list)

    # Extract features (parallel or sequential)
    if n_workers > 1 and len(to_extract) > 1:
        _extract_parallel(
            to_extract, images, masks, feature_classes, bin_width,
            n_workers, features_list,
        )
    else:
        for idx in to_extract:
            m = masks[idx] if masks is not None else None
            feats = extract_radiomic_features(
                images[idx], mask=m, feature_classes=feature_classes,
                bin_width=bin_width,
            )
            features_list[idx] = feats
            _save_to_cache(images[idx], feats, cache_dir)

    return np.stack(features_list)


def _extract_parallel(
    indices: list[int],
    images: list[NDArray],
    masks: Optional[list[NDArray]],
    feature_classes: list[str],
    bin_width: int,
    n_workers: int,
    features_list: list[NDArray],
) -> None:
    """Run parallel feature extraction."""
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for idx in indices:
            m = masks[idx] if masks is not None else None
            fut = executor.submit(
                _extract_single, images[idx], m, feature_classes, bin_width
            )
            futures[fut] = idx
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                feats = fut.result()
            except Exception as e:
                logger.warning(f"Extraction failed for image {idx}: {e}")
                feats = np.zeros(1, dtype=np.float64)
            features_list[idx] = feats


# ---------------------------------------------------------------------------
# Disk-based feature cache
# ---------------------------------------------------------------------------


def _image_hash(image: NDArray) -> str:
    """Compute a stable SHA-256 hash of an image array."""
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


def _load_from_cache(
    image: NDArray, cache_dir: Optional[Path]
) -> Optional[NDArray]:
    if cache_dir is None:
        return None
    h = _image_hash(image)
    fpath = cache_dir / f"frd_feat_{h}.npz"
    if fpath.exists():
        data = np.load(fpath)
        return data["features"]
    return None


def _save_to_cache(
    image: NDArray, features: NDArray, cache_dir: Optional[Path]
) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = _image_hash(image)
    fpath = cache_dir / f"frd_feat_{h}.npz"
    np.savez_compressed(fpath, features=features)


# ---------------------------------------------------------------------------
# Fréchet distance
# ---------------------------------------------------------------------------


def compute_frechet_distance(
    mu1: NDArray[np.floating],
    sigma1: NDArray[np.floating],
    mu2: NDArray[np.floating],
    sigma2: NDArray[np.floating],
) -> float:
    """Compute the Fréchet distance between two multivariate Gaussians.

    FD = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))

    Args:
        mu1: Mean vector of distribution 1.
        sigma1: Covariance matrix of distribution 1.
        mu2: Mean vector of distribution 2.
        sigma2: Covariance matrix of distribution 2.

    Returns:
        Fréchet distance value (non-negative).
    """
    diff = mu1 - mu2
    mean_diff_sq = float(np.dot(diff, diff))

    # Compute matrix square root of product of covariances
    covmean = linalg.sqrtm(sigma1 @ sigma2)

    # Numerical correction for possible imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0, atol=1e-3):
            logger.warning(
                "Imaginary component in sqrtm result "
                f"(max={np.max(np.abs(np.imag(covmean)))}). "
                "Taking real part."
            )
        covmean = np.real(covmean)

    trace_term = float(np.trace(sigma1 + sigma2 - 2.0 * covmean))

    return max(0.0, mean_diff_sq + trace_term)


def compute_frd(
    real_images: list[NDArray[np.floating]],
    synthetic_images: list[NDArray[np.floating]],
    masks_real: Optional[list[NDArray[np.bool_]]] = None,
    masks_synthetic: Optional[list[NDArray[np.bool_]]] = None,
    feature_classes: Optional[list[str]] = None,
    bin_width: int = FRD_DEFAULT_BIN_WIDTH,
    n_workers: int = 1,
    cache_dir: Optional[Path] = None,
) -> float:
    """Compute Fréchet Radiomics Distance (FRD) between real and synthetic sets.

    Extracts IBSI-compliant radiomic features from both image sets,
    fits multivariate Gaussians, and computes the Fréchet distance.

    Args:
        real_images: List of real post-contrast image arrays.
        synthetic_images: List of synthetic post-contrast image arrays.
        masks_real: Optional list of masks for real images.
        masks_synthetic: Optional list of masks for synthetic images.
        feature_classes: List of pyradiomics feature class names to use.
        bin_width: Discretisation bin width (default: 25).
        n_workers: Number of parallel workers for feature extraction.
        cache_dir: Optional directory for caching features.

    Returns:
        FRD value (non-negative). Lower is better (0 = identical distributions).

    Raises:
        ValueError: If fewer than 2 images are provided in either set.
    """
    if len(real_images) < 2:
        raise ValueError(f"Need at least 2 real images for FRD, got {len(real_images)}")
    if len(synthetic_images) < 2:
        raise ValueError(
            f"Need at least 2 synthetic images for FRD, got {len(synthetic_images)}"
        )

    logger.info(
        f"Computing FRD: {len(real_images)} real vs {len(synthetic_images)} synthetic"
    )

    # Extract features (with optional parallelism and caching)
    real_matrix = extract_radiomic_features_batch(
        real_images,
        masks=masks_real,
        feature_classes=feature_classes,
        bin_width=bin_width,
        n_workers=n_workers,
        cache_dir=cache_dir,
    )
    synth_matrix = extract_radiomic_features_batch(
        synthetic_images,
        masks=masks_synthetic,
        feature_classes=feature_classes,
        bin_width=bin_width,
        n_workers=n_workers,
        cache_dir=cache_dir,
    )

    # Replace NaN/inf with 0
    real_matrix = np.nan_to_num(real_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    synth_matrix = np.nan_to_num(synth_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute statistics
    mu_real = np.mean(real_matrix, axis=0)
    mu_synth = np.mean(synth_matrix, axis=0)

    sigma_real = np.cov(real_matrix, rowvar=False)
    sigma_synth = np.cov(synth_matrix, rowvar=False)

    # Handle single-feature case (cov returns scalar)
    if sigma_real.ndim == 0:
        sigma_real = np.array([[float(sigma_real)]])
        sigma_synth = np.array([[float(sigma_synth)]])

    # Regularize covariance matrices for numerical stability
    eps = 1e-6
    sigma_real += eps * np.eye(sigma_real.shape[0])
    sigma_synth += eps * np.eye(sigma_synth.shape[0])

    return compute_frechet_distance(mu_real, sigma_real, mu_synth, sigma_synth)


def compute_frd_from_features(
    real_features: NDArray[np.floating],
    synthetic_features: NDArray[np.floating],
) -> float:
    """Compute FRD from pre-extracted feature matrices.

    Convenience function when features are already extracted.

    Args:
        real_features: Feature matrix of shape (n_real, n_features).
        synthetic_features: Feature matrix of shape (n_synth, n_features).

    Returns:
        FRD value (non-negative).
    """
    if real_features.shape[0] < 2:
        raise ValueError("Need at least 2 real samples")
    if synthetic_features.shape[0] < 2:
        raise ValueError("Need at least 2 synthetic samples")
    if real_features.shape[1] != synthetic_features.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {real_features.shape[1]} vs "
            f"{synthetic_features.shape[1]}"
        )

    # Clean data
    real_features = np.nan_to_num(real_features, nan=0.0, posinf=0.0, neginf=0.0)
    synthetic_features = np.nan_to_num(
        synthetic_features, nan=0.0, posinf=0.0, neginf=0.0
    )

    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(synthetic_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(synthetic_features, rowvar=False)

    if sigma1.ndim == 0:
        sigma1 = np.array([[float(sigma1)]])
        sigma2 = np.array([[float(sigma2)]])

    eps = 1e-6
    sigma1 += eps * np.eye(sigma1.shape[0])
    sigma2 += eps * np.eye(sigma2.shape[0])

    return compute_frechet_distance(mu1, sigma1, mu2, sigma2)
