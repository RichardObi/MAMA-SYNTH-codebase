"""Shared fixtures for MAMA-SYNTH evaluation tests.

Creates realistic synthetic breast DCE-MRI test data including:
  - Pre-contrast images (uniform low-intensity breast tissue)
  - Post-contrast ground truth (bright tumor enhancement)
  - Synthetic post-contrast (approximated enhancement)
  - Tumor segmentation masks
"""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# Constants mimicking real DCE-MRI characteristics
# ---------------------------------------------------------------------------

IMG_SHAPE_2D = (64, 64)          # Downscaled from 512x512 for fast tests
IMG_SHAPE_3D = (5, 64, 64)      # Small 3D volume
TUMOR_CENTER_2D = (32, 32)
TUMOR_RADIUS_2D = 8
TUMOR_CENTER_3D = (2, 32, 32)
TUMOR_RADIUS_3D = (2, 8, 8)

# Intensity ranges (mimicking 16-bit MRI intensities, scaled down)
BG_INTENSITY = 10.0
TISSUE_INTENSITY = 150.0
GLANDULAR_INTENSITY = 200.0
TUMOR_ENHANCEMENT = 400.0
SYNTH_TUMOR_ENHANCEMENT = 360.0   # Slightly different from ground truth


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _make_breast_mask_2d(shape: tuple[int, int]) -> np.ndarray:
    """Create a semi-circular breast tissue mask."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 2 - 2
    mask = ((y - center_y) ** 2 + (x - center_x) ** 2) <= radius ** 2
    return mask


def _make_tumor_mask_2d(
    shape: tuple[int, int],
    center: tuple[int, int] = TUMOR_CENTER_2D,
    radius: int = TUMOR_RADIUS_2D,
) -> np.ndarray:
    """Create a circular tumor ROI mask."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    return ((y - center[0]) ** 2 + (x - center[1]) ** 2) <= radius ** 2


def _make_tumor_mask_3d(
    shape: tuple[int, int, int],
    center: tuple[int, int, int] = TUMOR_CENTER_3D,
    radii: tuple[int, int, int] = TUMOR_RADIUS_3D,
) -> np.ndarray:
    """Create an ellipsoidal tumor mask in 3D."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return (
        ((z - center[0]) / max(radii[0], 1)) ** 2
        + ((y - center[1]) / max(radii[1], 1)) ** 2
        + ((x - center[2]) / max(radii[2], 1)) ** 2
    ) <= 1.0


def make_precontrast_2d(shape: tuple[int, int] = IMG_SHAPE_2D) -> np.ndarray:
    """Generate a synthetic pre-contrast breast MRI slice.

    Low-intensity image with breast tissue and glandular patterns.
    """
    rng = np.random.RandomState(42)
    image = np.full(shape, BG_INTENSITY, dtype=np.float64)
    breast = _make_breast_mask_2d(shape)
    image[breast] = TISSUE_INTENSITY

    # Add glandular tissue pattern
    gland_noise = rng.normal(0, 15, shape)
    image[breast] += gland_noise[breast]

    # Slightly brighter central glandular region
    inner_mask = _make_tumor_mask_2d(shape, center=(shape[0] // 2, shape[1] // 2), radius=shape[1] // 4)
    image[breast & inner_mask] += 30

    return np.clip(image, 0, None)


def make_postcontrast_real_2d(shape: tuple[int, int] = IMG_SHAPE_2D) -> np.ndarray:
    """Generate a synthetic real post-contrast breast MRI slice.

    Pre-contrast base + strong tumor enhancement.
    """
    image = make_precontrast_2d(shape)
    tumor = _make_tumor_mask_2d(shape)
    image[tumor] = TUMOR_ENHANCEMENT

    # Add some enhancement gradient at tumor border
    rng = np.random.RandomState(43)
    image[tumor] += rng.normal(0, 20, np.sum(tumor))
    return np.clip(image, 0, None)


def make_postcontrast_synthetic_2d(shape: tuple[int, int] = IMG_SHAPE_2D) -> np.ndarray:
    """Generate a synthetic model output (imperfect reconstruction).

    Similar to real post-contrast but with slightly lower enhancement
    and some noise, simulating a generative model's output.
    """
    image = make_precontrast_2d(shape)
    tumor = _make_tumor_mask_2d(shape)
    image[tumor] = SYNTH_TUMOR_ENHANCEMENT

    rng = np.random.RandomState(44)
    image[tumor] += rng.normal(0, 25, np.sum(tumor))
    # Add slight global noise to simulate synthesis artifacts
    image += rng.normal(0, 5, shape)
    return np.clip(image, 0, None)


def make_precontrast_3d(shape: tuple[int, int, int] = IMG_SHAPE_3D) -> np.ndarray:
    """Generate a 3D pre-contrast breast MRI volume."""
    rng = np.random.RandomState(42)
    image = np.full(shape, BG_INTENSITY, dtype=np.float64)
    for s in range(shape[0]):
        breast = _make_breast_mask_2d(shape[1:])
        image[s][breast] = TISSUE_INTENSITY + rng.normal(0, 10, np.sum(breast))
    return np.clip(image, 0, None)


def make_postcontrast_real_3d(shape: tuple[int, int, int] = IMG_SHAPE_3D) -> np.ndarray:
    """Generate a 3D real post-contrast volume."""
    image = make_precontrast_3d(shape)
    tumor = _make_tumor_mask_3d(shape)
    rng = np.random.RandomState(43)
    image[tumor] = TUMOR_ENHANCEMENT + rng.normal(0, 20, np.sum(tumor))
    return np.clip(image, 0, None)


def make_postcontrast_synthetic_3d(shape: tuple[int, int, int] = IMG_SHAPE_3D) -> np.ndarray:
    """Generate a 3D synthetic post-contrast volume."""
    image = make_precontrast_3d(shape)
    tumor = _make_tumor_mask_3d(shape)
    rng = np.random.RandomState(44)
    image[tumor] = SYNTH_TUMOR_ENHANCEMENT + rng.normal(0, 25, np.sum(tumor))
    image += rng.normal(0, 5, shape)
    return np.clip(image, 0, None)


def save_sitk_image(array: np.ndarray, path: Path) -> None:
    """Save a numpy array as a SimpleITK image (NIfTI)."""
    image = sitk.GetImageFromArray(array.astype(np.float32))
    sitk.WriteImage(image, str(path))


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def img_2d_real():
    """Real post-contrast 2D slice."""
    return make_postcontrast_real_2d()


@pytest.fixture
def img_2d_synth():
    """Synthetic post-contrast 2D slice."""
    return make_postcontrast_synthetic_2d()


@pytest.fixture
def img_2d_precontrast():
    """Pre-contrast 2D slice."""
    return make_precontrast_2d()


@pytest.fixture
def tumor_mask_2d():
    """2D tumor segmentation mask."""
    return _make_tumor_mask_2d(IMG_SHAPE_2D)


@pytest.fixture
def img_3d_real():
    """Real post-contrast 3D volume."""
    return make_postcontrast_real_3d()


@pytest.fixture
def img_3d_synth():
    """Synthetic post-contrast 3D volume."""
    return make_postcontrast_synthetic_3d()


@pytest.fixture
def tumor_mask_3d():
    """3D tumor segmentation mask."""
    return _make_tumor_mask_3d(IMG_SHAPE_3D)


@pytest.fixture
def temp_dirs(tmp_path: Path):
    """Create temporary directories for ground truth, predictions, masks, output."""
    gt_dir = tmp_path / "ground-truth"
    pred_dir = tmp_path / "predictions"
    masks_dir = tmp_path / "masks"
    output_file = tmp_path / "output" / "metrics.json"

    gt_dir.mkdir()
    pred_dir.mkdir()
    masks_dir.mkdir()

    return gt_dir, pred_dir, masks_dir, output_file


@pytest.fixture
def populated_dirs(temp_dirs):
    """Temp dirs pre-populated with 3 realistic test cases."""
    gt_dir, pred_dir, masks_dir, output_file = temp_dirs

    for i in range(3):
        rng = np.random.RandomState(100 + i)
        real = make_postcontrast_real_2d()
        # Vary synthetic slightly per case
        synth = make_postcontrast_synthetic_2d() + rng.normal(0, 2, IMG_SHAPE_2D)
        synth = np.clip(synth, 0, None)
        mask = _make_tumor_mask_2d(IMG_SHAPE_2D).astype(np.float32)

        save_sitk_image(real, gt_dir / f"case{i:03d}.nii.gz")
        save_sitk_image(synth, pred_dir / f"case{i:03d}.nii.gz")
        save_sitk_image(mask, masks_dir / f"case{i:03d}.nii.gz")

    return gt_dir, pred_dir, masks_dir, output_file
