#  Copyright 2025 mama-sia-eval contributors
#  Licensed under the Apache License, Version 2.0

"""
Generate artificial test data for end-to-end pipeline testing.

Creates a toy dataset mimicking the MAMA-SYNTH challenge data layout:
  - Pre-contrast 2D breast MRI slices (512×512, 16-bit PNG)
  - Post-contrast ground truth slices
  - Synthetic post-contrast slices (slightly degraded)
  - Tumor segmentation masks
  - Molecular subtype labels CSV

Usage:
    python -m mama_sia_eval.generate_test_data --output-dir ./test_data --n-cases 20
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

__all__ = ["generate_case", "save_dataset"]

# ---------------------------------------------------------------------------
# Synthetic breast MRI generation
# ---------------------------------------------------------------------------

DEFAULT_SHAPE = (512, 512)  # Challenge standard resolution
DEFAULT_N_CASES = 20


def _breast_mask(shape: tuple[int, int], rng: np.random.RandomState) -> np.ndarray:
    """Semi-elliptical breast tissue mask."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2 + rng.randint(-10, 10), w // 2 + rng.randint(-10, 10)
    ry, rx = h // 2 - 20, w // 2 - 20
    return ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1.0


def _tumor_mask(
    shape: tuple[int, int],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Random elliptical tumor ROI."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    # Place tumor in a realistic region (central-ish)
    cy = rng.randint(h // 4, 3 * h // 4)
    cx = rng.randint(w // 4, 3 * w // 4)
    ry = rng.randint(10, 40)
    rx = rng.randint(10, 40)
    return ((y - cy) / max(ry, 1)) ** 2 + ((x - cx) / max(rx, 1)) ** 2 <= 1.0


def generate_case(
    case_id: str,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    seed: int = 0,
) -> dict:
    """Generate a single realistic synthetic DCE-MRI case.

    Returns dict with arrays: precontrast, postcontrast_real,
    postcontrast_synth, tumor_mask, and labels.
    """
    rng = np.random.RandomState(seed)

    # Background + tissue
    image_pre = np.full(shape, 50.0, dtype=np.float64)
    breast = _breast_mask(shape, rng)
    tissue_intensity = rng.uniform(800, 1200)
    image_pre[breast] = tissue_intensity + rng.normal(0, 80, np.sum(breast))

    # Fibroglandular tissue pattern (Perlin-like noise via upsampled grid)
    noise_small = rng.randn(shape[0] // 8, shape[1] // 8)
    from scipy.ndimage import zoom
    noise_large = zoom(noise_small, 8, order=3)[:shape[0], :shape[1]]
    image_pre[breast] += noise_large[breast] * 60

    image_pre = np.clip(image_pre, 0, 65535)

    # Post-contrast: tissue enhancement + strong tumor enhancement
    image_post = image_pre.copy()
    # Mild background enhancement
    image_post[breast] += rng.uniform(50, 150)
    # Tumor
    tumor = _tumor_mask(shape, rng)
    tumor_enhancement = rng.uniform(2000, 4000)
    image_post[tumor & breast] = tumor_enhancement + rng.normal(0, 200, np.sum(tumor & breast))
    image_post = np.clip(image_post, 0, 65535)

    # Synthetic post-contrast: slightly degraded version
    image_synth = image_post.copy()
    # Reduce tumor enhancement
    synth_factor = rng.uniform(0.7, 0.95)
    image_synth[tumor & breast] = (
        image_pre[tumor & breast]
        + (image_post[tumor & breast] - image_pre[tumor & breast]) * synth_factor
    )
    # Add synthesis noise
    image_synth += rng.normal(0, 30, shape)
    image_synth = np.clip(image_synth, 0, 65535)

    # Labels
    is_tnbc = int(rng.rand() < 0.25)  # ~25% TNBC prevalence
    is_luminal = int(rng.rand() < 0.60) if not is_tnbc else 0

    return {
        "case_id": case_id,
        "precontrast": image_pre.astype(np.float64),
        "postcontrast_real": image_post.astype(np.float64),
        "postcontrast_synth": image_synth.astype(np.float64),
        "tumor_mask": (tumor & breast).astype(np.uint8),
        "labels": {"tnbc": is_tnbc, "luminal": is_luminal},
    }


def save_dataset(
    output_dir: Path,
    n_cases: int = DEFAULT_N_CASES,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    fmt: str = "nii.gz",
) -> Path:
    """Generate and save a complete artificial dataset.

    Directory layout::

        output_dir/
        ├── ground-truth/      (post-contrast real)
        ├── predictions/       (post-contrast synthetic)
        ├── precontrast/       (pre-contrast input)
        ├── masks/             (tumor segmentation masks)
        ├── labels.csv
        └── labels.json

    Args:
        output_dir: Root directory for the dataset.
        n_cases: Number of cases to generate.
        shape: Image dimensions (H, W).
        fmt: Image format — ``"nii.gz"`` or ``"png"``.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    gt_dir = output_dir / "ground-truth"
    pred_dir = output_dir / "predictions"
    pre_dir = output_dir / "precontrast"
    mask_dir = output_dir / "masks"
    for d in [gt_dir, pred_dir, pre_dir, mask_dir]:
        d.mkdir(parents=True, exist_ok=True)

    labels_json: dict[str, dict[str, int]] = {}
    labels_rows: list[dict[str, str | int]] = []

    for i in range(n_cases):
        case_id = f"case{i:04d}"
        case = generate_case(case_id, shape=shape, seed=42 + i)

        ext = f".{fmt}"
        _save_image(case["precontrast"], pre_dir / f"{case_id}{ext}")
        _save_image(case["postcontrast_real"], gt_dir / f"{case_id}{ext}")
        _save_image(case["postcontrast_synth"], pred_dir / f"{case_id}{ext}")
        _save_image(case["tumor_mask"].astype(np.float32), mask_dir / f"{case_id}{ext}")

        labels_json[case_id] = case["labels"]
        labels_rows.append({"case_id": case_id, **case["labels"]})

    # Save labels in both JSON and CSV
    with open(output_dir / "labels.json", "w") as f:
        json.dump(labels_json, f, indent=2)
    with open(output_dir / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "tnbc", "luminal"])
        writer.writeheader()
        writer.writerows(labels_rows)

    logger.info(
        f"Generated {n_cases} cases in {output_dir} "
        f"(format={fmt}, shape={shape})"
    )
    return output_dir


def _save_image(array: np.ndarray, path: Path) -> None:
    """Save array as SimpleITK image (NIfTI/MHA) or 16-bit PNG."""
    if path.suffix == ".png":
        try:
            from PIL import Image

            img_16 = np.clip(array, 0, 65535).astype(np.uint16)
            Image.fromarray(img_16).save(str(path))
            return
        except ImportError:
            pass  # fall through to SimpleITK

    image = sitk.GetImageFromArray(array.astype(np.float32))
    sitk.WriteImage(image, str(path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate artificial MAMA-SYNTH test data."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Root output directory.",
    )
    parser.add_argument(
        "--n-cases", type=int, default=DEFAULT_N_CASES,
        help="Number of cases to generate.",
    )
    parser.add_argument(
        "--shape", type=int, nargs=2, default=list(DEFAULT_SHAPE),
        help="Image shape (H W).",
    )
    parser.add_argument(
        "--format", choices=["nii.gz", "png", "mha"], default="nii.gz",
        help="Image format.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    save_dataset(
        output_dir=args.output_dir,
        n_cases=args.n_cases,
        shape=tuple(args.shape),
        fmt=args.format,
    )
    print(f"Dataset generated at {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
