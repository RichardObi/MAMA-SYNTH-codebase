"""CLI tool to extract and visualise 2-D slices for inspection.

Produces two kinds of output for every extracted slice:

Raw files (.nii.gz)
    The exact 2-D image and mask arrays that the classifiers see during
    radiomic feature extraction — same normalisation, same slice indices.

Visual files (.png)
    Side-by-side matplotlib figure: raw image on the left, image with
    red mask overlay on the right.  Easy to open in any image viewer.

A summary CSV is written to the output directory listing every extracted
slice with its patient ID, task, phase, slice index and output paths.

Usage examples
--------------
# Contrast task, max_tumor mode, patients listed in a file
python -m eval.extract_slices \\
    --task contrast \\
    --mode max_tumor \\
    --data-dir /data/mama-mia \\
    --patient-ids P001 P002 P003 \\
    --output-dir /tmp/slices

# Tumor-ROI task, all_tumor mode, custom image / segmentation dirs
python -m eval.extract_slices \\
    --task tumor_roi \\
    --mode all_tumor \\
    --images-dir /data/mama-mia/images \\
    --segmentations-dir /data/mama-mia/segmentations \\
    --patient-ids P001 P002 \\
    --output-dir /tmp/slices

# Both tasks, with global normalisation stats
python -m eval.extract_slices \\
    --task both \\
    --mode all_tumor \\
    --data-dir /data/mama-mia \\
    --normalization-stats /models/norm_stats.json \\
    --output-dir /tmp/slices
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Constants (must match train_classifier.py) ────────────────────────────
IMAGES_SUBDIR = "images"
SEGMENTATIONS_SUBDIR = "segmentations"
DEFAULT_PHASE = 1
DEFAULT_PRE_PHASE = 0


# ─── Path helpers (mirrored from train_classifier.py) ─────────────────────
def _get_image_path(images_dir: Path, patient_id: str, phase: int) -> Path:
    return images_dir / patient_id / f"{patient_id}_{phase:04d}.nii.gz"


def _get_segmentation_path(segmentations_dir: Path, patient_id: str) -> Path:
    return segmentations_dir / f"{patient_id}.nii.gz"


def _load_nifti_as_array(filepath: Path) -> np.ndarray:
    import SimpleITK as sitk
    if not filepath.exists():
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")
    return sitk.GetArrayFromImage(sitk.ReadImage(str(filepath), sitk.sitkFloat32))


def _load_mask_as_array(filepath: Path) -> np.ndarray:
    import SimpleITK as sitk
    if not filepath.exists():
        raise FileNotFoundError(f"Mask file not found: {filepath}")
    return sitk.GetArrayFromImage(sitk.ReadImage(str(filepath), sitk.sitkUInt8)).astype(bool)


def _save_slice_as_nifti(array: np.ndarray, dest: Path) -> None:
    """Write a 2-D float32 array to a NIfTI file."""
    import SimpleITK as sitk
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(array.astype(np.float32))
    sitk.WriteImage(img, str(dest))


def _save_mask_as_nifti(array: np.ndarray, dest: Path) -> None:
    """Write a 2-D boolean/uint8 array to a NIfTI file."""
    import SimpleITK as sitk
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(array.astype(np.uint8))
    sitk.WriteImage(img, str(dest))


def _save_visual_overlay(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    dest: Path,
    title: str = "",
    mask_color: tuple = (1.0, 0.0, 0.0),  # RGB, default red
    mask_label: str = "Mask",
) -> None:
    """Save a side-by-side PNG: raw image | image with mask overlay.

    Parameters
    ----------
    mask_color:
        RGB float tuple for the filled mask overlay (default: red).
    mask_label:
        Short label describing the mask shown in the right panel title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize as MplNorm
    from matplotlib.patches import Patch

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Normalise display range robustly
    vmin, vmax = np.percentile(image_slice, [1, 99])
    norm = MplNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title, fontsize=10)

    for ax in axes:
        ax.imshow(image_slice, cmap="gray", norm=norm, origin="upper")
        ax.axis("off")

    # Right panel: coloured mask overlay (filled, semi-transparent)
    if np.any(mask_slice):
        r, g, b = mask_color
        overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
        overlay[mask_slice, 0] = r
        overlay[mask_slice, 1] = g
        overlay[mask_slice, 2] = b
        overlay[mask_slice, 3] = 0.45   # alpha
        axes[1].imshow(overlay, origin="upper")
        legend_patch = Patch(facecolor=mask_color, alpha=0.6, label=mask_label)
        axes[1].legend(
            handles=[legend_patch],
            loc="lower right",
            fontsize=8,
            framealpha=0.7,
        )

    axes[0].set_title("Raw slice")
    axes[1].set_title(f"Slice + {mask_label} overlay")

    fig.tight_layout()
    fig.savefig(str(dest), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_tumor_roi_combined_visual(
    image_slice: np.ndarray,
    tumor_mask: np.ndarray,
    mirror_mask: np.ndarray,
    dest: Path,
    title: str = "",
) -> None:
    """Save an annotated three-panel PNG for the tumor-ROI task.

    Panels (left → right):

    1. **Tumor ROI** — image + red filled overlay for the original tumor mask.
    2. **Both masks** — image + red tumor mask + blue mirrored mask shown
       together so spatial relationship is immediately visible.  The body
       midline is drawn as a dashed vertical line.
    3. **Mirrored ROI** — image + blue filled overlay for the mirrored mask.

    Each panel carries a plain-text annotation and a legend entry so the
    viewer can identify which region is which at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize as MplNorm
    from matplotlib.patches import Patch

    dest.parent.mkdir(parents=True, exist_ok=True)

    vmin, vmax = np.percentile(image_slice, [1, 99])
    norm = MplNorm(vmin=vmin, vmax=vmax)

    TUMOR_COLOR  = (1.0, 0.2, 0.2)   # red
    MIRROR_COLOR = (0.2, 0.5, 1.0)   # blue
    ALPHA = 0.45

    def _make_overlay(mask, rgb):
        ov = np.zeros((*mask.shape, 4), dtype=np.float32)
        ov[mask, 0] = rgb[0]
        ov[mask, 1] = rgb[1]
        ov[mask, 2] = rgb[2]
        ov[mask, 3] = ALPHA
        return ov

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=10)

    for ax in axes:
        ax.imshow(image_slice, cmap="gray", norm=norm, origin="upper")
        ax.axis("off")

    # ── Panel 0: Tumor ROI (red) ──────────────────────────────────────────
    if np.any(tumor_mask):
        axes[0].imshow(_make_overlay(tumor_mask, TUMOR_COLOR), origin="upper")
    axes[0].set_title("Tumor ROI", fontsize=11, fontweight="bold")
    axes[0].legend(
        handles=[Patch(facecolor=TUMOR_COLOR, alpha=0.6, label="Original tumor mask")],
        loc="lower right", fontsize=8, framealpha=0.7,
    )
    # Text annotation pinned to the mask centroid (if mask is non-empty)
    if np.any(tumor_mask):
        ys, xs = np.where(tumor_mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        axes[0].text(
            cx, cy, "Tumor",
            color="white", fontsize=8, ha="center", va="center",
            bbox=dict(facecolor=TUMOR_COLOR, alpha=0.6, boxstyle="round,pad=0.2"),
        )

    # ── Panel 1: Both masks together with midline ─────────────────────────
    if np.any(tumor_mask):
        axes[1].imshow(_make_overlay(tumor_mask, TUMOR_COLOR), origin="upper")
    if np.any(mirror_mask):
        axes[1].imshow(_make_overlay(mirror_mask, MIRROR_COLOR), origin="upper")
    # Estimate and draw body midline
    try:
        from eval.mirror_utils import detect_midline
        midline_col = detect_midline(image_slice)
        axes[1].axvline(
            x=midline_col, color="yellow", linestyle="--",
            linewidth=1.2, alpha=0.8, label="Body midline",
        )
    except Exception:
        pass
    axes[1].set_title("Both masks (comparison)", fontsize=11, fontweight="bold")
    axes[1].legend(
        handles=[
            Patch(facecolor=TUMOR_COLOR, alpha=0.6, label="Original tumor mask"),
            Patch(facecolor=MIRROR_COLOR, alpha=0.6, label="Mirrored (contralateral) mask"),
        ],
        loc="lower right", fontsize=7, framealpha=0.7,
    )

    # ── Panel 2: Mirrored ROI (blue) ──────────────────────────────────────
    if np.any(mirror_mask):
        axes[2].imshow(_make_overlay(mirror_mask, MIRROR_COLOR), origin="upper")
    axes[2].set_title("Mirrored (Contralateral) ROI", fontsize=11, fontweight="bold")
    axes[2].legend(
        handles=[Patch(facecolor=MIRROR_COLOR, alpha=0.6, label="Mirrored mask")],
        loc="lower right", fontsize=8, framealpha=0.7,
    )
    if np.any(mirror_mask):
        ys, xs = np.where(mirror_mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        axes[2].text(
            cx, cy, "Mirror",
            color="white", fontsize=8, ha="center", va="center",
            bbox=dict(facecolor=MIRROR_COLOR, alpha=0.6, boxstyle="round,pad=0.2"),
        )

    fig.tight_layout()
    fig.savefig(str(dest), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Normalisation  ───────────────────────────────────────────────────────
def _load_normalizer(stats_path: Optional[Path]):
    """Return a volume normaliser callable or None.

    Mirrors the logic from train_classifier.py's ``make_normalizer``  /
    ``load_normalization_stats``.
    """
    if stats_path is None:
        return None
    try:
        from eval.train_classifier import load_normalization_stats, make_normalizer
        stats = load_normalization_stats(stats_path)
        return make_normalizer(stats)
    except Exception as e:
        logger.warning(
            f"Could not load normalisation stats from {stats_path}: {e}. "
            "Falling back to per-slice z-score."
        )
        return None


# ─── Core extraction helpers ──────────────────────────────────────────────
def _extract_slices_for_volume(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    mode: str,
    per_slice_norm: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Return (img_slices, mask_slices, slice_indices) for one volume."""
    from eval.slice_extraction import SliceMode, extract_2d_slice, extract_all_tumor_slices

    sm = SliceMode(mode)
    if sm == SliceMode.ALL_TUMOR:
        imgs, masks, idxs = extract_all_tumor_slices(
            image_array, mask=mask_array, normalize=per_slice_norm
        )
        return imgs, masks, idxs
    else:
        # max_tumor (and centre_tumor / middle as bonus)
        img_2d, msk_2d, z_idx = extract_2d_slice(
            image_array, mask=mask_array, mode=sm, normalize=per_slice_norm
        )
        return [img_2d], [msk_2d], [z_idx]


# ─── Task: contrast  ──────────────────────────────────────────────────────
def run_contrast(
    patient_ids: list[str],
    images_dir: Path,
    segmentations_dir: Path,
    output_dir: Path,
    mode: str,
    phase: int,
    pre_phase: int,
    normalizer,
    save_raw: bool,
    save_visual: bool,
) -> list[dict]:
    """Extract pre- and post-contrast slices at matching slice indices."""
    rows: list[dict] = []

    for pid in patient_ids:
        seg_path = _get_segmentation_path(segmentations_dir, pid)
        post_path = _get_image_path(images_dir, pid, phase)
        pre_path = _get_image_path(images_dir, pid, pre_phase)

        # --- Load post-contrast (used to determine slice indices via mask) ---
        try:
            post_arr = _load_nifti_as_array(post_path)
        except Exception as e:
            logger.warning(f"[{pid}] Could not load post-contrast phase {phase}: {e} — skipping.")
            continue

        if normalizer is not None:
            post_arr = normalizer(post_arr)

        try:
            mask_arr = _load_mask_as_array(seg_path)
        except Exception as e:
            logger.warning(f"[{pid}] Could not load mask: {e} — skipping.")
            continue

        if not np.any(mask_arr):
            logger.warning(f"[{pid}] Empty mask — skipping.")
            continue

        per_slice_norm = normalizer is None

        # Extract from post-contrast to get canonical slice indices
        try:
            post_imgs, post_masks, z_idxs = _extract_slices_for_volume(
                post_arr, mask_arr, mode, per_slice_norm
            )
        except Exception as e:
            logger.warning(f"[{pid}] Slice extraction (post) failed: {e} — skipping.")
            continue

        # --- Load pre-contrast ---
        try:
            pre_arr = _load_nifti_as_array(pre_path)
        except Exception as e:
            logger.warning(f"[{pid}] Could not load pre-contrast phase {pre_phase}: {e} — skipping.")
            continue

        if normalizer is not None:
            pre_arr = normalizer(pre_arr)

        # Extract pre-contrast slices at the SAME z-indices (mirror training)
        from eval.slice_extraction import SliceMode, extract_2d_slice

        pre_imgs: list[np.ndarray] = []
        pre_masks: list[np.ndarray] = []

        for z_idx in z_idxs:
            pre_slice = pre_arr[z_idx]
            mask_slice = mask_arr[z_idx]
            if per_slice_norm:
                mu = pre_slice.mean()
                std = pre_slice.std()
                if std > 0:
                    pre_slice = (pre_slice - mu) / std
            pre_imgs.append(pre_slice)
            pre_masks.append(mask_slice)

        # --- Save outputs ---
        pid_dir = output_dir / "contrast" / pid
        pid_dir.mkdir(parents=True, exist_ok=True)

        for i, z_idx in enumerate(z_idxs):
            for label, img_sl, msk_sl, ph in [
                ("post", post_imgs[i], post_masks[i], phase),
                ("pre",  pre_imgs[i],  pre_masks[i],  pre_phase),
            ]:
                stem = f"{pid}_contrast_{label}_slice{z_idx:04d}"
                raw_img_path = pid_dir / f"{stem}_image.nii.gz"
                raw_msk_path = pid_dir / f"{stem}_mask.nii.gz"
                vis_path = pid_dir / f"{stem}.png"

                if save_raw:
                    _save_slice_as_nifti(img_sl, raw_img_path)
                    _save_mask_as_nifti(msk_sl, raw_msk_path)

                if save_visual:
                    _save_visual_overlay(
                        img_sl, msk_sl, vis_path,
                        title=f"{pid}  |  contrast={label}  |  "
                              f"phase={ph}  |  z={z_idx}",
                    )

                rows.append(dict(
                    patient_id=pid,
                    task="contrast",
                    label=label,
                    phase=ph,
                    slice_idx=z_idx,
                    raw_image=str(raw_img_path) if save_raw else "",
                    raw_mask=str(raw_msk_path) if save_raw else "",
                    visual=str(vis_path) if save_visual else "",
                ))

    return rows


# ─── Task: tumor-ROI vs mirrored-ROI ─────────────────────────────────────
def run_tumor_roi(
    patient_ids: list[str],
    images_dir: Path,
    segmentations_dir: Path,
    output_dir: Path,
    mode: str,
    phase: int,
    normalizer,
    save_raw: bool,
    save_visual: bool,
    search_fraction: float = 0.4,
    min_tissue_fraction: float = 0.3,
) -> list[dict]:
    """Extract tumor-ROI and mirrored contralateral slices."""
    from eval.mirror_utils import create_mirrored_mask

    rows: list[dict] = []

    for pid in patient_ids:
        seg_path = _get_segmentation_path(segmentations_dir, pid)
        img_path = _get_image_path(images_dir, pid, phase)

        try:
            image_arr = _load_nifti_as_array(img_path)
        except Exception as e:
            logger.warning(f"[{pid}] Could not load phase {phase}: {e} — skipping.")
            continue

        if normalizer is not None:
            image_arr = normalizer(image_arr)

        try:
            mask_arr = _load_mask_as_array(seg_path)
        except Exception as e:
            logger.warning(f"[{pid}] Could not load mask: {e} — skipping.")
            continue

        if not np.any(mask_arr):
            logger.warning(f"[{pid}] Empty mask — skipping.")
            continue

        # Create mirrored mask
        mirrored_mask = create_mirrored_mask(
            image_arr, mask_arr,
            search_fraction=search_fraction,
            min_tissue_fraction=min_tissue_fraction,
        )
        if mirrored_mask is None:
            logger.info(f"[{pid}] Mirrored mask validation failed — skipping.")
            continue

        per_slice_norm = normalizer is None

        # Extract tumor slices
        try:
            tumor_imgs, tumor_masks, z_idxs = _extract_slices_for_volume(
                image_arr, mask_arr, mode, per_slice_norm
            )
        except Exception as e:
            logger.warning(f"[{pid}] Slice extraction (tumor) failed: {e} — skipping.")
            continue

        # Build mirror slices at same z-indices
        mirror_imgs: list[np.ndarray] = []
        mirror_masks: list[np.ndarray] = []

        for z_idx in z_idxs:
            img_slice = image_arr[z_idx]
            if per_slice_norm:
                mu = img_slice.mean()
                std = img_slice.std()
                if std > 0:
                    img_slice = (img_slice - mu) / std
            mirror_imgs.append(img_slice)
            mirror_masks.append(mirrored_mask[z_idx])

        # --- Save outputs ---
        pid_dir = output_dir / "tumor_roi" / pid
        pid_dir.mkdir(parents=True, exist_ok=True)

        for i, z_idx in enumerate(z_idxs):
            # ── Raw NIfTI files ────────────────────────────────────────────
            tumor_raw_img  = pid_dir / f"{pid}_tumor_roi_tumor_slice{z_idx:04d}_image.nii.gz"
            tumor_raw_mask = pid_dir / f"{pid}_tumor_roi_tumor_slice{z_idx:04d}_mask.nii.gz"
            mirror_raw_img  = pid_dir / f"{pid}_tumor_roi_mirror_slice{z_idx:04d}_image.nii.gz"
            mirror_raw_mask = pid_dir / f"{pid}_tumor_roi_mirror_slice{z_idx:04d}_mask.nii.gz"

            if save_raw:
                _save_slice_as_nifti(tumor_imgs[i],   tumor_raw_img)
                _save_mask_as_nifti(tumor_masks[i],   tumor_raw_mask)
                _save_slice_as_nifti(mirror_imgs[i],  mirror_raw_img)
                _save_mask_as_nifti(mirror_masks[i],  mirror_raw_mask)

            # ── Visual files ──────────────────────────────────────────────
            combined_vis = pid_dir / f"{pid}_tumor_roi_combined_slice{z_idx:04d}.png"
            tumor_vis    = pid_dir / f"{pid}_tumor_roi_tumor_slice{z_idx:04d}.png"
            mirror_vis   = pid_dir / f"{pid}_tumor_roi_mirror_slice{z_idx:04d}.png"

            if save_visual:
                # Combined 3-panel annotated figure (primary inspection file)
                _save_tumor_roi_combined_visual(
                    image_slice=tumor_imgs[i],
                    tumor_mask=tumor_masks[i],
                    mirror_mask=mirror_masks[i],
                    dest=combined_vis,
                    title=(
                        f"{pid}  |  tumor-ROI vs mirrored-ROI  |  "
                        f"phase={phase}  |  z={z_idx}  |  "
                        "red=tumor  blue=mirror"
                    ),
                )
                # Individual per-ROI overlays (red / blue) for completeness
                _save_visual_overlay(
                    tumor_imgs[i], tumor_masks[i], tumor_vis,
                    title=f"{pid}  |  Tumor ROI  |  phase={phase}  |  z={z_idx}",
                    mask_color=(1.0, 0.2, 0.2),
                    mask_label="Original tumor mask",
                )
                _save_visual_overlay(
                    mirror_imgs[i], mirror_masks[i], mirror_vis,
                    title=f"{pid}  |  Mirrored (Contralateral) ROI  |  phase={phase}  |  z={z_idx}",
                    mask_color=(0.2, 0.5, 1.0),
                    mask_label="Mirrored mask",
                )

            for label, raw_img, raw_msk, vis in [
                ("tumor",  tumor_raw_img,  tumor_raw_mask,  tumor_vis),
                ("mirror", mirror_raw_img, mirror_raw_mask, mirror_vis),
            ]:
                rows.append(dict(
                    patient_id=pid,
                    task="tumor_roi",
                    label=label,
                    phase=phase,
                    slice_idx=z_idx,
                    raw_image=str(raw_img)  if save_raw    else "",
                    raw_mask=str(raw_msk)   if save_raw    else "",
                    visual=str(vis)         if save_visual else "",
                    visual_combined=str(combined_vis) if save_visual else "",
                ))

    return rows


# ─── Summary CSV ──────────────────────────────────────────────────────────
def _write_csv(rows: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        logger.warning("No rows to write to summary CSV.")
        return
    with open(dest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Summary CSV written to {dest} ({len(rows)} rows).")


# ─── CLI ──────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval.extract_slices",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Patient selection
    g_pid = p.add_mutually_exclusive_group(required=True)
    g_pid.add_argument(
        "--patient-ids", nargs="+", metavar="PID",
        help="One or more patient IDs to process.",
    )
    g_pid.add_argument(
        "--patient-ids-file", type=Path, metavar="FILE",
        help="Text file with one patient ID per line.",
    )

    # Data directories
    p.add_argument(
        "--data-dir", type=Path, metavar="DIR",
        help=(
            "Root directory of the MAMA-MIA dataset. "
            "Subfolders 'images/' and 'segmentations/' are used unless "
            "--images-dir / --segmentations-dir are given."
        ),
    )
    p.add_argument("--images-dir", type=Path, metavar="DIR",
                   help="Override path to images directory.")
    p.add_argument("--segmentations-dir", type=Path, metavar="DIR",
                   help="Override path to segmentations directory.")

    # Task / mode
    p.add_argument(
        "--task", required=True,
        choices=["contrast", "tumor_roi", "both"],
        help="Which classifier pipeline to extract slices for.",
    )
    p.add_argument(
        "--mode", required=True,
        choices=["max_tumor", "all_tumor", "center_tumor", "middle"],
        help="Slice extraction strategy (max_tumor or all_tumor recommended).",
    )

    # Phase selection
    p.add_argument(
        "--phase", type=int, default=DEFAULT_PHASE, metavar="N",
        help=f"Post-contrast phase index (default: {DEFAULT_PHASE}).",
    )
    p.add_argument(
        "--pre-phase", type=int, default=DEFAULT_PRE_PHASE, metavar="N",
        help=(
            f"Pre-contrast phase index used for --task contrast "
            f"(default: {DEFAULT_PRE_PHASE})."
        ),
    )

    # Normalisation
    p.add_argument(
        "--normalization-stats", type=Path, metavar="JSON",
        help=(
            "Path to a normalization_stats.json file produced by "
            "train_classifier.py.  When provided, global z-score "
            "normalisation is applied to the 3-D volume before slice "
            "extraction (matching the training pipeline exactly)."
        ),
    )
    p.add_argument(
        "--no-normalization", action="store_true",
        help="Disable ALL normalisation (both global and per-slice).",
    )

    # Mirror-utils params
    p.add_argument("--search-fraction", type=float, default=0.4,
                   help="Midline detection search fraction (default: 0.4).")
    p.add_argument("--min-tissue-fraction", type=float, default=0.3,
                   help="Minimum tissue fraction for mirrored region (default: 0.3).")

    # Output
    p.add_argument(
        "--output-dir", type=Path, required=True, metavar="DIR",
        help="Directory where extracted files and the summary CSV are written.",
    )
    p.add_argument(
        "--no-raw", action="store_true",
        help="Skip saving raw NIfTI files.",
    )
    p.add_argument(
        "--no-visual", action="store_true",
        help="Skip saving PNG overlay files.",
    )

    # Misc
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Resolve patient IDs ---
    if args.patient_ids:
        patient_ids = args.patient_ids
    else:
        pif: Path = args.patient_ids_file
        if not pif.exists():
            logger.error(f"Patient IDs file not found: {pif}")
            return 1
        patient_ids = [ln.strip() for ln in pif.read_text().splitlines() if ln.strip()]

    if not patient_ids:
        logger.error("No patient IDs provided.")
        return 1

    logger.info(f"Processing {len(patient_ids)} patient(s): {patient_ids[:5]}"
                f"{'...' if len(patient_ids) > 5 else ''}")

    # --- Resolve data directories ---
    if args.images_dir:
        images_dir = args.images_dir
    elif args.data_dir:
        images_dir = args.data_dir / IMAGES_SUBDIR
    else:
        logger.error("Provide either --data-dir or --images-dir.")
        return 1

    if args.segmentations_dir:
        segmentations_dir = args.segmentations_dir
    elif args.data_dir:
        segmentations_dir = args.data_dir / SEGMENTATIONS_SUBDIR
    else:
        logger.error("Provide either --data-dir or --segmentations-dir.")
        return 1

    # --- Normaliser ---
    if args.no_normalization:
        normalizer = None
        # Monkey-patch: we also need to disable per-slice norm later.
        # We signal this by setting a sentinel.
        _force_no_norm = True
    else:
        normalizer = _load_normalizer(
            args.normalization_stats if args.normalization_stats else None
        )
        _force_no_norm = False

    # When --no-normalization is active we must pass normalize=False to the
    # slice extraction functions.  We achieve this by wrapping the extractor.
    if _force_no_norm:
        import eval.slice_extraction as _se
        _orig_extract_2d = _se.extract_2d_slice
        _orig_extract_all = _se.extract_all_tumor_slices

        def _patched_2d(volume, mask=None, mode=None, axis=0, normalize=True, **kw):
            return _orig_extract_2d(volume, mask=mask, mode=mode, axis=axis, normalize=False, **kw)

        def _patched_all(volume, mask=None, axis=0, normalize=True, **kw):
            return _orig_extract_all(volume, mask=mask, axis=axis, normalize=False, **kw)

        _se.extract_2d_slice = _patched_2d
        _se.extract_all_tumor_slices = _patched_all

    # --- Determine per_slice_norm flag for mirror slice construction ---
    # (matches the logic in train_classifier.py)
    _per_slice_norm_active = (normalizer is None) and not _force_no_norm

    save_raw = not args.no_raw
    save_visual = not args.no_visual

    if not save_raw and not save_visual:
        logger.warning("Both --no-raw and --no-visual are set. Nothing will be saved.")

    all_rows: list[dict] = []

    if args.task in ("contrast", "both"):
        logger.info("=== Contrast task ===")
        rows = run_contrast(
            patient_ids=patient_ids,
            images_dir=images_dir,
            segmentations_dir=segmentations_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            phase=args.phase,
            pre_phase=args.pre_phase,
            normalizer=normalizer,
            save_raw=save_raw,
            save_visual=save_visual,
        )
        all_rows.extend(rows)
        logger.info(f"Contrast: extracted {len(rows)} slice entries.")

    if args.task in ("tumor_roi", "both"):
        logger.info("=== Tumor-ROI task ===")
        rows = run_tumor_roi(
            patient_ids=patient_ids,
            images_dir=images_dir,
            segmentations_dir=segmentations_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            phase=args.phase,
            normalizer=normalizer,
            save_raw=save_raw,
            save_visual=save_visual,
            search_fraction=args.search_fraction,
            min_tissue_fraction=args.min_tissue_fraction,
        )
        all_rows.extend(rows)
        logger.info(f"Tumor-ROI: extracted {len(rows)} slice entries.")

    # --- Summary CSV ---
    csv_path = args.output_dir / "slice_summary.csv"
    _write_csv(all_rows, csv_path)

    logger.info(
        f"Done. Total slice entries: {len(all_rows)}. "
        f"Output directory: {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
