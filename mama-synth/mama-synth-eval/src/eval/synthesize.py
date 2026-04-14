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
Synthesis pipeline for generating post-contrast DCE-MRI from pre-contrast
images.

Provides CLI tools to run the **MAMA-SYNTH baseline model** (Pix2PixHD
via the `medigan <https://medigan.readthedocs.io>`_ library) and/or
evaluate already-generated synthetic images against ground truth.

Participants can also point the CLI at their own predictions directory,
bypassing synthesis entirely, so that the same unified command covers
both baseline and custom models.

Console scripts
---------------
``mamasynth-synthesize``
    Generate synthetic post-contrast images from pre-contrast inputs.

``mamasynth-synthesize-and-evaluate``
    Generate (or point to pre-generated) synthetic images **and** run
    the full MAMA-SYNTH evaluation pipeline in a single command.

Usage examples::

    # 1. Baseline synthesis only
    mamasynth-synthesize \\
        --data-dir /path/to/mama-mia \\
        --output-dir /path/to/synthetic-images

    # 2. Evaluate pre-generated predictions
    mamasynth-synthesize-and-evaluate \\
        --predictions-dir /path/to/existing-predictions \\
        --ground-truth-path /path/to/ground-truth \\
        --output-file metrics.json

    # 3. Baseline synthesis + full evaluation
    mamasynth-synthesize-and-evaluate \\
        --data-dir /path/to/mama-mia \\
        --output-dir /path/to/synthetic-images \\
        --clf-model-dir /path/to/classifiers \\
        --output-file metrics.json
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default medigan model ID for breast DCE-MRI Pix2PixHD
MEDIGAN_MODEL_ID = "00023_PIX2PIXHD_BREAST_DCEMRI"
MEDIGAN_MODEL_ID_NUMERIC = 23

# Default directory names matching MAMA-MIA / evaluation pipeline layout
IMAGES_SUBDIR = "images"
SEGMENTATIONS_SUBDIR = "segmentations"
DEFAULT_PHASE_PRE = 0   # pre-contrast
DEFAULT_PHASE_POST = 1  # first post-contrast

# Staging sub-directory placed *inside* output_dir so that intermediate
# PNG slices live on the same filesystem as the final outputs and
# survive failures for debugging.
WORK_SUBDIR = ".synthesis_work"

# Sub-directories for extracted ground-truth / mask slices used during
# the combined synthesize-and-evaluate pipeline.
GT_SLICES_SUBDIR = ".gt_slices"
MASK_SLICES_SUBDIR = ".mask_slices"


# ---------------------------------------------------------------------------
# Medigan-based synthesis
# ---------------------------------------------------------------------------


def _normalize_gpu_id(gpu_id: str) -> str:
    """Convert a CLI-style GPU identifier to a valid PyTorch device string.

    medigan model 23 passes the ``gpu_id`` value directly to
    ``torch.device()``, which requires strings like ``"cuda:0"`` or
    ``"cpu"`` — bare integers such as ``"0"`` are rejected with
    *"Invalid device string"*.

    Conversion rules:

    * ``"-1"``  → ``"cpu"``
    * ``"cpu"`` → ``"cpu"``  (already valid)
    * ``"0"``   → ``"cuda:0"``
    * ``"1"``   → ``"cuda:1"``
    * ``"cuda:0"`` / ``"cuda:1"`` → returned as-is
    """
    gpu_id = gpu_id.strip()

    # Already a valid full device string
    if gpu_id in ("cpu", "cuda"):
        return gpu_id
    if gpu_id.startswith("cuda:"):
        return gpu_id

    # Bare integer
    try:
        idx = int(gpu_id)
    except ValueError:
        # Unknown format — return as-is and let PyTorch raise if invalid
        return gpu_id

    if idx < 0:
        return "cpu"
    return f"cuda:{idx}"


def _ensure_medigan_importable() -> None:
    """Add CWD to ``sys.path`` so medigan model imports succeed.

    medigan stores downloaded model packages under ``./models/`` (relative
    to the current working directory) and loads them at runtime via
    ``importlib.import_module("models.<model_id>.<package>")``.  Two things
    must be true for that import to work:

    1. CWD is on ``sys.path`` so that the top-level ``models`` package is
       discoverable.
    2. ``models/`` contains an ``__init__.py`` so Python treats it as a
       regular package (namespace packages *can* work on ≥ 3.3 but adding
       the file is safer and avoids edge-cases in importlib).
    """
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    models_pkg = Path(cwd) / "models"
    models_pkg.mkdir(exist_ok=True)
    init_file = models_pkg / "__init__.py"
    if not init_file.exists():
        init_file.touch()


def _nifti_to_png_slices(
    nifti_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Slice a 3D NIfTI volume into 2D axial PNG files for medigan.

    Each axial slice is normalised to ``[0, 255]`` and saved as an 8-bit
    grayscale PNG.  The returned *metadata* dictionary captures the
    spatial information needed by :func:`_png_slices_to_nifti` to
    reassemble the generated slices into a proper 3D NIfTI volume.

    Parameters
    ----------
    nifti_path : Path
        Input NIfTI file (``*.nii.gz`` or ``*.nii``).
    output_dir : Path
        Directory into which PNG slices are written.

    Returns
    -------
    dict
        Keys: ``shape``, ``spacing``, ``origin``, ``direction``,
        ``min_val``, ``max_val``, ``num_slices``.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError(
            "NIfTI I/O requires SimpleITK. "
            "Install with: pip install SimpleITK"
        )
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "PNG I/O requires Pillow. "
            "Install with: pip install Pillow"
        )

    img = sitk.ReadImage(str(nifti_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)

    # Handle 4D volumes — take the first time-point
    if arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(
            f"Expected 3D or 4D NIfTI, got {arr.ndim}D from {nifti_path}"
        )

    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        norm = (arr - vmin) / (vmax - vmin) * 255.0
    else:
        norm = np.zeros_like(arr)

    output_dir.mkdir(parents=True, exist_ok=True)
    num_slices = arr.shape[0]
    for i in range(num_slices):
        pil_img = Image.fromarray(norm[i].astype(np.uint8), mode="L")
        pil_img.save(output_dir / f"slice_{i:04d}.png")

    return {
        "shape": arr.shape,
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
        "min_val": vmin,
        "max_val": vmax,
        "num_slices": num_slices,
    }


def _find_generated_images(root: Path) -> list[Path]:
    """Collect all generated image files under *root*, recursively.

    medigan may save output images in a subdirectory of the provided
    ``output_path`` (e.g. ``output/batch_0/``) and may use different
    extensions depending on the model.  This helper walks *root* and
    collects every file whose suffix matches a common image format.

    The returned list is **sorted** so that reassembly order is
    deterministic (critical for correct slice ordering).

    Raises
    ------
    FileNotFoundError
        If no image files are found anywhere under *root*.
    """
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        # List actual contents for debugging
        all_files = list(root.rglob("*"))
        logger.error(
            f"No image files found under {root}. "
            f"Directory contains {len(all_files)} entries: "
            f"{[str(f.relative_to(root)) for f in all_files[:20]]}"
        )
        raise FileNotFoundError(
            f"No image files found in {root} or any of its "
            "subdirectories — the model may not have produced output."
        )
    return images


def _find_png_output_dir(root: Path) -> Path:
    """Locate the directory that actually contains generated PNGs.

    medigan may save output images in a subdirectory of the provided
    ``output_path`` (e.g. ``output/batch_0/``).  This helper walks the
    tree rooted at *root* and returns the first directory that contains
    at least one ``*.png`` file.  If *root* itself contains PNGs it is
    returned immediately.

    Raises
    ------
    FileNotFoundError
        If no directory under *root* contains any PNG files.
    """
    # Check root itself first
    if list(root.glob("*.png")):
        return root

    # Walk subdirectories (breadth-first / sorted for determinism)
    for dirpath in sorted(root.rglob("*")):
        if dirpath.is_dir() and list(dirpath.glob("*.png")):
            return dirpath

    raise FileNotFoundError(
        f"No PNG files found in {root} or any of its subdirectories — "
        "the model may not have produced output images."
    )


def _png_slices_to_nifti(
    png_dir: Path,
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    """Reassemble 2D PNG slices into a 3D NIfTI volume.

    Output intensities are rescaled from ``[0, 255]`` back to the
    original intensity range recorded in *metadata*.

    Parameters
    ----------
    png_dir : Path
        Root directory containing the generated image slices.  The
        function searches recursively for image files (PNG, JPEG, TIFF,
        BMP) — medigan may place them in a subdirectory such as
        ``batch_0/`` or rename them with a ``batch_N_`` prefix.
    output_path : Path
        Destination NIfTI file path (e.g. ``patient_0001.nii.gz``).
    metadata : dict
        Metadata dict returned by :func:`_nifti_to_png_slices`.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("NIfTI I/O requires SimpleITK.")
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PNG I/O requires Pillow.")

    # Find all image files medigan produced (may be in subdirectories
    # and/or renamed with a batch_N_ prefix).
    image_files = _find_generated_images(png_dir)

    slices = []
    for pf in image_files:
        pil_img = Image.open(pf).convert("L")
        slices.append(np.array(pil_img, dtype=np.float32))

    arr = np.stack(slices, axis=0)  # [Z, Y, X]

    # Rescale to original intensity range
    vmin, vmax = metadata["min_val"], metadata["max_val"]
    if vmax > vmin:
        arr = arr / 255.0 * (vmax - vmin) + vmin

    # Resize slices if spatial dimensions changed
    orig = metadata["shape"]
    if arr.shape[1:] != orig[1:]:
        resized = []
        for i in range(arr.shape[0]):
            pil_tmp = Image.fromarray(arr[i])
            pil_tmp = pil_tmp.resize(
                (orig[2], orig[1]),  # PIL expects (width, height)
                Image.BILINEAR,
            )
            resized.append(np.array(pil_tmp, dtype=np.float32))
        arr = np.stack(resized, axis=0)

    # Pad / truncate slice count to match original volume
    if arr.shape[0] < orig[0]:
        pad = np.full(
            (orig[0] - arr.shape[0],) + arr.shape[1:],
            vmin,
            dtype=arr.dtype,
        )
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > orig[0]:
        arr = arr[: orig[0]]

    sitk_img = sitk.GetImageFromArray(arr)
    sitk_img.SetSpacing(metadata["spacing"])
    sitk_img.SetOrigin(metadata["origin"])
    sitk_img.SetDirection(metadata["direction"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk_img, str(output_path))


def synthesize_with_medigan(
    input_dir: Path,
    output_dir: Path,
    masks_dir: Optional[Path] = None,
    slice_mode: str = "max_tumor",
    model_id: str = MEDIGAN_MODEL_ID,
    phase: int = DEFAULT_PHASE_PRE,
    batch_size: int = 1,
    gpu_id: str = "0",
    image_size: int = 512,
    keep_work_dir: bool = False,
) -> list[Path]:
    """Generate synthetic post-contrast **2D PNG slices** using medigan.

    For each pre-contrast NIfTI volume found in *input_dir* the pipeline:

    1. Loads the volume and an optional tumour segmentation mask from
       *masks_dir*.
    2. Selects specific axial slices based on *slice_mode* and the mask.
    3. Saves the selected slices as 8-bit grayscale PNG files in a
       per-patient staging directory.
    4. Calls ``medigan.Generators.generate()`` on the PNG slices.
    5. Copies the generated output PNGs to *output_dir* with
       patient-based filenames.

    The output is a flat directory of **2D PNG images** (not 3D NIfTI
    volumes).  File naming convention:

    * Single-slice modes (``max_tumor`` / ``center_tumor``):
      ``{patient_id}.png``
    * Multi-slice mode (``all_tumor``):
      ``{patient_id}_s{slice_index:04d}.png``

    Intermediate files are staged in
    ``<output_dir>/.synthesis_work/<patient_id>/`` and cleaned up on
    success unless *keep_work_dir* is ``True``.  On failure the staging
    directory is **kept** for debugging.

    Parameters
    ----------
    input_dir : Path
        Directory containing pre-contrast NIfTI images (or patient
        sub-folders with ``{pid}_0000.nii.gz`` files).
    output_dir : Path
        Target directory for generated PNG images.
    masks_dir : Path or None
        Directory with tumour segmentation masks for slice selection.
        When ``None``, single-slice modes fall back to the middle slice.
    slice_mode : str
        ``"max_tumor"`` (default), ``"center_tumor"``, or
        ``"all_tumor"``.
    model_id : str
        Medigan model identifier (default: Pix2PixHD breast DCE-MRI).
    phase : int
        MRI phase index of the **input** images (default: 0 = pre).
    batch_size : int
        Number of images to generate per batch.
    gpu_id : str
        GPU identifier.  Bare integers are normalised to ``"cuda:<N>"``
        (or ``"cpu"`` when negative).
    image_size : int
        Spatial resolution expected by the model (default: 512).
    keep_work_dir : bool
        If ``True`` staging directories are never deleted.

    Returns
    -------
    list[Path]
        Paths to the generated PNG files.
    """
    try:
        from medigan import Generators
    except ImportError:
        raise ImportError(
            "Synthesis with medigan requires the 'medigan' package. "
            "Install with: pip install 'mama-synth-eval[synthesis]'"
        )

    # Ensure medigan can import its downloaded model packages
    _ensure_medigan_importable()

    # Normalise gpu_id to a valid PyTorch device string
    device_str = _normalize_gpu_id(gpu_id)

    generators = Generators()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Staging area inside output_dir — same filesystem, survives failures
    work_root = output_dir / WORK_SUBDIR
    work_root.mkdir(exist_ok=True)

    generated_files: list[Path] = []

    logger.info(
        f"Running medigan synthesis (model={model_id}, "
        f"gpu_id={device_str}, image_size={image_size}, "
        f"slice_mode={slice_mode}) on {input_dir}"
    )
    logger.info(f"Staging directory: {work_root}")

    # Discover input images
    input_images = _discover_input_images(input_dir, phase=phase)
    if not input_images:
        raise FileNotFoundError(
            f"No pre-contrast images found in {input_dir}. "
            f"Expected files matching *_{phase:04d}.nii.gz or similar."
        )

    logger.info(f"Found {len(input_images)} input images.")

    for img_path in input_images:
        patient_id = _extract_patient_id(img_path)
        patient_work = work_root / patient_id
        work_input = patient_work / "input_slices"
        work_output = patient_work / "output_slices"

        try:
            # Prepare fresh per-patient staging dirs
            if patient_work.exists():
                shutil.rmtree(patient_work)
            work_input.mkdir(parents=True)
            work_output.mkdir(parents=True)

            # Load mask for slice selection
            mask_arr = _load_mask_for_patient(patient_id, masks_dir)

            # 1. Extract selected slices as PNG (sequential naming for
            #    medigan, which processes a directory of PNGs).
            slice_infos = _extract_and_save_slices(
                img_path,
                work_input,
                patient_id,
                mask_arr=mask_arr,
                slice_mode=slice_mode,
                sequential_naming=True,
            )
            logger.debug(
                f"{patient_id}: {len(slice_infos)} slice(s) extracted "
                f"(indices: {[idx for _, idx in slice_infos]})"
            )

            # 2. Run medigan synthesis
            generators.generate(
                model_id=model_id,
                input_path=str(work_input),
                output_path=str(work_output),
                num_samples=len(slice_infos),
                save_images=True,
                image_size=str(image_size),
                gpu_id=device_str,
            )

            # Collect output PNGs
            produced = sorted(_find_generated_images(work_output))
            logger.debug(
                f"{patient_id}: medigan produced {len(produced)} file(s)"
            )

            if len(produced) != len(slice_infos):
                logger.warning(
                    f"{patient_id}: expected {len(slice_infos)} output "
                    f"image(s) but medigan produced {len(produced)}"
                )

            # 3. Copy output PNGs to output_dir with patient-based naming
            for i, (_, slice_idx) in enumerate(slice_infos):
                if i >= len(produced):
                    break
                if len(slice_infos) == 1:
                    out_name = f"{patient_id}.png"
                else:
                    out_name = f"{patient_id}_s{slice_idx:04d}.png"
                dest = output_dir / out_name
                shutil.copy2(str(produced[i]), str(dest))
                generated_files.append(dest)

            logger.debug(f"Generated PNG(s) for {patient_id}")

            # Clean up staging dir on success (unless user wants to keep it)
            if not keep_work_dir:
                shutil.rmtree(patient_work, ignore_errors=True)

        except Exception as e:
            logger.warning(
                f"Synthesis failed for {patient_id}: {e}  "
                f"(intermediate files kept in {patient_work})"
            )

    # Remove the work root if empty and we're not keeping it
    if not keep_work_dir:
        try:
            work_root.rmdir()  # only succeeds if empty (all patients OK)
        except OSError:
            remaining = list(work_root.iterdir())
            logger.info(
                f"Staging directory retained ({len(remaining)} failed "
                f"patient(s)): {work_root}"
            )

    logger.info(
        f"Synthesis complete: {len(generated_files)} PNG(s) from "
        f"{len(input_images)} input volume(s) → {output_dir}"
    )
    return generated_files


def _discover_input_images(
    input_dir: Path,
    phase: int = DEFAULT_PHASE_PRE,
) -> list[Path]:
    """Find pre-contrast images in *input_dir*.

    Supports both flat layouts (``{pid}_0000.nii.gz``) and nested layouts
    (``{pid}/{pid}_0000.nii.gz``).
    """
    phase_suffix = f"_{phase:04d}"

    # Try nested layout first (MAMA-MIA style)
    images: list[Path] = []
    for patient_dir in sorted(input_dir.iterdir()):
        if patient_dir.is_dir():
            for f in sorted(patient_dir.iterdir()):
                if phase_suffix in f.stem and f.suffix in (".gz", ".nii", ".mha"):
                    images.append(f)

    if images:
        return images

    # Fall back to flat layout
    for f in sorted(input_dir.iterdir()):
        if (
            f.is_file()
            and phase_suffix in f.stem
            and f.suffix in (".gz", ".nii", ".mha")
        ):
            images.append(f)

    return images


def _extract_patient_id(path: Path) -> str:
    """Extract patient ID from a NIfTI filename.

    ``ISPY1_1001_0000.nii.gz`` → ``ISPY1_1001``
    """
    stem = path.name
    # Remove all extensions (.nii.gz, .nii, .mha)
    for ext in (".nii.gz", ".nii", ".mha", ".mhd", ".png"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    # Remove trailing _DDDD phase suffix
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        return parts[0]
    return stem


def _strip_extensions(filename: str) -> str:
    """Remove medical-image extensions from a filename."""
    for ext in (".nii.gz", ".nii", ".mha", ".mhd", ".png"):
        if filename.endswith(ext):
            return filename[: -len(ext)]
    return Path(filename).stem


def _stem_matches_patient(filename: str, patient_id: str) -> bool:
    """Test whether *filename* belongs to *patient_id*.

    Strips medical-image extensions, then checks for:

    1. Exact match (e.g. ``DUKE_055.nii.gz`` for patient ``DUKE_055``).
    2. Match with a trailing ``_DDDD`` phase / channel suffix
       (e.g. ``DUKE_055_0000.nii.gz``).

    This avoids the ambiguity in :func:`_extract_patient_id`, which
    cannot distinguish a 4-digit patient-ID suffix from a phase suffix
    when the file has no phase tag.
    """
    stem = _strip_extensions(filename)
    if stem == patient_id:
        return True
    # Check for {patient_id}_{4-digit-suffix} pattern
    if stem.startswith(patient_id + "_"):
        suffix = stem[len(patient_id) + 1 :]
        if suffix.isdigit() and len(suffix) == 4:
            return True
    return False


# ---------------------------------------------------------------------------
# 2D slice selection and extraction helpers
# ---------------------------------------------------------------------------

# Slice selection modes exposed via CLI (a subset of SliceMode in
# slice_extraction.py, limited to the three modes that make sense for
# the synthesis→evaluation pipeline).
SLICE_MODE_CHOICES = ("max_tumor", "center_tumor", "all_tumor")


def _select_slices(
    volume: np.ndarray,
    mask: Optional[np.ndarray],
    slice_mode: str,
) -> list[int]:
    """Select axial slice indices from a 3D volume.

    Parameters
    ----------
    volume : ndarray
        3D array ``(D, H, W)``.
    mask : ndarray or None
        3D boolean mask.  Required for tumour-based modes; when
        ``None`` the function falls back to the middle slice.
    slice_mode : str
        ``"max_tumor"``, ``"center_tumor"``, or ``"all_tumor"``.

    Returns
    -------
    list[int]
        Sorted slice indices.
    """
    from eval.slice_extraction import (
        find_max_tumor_slice,
        find_center_tumor_slice,
    )

    has_mask = mask is not None and mask.ndim == 3 and np.any(mask)
    n_slices = volume.shape[0]

    if slice_mode == "max_tumor":
        if has_mask:
            return [find_max_tumor_slice(mask)]
        logger.warning(
            "No mask for max_tumor mode; falling back to middle slice."
        )
        return [n_slices // 2]

    if slice_mode == "center_tumor":
        if has_mask:
            return [find_center_tumor_slice(mask)]
        logger.warning(
            "No mask for center_tumor mode; falling back to middle slice."
        )
        return [n_slices // 2]

    if slice_mode == "all_tumor":
        if not has_mask:
            raise ValueError(
                "all_tumor slice mode requires a segmentation mask "
                "but none was provided or the mask is empty."
            )
        areas = np.sum(mask, axis=(1, 2))
        return sorted(int(i) for i in np.nonzero(areas)[0])

    raise ValueError(f"Unknown slice_mode: {slice_mode!r}")


def _load_mask_for_patient(
    patient_id: str,
    masks_dir: Optional[Path],
) -> Optional[np.ndarray]:
    """Load a 3D segmentation mask for *patient_id* from *masks_dir*.

    Searches for files whose extracted patient ID matches, trying common
    medical-image extensions in both flat and nested directory layouts.

    Returns ``None`` when *masks_dir* is ``None``, does not exist, or no
    matching mask file is found.
    """
    if masks_dir is None or not masks_dir.exists():
        return None

    try:
        import SimpleITK as sitk
    except ImportError:
        logger.warning("SimpleITK not available — cannot load masks.")
        return None

    for ext in (".nii.gz", ".nii", ".mha", ".mhd"):
        # Flat layout: masks_dir/{pid}*{ext}
        for candidate in sorted(masks_dir.glob(f"{patient_id}*{ext}")):
            if _stem_matches_patient(candidate.name, patient_id):
                try:
                    arr = sitk.GetArrayFromImage(
                        sitk.ReadImage(str(candidate), sitk.sitkUInt8)
                    )
                    if arr.ndim == 4:
                        arr = arr[0]
                    return arr.astype(bool)
                except Exception as e:
                    logger.warning(f"Failed to load mask {candidate}: {e}")

        # Nested layout: masks_dir/{pid}/{pid}*{ext}
        nested = masks_dir / patient_id
        if nested.is_dir():
            for candidate in sorted(nested.glob(f"{patient_id}*{ext}")):
                if _stem_matches_patient(candidate.name, patient_id):
                    try:
                        arr = sitk.GetArrayFromImage(
                            sitk.ReadImage(str(candidate), sitk.sitkUInt8)
                        )
                        if arr.ndim == 4:
                            arr = arr[0]
                        return arr.astype(bool)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load mask {candidate}: {e}"
                        )

    return None


def _extract_and_save_slices(
    nifti_path: Path,
    output_dir: Path,
    patient_id: str,
    mask_arr: Optional[np.ndarray] = None,
    slice_mode: str = "max_tumor",
    sequential_naming: bool = False,
) -> list[tuple[Path, int]]:
    """Extract selected axial slices from a NIfTI volume and save as PNG.

    The full 3D volume is normalised to ``[0, 255]`` using the global
    min/max, and the selected slices are saved as 8-bit grayscale PNGs.

    Parameters
    ----------
    nifti_path : Path
        Input NIfTI file.
    output_dir : Path
        Target directory for PNG files.
    patient_id : str
        Patient identifier (used in filenames).
    mask_arr : ndarray or None
        3D boolean mask for slice selection.
    slice_mode : str
        ``"max_tumor"``, ``"center_tumor"``, or ``"all_tumor"``.
    sequential_naming : bool
        If ``True`` files are named ``slice_0000.png``, ``slice_0001.png``,
        … (positional, for medigan input staging).  If ``False`` files use
        ``{patient_id}.png`` (single-slice) or
        ``{patient_id}_s{idx:04d}.png`` (multi-slice).

    Returns
    -------
    list[tuple[Path, int]]
        ``(png_path, slice_index)`` pairs sorted by slice index.
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError("NIfTI I/O requires SimpleITK.")
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PNG I/O requires Pillow.")

    img = sitk.ReadImage(str(nifti_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    if arr.ndim == 4:
        arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(
            f"Expected 3D or 4D NIfTI, got {arr.ndim}D from {nifti_path}"
        )

    # Select slices
    indices = _select_slices(arr, mask_arr, slice_mode)

    # Normalise to [0, 255] based on full volume range
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        norm = (arr - vmin) / (vmax - vmin) * 255.0
    else:
        norm = np.zeros_like(arr)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Path, int]] = []

    for seq_i, idx in enumerate(indices):
        slice_arr = norm[idx].astype(np.uint8)
        if sequential_naming:
            name = f"slice_{seq_i:04d}.png"
        elif len(indices) == 1:
            name = f"{patient_id}.png"
        else:
            name = f"{patient_id}_s{idx:04d}.png"

        png_path = output_dir / name
        Image.fromarray(slice_arr, mode="L").save(png_path)
        results.append((png_path, int(idx)))

    return results


def extract_ground_truth_slices(
    gt_dir: Path,
    masks_dir: Optional[Path],
    output_dir: Path,
    slice_mode: str = "max_tumor",
    phase: int = DEFAULT_PHASE_POST,
    masks_output_dir: Optional[Path] = None,
) -> list[Path]:
    """Extract 2D ground-truth slices from 3D NIfTI volumes.

    Uses the same slice-selection logic (mask + *slice_mode*) as
    :func:`synthesize_with_medigan` so that predictions and ground truth
    are compared at the same anatomical positions.

    When *masks_output_dir* is provided, the corresponding 2D binary
    mask slices are also saved there (as 8-bit grayscale PNGs with the
    same filenames) so that ROI-based evaluation metrics can consume
    them directly.

    Parameters
    ----------
    gt_dir : Path
        Directory with ground-truth post-contrast NIfTI volumes.
    masks_dir : Path or None
        Segmentation masks directory (for slice selection).
    output_dir : Path
        Target directory for extracted GT PNG slices.
    slice_mode : str
        Slice selection strategy (must match what was used for synthesis).
    phase : int
        Phase index of GT images (default 1 = post-contrast).
    masks_output_dir : Path or None
        If given, matching 2D mask slices are saved here.

    Returns
    -------
    list[Path]
        Paths to saved GT PNG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if masks_output_dir is not None:
        masks_output_dir.mkdir(parents=True, exist_ok=True)

    gt_images = _discover_input_images(gt_dir, phase=phase)
    if not gt_images:
        # Fall back: try all NIfTI-like files (GT may lack phase suffix)
        gt_images = sorted(
            f for f in gt_dir.iterdir()
            if f.is_file()
            and any(f.name.endswith(e) for e in (".nii.gz", ".nii", ".mha"))
        )

    saved: list[Path] = []

    for gt_path in gt_images:
        patient_id = _extract_patient_id(gt_path)
        mask_arr = _load_mask_for_patient(patient_id, masks_dir)

        try:
            slice_infos = _extract_and_save_slices(
                gt_path, output_dir, patient_id,
                mask_arr=mask_arr,
                slice_mode=slice_mode,
                sequential_naming=False,
            )
            saved.extend(p for p, _ in slice_infos)

            # Save matching mask slices
            if masks_output_dir is not None and mask_arr is not None:
                try:
                    from PIL import Image as _PIL

                    for _, idx in slice_infos:
                        mask_slice = (mask_arr[idx] > 0).astype(np.uint8) * 255
                        if len(slice_infos) == 1:
                            mname = f"{patient_id}.png"
                        else:
                            mname = f"{patient_id}_s{idx:04d}.png"
                        _PIL.fromarray(mask_slice, mode="L").save(
                            masks_output_dir / mname
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to save mask slices for {patient_id}: {e}"
                    )

        except Exception as e:
            logger.warning(
                f"Failed to extract GT slices for {patient_id}: {e}"
            )

    logger.info(
        f"Extracted {len(saved)} ground-truth PNG slice(s) → {output_dir}"
    )
    return saved


# ---------------------------------------------------------------------------
# Evaluate predictions (thin wrapper around MamaSynthEval)
# ---------------------------------------------------------------------------

def run_evaluation(
    predictions_dir: Path,
    ground_truth_dir: Path,
    output_file: Path,
    masks_dir: Optional[Path] = None,
    labels_path: Optional[Path] = None,
    clf_model_dir: Optional[Path] = None,
    seg_model_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    disable_lpips: bool = False,
    disable_frd: bool = False,
    disable_segmentation: bool = False,
    disable_classification: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full MAMA-SYNTH evaluation pipeline.

    Parameters
    ----------
    predictions_dir : Path
        Directory with generated post-contrast images.
    ground_truth_dir : Path
        Directory with reference post-contrast images.
    output_file : Path
        Output JSON file path.
    masks_dir, labels_path, clf_model_dir, seg_model_path, cache_dir :
        Optional paths for ROI / classification / segmentation evaluation.
    disable_lpips, disable_frd, disable_segmentation, disable_classification :
        Flags to skip specific metric groups.
    verbose : bool
        Enable DEBUG logging.

    Returns
    -------
    dict
        Evaluation results (same structure as ``metrics.json``).
    """
    from eval.evaluation import MamaSynthEval

    evaluator = MamaSynthEval(
        ground_truth_path=str(ground_truth_dir),
        predictions_path=str(predictions_dir),
        output_file=str(output_file),
        masks_path=str(masks_dir) if masks_dir else None,
        labels_path=str(labels_path) if labels_path else None,
        clf_model_dir=str(clf_model_dir) if clf_model_dir else None,
        seg_model_path=str(seg_model_path) if seg_model_path else None,
        cache_dir=str(cache_dir) if cache_dir else None,
        enable_lpips=not disable_lpips,
        enable_frd=not disable_frd,
        enable_segmentation=not disable_segmentation,
        enable_classification=not disable_classification,
    )

    results = evaluator.evaluate()
    logger.info(f"Evaluation results saved to {output_file}")
    return results


# ---------------------------------------------------------------------------
# CLI — synthesize
# ---------------------------------------------------------------------------

def parse_synthesize_args(
    argv: Optional[list[str]] = None,
) -> argparse.Namespace:
    """Parse arguments for ``mamasynth-synthesize``."""
    parser = argparse.ArgumentParser(
        prog="mamasynth-synthesize",
        description=(
            "Generate synthetic post-contrast DCE-MRI images using the "
            "MAMA-SYNTH baseline model (Pix2PixHD via medigan) or prepare "
            "predictions from a custom model for evaluation."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Root MAMA-MIA dataset directory. Pre-contrast images are "
            "loaded from <data-dir>/images/."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Override path to pre-contrast input images. "
            "If --data-dir is provided, defaults to <data-dir>/images."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the generated post-contrast images.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medigan",
        help=(
            "Synthesis model to use. "
            "'medigan' (default) uses the Pix2PixHD baseline. "
            "Specify a custom model path or identifier for other models."
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MEDIGAN_MODEL_ID,
        help=f"Medigan model ID. Default: {MEDIGAN_MODEL_ID}.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=DEFAULT_PHASE_PRE,
        help="Input MRI phase index (0 = pre-contrast). Default: 0.",
    )
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="0",
        help=(
            "GPU device for the synthesis model. Accepts a bare "
            "integer (0, 1, …) which maps to 'cuda:N', '-1' for "
            "CPU, or a full device string like 'cuda:0' / 'cpu'. "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Spatial resolution for the model. Default: 512.",
    )
    parser.add_argument(
        "--slice-mode",
        type=str,
        choices=SLICE_MODE_CHOICES,
        default="max_tumor",
        help=(
            "Slice selection strategy for extracting 2D slices from "
            "3D input volumes. 'max_tumor' (default): slice with "
            "largest tumour area. 'center_tumor': slice through "
            "tumour centre of mass. 'all_tumor': every slice with "
            "tumour."
        ),
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help=(
            "Directory with segmentation masks for slice selection. "
            "Required for accurate tumour-based slice modes. "
            "Default: <data-dir>/segmentations if --data-dir is set."
        ),
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help=(
            "Keep the intermediate slice directories inside "
            "<output-dir>/.synthesis_work/ after successful synthesis "
            "(always kept on failure for debugging)."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG level) logging.",
    )

    args = parser.parse_args(argv)

    # Resolve input directory
    if args.input_dir is None:
        if args.data_dir is not None:
            args.input_dir = args.data_dir / IMAGES_SUBDIR
        else:
            parser.error(
                "At least one of --data-dir or --input-dir is required."
            )

    # Resolve masks directory
    if args.masks_dir is None and args.data_dir is not None:
        candidate = args.data_dir / SEGMENTATIONS_SUBDIR
        if candidate.exists():
            args.masks_dir = candidate

    return args


def synthesize_main(argv: Optional[list[str]] = None) -> None:
    """Entry point for ``mamasynth-synthesize``."""
    args = parse_synthesize_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("MAMA-SYNTH Synthesis Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input:  {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Model:  {args.model}")

    if args.model == "medigan":
        generated = synthesize_with_medigan(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            masks_dir=args.masks_dir,
            slice_mode=args.slice_mode,
            model_id=args.model_id,
            phase=args.phase,
            gpu_id=args.gpu_id,
            image_size=args.image_size,
            keep_work_dir=args.keep_work_dir,
        )
        logger.info(f"Generated {len(generated)} synthetic PNG(s).")
    else:
        logger.error(
            f"Unknown model '{args.model}'. Currently supported: 'medigan'. "
            "For custom models, generate predictions externally and use "
            "mamasynth-synthesize-and-evaluate --predictions-dir instead."
        )
        sys.exit(1)

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI — synthesize-and-evaluate
# ---------------------------------------------------------------------------

def parse_synthesize_and_evaluate_args(
    argv: Optional[list[str]] = None,
) -> argparse.Namespace:
    """Parse arguments for ``mamasynth-synthesize-and-evaluate``."""
    parser = argparse.ArgumentParser(
        prog="mamasynth-synthesize-and-evaluate",
        description=(
            "Synthesize post-contrast DCE-MRI images and/or evaluate "
            "predictions against ground-truth. Supports the medigan "
            "baseline, custom models, or pre-generated prediction "
            "directories."
        ),
        epilog=(
            "Examples:\n"
            "  # Evaluate pre-generated predictions\n"
            "  mamasynth-synthesize-and-evaluate \\\n"
            "      --predictions-dir /path/to/predictions \\\n"
            "      --ground-truth-path /path/to/ground-truth \\\n"
            "      --output-file metrics.json\n"
            "\n"
            "  # Synthesize with medigan baseline + evaluate\n"
            "  mamasynth-synthesize-and-evaluate \\\n"
            "      --data-dir /path/to/mama-mia \\\n"
            "      --output-dir /path/to/synthetic \\\n"
            "      --clf-model-dir /path/to/classifiers \\\n"
            "      --output-file metrics.json\n"
        ),
    )

    # --- Synthesis options -------------------------------------------------
    synth = parser.add_argument_group("synthesis options")
    synth.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root MAMA-MIA dataset directory.",
    )
    synth.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override input images directory.",
    )
    synth.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for generated synthetic images (synthesis mode). "
            "Also used as --predictions-dir if the latter is not set."
        ),
    )
    synth.add_argument(
        "--model",
        type=str,
        default="medigan",
        help="Synthesis model to use. Default: 'medigan'.",
    )
    synth.add_argument(
        "--model-id",
        type=str,
        default=MEDIGAN_MODEL_ID,
        help=f"Medigan model ID. Default: {MEDIGAN_MODEL_ID}.",
    )
    synth.add_argument(
        "--phase",
        type=int,
        default=DEFAULT_PHASE_PRE,
        help="Input MRI phase index. Default: 0 (pre-contrast).",
    )
    synth.add_argument(
        "--gpu-id",
        type=str,
        default="0",
        help=(
            "GPU device for the synthesis model. Accepts a bare "
            "integer (0, 1, …) which maps to 'cuda:N', '-1' for "
            "CPU, or a full device string like 'cuda:0' / 'cpu'. "
            "Default: 0."
        ),
    )
    synth.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Spatial resolution for the model. Default: 512.",
    )
    synth.add_argument(
        "--slice-mode",
        type=str,
        choices=SLICE_MODE_CHOICES,
        default="max_tumor",
        help=(
            "Slice selection strategy: 'max_tumor' (default), "
            "'center_tumor', or 'all_tumor'."
        ),
    )
    synth.add_argument(
        "--keep-work-dir",
        action="store_true",
        help=(
            "Keep intermediate slice directories inside "
            "<output-dir>/.synthesis_work/ after synthesis "
            "(always kept on failure for debugging)."
        ),
    )
    synth.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help=(
            "Segmentation masks directory used for slice selection "
            "during synthesis. Falls back to --masks-path (evaluation "
            "masks) when not set. Auto-resolved from "
            "<data-dir>/segmentations if --data-dir is given."
        ),
    )
    synth.add_argument(
        "--skip-synthesis",
        action="store_true",
        help=(
            "Skip the synthesis step and only evaluate. "
            "Requires --predictions-dir."
        ),
    )

    # --- Evaluation options ------------------------------------------------
    evl = parser.add_argument_group("evaluation options")
    evl.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help=(
            "Directory with pre-generated predictions. If provided, "
            "synthesis is skipped and these predictions are evaluated "
            "directly."
        ),
    )
    evl.add_argument(
        "--ground-truth-path", "-g",
        type=Path,
        default=None,
        help=(
            "Ground truth post-contrast images. "
            "Default: <data-dir>/images if --data-dir is set."
        ),
    )
    evl.add_argument(
        "--output-file", "-o",
        type=Path,
        default=Path("metrics.json"),
        help="Output JSON file path. Default: metrics.json.",
    )
    evl.add_argument(
        "--masks-path", "-m",
        type=Path,
        default=None,
        help=(
            "Tumor segmentation masks directory. "
            "Default: <data-dir>/segmentations if --data-dir is set."
        ),
    )
    evl.add_argument(
        "--labels-path", "-l",
        type=Path,
        default=None,
        help="Labels file (JSON or CSV) for classification evaluation.",
    )
    evl.add_argument(
        "--clf-model-dir",
        type=Path,
        default=None,
        help="Directory with pre-trained classifier models (.pkl or .pt).",
    )
    evl.add_argument(
        "--seg-model-path",
        type=Path,
        default=None,
        help="Pre-trained nnUNet model directory for segmentation.",
    )
    evl.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Feature cache directory.",
    )
    evl.add_argument(
        "--disable-lpips",
        action="store_true",
        help="Skip LPIPS computation.",
    )
    evl.add_argument(
        "--disable-frd",
        action="store_true",
        help="Skip FRD computation.",
    )
    evl.add_argument(
        "--disable-segmentation",
        action="store_true",
        help="Skip segmentation evaluation.",
    )
    evl.add_argument(
        "--disable-classification",
        action="store_true",
        help="Skip classification evaluation.",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG level) logging.",
    )

    args = parser.parse_args(argv)

    # --- Resolve paths / modes -------------------------------------------
    has_predictions = args.predictions_dir is not None
    skip_synthesis = args.skip_synthesis or has_predictions

    if not skip_synthesis:
        # Need input for synthesis
        if args.input_dir is None:
            if args.data_dir is not None:
                args.input_dir = args.data_dir / IMAGES_SUBDIR
            else:
                parser.error(
                    "Synthesis mode requires --data-dir or --input-dir."
                )
        if args.output_dir is None:
            parser.error(
                "Synthesis mode requires --output-dir to save generated images."
            )

    # Predictions dir for evaluation
    if args.predictions_dir is None:
        if args.output_dir is not None:
            args.predictions_dir = args.output_dir
        else:
            parser.error(
                "Evaluation requires --predictions-dir or --output-dir."
            )

    # Ground truth
    if args.ground_truth_path is None:
        if args.data_dir is not None:
            args.ground_truth_path = args.data_dir / IMAGES_SUBDIR
        else:
            parser.error(
                "Evaluation requires --ground-truth-path or --data-dir."
            )

    # Defaults from data-dir
    if args.data_dir is not None:
        if args.masks_path is None:
            candidate = args.data_dir / SEGMENTATIONS_SUBDIR
            if candidate.exists():
                args.masks_path = candidate

    # Resolve masks_dir for synthesis slice selection:
    #   explicit --masks-dir  >  --masks-path  >  <data-dir>/segmentations
    if args.masks_dir is None:
        args.masks_dir = args.masks_path

    args._skip_synthesis = skip_synthesis
    return args


def synthesize_and_evaluate_main(
    argv: Optional[list[str]] = None,
) -> None:
    """Entry point for ``mamasynth-synthesize-and-evaluate``."""
    args = parse_synthesize_and_evaluate_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("MAMA-SYNTH Synthesis & Evaluation Pipeline")
    logger.info("=" * 60)

    # --- Step 1: Synthesis ------------------------------------------------
    if not args._skip_synthesis:
        logger.info("\n--- Step 1: Synthesis ---")
        logger.info(f"Input:  {args.input_dir}")
        logger.info(f"Output: {args.output_dir}")
        logger.info(f"Model:  {args.model}")
        logger.info(f"Slice mode: {args.slice_mode}")

        if args.model == "medigan":
            synthesize_with_medigan(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                masks_dir=args.masks_dir,
                slice_mode=args.slice_mode,
                model_id=args.model_id,
                phase=args.phase,
                gpu_id=args.gpu_id,
                image_size=args.image_size,
                keep_work_dir=args.keep_work_dir,
            )

    # --- Step 1b: Extract matching GT and mask slices ---------------------
    gt_eval_dir: Path = args.ground_truth_path
    mask_eval_dir: Optional[Path] = args.masks_path

    if not args._skip_synthesis:
        gt_eval_dir = args.output_dir / GT_SLICES_SUBDIR
        mask_eval_dir = (
            args.output_dir / MASK_SLICES_SUBDIR
            if args.masks_dir else None
        )
        logger.info("\n--- Step 1b: Extracting GT slices ---")
        extract_ground_truth_slices(
            gt_dir=args.ground_truth_path,
            masks_dir=args.masks_dir,
            output_dir=gt_eval_dir,
            slice_mode=args.slice_mode,
            phase=DEFAULT_PHASE_POST,
            masks_output_dir=mask_eval_dir,
        )

    # --- Step 2: Evaluation -----------------------------------------------
    logger.info("\n--- Step 2: Evaluation ---")
    logger.info(f"Predictions:   {args.predictions_dir}")
    logger.info(f"Ground truth:  {gt_eval_dir}")
    logger.info(f"Output:        {args.output_file}")

    results = run_evaluation(
        predictions_dir=args.predictions_dir,
        ground_truth_dir=gt_eval_dir,
        output_file=args.output_file,
        masks_dir=mask_eval_dir,
        labels_path=args.labels_path,
        clf_model_dir=args.clf_model_dir,
        seg_model_path=args.seg_model_path,
        cache_dir=args.cache_dir,
        disable_lpips=args.disable_lpips,
        disable_frd=args.disable_frd,
        disable_segmentation=args.disable_segmentation,
        disable_classification=args.disable_classification,
        verbose=args.verbose,
    )

    # Print summary
    aggregates = results.get("aggregates", {})
    if aggregates:
        logger.info("\n--- Evaluation Summary ---")
        for key, value in sorted(aggregates.items()):
            if isinstance(value, dict):
                logger.info(
                    f"  {key}: {value.get('mean', 'N/A'):.4f} "
                    f"(±{value.get('std', 0):.4f})"
                )
            elif isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    logger.info("=" * 60)


if __name__ == "__main__":
    synthesize_and_evaluate_main()
