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
import sys
import tempfile
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
        Root directory containing the generated PNG slices.  If the
        model saved images in a subdirectory (e.g. ``batch_0/``), this
        function searches recursively for the first directory that
        actually contains PNG files.
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

    png_files = sorted(png_dir.glob("*.png"))
    if not png_files:
        # medigan may save in a subdirectory (e.g. output/batch_0/)
        try:
            actual_dir = _find_png_output_dir(png_dir)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No PNG files found in {png_dir} or its subdirectories "
                "— the model may not have produced output images."
            )
        png_files = sorted(actual_dir.glob("*.png"))

    slices = []
    for pf in png_files:
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
    model_id: str = MEDIGAN_MODEL_ID,
    phase: int = DEFAULT_PHASE_PRE,
    batch_size: int = 1,
    gpu_id: str = "0",
    image_size: int = 512,
) -> list[Path]:
    """Generate synthetic post-contrast images using medigan.

    For each pre-contrast NIfTI volume found in *input_dir* the pipeline:

    1. Slices the 3D volume into 2D axial PNG files (the format expected
       by medigan model 23 / Pix2PixHD).
    2. Calls ``medigan.Generators.generate()`` with the PNG directory.
    3. Reassembles the generated 2D slices into a 3D NIfTI volume and
       saves it to *output_dir*.

    Parameters
    ----------
    input_dir : Path
        Directory containing pre-contrast NIfTI images (or patient
        sub-folders with ``{pid}_0000.nii.gz`` files).
    output_dir : Path
        Target directory for generated post-contrast images.
    model_id : str
        Medigan model identifier (default: Pix2PixHD breast DCE-MRI).
    phase : int
        MRI phase index of the **input** images (default: 0 = pre-contrast).
    batch_size : int
        Number of images to generate per batch.
    gpu_id : str
        GPU identifier.  Accepts bare integers (``"0"``, ``"1"``,
        ``"-1"`` for CPU) or full PyTorch device strings
        (``"cuda:0"``, ``"cpu"``).  Bare integers are normalised to
        ``"cuda:<N>"`` (or ``"cpu"`` when negative) before being
        forwarded to medigan.
    image_size : int
        Spatial resolution expected by the model (default: 512).

    Returns
    -------
    list[Path]
        Paths to the generated NIfTI files.
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

    generated_files: list[Path] = []

    logger.info(
        f"Running medigan synthesis (model={model_id}, "
        f"gpu_id={device_str}, image_size={image_size}) on {input_dir}"
    )

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
        out_nifti = output_dir / f"{patient_id}_0001.nii.gz"

        try:
            with tempfile.TemporaryDirectory(
                prefix="mamasynth_",
            ) as tmpdir:
                tmp = Path(tmpdir)
                tmp_input = tmp / "input"
                tmp_output = tmp / "output"
                tmp_input.mkdir()
                tmp_output.mkdir()

                # 1. Slice NIfTI → 2D PNGs
                slice_meta = _nifti_to_png_slices(img_path, tmp_input)
                logger.debug(
                    f"{patient_id}: {slice_meta['num_slices']} slices "
                    f"extracted to {tmp_input}"
                )

                # 2. Run medigan synthesis
                generators.generate(
                    model_id=model_id,
                    input_path=str(tmp_input),
                    output_path=str(tmp_output),
                    num_samples=1,
                    save_images=True,
                    image_size=str(image_size),
                    gpu_id=device_str,
                )

                # 3. Reassemble output PNGs → 3D NIfTI
                _png_slices_to_nifti(tmp_output, out_nifti, slice_meta)

            generated_files.append(out_nifti)
            logger.debug(f"Generated: {out_nifti.name}")
        except Exception as e:
            logger.warning(f"Synthesis failed for {patient_id}: {e}")

    logger.info(
        f"Synthesis complete: {len(generated_files)}/{len(input_images)} "
        f"images generated → {output_dir}"
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
        if f.is_file() and phase_suffix in f.stem:
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
            model_id=args.model_id,
            phase=args.phase,
            gpu_id=args.gpu_id,
            image_size=args.image_size,
        )
        logger.info(f"Generated {len(generated)} synthetic images.")
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

        if args.model == "medigan":
            synthesize_with_medigan(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                model_id=args.model_id,
                phase=args.phase,
                gpu_id=args.gpu_id,
                image_size=args.image_size,
            )
        else:
            logger.error(
                f"Unknown model '{args.model}'. "
                "For custom models, use --predictions-dir with pre-generated "
                "images."
            )
            sys.exit(1)
    else:
        logger.info(
            f"\n--- Skipping synthesis — "
            f"using predictions from {args.predictions_dir} ---"
        )

    # --- Step 2: Evaluation -----------------------------------------------
    logger.info("\n--- Step 2: Evaluation ---")
    logger.info(f"Predictions:   {args.predictions_dir}")
    logger.info(f"Ground truth:  {args.ground_truth_path}")
    logger.info(f"Output:        {args.output_file}")

    results = run_evaluation(
        predictions_dir=args.predictions_dir,
        ground_truth_dir=args.ground_truth_path,
        output_file=args.output_file,
        masks_dir=args.masks_path,
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
