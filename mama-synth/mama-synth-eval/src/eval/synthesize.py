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


# ---------------------------------------------------------------------------
# Medigan-based synthesis
# ---------------------------------------------------------------------------

def synthesize_with_medigan(
    input_dir: Path,
    output_dir: Path,
    model_id: str = MEDIGAN_MODEL_ID,
    phase: int = DEFAULT_PHASE_PRE,
    batch_size: int = 1,
) -> list[Path]:
    """Generate synthetic post-contrast images using medigan.

    Loads pre-contrast images from *input_dir*, runs them through the
    specified medigan model, and saves results to *output_dir*.

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

    Returns
    -------
    list[Path]
        Paths to the generated files.
    """
    try:
        from medigan import Generators
    except ImportError:
        raise ImportError(
            "Synthesis with medigan requires the 'medigan' package. "
            "Install with: pip install 'mama-synth-eval[synthesis]'"
        )

    generators = Generators()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[Path] = []

    logger.info(
        f"Running medigan synthesis (model={model_id}) on {input_dir}"
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
        out_path = output_dir / f"{patient_id}_0001.nii.gz"

        try:
            # Use medigan's generate method
            generators.generate(
                model_id=model_id,
                input_path=str(img_path),
                output_path=str(out_path),
                num_samples=1,
            )
            generated_files.append(out_path)
            logger.debug(f"Generated: {out_path.name}")
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
