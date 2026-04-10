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
Command-line interface for the MAMA-SYNTH evaluation suite.

Usage:
    python -m eval \\
        --ground-truth-path /path/to/gt \\
        --predictions-path /path/to/pred \\
        --output-file /path/to/metrics.json \\
        [--masks-path /path/to/masks] \\
        [--labels-path /path/to/labels.json] \\
        [--roi-margin-mm 10.0] \\
        [--disable-lpips] [--disable-frd] \\
        [--disable-segmentation] [--disable-classification] \\
        [--seg-model-path /path/to/nnunet] \\
        [--clf-model-dir /path/to/classifiers] \\
        [--cache-dir /path/to/cache] \\
        [-v]
"""

import argparse
import logging
import sys
from pathlib import Path

from eval import MamaSynthEval


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "MAMA-SYNTH Challenge Evaluation: evaluate pre- to post-contrast "
            "breast DCE-MRI synthesis across 8 equally-weighted metrics — "
            "MSE, LPIPS (full image), SSIM, FRD (ROI), AUROC luminal, "
            "AUROC TNBC (classification), Dice, HD95 (segmentation)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-g", "--ground-truth-path",
        type=Path,
        default=Path("/opt/ml/input/data/ground_truth/"),
        help="Path to the ground truth directory containing reference images.",
    )
    parser.add_argument(
        "-i", "--predictions-path",
        type=Path,
        default=Path("/input/"),
        help="Path to the predictions directory containing generated images.",
    )
    parser.add_argument(
        "-o", "--output-file",
        type=Path,
        default=Path("/output/metrics.json"),
        help="Path to the output JSON file for metrics.",
    )
    parser.add_argument(
        "-m", "--masks-path",
        type=Path,
        default=None,
        help="Path to directory containing tumor segmentation masks (for ROI & SEG).",
    )
    parser.add_argument(
        "-l", "--labels-path",
        type=Path,
        default=None,
        help="Path to JSON/CSV file with molecular subtype labels (for CLF).",
    )
    parser.add_argument(
        "--roi-margin-mm",
        type=float,
        default=10.0,
        help="Dilation margin (mm) around tumor mask for ROI evaluation.",
    )
    parser.add_argument(
        "--seg-model-path",
        type=Path,
        default=None,
        help="Path to pre-trained nnUNet model directory for segmentation.",
    )
    parser.add_argument(
        "--clf-model-dir",
        type=Path,
        default=None,
        help="Directory with pre-trained classifier .pkl files (tnbc_classifier.pkl, luminal_classifier.pkl).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for caching intermediate radiomic features.",
    )
    parser.add_argument(
        "--disable-lpips",
        action="store_true",
        help="Skip LPIPS computation (useful if torch is not available).",
    )
    parser.add_argument(
        "--disable-frd",
        action="store_true",
        help="Skip FRD computation (useful if pyradiomics is not available).",
    )
    parser.add_argument(
        "--disable-segmentation",
        action="store_true",
        help="Skip segmentation evaluation.",
    )
    parser.add_argument(
        "--disable-classification",
        action="store_true",
        help="Skip classification evaluation.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        evaluator = MamaSynthEval(
            ground_truth_path=args.ground_truth_path,
            predictions_path=args.predictions_path,
            output_file=args.output_file,
            masks_path=args.masks_path,
            labels_path=args.labels_path,
            roi_margin_mm=args.roi_margin_mm,
            enable_lpips=not args.disable_lpips,
            enable_frd=not args.disable_frd,
            enable_segmentation=not args.disable_segmentation,
            enable_classification=not args.disable_classification,
            seg_model_path=args.seg_model_path,
            clf_model_dir=args.clf_model_dir,
            cache_dir=args.cache_dir,
        )
        results = evaluator.evaluate()

        # Print Grand-Challenge aggregates (the 8 official metrics)
        if "aggregates" in results and results["aggregates"]:
            print("\n=== Challenge Metrics (aggregates) ===")
            for metric, value in results["aggregates"].items():
                if isinstance(value, dict) and "mean" in value:
                    print(f"  {metric}: {value['mean']:.4f} +/- {value.get('std', 0):.4f}")
                elif isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")

        # Print per-task detail
        if "full_image" in results:
            print("\n=== Task 1: Full-Image Metrics ===")
            for metric, stats in results["full_image"].items():
                if isinstance(stats, dict) and "mean" in stats:
                    print(f"  {metric.upper()}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
                elif isinstance(stats, (int, float)):
                    print(f"  {metric.upper()}: {stats:.4f}")

        if "roi" in results:
            print("\n=== Task 2: Tumor ROI Metrics ===")
            for metric, stats in results["roi"].items():
                if isinstance(stats, dict) and "mean" in stats:
                    print(f"  {metric.upper()}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
                elif isinstance(stats, (int, float)):
                    print(f"  {metric.upper()}: {stats:.4f}")

        if "classification" in results:
            print("\n=== Task 3: Classification Metrics ===")
            for key, val in results["classification"].items():
                if isinstance(val, (int, float)):
                    print(f"  {key}: {val:.4f}")
                else:
                    print(f"  {key}: {val}")

        if "segmentation" in results:
            print("\n=== Task 4: Segmentation Metrics ===")
            for metric, stats in results["segmentation"].items():
                if isinstance(stats, dict) and "mean" in stats:
                    print(f"  {metric.upper()}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

        # Legacy per-case aggregate
        if "aggregate" in results:
            print("\n=== Per-Case Image Metrics (legacy aggregate) ===")
            for metric, stats in results["aggregate"].items():
                if isinstance(stats, dict) and "mean" in stats:
                    print(f"  {metric.upper()}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

        if "missing_predictions" in results:
            n_missing = len(results["missing_predictions"])
            print(f"\n  WARNING: {n_missing} ground truth images had no predictions (worst-score imputed)")

        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except ValueError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
