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
mama-synth-eval: Evaluation suite for the MAMA-SYNTH challenge.

Pre- to post-contrast translation evaluation for breast DCE-MRI with
8 equally-weighted metrics across four tasks: full-image comparison
(MSE, LPIPS), tumor ROI comparison (SSIM, FRD), classification (AUROC),
and segmentation (Dice, HD95).
"""

from eval.evaluation import (
    DatasetNormalizer,
    METRIC_AUROC_LUMINAL,
    METRIC_AUROC_TNBC,
    METRIC_DICE,
    METRIC_FRD_ROI,
    METRIC_HD95,
    METRIC_LPIPS_FULL,
    METRIC_MSE_FULL,
    METRIC_SSIM_ROI,
    MamaSynthEval,
    normalize_intensity,
)
from eval.metrics import (
    compute_dice,
    compute_hd95,
    compute_mae,
    compute_mse,
    compute_ncc,
    compute_nmse,
    compute_psnr,
    compute_ssim,
)
from eval.slice_extraction import (
    SliceMode,
    extract_2d_slice,
    extract_multi_slices,
)
from eval.classification import (
    CNNClassifier,
    EnsembleClassifier,
    RadiomicsClassifier,
    evaluate_classification,
)
from eval.training_visualization import TrainingVisualizer

__all__ = [
    "MamaSynthEval",
    "DatasetNormalizer",
    "normalize_intensity",
    # Challenge metric name constants
    "METRIC_MSE_FULL",
    "METRIC_LPIPS_FULL",
    "METRIC_SSIM_ROI",
    "METRIC_FRD_ROI",
    "METRIC_AUROC_LUMINAL",
    "METRIC_AUROC_TNBC",
    "METRIC_DICE",
    "METRIC_HD95",
    # Image-to-image metrics
    "compute_mae",
    "compute_mse",
    "compute_nmse",
    "compute_psnr",
    "compute_ssim",
    "compute_ncc",
    # Segmentation metrics
    "compute_dice",
    "compute_hd95",
    # Classification
    "RadiomicsClassifier",
    "CNNClassifier",
    "EnsembleClassifier",
    "evaluate_classification",
    # Classifier training
    "train_classifier",
    # 2D slice extraction
    "SliceMode",
    "extract_2d_slice",
    "extract_multi_slices",
    # Training visualisation
    "TrainingVisualizer",
]

__version__ = "0.7.0"