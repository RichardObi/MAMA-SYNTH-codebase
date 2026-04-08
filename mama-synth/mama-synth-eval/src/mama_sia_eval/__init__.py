#  Copyright 2025 mama-sia-eval contributors
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
mama-sia-eval: Evaluation suite for the MAMA-SYNTH challenge.

Pre- to post-contrast translation evaluation for breast DCE-MRI,
covering image fidelity, distributional realism (FRD), downstream
segmentation, and downstream classification metrics.
"""

from mama_sia_eval.evaluation import (
    DatasetNormalizer,
    MamaSiaEval,
    normalize_intensity,
)
from mama_sia_eval.metrics import (
    compute_dice,
    compute_hd95,
    compute_mae,
    compute_mse,
    compute_ncc,
    compute_nmse,
    compute_psnr,
    compute_ssim,
)

__all__ = [
    "MamaSiaEval",
    "DatasetNormalizer",
    "normalize_intensity",
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
    # Classifier training
    "train_classifier",
]

__version__ = "0.3.0"