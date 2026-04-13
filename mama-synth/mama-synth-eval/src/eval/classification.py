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
Classification evaluation for breast cancer molecular subtype prediction.

Evaluates whether synthesized DCE-MRI images preserve biologically meaningful
contrast patterns for downstream classification tasks:
  - TNBC vs non-TNBC (Triple-Negative Breast Cancer)
  - Luminal vs non-Luminal

Classification is performed using fixed, pre-trained evaluators applied to
the synthesized post-contrast images. For local evaluation, a radiomics-based
classifier is provided as a baseline.

Reference: MAMA-SYNTH Challenge Assessment Methods, Metric (1) Classification.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classifier protocol
# ---------------------------------------------------------------------------


class Classifier(Protocol):
    """Protocol for classification models compatible with evaluation."""

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return class probabilities for samples in X."""
        ...


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def compute_auroc(
    y_true: NDArray[np.integer],
    y_score: NDArray[np.floating],
) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (AUROC).

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_score: Predicted probability of the positive class.

    Returns:
        AUROC value in [0, 1].
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Need at least two classes to compute AUROC
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true; AUROC is undefined.")
        return float("nan")

    return float(roc_auc_score(y_true, y_score))


def compute_balanced_accuracy(
    y_true: NDArray[np.integer],
    y_pred: NDArray[np.integer],
) -> float:
    """Compute Balanced Accuracy (macro-averaged recall).

    Args:
        y_true: Binary ground truth labels.
        y_pred: Predicted binary labels.

    Returns:
        Balanced accuracy in [0, 1].
    """
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Radiomics-based classifier (baseline / local evaluation)
# ---------------------------------------------------------------------------


class RadiomicsClassifier:
    """Radiomics-based molecular subtype classifier.

    Extracts radiomic features from post-contrast DCE-MRI images and
    classifies molecular subtypes using a scikit-learn-compatible model.

    This serves as the baseline classification evaluator for the challenge.
    Organizers may replace the underlying model with a more advanced one.

    Attributes:
        task: Classification task ('tnbc' or 'luminal').
        model: Underlying scikit-learn classifier.
    """

    def __init__(
        self,
        task: str = "tnbc",
        model: Optional[Any] = None,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the classifier.

        Args:
            task: One of 'tnbc', 'luminal', 'contrast', or 'tumor_roi'.
            model: Pre-trained scikit-learn model with predict_proba method.
                   If None, a default XGBoost model is created.
            model_path: Path to a pickled model file. Takes precedence over model.
        """
        valid_tasks = {"tnbc", "luminal", "contrast", "tumor_roi"}
        if task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got '{task}'")
        self.task = task

        if model_path is not None:
            self.model = self._load_model(Path(model_path))
        elif model is not None:
            self.model = model
        else:
            self.model = self._create_default_model()

    @staticmethod
    def _load_model(path: Path) -> Any:
        """Load a pickled model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _create_default_model() -> Any:
        """Create a default XGBoost classifier.

        Falls back to RandomForest if XGBoost is not available.
        """
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=100,
                max_depth=5,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            )
        except Exception:
            from sklearn.ensemble import RandomForestClassifier

            logger.info("XGBoost not available, falling back to RandomForest.")
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
            )

    def train(
        self,
        features: NDArray[np.floating],
        labels: NDArray[np.integer],
    ) -> None:
        """Train the classifier on extracted radiomic features.

        Args:
            features: Feature matrix of shape (n_samples, n_features).
            labels: Binary labels (0/1) of shape (n_samples,).
        """
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(features, labels)
        logger.info(
            f"Trained {self.task} classifier on {features.shape[0]} samples "
            f"({features.shape[1]} features)"
        )

    def predict_proba(
        self,
        features: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Predict class probabilities.

        Args:
            features: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probabilities for the positive class, shape (n_samples,).
        """
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        proba = self.model.predict_proba(features)
        # Return probability of positive class
        return proba[:, 1] if proba.ndim == 2 else proba

    def predict(
        self,
        features: NDArray[np.floating],
        threshold: float = 0.5,
    ) -> NDArray[np.integer]:
        """Predict binary class labels.

        Args:
            features: Feature matrix of shape (n_samples, n_features).
            threshold: Decision threshold for positive class.

        Returns:
            Binary predictions shape (n_samples,).
        """
        proba = self.predict_proba(features)
        return (proba >= threshold).astype(np.int64)

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model to disk.

        Args:
            path: Output file path (.pkl).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {path}")


# ---------------------------------------------------------------------------
# CNN-based classifier (alternative to radiomics)
# ---------------------------------------------------------------------------


class CNNClassifier:
    """EfficientNet-based molecular subtype classifier.

    Classifies molecular subtypes directly from raw post-contrast DCE-MRI
    images (or 2D slices), without explicit radiomic feature extraction.

    The model is loaded from a .pt checkpoint saved by
    eval.train_cnn_classifier.train_cnn.

    Attributes:
        task: Classification task ('tnbc' or 'luminal').
        model: PyTorch CNN model.
        image_size: Expected input spatial resolution.
    """

    def __init__(
        self,
        task: str = "tnbc",
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        valid_tasks = {"tnbc", "luminal", "contrast", "tumor_roi"}
        if task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got '{task}'")
        self.task = task

        if model_path is None:
            raise ValueError("CNNClassifier requires a model_path.")

        self._load_model(Path(model_path))

    def _load_model(self, path: Path) -> None:
        """Load a CNN checkpoint from disk."""
        try:
            from eval.train_cnn_classifier import load_cnn_model
        except ImportError as e:
            raise ImportError(
                "CNN classifier requires PyTorch, torchvision, and timm. "
                "Install with: pip install 'mama-synth-eval[cnn]'. "
                f"Error: {e}"
            )

        self.model, config = load_cnn_model(path, device="cpu")
        self.image_size = config.get("image_size", 224)
        self.model_name = config.get("model_name", "efficientnet_b0")
        self.use_mask_channel = config.get("use_mask_channel", False)
        self.in_chans = config.get("in_chans", 3)

    def predict_proba_from_images(
        self,
        images: list[NDArray[np.floating]],
        masks: Optional[list[Optional[NDArray[np.bool_]]]] = None,
    ) -> NDArray[np.floating]:
        """Predict class probabilities from raw 2D or 3D images.

        For 3D volumes, the slice with the largest tumour cross-section
        is automatically extracted (if a mask is provided), or the middle
        slice is used as a fallback.

        Args:
            images: List of 2D/3D image arrays.
            masks: Optional list of corresponding binary masks.

        Returns:
            1-D array of positive-class probabilities, shape (N,).
        """
        import torch
        import torch.nn.functional as F

        self.model.eval()
        device = next(self.model.parameters()).device
        probabilities: list[float] = []

        for idx, img in enumerate(images):
            mask = masks[idx] if masks is not None and idx < len(masks) else None

            slice_2d = self._extract_best_slice(img, mask)
            tensor = self._preprocess_slice(slice_2d)

            if self.use_mask_channel:
                mask_2d = self._extract_best_mask_slice(img, mask)
                mask_t = self._preprocess_mask(mask_2d)
                tensor = torch.cat([tensor, mask_t], dim=0)

            tensor = tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                logit = self.model(tensor).squeeze()
                prob = torch.sigmoid(logit).item()

            probabilities.append(prob)

        return np.array(probabilities, dtype=np.float64)

    def _extract_best_slice(
        self, image: NDArray, mask: Optional[NDArray] = None,
    ) -> NDArray:
        """Extract a single representative 2D slice."""
        if image.ndim == 2:
            return image
        if image.ndim == 3:
            if mask is not None and mask.ndim == 3 and np.any(mask):
                from eval.slice_extraction import find_max_tumor_slice

                idx = find_max_tumor_slice(mask.astype(bool))
                return image[idx]
            return image[image.shape[0] // 2]
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D.")

    def _extract_best_mask_slice(
        self, image: NDArray, mask: Optional[NDArray] = None,
    ) -> NDArray:
        """Extract the 2D mask slice matching the chosen image slice."""
        if mask is None:
            if image.ndim == 2:
                return np.zeros_like(image, dtype=np.float32)
            elif image.ndim == 3:
                return np.zeros(image.shape[1:], dtype=np.float32)
            return np.zeros((1, 1), dtype=np.float32)
        if mask.ndim == 2:
            return mask.astype(np.float32)
        if mask.ndim == 3:
            if np.any(mask):
                from eval.slice_extraction import find_max_tumor_slice

                idx = find_max_tumor_slice(mask.astype(bool))
                return mask[idx].astype(np.float32)
            return mask[mask.shape[0] // 2].astype(np.float32)
        return np.zeros((1, 1), dtype=np.float32)

    def _preprocess_mask(self, mask_2d: NDArray) -> "torch.Tensor":
        """Convert a 2D mask to a resized (1, H, W) tensor."""
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(mask_2d.astype(np.float32)).unsqueeze(0)
        t = F.interpolate(
            t.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0)
        return t

    def _preprocess_slice(self, slice_2d: NDArray) -> "torch.Tensor":
        """Normalise and convert a 2D slice to a model-ready tensor."""
        import torch
        import torch.nn.functional as F

        img = slice_2d.astype(np.float32)
        vmin, vmax = float(img.min()), float(img.max())
        if vmax - vmin > 1e-8:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img)

        t = torch.from_numpy(img).unsqueeze(0)
        t = F.interpolate(
            t.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        t = t.repeat(3, 1, 1)
        return t


# ---------------------------------------------------------------------------
# Ensemble classifier (combines multiple models)
# ---------------------------------------------------------------------------


class EnsembleClassifier:
    """Ensemble classifier that averages probabilities from multiple models.

    Supports mixing RadiomicsClassifier and CNNClassifier instances.
    Each model produces a 1-D array of positive-class probabilities;
    the ensemble prediction is their arithmetic mean.
    """

    def __init__(self, task: str) -> None:
        valid_tasks = {"tnbc", "luminal", "contrast", "tumor_roi"}
        if task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got '{task}'")
        self.task = task
        self._radiomics_models: list[RadiomicsClassifier] = []
        self._cnn_models: list[CNNClassifier] = []

    def add_radiomics_model(self, clf: RadiomicsClassifier) -> "EnsembleClassifier":
        self._radiomics_models.append(clf)
        return self

    def add_cnn_model(self, clf: CNNClassifier) -> "EnsembleClassifier":
        self._cnn_models.append(clf)
        return self

    @property
    def n_models(self) -> int:
        return len(self._radiomics_models) + len(self._cnn_models)

    @property
    def has_radiomics(self) -> bool:
        return len(self._radiomics_models) > 0

    @property
    def has_cnn(self) -> bool:
        return len(self._cnn_models) > 0

    def description(self) -> str:
        parts: list[str] = []
        if self._radiomics_models:
            parts.append(f"{len(self._radiomics_models)} radiomics")
        if self._cnn_models:
            parts.append(f"{len(self._cnn_models)} CNN")
        return f"Ensemble({', '.join(parts)})"

    def predict_proba(
        self,
        features: Optional[NDArray[np.floating]] = None,
        images: Optional[list[NDArray[np.floating]]] = None,
        masks: Optional[list[Optional[NDArray[np.bool_]]]] = None,
    ) -> NDArray[np.floating]:
        """Average predicted probabilities across all ensemble members."""
        all_probs: list[NDArray[np.floating]] = []

        for clf in self._radiomics_models:
            if features is None:
                raise ValueError(
                    "Ensemble contains radiomics models but no features provided."
                )
            all_probs.append(np.asarray(clf.predict_proba(features)))

        for clf in self._cnn_models:
            if images is None:
                raise ValueError(
                    "Ensemble contains CNN models but no images provided."
                )
            all_probs.append(
                np.asarray(clf.predict_proba_from_images(images, masks))
            )

        if not all_probs:
            raise ValueError("Ensemble has no models.")

        stacked = np.stack(all_probs, axis=0)
        return np.mean(stacked, axis=0)

    @staticmethod
    def discover_models(task: str, model_dir: Path) -> "EnsembleClassifier":
        """Auto-discover all model files for *task* in *model_dir*."""
        ensemble = EnsembleClassifier(task=task)
        model_dir = Path(model_dir)

        for pkl_path in sorted(model_dir.glob(f"{task}_classifier*.pkl")):
            try:
                clf = RadiomicsClassifier(task=task, model_path=pkl_path)
                ensemble.add_radiomics_model(clf)
                logger.info(f"Ensemble: loaded radiomics model {pkl_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {pkl_path.name}: {e}")

        for pt_path in sorted(model_dir.glob(f"{task}_classifier*.pt")):
            try:
                clf = CNNClassifier(task=task, model_path=pt_path)
                ensemble.add_cnn_model(clf)
                logger.info(f"Ensemble: loaded CNN model {pt_path.name}")
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to load {pt_path.name}: {e}")

        return ensemble


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------


def evaluate_classification(
    y_true: NDArray[np.integer],
    y_score: NDArray[np.floating],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate classification performance.

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted probabilities for the positive class.
        threshold: Decision threshold for balanced accuracy computation.

    Returns:
        Dictionary with 'auroc' and 'balanced_accuracy'.
    """
    y_pred = (np.asarray(y_score) >= threshold).astype(np.int64)

    return {
        "auroc": compute_auroc(y_true, y_score),
        "balanced_accuracy": compute_balanced_accuracy(y_true, y_pred),
    }
