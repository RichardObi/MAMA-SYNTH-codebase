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

Reference: MAMA-SYNTH Challenge §Assessment Methods, Metric (1) Classification.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classifier protocol — any model that implements predict_proba is valid.
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
            task: One of 'tnbc' (TNBC vs non-TNBC) or 'luminal' (Luminal vs non-Luminal).
            model: Pre-trained scikit-learn model with predict_proba method.
                   If None, a default XGBoost model is created.
            model_path: Path to a pickled model file. Takes precedence over model.
        """
        valid_tasks = {"tnbc", "luminal"}
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
        except ImportError:
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
