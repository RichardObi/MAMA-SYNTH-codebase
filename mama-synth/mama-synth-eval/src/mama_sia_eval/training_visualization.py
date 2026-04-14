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
Training-specific visualisation for the MAMA-SYNTH classifier pipeline.

Generates publication-quality plots for classifier evaluation:
  - Confusion matrix heatmap
  - ROC curve with AUC
  - Precision–Recall curve with AP
  - Feature importance bar chart
  - Classification report summary table
  - Combined dashboard figure

Usage::

    from mama_sia_eval.training_visualization import TrainingVisualizer

    viz = TrainingVisualizer(output_dir="./training_reports")
    viz.confusion_matrix(y_true, y_pred, task="tnbc")
    viz.roc_curve(y_true, y_score, task="tnbc")
    viz.precision_recall_curve(y_true, y_score, task="tnbc")
    viz.feature_importance(model, feature_names, task="tnbc")
    viz.generate_dashboard(y_true, y_pred, y_score, model, task="tnbc")
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = ["TrainingVisualizer"]

# Task display labels
_TASK_LABELS = {
    "tnbc": "TNBC vs non-TNBC",
    "luminal": "Luminal vs non-Luminal",
}


def _task_label(task: str) -> str:
    return _TASK_LABELS.get(task, task)


class TrainingVisualizer:
    """Generate visualisation artefacts for classifier training results.

    All methods are safe to call even when matplotlib is unavailable —
    they will log a warning and return empty lists.

    Args:
        output_dir: Directory to save generated files.
    """

    def __init__(self, output_dir: str | Path = "training_reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_plt():
        """Import and configure matplotlib; returns plt or None."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            return plt
        except ImportError:
            logger.warning(
                "matplotlib not installed — skipping visualisation."
            )
            return None

    # ------------------------------------------------------------------
    # 1. Confusion Matrix
    # ------------------------------------------------------------------

    def confusion_matrix(
        self,
        y_true: NDArray[np.integer],
        y_pred: NDArray[np.integer],
        task: str = "classifier",
        class_names: Optional[list[str]] = None,
        normalize: bool = False,
    ) -> list[Path]:
        """Generate and save a confusion matrix heatmap.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            task: Task identifier (used in filename and title).
            class_names: Display labels for classes.
                Defaults to ['Negative', 'Positive'].
            normalize: If True, normalise rows to show proportions.

        Returns:
            List of saved file paths.
        """
        plt = self._get_plt()
        if plt is None:
            return []

        from sklearn.metrics import confusion_matrix as sk_cm

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if class_names is None:
            if task == "tnbc":
                class_names = ["non-TNBC", "TNBC"]
            elif task == "luminal":
                class_names = ["non-Luminal", "Luminal"]
            else:
                class_names = ["Negative", "Positive"]

        cm = sk_cm(y_true, y_pred)

        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_plot = np.where(row_sums > 0, cm / row_sums, 0.0)
            fmt = ".2f"
            title_suffix = " (Normalised)"
        else:
            cm_plot = cm.astype(float)
            fmt = "d"
            title_suffix = ""

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
        ax.set_title(f"Confusion Matrix — {_task_label(task)}{title_suffix}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Text annotations
        thresh = cm_plot.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j] if not normalize else cm_plot[i, j]
                text = format(val, fmt)
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color="white" if cm_plot[i, j] > thresh else "black",
                    fontsize=14,
                )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)
        fig.tight_layout()

        paths: list[Path] = []

        # Save raw confusion matrix
        path = self.output_dir / f"confusion_matrix_{task}.png"
        fig.savefig(path, dpi=150)
        paths.append(path)
        plt.close(fig)

        # Also save a normalised version if not already normalised
        if not normalize:
            norm_paths = self.confusion_matrix(
                y_true, y_pred, task=task,
                class_names=class_names, normalize=True,
            )
            paths.extend(norm_paths)

        # Save confusion matrix as JSON for programmatic access
        cm_dict = {
            "task": task,
            "class_names": class_names,
            "matrix": cm.tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
        json_path = self.output_dir / f"confusion_matrix_{task}.json"
        with open(json_path, "w") as f:
            json.dump(cm_dict, f, indent=2)
        paths.append(json_path)

        logger.info(f"Confusion matrix saved: {path}")
        return paths

    # ------------------------------------------------------------------
    # 2. ROC Curve
    # ------------------------------------------------------------------

    def roc_curve(
        self,
        y_true: NDArray[np.integer],
        y_score: NDArray[np.floating],
        task: str = "classifier",
    ) -> list[Path]:
        """Generate and save an ROC curve with AUC.

        Args:
            y_true: Ground truth labels.
            y_score: Predicted probabilities for the positive class.
            task: Task identifier.

        Returns:
            List of saved file paths.
        """
        plt = self._get_plt()
        if plt is None:
            return []

        from sklearn.metrics import auc, roc_curve as sk_roc

        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class in y_true; skipping ROC curve.")
            return []

        fpr, tpr, _ = sk_roc(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(
            fpr, tpr,
            color="#4C72B0", lw=2,
            label=f"ROC (AUC = {roc_auc:.3f})",
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {_task_label(task)}")
        ax.legend(loc="lower right")
        ax.set_aspect("equal")
        fig.tight_layout()

        path = self.output_dir / f"roc_curve_{task}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

        logger.info(f"ROC curve saved: {path} (AUC={roc_auc:.3f})")
        return [path]

    # ------------------------------------------------------------------
    # 3. Precision–Recall Curve
    # ------------------------------------------------------------------

    def precision_recall_curve(
        self,
        y_true: NDArray[np.integer],
        y_score: NDArray[np.floating],
        task: str = "classifier",
    ) -> list[Path]:
        """Generate and save a Precision–Recall curve with AP.

        Args:
            y_true: Ground truth labels.
            y_score: Predicted probabilities for the positive class.
            task: Task identifier.

        Returns:
            List of saved file paths.
        """
        plt = self._get_plt()
        if plt is None:
            return []

        from sklearn.metrics import (
            average_precision_score,
            precision_recall_curve as sk_pr,
        )

        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        if len(np.unique(y_true)) < 2:
            logger.warning("Only one class in y_true; skipping PR curve.")
            return []

        precision, recall, _ = sk_pr(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(
            recall, precision,
            color="#DD8452", lw=2,
            label=f"PR (AP = {ap:.3f})",
        )
        prevalence = float(np.mean(y_true))
        ax.axhline(
            y=prevalence, color="k", linestyle="--", lw=1, alpha=0.5,
            label=f"Baseline ({prevalence:.2f})",
        )
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision–Recall Curve — {_task_label(task)}")
        ax.legend(loc="upper right")
        fig.tight_layout()

        path = self.output_dir / f"pr_curve_{task}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

        logger.info(f"PR curve saved: {path} (AP={ap:.3f})")
        return [path]

    # ------------------------------------------------------------------
    # 4. Feature Importance
    # ------------------------------------------------------------------

    def feature_importance(
        self,
        model: Any,
        task: str = "classifier",
        feature_names: Optional[list[str]] = None,
        top_k: int = 20,
    ) -> list[Path]:
        """Plot feature importance for tree-based models.

        Supports XGBoost (``feature_importances_``), RandomForest, and
        any model with a ``feature_importances_`` attribute.

        Args:
            model: Trained model with ``feature_importances_`` attribute.
            task: Task identifier.
            feature_names: Optional feature names.
            top_k: Number of top features to show.

        Returns:
            List of saved file paths.
        """
        plt = self._get_plt()
        if plt is None:
            return []

        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            logger.info(
                f"Model has no feature_importances_ — skipping importance plot."
            )
            return []

        importances = np.asarray(importances)
        n_features = len(importances)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Sort by importance and take top-k
        idx = np.argsort(importances)[::-1][:top_k]
        top_names = [feature_names[i] for i in idx]
        top_values = importances[idx]

        fig, ax = plt.subplots(figsize=(7, max(4, top_k * 0.3)))
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_values, color="#55A868", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Top-{top_k} Feature Importance — {_task_label(task)}")
        fig.tight_layout()

        path = self.output_dir / f"feature_importance_{task}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

        logger.info(f"Feature importance plot saved: {path}")
        return [path]

    # ------------------------------------------------------------------
    # 5. Classification Report
    # ------------------------------------------------------------------

    def classification_report(
        self,
        y_true: NDArray[np.integer],
        y_pred: NDArray[np.integer],
        y_score: NDArray[np.floating],
        task: str = "classifier",
        class_names: Optional[list[str]] = None,
    ) -> list[Path]:
        """Save a text and JSON classification report.

        Includes precision, recall, F1-score, support, AUROC, balanced
        accuracy, and Matthews Correlation Coefficient (MCC).

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_score: Predicted probabilities.
            task: Task identifier.
            class_names: Display labels for classes.

        Returns:
            List of saved file paths.
        """
        from sklearn.metrics import (
            balanced_accuracy_score,
            classification_report as sk_report,
            matthews_corrcoef,
            roc_auc_score,
        )

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_score = np.asarray(y_score)

        if class_names is None:
            if task == "tnbc":
                class_names = ["non-TNBC", "TNBC"]
            elif task == "luminal":
                class_names = ["non-Luminal", "Luminal"]
            else:
                class_names = ["Negative", "Positive"]

        # Compute aggregate metrics
        n_unique = len(np.unique(y_true))
        auroc = float(roc_auc_score(y_true, y_score)) if n_unique >= 2 else float("nan")
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        mcc = float(matthews_corrcoef(y_true, y_pred))

        # Sklearn classification report (text)
        text_report = sk_report(
            y_true, y_pred,
            target_names=class_names,
            digits=4,
        )

        header = (
            f"Classification Report — {_task_label(task)}\n"
            f"{'=' * 55}\n"
        )
        footer = (
            f"\n{'─' * 55}\n"
            f"AUROC:                {auroc:.4f}\n"
            f"Balanced Accuracy:    {bal_acc:.4f}\n"
            f"Matthews Corr Coef:   {mcc:.4f}\n"
            f"Total Samples:        {len(y_true)}\n"
            f"Positive Samples:     {int(y_true.sum())}\n"
            f"Negative Samples:     {int((1 - y_true).sum())}\n"
        )

        full_text = header + text_report + footer

        paths: list[Path] = []

        # Save text report
        txt_path = self.output_dir / f"classification_report_{task}.txt"
        txt_path.write_text(full_text)
        paths.append(txt_path)

        # Save JSON report
        from sklearn.metrics import classification_report as sk_report_dict
        report_dict = sk_report_dict(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
        )
        report_dict["auroc"] = auroc
        report_dict["balanced_accuracy"] = bal_acc
        report_dict["mcc"] = mcc

        json_path = self.output_dir / f"classification_report_{task}.json"
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        paths.append(json_path)

        logger.info(f"Classification report saved: {txt_path}")
        return paths

    # ------------------------------------------------------------------
    # 6. Combined Dashboard
    # ------------------------------------------------------------------

    def generate_dashboard(
        self,
        y_true: NDArray[np.integer],
        y_pred: NDArray[np.integer],
        y_score: NDArray[np.floating],
        model: Any = None,
        task: str = "classifier",
        class_names: Optional[list[str]] = None,
        feature_names: Optional[list[str]] = None,
        dataset_label: str = "Validation",
    ) -> list[Path]:
        """Generate a combined 2×2 dashboard figure.

        Combines confusion matrix, ROC curve, PR curve, and either
        feature importance or a metrics summary in a single figure.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_score: Predicted probabilities.
            model: Trained model (for feature importance).
            task: Task identifier.
            class_names: Display labels.
            feature_names: Feature names for importance plot.
            dataset_label: Label for the dataset (e.g., "Validation", "Test").

        Returns:
            List of saved file paths.
        """
        plt = self._get_plt()
        if plt is None:
            return []

        from sklearn.metrics import (
            auc,
            average_precision_score,
            balanced_accuracy_score,
            confusion_matrix as sk_cm,
            precision_recall_curve as sk_pr,
            roc_auc_score,
            roc_curve as sk_roc,
        )

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_score = np.asarray(y_score)

        if class_names is None:
            if task == "tnbc":
                class_names = ["non-TNBC", "TNBC"]
            elif task == "luminal":
                class_names = ["non-Luminal", "Luminal"]
            else:
                class_names = ["Negative", "Positive"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            f"{_task_label(task)} — {dataset_label} Set Results",
            fontsize=14, fontweight="bold",
        )

        # --- Subplot 1: Confusion Matrix ---
        ax = axes[0, 0]
        cm = sk_cm(y_true, y_pred)
        cm_norm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        thresh = cm_norm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i,
                    f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=11,
                )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(class_names)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(class_names)
        ax.set_title("Confusion Matrix")

        # --- Subplot 2: ROC Curve ---
        ax = axes[0, 1]
        n_unique = len(np.unique(y_true))
        if n_unique >= 2:
            fpr, tpr, _ = sk_roc(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
            ax.legend(loc="lower right")
        else:
            ax.text(0.5, 0.5, "Insufficient classes\nfor ROC",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.set_aspect("equal")

        # --- Subplot 3: PR Curve ---
        ax = axes[1, 0]
        if n_unique >= 2:
            precision, recall, _ = sk_pr(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            ax.plot(recall, precision, color="#DD8452", lw=2, label=f"AP = {ap:.3f}")
            prev = float(np.mean(y_true))
            ax.axhline(y=prev, color="k", ls="--", lw=1, alpha=0.5,
                        label=f"Baseline ({prev:.2f})")
            ax.legend(loc="upper right")
        else:
            ax.text(0.5, 0.5, "Insufficient classes\nfor PR",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curve")

        # --- Subplot 4: Feature importance or metrics summary ---
        ax = axes[1, 1]
        importances = getattr(model, "feature_importances_", None) if model else None
        if importances is not None:
            importances = np.asarray(importances)
            if feature_names is None:
                feature_names = [f"f_{i}" for i in range(len(importances))]
            top_k = min(15, len(importances))
            idx = np.argsort(importances)[::-1][:top_k]
            names = [feature_names[i] for i in idx]
            vals = importances[idx]

            y_pos = np.arange(len(names))
            ax.barh(y_pos, vals, color="#55A868", height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            ax.set_title(f"Top-{top_k} Features")
        else:
            # Show a metrics summary table instead
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            n_pos = int(y_true.sum())
            n_neg = len(y_true) - n_pos
            lines = [
                f"Samples:    {len(y_true)}",
                f"Positive:   {n_pos} ({n_pos/len(y_true):.1%})",
                f"Negative:   {n_neg} ({n_neg/len(y_true):.1%})",
                "",
                f"Bal. Acc:   {bal_acc:.4f}",
            ]
            if n_unique >= 2:
                lines.append(f"AUROC:      {roc_auc_score(y_true, y_score):.4f}")
            ax.text(
                0.1, 0.5, "\n".join(lines),
                transform=ax.transAxes, fontsize=12,
                verticalalignment="center", fontfamily="monospace",
            )
            ax.axis("off")
            ax.set_title("Metrics Summary")

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        path = self.output_dir / f"dashboard_{task}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

        logger.info(f"Dashboard saved: {path}")
        return [path]

    # ------------------------------------------------------------------
    # 7. Generate all visualisations for a task
    # ------------------------------------------------------------------

    def generate_all(
        self,
        y_true: NDArray[np.integer],
        y_pred: NDArray[np.integer],
        y_score: NDArray[np.floating],
        model: Any = None,
        task: str = "classifier",
        class_names: Optional[list[str]] = None,
        feature_names: Optional[list[str]] = None,
        dataset_label: str = "Validation",
    ) -> list[Path]:
        """Generate all available visualisations for a task.

        Calls confusion_matrix, roc_curve, precision_recall_curve,
        feature_importance, classification_report, and generate_dashboard.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_score: Predicted probabilities.
            model: Trained model.
            task: Task identifier.
            class_names: Display labels.
            feature_names: Feature names.
            dataset_label: Label for the dataset.

        Returns:
            Aggregated list of all saved file paths.
        """
        paths: list[Path] = []
        paths.extend(self.confusion_matrix(y_true, y_pred, task=task, class_names=class_names))
        paths.extend(self.roc_curve(y_true, y_score, task=task))
        paths.extend(self.precision_recall_curve(y_true, y_score, task=task))
        paths.extend(self.classification_report(y_true, y_pred, y_score, task=task, class_names=class_names))

        if model is not None:
            paths.extend(self.feature_importance(model, task=task, feature_names=feature_names))

        paths.extend(self.generate_dashboard(
            y_true, y_pred, y_score,
            model=model, task=task,
            class_names=class_names,
            feature_names=feature_names,
            dataset_label=dataset_label,
        ))

        logger.info(
            f"Generated {len(paths)} visualisation artefacts for task '{task}' "
            f"in {self.output_dir}"
        )
        return paths
