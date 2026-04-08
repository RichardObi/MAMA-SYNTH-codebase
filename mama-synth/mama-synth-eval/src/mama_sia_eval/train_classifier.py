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
Classifier training pipeline for MAMA-MIA dataset.

Trains TNBC and luminal molecular subtype classifiers using radiomic
features extracted from post-contrast DCE-MRI images. Trained models are
directly compatible with the evaluation pipeline (evaluation.py).

The MAMA-MIA dataset (1,506 breast cancer DCE-MRI cases from 4 TCIA
collections) is hosted on Synapse: https://www.synapse.org/#!Synapse:syn60868042

Usage
-----
Single-command training::

    python -m mama_sia_eval.train_classifier \\
        --data-dir /path/to/mama_mia_dataset \\
        --output-dir /path/to/trained_models

Expected dataset layout::

    data-dir/
        clinical_and_imaging_info.xlsx       # Clinical metadata (Excel)
        images/                              # NIfTI images per patient
            {patient_id}/
                {patient_id}_0000.nii.gz     # Pre-contrast (phase 0)
                {patient_id}_0001.nii.gz     # 1st post-contrast (phase 1)
                ...
        segmentations/                       # Breast/tumor segmentation masks
            {patient_id}.nii.gz

Output::

    output-dir/
        tnbc_classifier.pkl                  # TNBC vs non-TNBC model
        luminal_classifier.pkl               # Luminal vs non-Luminal model

The output .pkl files are directly usable by the evaluation pipeline::

    mamasia-eval --clf-model-dir /path/to/trained_models ...

Reference: MAMA-SYNTH Challenge, Classification assessment.
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MAMA-MIA clinical data defaults
CLINICAL_EXCEL_FILENAME = "clinical_and_imaging_info.xlsx"
CLINICAL_SHEET_NAME = "dataset_info"
IMAGES_SUBDIR = "images"
SEGMENTATIONS_SUBDIR = "segmentations"

# tumor_subtype values in MAMA-MIA clinical data
TNBC_SUBTYPES = {"triple_negative"}
LUMINAL_SUBTYPES = {"luminal", "luminal_a", "luminal_b"}

# All valid (non-NaN) subtypes for filtering
ALL_SUBTYPES = {
    "triple_negative",
    "luminal",
    "luminal_a",
    "luminal_b",
    "her2_pure",
    "her2_enriched",
}

# Default MRI phase for feature extraction (1 = first post-contrast)
DEFAULT_PHASE = 1

# Default train/validation split ratio
DEFAULT_VAL_RATIO = 0.2

# Random seed for reproducibility
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

def _get_model_configs() -> list[dict[str, Any]]:
    """Return a list of model configurations to try during training.

    Each config is a dict with 'name', 'create_fn' (callable returning a model).
    The search tries XGBoost configs first, falling back to RandomForest if
    XGBoost is unavailable.
    """
    configs: list[dict[str, Any]] = []

    try:
        from xgboost import XGBClassifier

        configs.extend([
            {
                "name": "XGBoost(n=100, depth=3, lr=0.1)",
                "create_fn": lambda: XGBClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=DEFAULT_SEED,
                ),
            },
            {
                "name": "XGBoost(n=200, depth=5, lr=0.05)",
                "create_fn": lambda: XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=DEFAULT_SEED,
                ),
            },
            {
                "name": "XGBoost(n=300, depth=4, lr=0.05)",
                "create_fn": lambda: XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=DEFAULT_SEED,
                ),
            },
            {
                "name": "XGBoost(n=150, depth=6, lr=0.1)",
                "create_fn": lambda: XGBClassifier(
                    n_estimators=150, max_depth=6, learning_rate=0.1,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=DEFAULT_SEED,
                ),
            },
            {
                "name": "XGBoost(n=200, depth=3, lr=0.1, subsample=0.8)",
                "create_fn": lambda: XGBClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=DEFAULT_SEED,
                ),
            },
        ])
    except Exception:
        logger.info("XGBoost not available. Using RandomForest only.")

    from sklearn.ensemble import RandomForestClassifier

    configs.extend([
        {
            "name": "RandomForest(n=100, depth=10)",
            "create_fn": lambda: RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=DEFAULT_SEED,
            ),
        },
        {
            "name": "RandomForest(n=200, depth=15)",
            "create_fn": lambda: RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=DEFAULT_SEED,
            ),
        },
        {
            "name": "RandomForest(n=300, depth=None)",
            "create_fn": lambda: RandomForestClassifier(
                n_estimators=300, max_depth=None, random_state=DEFAULT_SEED,
            ),
        },
    ])

    return configs


# ---------------------------------------------------------------------------
# Clinical data loading
# ---------------------------------------------------------------------------

def load_clinical_data(
    data_dir: Path,
    clinical_data_path: Optional[Path] = None,
) -> "pd.DataFrame":
    """Load MAMA-MIA clinical data from Excel file.

    Args:
        data_dir: Root directory of the MAMA-MIA dataset.
        clinical_data_path: Override path to the clinical Excel file.

    Returns:
        DataFrame with clinical data.
    """
    import pandas as pd

    if clinical_data_path is None:
        clinical_data_path = data_dir / CLINICAL_EXCEL_FILENAME

    if not clinical_data_path.exists():
        raise FileNotFoundError(
            f"Clinical data file not found: {clinical_data_path}\n"
            f"Expected Excel file '{CLINICAL_EXCEL_FILENAME}' in {data_dir}"
        )

    logger.info(f"Loading clinical data from {clinical_data_path}")
    df = pd.read_excel(clinical_data_path, sheet_name=CLINICAL_SHEET_NAME)
    logger.info(
        f"Loaded {len(df)} patients, "
        f"columns: {', '.join(df.columns[:10].tolist())}..."
    )
    return df


def create_labels(
    clinical_df: "pd.DataFrame",
    task: str,
) -> tuple[list[str], NDArray[np.integer]]:
    """Create binary classification labels from clinical data.

    Filters patients with valid tumor_subtype and creates binary labels.

    Args:
        clinical_df: Clinical data DataFrame with 'patient_id' and
            'tumor_subtype' columns.
        task: Classification task ('tnbc' or 'luminal').

    Returns:
        Tuple of (patient_ids, labels) where labels is a binary array.
    """
    import pandas as pd

    valid_tasks = {"tnbc", "luminal"}
    if task not in valid_tasks:
        raise ValueError(f"task must be one of {valid_tasks}, got '{task}'")

    # Filter rows with valid tumor_subtype
    df = clinical_df[clinical_df["tumor_subtype"].notna()].copy()
    df = df[df["tumor_subtype"].isin(ALL_SUBTYPES)].copy()

    if len(df) == 0:
        raise ValueError(
            "No patients with valid tumor_subtype found in clinical data."
        )

    patient_ids = df["patient_id"].tolist()

    if task == "tnbc":
        labels = np.array(
            [1 if st in TNBC_SUBTYPES else 0 for st in df["tumor_subtype"]],
            dtype=np.int64,
        )
    else:  # luminal
        labels = np.array(
            [1 if st in LUMINAL_SUBTYPES else 0 for st in df["tumor_subtype"]],
            dtype=np.int64,
        )

    n_pos = int(np.sum(labels))
    n_neg = len(labels) - n_pos
    logger.info(
        f"Task '{task}': {len(labels)} patients with valid labels "
        f"(positive={n_pos}, negative={n_neg}, prevalence={n_pos/len(labels):.2%})"
    )

    return patient_ids, labels


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _load_nifti_as_array(filepath: Path) -> NDArray[np.floating]:
    """Load a NIfTI file and return as numpy array."""
    import SimpleITK as sitk

    if not filepath.exists():
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")

    sitk_image = sitk.ReadImage(str(filepath), sitk.sitkFloat32)
    return sitk.GetArrayFromImage(sitk_image).astype(np.float64)


def _load_mask_as_array(filepath: Path) -> NDArray[np.bool_]:
    """Load a NIfTI segmentation mask and return as boolean array."""
    import SimpleITK as sitk

    if not filepath.exists():
        raise FileNotFoundError(f"Mask file not found: {filepath}")

    sitk_image = sitk.ReadImage(str(filepath), sitk.sitkUInt8)
    return sitk.GetArrayFromImage(sitk_image).astype(bool)


def _get_image_path(
    images_dir: Path,
    patient_id: str,
    phase: int,
) -> Path:
    """Construct the path to a patient's MRI phase image."""
    return images_dir / patient_id / f"{patient_id}_{phase:04d}.nii.gz"


def _get_segmentation_path(
    segmentations_dir: Path,
    patient_id: str,
) -> Path:
    """Construct the path to a patient's segmentation mask."""
    return segmentations_dir / f"{patient_id}.nii.gz"


def extract_features_for_patients(
    patient_ids: list[str],
    data_dir: Path,
    images_dir: Optional[Path] = None,
    segmentations_dir: Optional[Path] = None,
    phase: int = DEFAULT_PHASE,
    cache_dir: Optional[Path] = None,
    n_workers: int = 1,
) -> tuple[NDArray[np.floating], list[str], list[int]]:
    """Extract radiomic features for a list of patients.

    Loads each patient's post-contrast MRI image and segmentation mask,
    then extracts radiomic features using pyradiomics (same pipeline as
    the evaluation code).

    Args:
        patient_ids: List of patient identifiers.
        data_dir: Root directory of the MAMA-MIA dataset.
        images_dir: Override path to images folder.
        segmentations_dir: Override path to segmentations folder.
        phase: MRI phase to use (default: 1 = first post-contrast).
        cache_dir: Directory for caching per-patient feature files.
        n_workers: Number of parallel workers (currently sequential).

    Returns:
        Tuple of:
            - feature_matrix: Shape (n_valid_patients, n_features).
            - valid_patient_ids: Patient IDs that were successfully processed.
            - valid_indices: Original indices of valid patients.
    """
    from mama_sia_eval.frd import (
        FRD_DEFAULT_BIN_WIDTH,
        FRD_FEATURE_CLASSES,
        extract_radiomic_features,
    )

    if images_dir is None:
        images_dir = data_dir / IMAGES_SUBDIR
    if segmentations_dir is None:
        segmentations_dir = data_dir / SEGMENTATIONS_SUBDIR

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Try to import tqdm for progress bars
    try:
        from tqdm import tqdm
        iterator = tqdm(
            enumerate(patient_ids),
            total=len(patient_ids),
            desc="Extracting features",
            unit="patient",
        )
    except ImportError:
        iterator = enumerate(patient_ids)

    features_list: list[NDArray[np.floating]] = []
    valid_patient_ids: list[str] = []
    valid_indices: list[int] = []
    n_cached = 0
    n_extracted = 0
    n_failed = 0

    for i, pid in iterator:
        # Check cache first
        if cache_dir is not None:
            cache_file = cache_dir / f"{pid}_phase{phase}.npy"
            if cache_file.exists():
                try:
                    feat = np.load(cache_file)
                    features_list.append(feat)
                    valid_patient_ids.append(pid)
                    valid_indices.append(i)
                    n_cached += 1
                    continue
                except Exception:
                    pass  # Re-extract if cache is corrupted

        # Load image
        image_path = _get_image_path(images_dir, pid, phase)
        seg_path = _get_segmentation_path(segmentations_dir, pid)

        try:
            image_array = _load_nifti_as_array(image_path)
        except FileNotFoundError:
            logger.warning(f"Image not found for {pid}: {image_path}, skipping.")
            n_failed += 1
            continue
        except Exception as e:
            logger.warning(f"Failed to load image for {pid}: {e}, skipping.")
            n_failed += 1
            continue

        # Load mask (optional - use full image if not available)
        mask_array: Optional[NDArray[np.bool_]] = None
        try:
            mask_array = _load_mask_as_array(seg_path)
        except FileNotFoundError:
            logger.debug(f"No segmentation for {pid}, using full image.")
        except Exception as e:
            logger.warning(
                f"Failed to load segmentation for {pid}: {e}, using full image."
            )

        # Extract radiomic features
        try:
            feat = extract_radiomic_features(
                image_array,
                mask=mask_array,
                feature_classes=FRD_FEATURE_CLASSES,
                bin_width=FRD_DEFAULT_BIN_WIDTH,
            )
        except Exception as e:
            logger.warning(f"Feature extraction failed for {pid}: {e}, skipping.")
            n_failed += 1
            continue

        # Validate feature vector
        if feat.size == 0 or np.all(feat == 0):
            logger.warning(f"Empty features for {pid}, skipping.")
            n_failed += 1
            continue

        # Cache features
        if cache_dir is not None:
            try:
                np.save(cache_dir / f"{pid}_phase{phase}.npy", feat)
            except Exception as e:
                logger.debug(f"Failed to cache features for {pid}: {e}")

        features_list.append(feat)
        valid_patient_ids.append(pid)
        valid_indices.append(i)
        n_extracted += 1

    logger.info(
        f"Feature extraction complete: {n_extracted} extracted, "
        f"{n_cached} from cache, {n_failed} failed "
        f"({len(valid_patient_ids)}/{len(patient_ids)} patients valid)"
    )

    if len(features_list) == 0:
        raise RuntimeError(
            "No features could be extracted. Check image paths and data."
        )

    # Stack into matrix, handling potential dimension mismatches
    n_features = features_list[0].shape[0]
    uniform = all(f.shape[0] == n_features for f in features_list)

    if uniform:
        feature_matrix = np.stack(features_list, axis=0)
    else:
        # Pad shorter feature vectors with zeros
        max_features = max(f.shape[0] for f in features_list)
        logger.warning(
            f"Feature dimension mismatch (min={min(f.shape[0] for f in features_list)}, "
            f"max={max_features}). Padding with zeros."
        )
        feature_matrix = np.zeros(
            (len(features_list), max_features), dtype=np.float64
        )
        for j, feat in enumerate(features_list):
            feature_matrix[j, : feat.shape[0]] = feat

    # Clean NaN/Inf values
    feature_matrix = np.nan_to_num(
        feature_matrix, nan=0.0, posinf=0.0, neginf=0.0
    )

    logger.info(
        f"Feature matrix shape: {feature_matrix.shape} "
        f"({feature_matrix.shape[0]} patients × {feature_matrix.shape[1]} features)"
    )

    return feature_matrix, valid_patient_ids, valid_indices


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X: NDArray[np.floating],
    y: NDArray[np.integer],
) -> dict[str, float]:
    """Evaluate a trained model on a dataset.

    Args:
        model: Trained sklearn-compatible classifier.
        X: Feature matrix.
        y: Ground truth labels.

    Returns:
        Dict with 'auroc' and 'balanced_accuracy'.
    """
    proba = model.predict_proba(X)
    y_score = proba[:, 1] if proba.ndim == 2 else proba
    y_pred = (y_score >= 0.5).astype(np.int64)

    n_unique = len(np.unique(y))
    auroc = float(roc_auc_score(y, y_score)) if n_unique >= 2 else float("nan")
    bal_acc = float(balanced_accuracy_score(y, y_pred))

    return {"auroc": auroc, "balanced_accuracy": bal_acc}


def train_single_model(
    config: dict[str, Any],
    X_train: NDArray[np.floating],
    y_train: NDArray[np.integer],
    X_val: NDArray[np.floating],
    y_val: NDArray[np.integer],
) -> tuple[Any, dict[str, float], dict[str, float]]:
    """Train a single model configuration and evaluate.

    Args:
        config: Model configuration with 'name' and 'create_fn'.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Tuple of (trained_model, train_metrics, val_metrics).
    """
    model = config["create_fn"]()

    logger.info(f"  Training: {config['name']}")
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info(f"  Training took {elapsed:.1f}s")

    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)

    logger.info(
        f"  Train AUROC={train_metrics['auroc']:.4f}, "
        f"Bal.Acc={train_metrics['balanced_accuracy']:.4f}"
    )
    logger.info(
        f"  Val   AUROC={val_metrics['auroc']:.4f}, "
        f"Bal.Acc={val_metrics['balanced_accuracy']:.4f}"
    )

    return model, train_metrics, val_metrics


def train_with_model_selection(
    X_train: NDArray[np.floating],
    y_train: NDArray[np.integer],
    X_val: NDArray[np.floating],
    y_val: NDArray[np.integer],
    task: str,
) -> tuple[Any, str, dict[str, float]]:
    """Train multiple model configurations and select the best.

    Tries all configurations from _get_model_configs() and selects the
    model with the highest validation AUROC.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        task: Task name for logging.

    Returns:
        Tuple of (best_model, best_config_name, best_val_metrics).
    """
    configs = _get_model_configs()
    logger.info(
        f"\n{'='*60}\n"
        f"Training '{task}' classifier — trying {len(configs)} configurations\n"
        f"{'='*60}"
    )
    logger.info(
        f"Train set: {X_train.shape[0]} samples "
        f"(pos={int(y_train.sum())}, neg={int((1-y_train).sum())})"
    )
    logger.info(
        f"Val   set: {X_val.shape[0]} samples "
        f"(pos={int(y_val.sum())}, neg={int((1-y_val).sum())})"
    )

    best_model = None
    best_name = ""
    best_val_auroc = -1.0
    best_val_metrics: dict[str, float] = {}
    results_log: list[dict[str, Any]] = []

    for config in configs:
        try:
            model, train_m, val_m = train_single_model(
                config, X_train, y_train, X_val, y_val
            )
        except Exception as e:
            logger.warning(f"  Failed: {config['name']}: {e}")
            continue

        results_log.append({
            "name": config["name"],
            "train_auroc": train_m["auroc"],
            "train_bal_acc": train_m["balanced_accuracy"],
            "val_auroc": val_m["auroc"],
            "val_bal_acc": val_m["balanced_accuracy"],
        })

        val_auroc = val_m["auroc"]
        if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model = model
            best_name = config["name"]
            best_val_metrics = val_m

    if best_model is None:
        raise RuntimeError(
            f"All model configurations failed for task '{task}'."
        )

    logger.info(f"\n{'─'*60}")
    logger.info(f"Best model for '{task}': {best_name}")
    logger.info(
        f"  Val AUROC={best_val_metrics['auroc']:.4f}, "
        f"Bal.Acc={best_val_metrics['balanced_accuracy']:.4f}"
    )
    logger.info(f"{'─'*60}\n")

    return best_model, best_name, best_val_metrics


# ---------------------------------------------------------------------------
# Cross-validation training (alternative mode)
# ---------------------------------------------------------------------------

def train_with_cross_validation(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    task: str,
    n_folds: int = 5,
    seed: int = DEFAULT_SEED,
) -> tuple[Any, str, dict[str, float]]:
    """Train with k-fold cross-validation for model selection.

    Uses stratified k-fold CV to evaluate model configs, then retrains
    the best config on the full dataset.

    Args:
        X: Full feature matrix.
        y: Full label vector.
        task: Task name for logging.
        n_folds: Number of CV folds.
        seed: Random seed.

    Returns:
        Tuple of (best_model, best_config_name, avg_cv_metrics).
    """
    configs = _get_model_configs()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    logger.info(
        f"\n{'='*60}\n"
        f"Cross-validation for '{task}' — {n_folds} folds, "
        f"{len(configs)} configurations\n"
        f"{'='*60}"
    )

    best_config = None
    best_avg_auroc = -1.0

    for config in configs:
        fold_aurocs = []
        fold_bal_accs = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            try:
                model = config["create_fn"]()
                model.fit(X_tr, y_tr)
                metrics = evaluate_model(model, X_va, y_va)
                fold_aurocs.append(metrics["auroc"])
                fold_bal_accs.append(metrics["balanced_accuracy"])
            except Exception as e:
                logger.warning(
                    f"  {config['name']} fold {fold} failed: {e}"
                )

        if fold_aurocs:
            avg_auroc = float(np.nanmean(fold_aurocs))
            avg_bal_acc = float(np.nanmean(fold_bal_accs))
            logger.info(
                f"  {config['name']}: "
                f"CV AUROC={avg_auroc:.4f}±{np.nanstd(fold_aurocs):.4f}, "
                f"Bal.Acc={avg_bal_acc:.4f}"
            )

            if avg_auroc > best_avg_auroc:
                best_avg_auroc = avg_auroc
                best_config = config

    if best_config is None:
        raise RuntimeError(
            f"All configurations failed in cross-validation for '{task}'."
        )

    # Retrain best config on full data
    logger.info(f"\nBest CV config: {best_config['name']} (AUROC={best_avg_auroc:.4f})")
    logger.info("Retraining on full dataset...")
    final_model = best_config["create_fn"]()
    final_model.fit(X, y)
    full_metrics = evaluate_model(final_model, X, y)
    logger.info(
        f"Full-data training: AUROC={full_metrics['auroc']:.4f}, "
        f"Bal.Acc={full_metrics['balanced_accuracy']:.4f}"
    )

    return final_model, best_config["name"], {"cv_auroc": best_avg_auroc}


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    task: str,
    output_dir: Path,
) -> Path:
    """Save a trained model in the format expected by the evaluation pipeline.

    The evaluation code (RadiomicsClassifier._load_model) loads models
    using pickle.load(). This function saves the raw sklearn/xgboost
    model object (not a RadiomicsClassifier wrapper).

    Args:
        model: Trained sklearn-compatible classifier.
        task: Task name ('tnbc' or 'luminal').
        output_dir: Directory to save the model.

    Returns:
        Path to the saved model file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{task}_classifier.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Saved {task} classifier to {model_path}")
    return model_path


def save_training_report(
    report: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Save a JSON training report with metadata and metrics.

    Args:
        report: Dictionary with training results.
        output_dir: Output directory.

    Returns:
        Path to the saved report file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "training_report.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Training report saved to {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Train TNBC and luminal molecular subtype classifiers on the "
            "MAMA-MIA dataset. Trained models are compatible with the "
            "mama-sia-eval evaluation pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m mama_sia_eval.train_classifier \\\n"
            "      --data-dir /datasets/mama_mia_dataset \\\n"
            "      --output-dir ./trained_models\n"
            "\n"
            "After training, the models can be used for evaluation:\n"
            "  mamasia-eval --clf-model-dir ./trained_models ...\n"
        ),
    )

    # Required arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Root directory of the MAMA-MIA dataset containing "
            "'clinical_and_imaging_info.xlsx', 'images/', 'segmentations/'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save trained model files (.pkl).",
    )

    # Optional data arguments
    parser.add_argument(
        "--clinical-data",
        type=Path,
        default=None,
        help=(
            "Override path to clinical Excel file. "
            f"Default: <data-dir>/{CLINICAL_EXCEL_FILENAME}"
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help=f"Override path to images folder. Default: <data-dir>/{IMAGES_SUBDIR}",
    )
    parser.add_argument(
        "--segmentations-dir",
        type=Path,
        default=None,
        help=(
            f"Override path to segmentations folder. "
            f"Default: <data-dir>/{SEGMENTATIONS_SUBDIR}"
        ),
    )

    # Training arguments
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["tnbc", "luminal"],
        default=["tnbc", "luminal"],
        help="Classification tasks to train. Default: tnbc luminal",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=DEFAULT_PHASE,
        help=f"MRI phase index for feature extraction. Default: {DEFAULT_PHASE} (1st post-contrast).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Validation set fraction. Default: {DEFAULT_VAL_RATIO}.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help=(
            "If >0, use k-fold cross-validation for model selection instead "
            "of a single train/val split. The final model is retrained on "
            "all data. Default: 0 (use train/val split)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility. Default: {DEFAULT_SEED}.",
    )

    # Performance arguments
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for caching extracted features per patient. "
            "Speeds up re-runs. Default: <output-dir>/feature_cache"
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers for feature extraction. Default: 1.",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG level) logging.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """Run the classifier training pipeline."""
    args = parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("MAMA-MIA Classifier Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data directory:   {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Tasks:            {args.tasks}")
    logger.info(f"MRI phase:        {args.phase}")
    logger.info(f"Val ratio:        {args.val_ratio}")
    logger.info(f"CV folds:         {args.cv_folds}")
    logger.info(f"Seed:             {args.seed}")

    # Set default cache dir
    if args.cache_dir is None:
        args.cache_dir = args.output_dir / "feature_cache"

    # 1. Load clinical data
    logger.info("\n--- Step 1: Loading clinical data ---")
    clinical_df = load_clinical_data(args.data_dir, args.clinical_data)

    # Training report to save at the end
    report: dict[str, Any] = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "phase": args.phase,
        "val_ratio": args.val_ratio,
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "total_patients": len(clinical_df),
        "tasks": {},
    }

    # 2. For each task, create labels → extract features → train → save
    for task in args.tasks:
        task_start = time.time()
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Task: {task}")
        logger.info(f"{'#'*60}")

        # 2a. Create labels
        logger.info("\n--- Step 2: Creating labels ---")
        patient_ids, labels = create_labels(clinical_df, task)

        # 2b. Extract features
        logger.info("\n--- Step 3: Extracting radiomic features ---")
        feature_matrix, valid_pids, valid_indices = extract_features_for_patients(
            patient_ids=patient_ids,
            data_dir=args.data_dir,
            images_dir=args.images_dir,
            segmentations_dir=args.segmentations_dir,
            phase=args.phase,
            cache_dir=args.cache_dir,
            n_workers=args.n_workers,
        )

        # Filter labels to match valid patients
        valid_labels = labels[valid_indices]

        if len(valid_labels) < 10:
            logger.error(
                f"Too few valid patients ({len(valid_labels)}) for task '{task}'. "
                "Check image paths and data integrity."
            )
            continue

        # 2c. Train classifier
        if args.cv_folds > 0:
            # Cross-validation mode
            logger.info(f"\n--- Step 4: Training with {args.cv_folds}-fold CV ---")
            best_model, best_name, best_metrics = train_with_cross_validation(
                X=feature_matrix,
                y=valid_labels,
                task=task,
                n_folds=args.cv_folds,
                seed=args.seed,
            )
        else:
            # Train/val split mode
            logger.info("\n--- Step 4: Training with train/val split ---")
            X_train, X_val, y_train, y_val = train_test_split(
                feature_matrix,
                valid_labels,
                test_size=args.val_ratio,
                random_state=args.seed,
                stratify=valid_labels,
            )

            best_model, best_name, best_metrics = train_with_model_selection(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                task=task,
            )

        # 2d. Save model
        logger.info("\n--- Step 5: Saving model ---")
        model_path = save_model(best_model, task, args.output_dir)

        task_elapsed = time.time() - task_start
        report["tasks"][task] = {
            "best_model": best_name,
            "val_metrics": best_metrics,
            "n_patients_total": len(patient_ids),
            "n_patients_valid": len(valid_labels),
            "n_features": feature_matrix.shape[1],
            "n_positive": int(valid_labels.sum()),
            "n_negative": int((1 - valid_labels).sum()),
            "model_path": str(model_path),
            "elapsed_seconds": round(task_elapsed, 1),
        }

        logger.info(f"Task '{task}' completed in {task_elapsed:.1f}s")

    # 3. Save training report
    save_training_report(report, args.output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Training pipeline complete!")
    logger.info(f"Models saved in: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
