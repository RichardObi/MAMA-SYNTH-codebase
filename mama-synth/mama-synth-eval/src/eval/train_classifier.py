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
Classifier training pipeline for MAMA-MIA dataset.

Trains TNBC and luminal molecular subtype classifiers using radiomic
features extracted from post-contrast DCE-MRI images. Trained models are
directly compatible with the evaluation pipeline (evaluation.py).

The MAMA-MIA dataset (1,506 breast cancer DCE-MRI cases from 4 TCIA
collections) is hosted on Synapse: https://www.synapse.org/#!Synapse:syn60868042

Usage
-----
Single-command training::

    python -m eval.train_classifier \\
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

    eval --clf-model-dir /path/to/trained_models ...

Reference: MAMA-SYNTH Challenge, Classification assessment.
"""

import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eval.slice_extraction import (
    SliceMode,
    extract_2d_slice,
    extract_all_tumor_slices,
    extract_multi_slices,
)

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

# Default 2D slice extraction mode (None means use full 3D volume)
DEFAULT_SLICE_MODE: Optional[str] = None

# Number of slices for multi-slice mode
DEFAULT_N_SLICES = 5

# MAMA-MIA dataset split column used to separate train/test patients.
# The clinical Excel may use 'dataset_split', 'split', or 'dataset'.
# We try each in order.
SPLIT_COLUMN_CANDIDATES = ["dataset_split", "split", "dataset"]

# Default run-directory prefix inside --output-dir
RUN_DIR_PREFIX = "run"

# Auto-detected filenames for the CSV-based train/test split shipped with
# the MAMA-MIA dataset.  The file has two ragged columns: ``train_split``
# and ``test_split``, each cell is a patient ID.
SPLIT_CSV_CANDIDATES = [
    "train_test_splits.csv",
    "train_test_split.csv",
]

# Values that identify test-set patients in the split column.
# Matching is done *case-insensitively* — only lowercase forms are needed here.
TEST_SPLIT_VALUES = {"test", "testing"}
TRAIN_SPLIT_VALUES = {"train", "training"}


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

def _make_pipeline(clf: Any) -> Pipeline:
    """Wrap a classifier in a preprocessing pipeline.

    The pipeline applies:
    1. StandardScaler — zero-mean, unit-variance normalisation.
    2. VarianceThreshold — remove near-constant features.
    3. The classifier itself.

    Because the entire Pipeline is serialised with ``pickle``, the
    saved ``.pkl`` model is self-contained: it will scale and select
    features automatically at inference time.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("var_thresh", VarianceThreshold(threshold=0.0)),
        ("clf", clf),
    ])


def _get_model_configs(
    scale_pos_weight: float = 1.0,
    model_filter: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return a list of model configurations to try during training.

    Each config is a dict with 'name' and 'create_fn' (callable returning
    a sklearn :class:`Pipeline` with preprocessing + classifier).

    Args:
        scale_pos_weight: Ratio *n_negative / n_positive* for the current
            task.  Passed directly to XGBoost's ``scale_pos_weight`` and
            used to inform other class-imbalance parameters.
        model_filter: If not None/\"all\", only include configs whose
            family matches this value.  Accepted values:
            ``\"xgboost\"``, ``\"random_forest\"``, ``\"logistic_regression\"``,
            ``\"svm\"``, ``\"all\"`` (or ``None``).
    """
    configs: list[dict[str, Any]] = []

    # --- XGBoost (if available) -------------------------------------------
    try:
        from xgboost import XGBClassifier

        _spw = scale_pos_weight  # capture for lambdas

        configs.extend([
            {
                "name": "XGB(n=200, d=3, lr=0.1, bal)",
                "family": "xgboost",
                "create_fn": lambda: _make_pipeline(XGBClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.1,
                    scale_pos_weight=_spw,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=DEFAULT_SEED,
                )),
            },
            {
                "name": "XGB(n=300, d=4, lr=0.05, bal)",
                "family": "xgboost",
                "create_fn": lambda: _make_pipeline(XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    scale_pos_weight=_spw,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=DEFAULT_SEED,
                )),
            },
            {
                "name": "XGB(n=200, d=5, lr=0.05, sub=0.8, bal)",
                "family": "xgboost",
                "create_fn": lambda: _make_pipeline(XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=_spw,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=DEFAULT_SEED,
                )),
            },
            {
                "name": "XGB(n=150, d=6, lr=0.1, sub=0.8, bal)",
                "family": "xgboost",
                "create_fn": lambda: _make_pipeline(XGBClassifier(
                    n_estimators=150, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=_spw,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=DEFAULT_SEED,
                )),
            },
        ])
    except Exception:
        logger.info("XGBoost not available.")

    # --- Random Forest ----------------------------------------------------
    from sklearn.ensemble import RandomForestClassifier

    configs.extend([
        {
            "name": "RF(n=200, d=15, bal)",
            "family": "random_forest",
            "create_fn": lambda: _make_pipeline(RandomForestClassifier(
                n_estimators=200, max_depth=15,
                class_weight="balanced",
                random_state=DEFAULT_SEED,
            )),
        },
        {
            "name": "RF(n=300, d=None, bal)",
            "family": "random_forest",
            "create_fn": lambda: _make_pipeline(RandomForestClassifier(
                n_estimators=300, max_depth=None,
                class_weight="balanced",
                random_state=DEFAULT_SEED,
            )),
        },
    ])

    # --- Logistic Regression (often strong on small N, many features) -----
    from sklearn.linear_model import LogisticRegression

    for C in (0.01, 0.1, 1.0, 10.0):
        configs.append({
            "name": f"LogReg(C={C}, bal)",
            "family": "logistic_regression",
            "create_fn": (
                lambda _C=C: _make_pipeline(LogisticRegression(
                    C=_C, class_weight="balanced", max_iter=2000,
                    solver="lbfgs", random_state=DEFAULT_SEED,
                ))
            ),
        })

    # --- SVM with RBF kernel (good for moderate N) ------------------------
    from sklearn.svm import SVC

    for C in (0.1, 1.0, 10.0):
        configs.append({
            "name": f"SVM-RBF(C={C}, bal)",
            "family": "svm",
            "create_fn": (
                lambda _C=C: _make_pipeline(SVC(
                    C=_C, kernel="rbf", probability=True,
                    class_weight="balanced", random_state=DEFAULT_SEED,
                ))
            ),
        })

    # --- Apply family filter ----------------------------------------------
    if model_filter and model_filter != "all":
        configs = [c for c in configs if c.get("family") == model_filter]
        if not configs:
            raise ValueError(
                f"No model configs match --radiomics-model '{model_filter}'. "
                f"Valid choices: all, xgboost, random_forest, "
                f"logistic_regression, svm."
            )

    return configs


# ---------------------------------------------------------------------------
# Structured output directory
# ---------------------------------------------------------------------------


def _build_run_dir(
    output_dir: Path,
    classifier_type: str,
    tasks: list[str],
    run_name: Optional[str] = None,
) -> Path:
    """Create a versioned run directory under *output_dir*.

    The directory name is::

        run_NNN_YYYYMMDD_HHMMSS_{classifier_type}_{tasks}

    where *NNN* is an auto-incremented 3-digit run counter.  An optional
    *run_name* replaces the auto-generated descriptor suffix.

    A ``latest`` symbolic link is updated to point at the new directory.

    Parameters
    ----------
    output_dir : Path
        Base output directory (e.g. ``./trained_models``).
    classifier_type : str
        ``"radiomics"`` or ``"cnn"``.
    tasks : list[str]
        Active classification tasks.
    run_name : str | None
        Optional custom suffix for the run directory.

    Returns
    -------
    Path to the newly-created run directory.
    """
    from datetime import datetime

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine next run number by scanning existing directories.
    existing_nums: list[int] = []
    for child in output_dir.iterdir():
        if child.is_dir() and child.name.startswith(f"{RUN_DIR_PREFIX}_"):
            parts = child.name.split("_")
            if len(parts) >= 2:
                try:
                    existing_nums.append(int(parts[1]))
                except ValueError:
                    pass
    next_num = max(existing_nums, default=0) + 1

    # Build directory name.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        # Sanitise user-supplied name.
        safe_name = (
            run_name.strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )[:60]
        dir_name = f"{RUN_DIR_PREFIX}_{next_num:03d}_{ts}_{safe_name}"
    else:
        tasks_tag = "_".join(sorted(tasks))
        dir_name = f"{RUN_DIR_PREFIX}_{next_num:03d}_{ts}_{classifier_type}_{tasks_tag}"

    run_dir = output_dir / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Update 'latest' symlink.
    latest_link = output_dir / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)
    except OSError:
        # May fail on some file-systems (e.g. FAT32); non-fatal.
        logger.debug(f"Could not create 'latest' symlink: {latest_link}")

    return run_dir


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

    For the ``contrast`` task, every patient is included regardless of
    tumor_subtype (no filtering needed — the label is the MRI phase, not
    clinical metadata).  The returned list contains each patient **once**;
    actual sample doubling (phase 0 + phase 1) is performed downstream.

    For the ``tumor_roi`` task, every patient is included (like contrast).
    Labels are placeholder 0s — the real 0/1 labels (tumor ROI vs
    contralateral mirrored ROI) are created by
    ``create_tumor_roi_dataset()`` downstream.

    Args:
        clinical_df: Clinical data DataFrame with 'patient_id' and
            'tumor_subtype' columns.
        task: Classification task ('tnbc', 'luminal', 'contrast', or
            'tumor_roi').

    Returns:
        Tuple of (patient_ids, labels) where labels is a binary array.
    """
    import pandas as pd

    valid_tasks = {"tnbc", "luminal", "contrast", "tumor_roi"}
    if task not in valid_tasks:
        raise ValueError(f"task must be one of {valid_tasks}, got '{task}'")

    if task == "contrast":
        # Contrast task: every patient with a ``patient_id`` is valid.
        # Labels are placeholder 0s — the real 0/1 labels are created by
        # ``create_contrast_dataset()`` which doubles the dataset (one
        # sample per phase per patient).
        patient_ids = clinical_df["patient_id"].dropna().unique().tolist()
        labels = np.zeros(len(patient_ids), dtype=np.int64)
        logger.info(
            f"Task 'contrast': {len(patient_ids)} patients available "
            f"(each contributes phase-0 and phase-1 samples)."
        )
        return patient_ids, labels

    if task == "tumor_roi":
        # Tumor ROI task: every patient participates.  Labels are
        # placeholder 0s — real labels (1=tumor, 0=mirror) are created
        # by ``create_tumor_roi_dataset()``.
        patient_ids = clinical_df["patient_id"].dropna().unique().tolist()
        labels = np.zeros(len(patient_ids), dtype=np.int64)
        logger.info(
            f"Task 'tumor_roi': {len(patient_ids)} patients available "
            f"(each contributes tumor-ROI and mirrored-ROI samples)."
        )
        return patient_ids, labels

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


def create_contrast_dataset(
    patient_ids: list[str],
    data_dir: Path,
    images_dir: Optional[Path] = None,
    segmentations_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    n_workers: int = 1,
    slice_mode: Optional[str] = None,
    n_slices: int = DEFAULT_N_SLICES,
) -> tuple[NDArray[np.floating], NDArray[np.integer], list[str], list[int]]:
    """Build a feature matrix for the pre-/post-contrast classification task.

    Each patient contributes **two** feature vectors: one extracted from
    phase 0 (pre-contrast, label=0) and one from phase 1 (post-contrast,
    label=1).  This effectively doubles the dataset size compared to a
    single-phase subtype task.

    Parameters
    ----------
    patient_ids : list[str]
        Patient identifiers (each will appear twice in the output).
    data_dir, images_dir, segmentations_dir : Path
        Standard MAMA-MIA path arguments.
    cache_dir : Path | None
        Feature cache directory.
    n_workers, slice_mode, n_slices : int | str | None
        Feature extraction parameters (passed through).

    Returns
    -------
    feature_matrix : NDArray
        Shape ``(2*N_valid, n_features)``.
    labels : NDArray
        Binary array — ``0`` for pre-contrast, ``1`` for post-contrast.
    valid_pids : list[str]
        Patient ID for each row (may repeat).
    valid_original_indices : list[int]
        Index into *patient_ids* for each valid row.
    """
    all_feats: list[NDArray[np.floating]] = []
    all_labels: list[int] = []
    all_pids: list[str] = []
    all_indices: list[int] = []

    for phase_val, label_val in [(0, 0), (1, 1)]:
        logger.info(
            f"Contrast task: extracting features from phase {phase_val} "
            f"(label={label_val}) for {len(patient_ids)} patients."
        )
        feat_matrix, valid_pids, valid_idx = extract_features_for_patients(
            patient_ids=patient_ids,
            data_dir=data_dir,
            images_dir=images_dir,
            segmentations_dir=segmentations_dir,
            phase=phase_val,
            cache_dir=cache_dir,
            n_workers=n_workers,
            slice_mode=slice_mode,
            n_slices=n_slices,
            dual_phase=False,  # never dual-phase for contrast task itself
        )
        n_samples = feat_matrix.shape[0]
        all_feats.append(feat_matrix)
        all_labels.extend([label_val] * n_samples)
        all_pids.extend(valid_pids)
        all_indices.extend(valid_idx)

    feature_matrix = np.vstack(all_feats)
    labels = np.array(all_labels, dtype=np.int64)

    n_pre = int((labels == 0).sum())
    n_post = int((labels == 1).sum())
    logger.info(
        f"Contrast dataset: {len(labels)} samples "
        f"(pre-contrast={n_pre}, post-contrast={n_post})"
    )

    return feature_matrix, labels, all_pids, all_indices


def create_tumor_roi_dataset(
    patient_ids: list[str],
    data_dir: Path,
    images_dir: Optional[Path] = None,
    segmentations_dir: Optional[Path] = None,
    phase: int = 1,
    cache_dir: Optional[Path] = None,
    n_workers: int = 1,
    slice_mode: Optional[str] = None,
    n_slices: int = 5,
    search_fraction: float = 0.4,
    min_tissue_fraction: float = 0.3,
) -> tuple[NDArray[np.floating], NDArray[np.integer], list[str], list[int]]:
    """Build a feature matrix for the tumor-ROI vs mirrored-ROI task.

    For each patient, this function:
      1. Loads the post-contrast image and tumor segmentation mask.
      2. Extracts radiomic features from the **tumor ROI** (label=1).
      3. Creates a contralateral mirrored mask (reflected about the
         body midline) and extracts radiomic features from that
         **mirrored ROI** (label=0).

    Patients whose mirrored mask fails tissue validation are skipped
    (they contribute neither a positive nor a negative sample).

    Parameters
    ----------
    patient_ids : list[str]
        Patient identifiers.
    data_dir : Path
        Root directory of the MAMA-MIA dataset.
    images_dir, segmentations_dir : Path | None
        Override paths.
    phase : int
        MRI phase to use (default: 1 = first post-contrast).
    cache_dir : Path | None
        Feature cache directory.  Tumor-ROI and mirrored-ROI features
        are cached separately with ``_tumor`` and ``_mirror`` suffixes.
    n_workers : int
        Currently unused (sequential extraction).
    slice_mode : str | None
        2D slice extraction strategy.  Only ``"all_tumor"`` is fully
        supported; when set, each tumour slice becomes an independent
        sample.  ``None`` → full 3D volume.
    n_slices : int
        Number of slices for multi-slice mode (passed through).
    search_fraction : float
        Passed to :func:`eval.mirror_utils.detect_midline`.
    min_tissue_fraction : float
        Passed to :func:`eval.mirror_utils.validate_mirrored_region`.

    Returns
    -------
    feature_matrix : NDArray
        Shape ``(2*N_valid, n_features)`` (tumor + mirror per patient).
    labels : NDArray
        Binary array — ``1`` for tumor ROI, ``0`` for mirrored ROI.
    valid_pids : list[str]
        Patient ID for each row (each valid patient appears twice or
        more in ``all_tumor`` mode).
    valid_original_indices : list[int]
        Index into *patient_ids* for each valid row.
    """
    from eval.frd import (
        FRD_DEFAULT_BIN_WIDTH,
        FRD_FEATURE_CLASSES,
        extract_radiomic_features,
    )
    from eval.mirror_utils import create_mirrored_mask

    if images_dir is None:
        images_dir = data_dir / IMAGES_SUBDIR
    if segmentations_dir is None:
        segmentations_dir = data_dir / SEGMENTATIONS_SUBDIR

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Resolve slice mode
    _use_all_tumor = (
        slice_mode is not None and SliceMode(slice_mode) == SliceMode.ALL_TUMOR
    )

    # Cache suffix
    _sm_tag = f"_{slice_mode}" if slice_mode else ""
    cache_suffix_tumor = f"_phase{phase}{_sm_tag}_tumor.npy"
    cache_suffix_mirror = f"_phase{phase}{_sm_tag}_mirror.npy"

    try:
        from tqdm import tqdm
        iterator = tqdm(
            enumerate(patient_ids),
            total=len(patient_ids),
            desc="Tumor ROI dataset",
            unit="patient",
        )
    except ImportError:
        iterator = enumerate(patient_ids)

    all_feats: list[NDArray[np.floating]] = []
    all_labels: list[int] = []
    all_pids: list[str] = []
    all_indices: list[int] = []
    n_valid = 0
    n_skipped = 0

    for i, pid in iterator:
        # --- Try loading from cache first ---
        if cache_dir is not None:
            tumor_cache = cache_dir / f"{pid}{cache_suffix_tumor}"
            mirror_cache = cache_dir / f"{pid}{cache_suffix_mirror}"
            if tumor_cache.exists() and mirror_cache.exists():
                try:
                    t_feat = np.load(tumor_cache, allow_pickle=False)
                    m_feat = np.load(mirror_cache, allow_pickle=False)

                    # Handle 2D cached arrays (all_tumor mode)
                    if t_feat.ndim == 2 and m_feat.ndim == 2:
                        for row in t_feat:
                            all_feats.append(row)
                            all_labels.append(1)
                            all_pids.append(pid)
                            all_indices.append(i)
                        for row in m_feat:
                            all_feats.append(row)
                            all_labels.append(0)
                            all_pids.append(pid)
                            all_indices.append(i)
                    else:
                        all_feats.append(t_feat)
                        all_labels.append(1)
                        all_pids.append(pid)
                        all_indices.append(i)
                        all_feats.append(m_feat)
                        all_labels.append(0)
                        all_pids.append(pid)
                        all_indices.append(i)
                    n_valid += 1
                    continue
                except Exception:
                    pass  # Re-extract if cache is corrupted

        # --- Load image and mask ---
        image_path = _get_image_path(images_dir, pid, phase)
        seg_path = _get_segmentation_path(segmentations_dir, pid)

        try:
            image_array = _load_nifti_as_array(image_path)
        except Exception as e:
            logger.warning(f"Failed to load image for {pid}: {e}, skipping.")
            n_skipped += 1
            continue

        try:
            mask_array = _load_mask_as_array(seg_path)
        except Exception as e:
            logger.warning(
                f"No segmentation mask for {pid}: {e}, skipping."
            )
            n_skipped += 1
            continue

        if not np.any(mask_array):
            logger.warning(f"Empty mask for {pid}, skipping.")
            n_skipped += 1
            continue

        # --- Create mirrored mask ---
        mirrored_mask = create_mirrored_mask(
            image_array, mask_array,
            search_fraction=search_fraction,
            min_tissue_fraction=min_tissue_fraction,
        )
        if mirrored_mask is None:
            logger.info(
                f"Mirrored mask validation failed for {pid}, skipping."
            )
            n_skipped += 1
            continue

        # --- Extract features ---
        try:
            if _use_all_tumor and image_array.ndim == 3:
                # All-tumour slices: extract per-slice features
                tumor_slices_img, tumor_slices_msk, tumor_idxs = (
                    extract_all_tumor_slices(
                        image_array, mask=mask_array, normalize=True,
                    )
                )
                # Extract mirror slices at the same z-indices
                mirror_slices_img = []
                mirror_slices_msk = []
                for z_idx in tumor_idxs:
                    img_slice = image_array[z_idx]
                    # Normalise the same way as extract_all_tumor_slices
                    mu = img_slice.mean()
                    std = img_slice.std()
                    if std > 0:
                        img_slice = (img_slice - mu) / std
                    mirror_slices_img.append(img_slice)
                    mirror_slices_msk.append(mirrored_mask[z_idx])

                t_feats: list[NDArray[np.floating]] = []
                m_feats: list[NDArray[np.floating]] = []

                for s_img, s_msk in zip(tumor_slices_img, tumor_slices_msk):
                    sf = extract_radiomic_features(
                        s_img, mask=s_msk,
                        feature_classes=FRD_FEATURE_CLASSES,
                        bin_width=FRD_DEFAULT_BIN_WIDTH,
                    )
                    if sf.size > 0 and not np.all(sf == 0):
                        t_feats.append(sf)

                for s_img, s_msk in zip(mirror_slices_img, mirror_slices_msk):
                    if not np.any(s_msk):
                        continue
                    sf = extract_radiomic_features(
                        s_img, mask=s_msk,
                        feature_classes=FRD_FEATURE_CLASSES,
                        bin_width=FRD_DEFAULT_BIN_WIDTH,
                    )
                    if sf.size > 0 and not np.all(sf == 0):
                        m_feats.append(sf)

                if not t_feats or not m_feats:
                    logger.warning(
                        f"Empty features for {pid} (tumor or mirror), "
                        "skipping."
                    )
                    n_skipped += 1
                    continue

                for f in t_feats:
                    all_feats.append(f)
                    all_labels.append(1)
                    all_pids.append(pid)
                    all_indices.append(i)
                for f in m_feats:
                    all_feats.append(f)
                    all_labels.append(0)
                    all_pids.append(pid)
                    all_indices.append(i)

                # Cache
                if cache_dir is not None:
                    try:
                        np.save(
                            cache_dir / f"{pid}{cache_suffix_tumor}",
                            np.stack(t_feats, axis=0),
                        )
                        np.save(
                            cache_dir / f"{pid}{cache_suffix_mirror}",
                            np.stack(m_feats, axis=0),
                        )
                    except Exception as e:
                        logger.debug(f"Failed to cache for {pid}: {e}")

            else:
                # Full 3D or single-slice extraction
                t_feat = extract_radiomic_features(
                    image_array, mask=mask_array,
                    feature_classes=FRD_FEATURE_CLASSES,
                    bin_width=FRD_DEFAULT_BIN_WIDTH,
                )
                m_feat = extract_radiomic_features(
                    image_array, mask=mirrored_mask,
                    feature_classes=FRD_FEATURE_CLASSES,
                    bin_width=FRD_DEFAULT_BIN_WIDTH,
                )

                if t_feat.size == 0 or np.all(t_feat == 0):
                    logger.warning(f"Empty tumor features for {pid}.")
                    n_skipped += 1
                    continue
                if m_feat.size == 0 or np.all(m_feat == 0):
                    logger.warning(f"Empty mirror features for {pid}.")
                    n_skipped += 1
                    continue

                all_feats.append(t_feat)
                all_labels.append(1)
                all_pids.append(pid)
                all_indices.append(i)
                all_feats.append(m_feat)
                all_labels.append(0)
                all_pids.append(pid)
                all_indices.append(i)

                # Cache
                if cache_dir is not None:
                    try:
                        np.save(
                            cache_dir / f"{pid}{cache_suffix_tumor}",
                            t_feat,
                        )
                        np.save(
                            cache_dir / f"{pid}{cache_suffix_mirror}",
                            m_feat,
                        )
                    except Exception as e:
                        logger.debug(f"Failed to cache for {pid}: {e}")

            n_valid += 1

        except Exception as e:
            logger.warning(
                f"Feature extraction failed for {pid}: {e}, skipping."
            )
            n_skipped += 1
            continue

    logger.info(
        f"Tumor ROI dataset: {n_valid} patients valid, "
        f"{n_skipped} skipped, {len(all_feats)} total samples "
        f"(tumor={sum(1 for l in all_labels if l == 1)}, "
        f"mirror={sum(1 for l in all_labels if l == 0)})"
    )

    if len(all_feats) == 0:
        raise RuntimeError(
            "No features could be extracted for tumor_roi task. "
            "Check image/mask paths and data."
        )

    feature_matrix = np.vstack([f[np.newaxis, :] if f.ndim == 1 else f for f in all_feats])
    labels = np.array(all_labels, dtype=np.int64)

    # Clean NaN/Inf
    feature_matrix = np.nan_to_num(
        feature_matrix, nan=0.0, posinf=0.0, neginf=0.0
    )

    return feature_matrix, labels, all_pids, all_indices


def load_split_csv(
    csv_path: Path,
) -> tuple[list[str], list[str]]:
    """Load train/test patient IDs from a MAMA-MIA split CSV file.

    The expected format has two columns — ``train_split`` and
    ``test_split`` — each cell containing a patient ID.  Columns may
    have different lengths (ragged); empty cells are silently ignored.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Tuple of ``(train_ids, test_ids)``.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError: If the expected columns are not found.
    """
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalise column names
    col_map = {c.lower().strip(): c for c in df.columns}

    train_col = col_map.get("train_split")
    test_col = col_map.get("test_split")

    if train_col is None and test_col is None:
        raise ValueError(
            f"Split CSV {csv_path} does not contain expected columns "
            f"'train_split' and/or 'test_split'.  Found: {list(df.columns)}"
        )

    train_ids: list[str] = []
    test_ids: list[str] = []

    if train_col is not None:
        train_ids = (
            df[train_col].dropna().astype(str).str.strip()
            .loc[lambda s: s != ""].tolist()
        )
    if test_col is not None:
        test_ids = (
            df[test_col].dropna().astype(str).str.strip()
            .loc[lambda s: s != ""].tolist()
        )

    logger.info(
        f"Loaded split CSV: {len(train_ids)} train, {len(test_ids)} test "
        f"patients from {csv_path.name}"
    )
    return train_ids, test_ids


def _find_split_csv(data_dir: Path) -> Optional[Path]:
    """Auto-detect a split CSV file in *data_dir*.

    Tries each filename in :data:`SPLIT_CSV_CANDIDATES` (case-sensitive)
    and returns the first match, or ``None``.
    """
    for candidate in SPLIT_CSV_CANDIDATES:
        p = data_dir / candidate
        if p.exists():
            logger.info(f"Auto-detected split CSV: {p}")
            return p
    return None


def detect_split_column(
    clinical_df: "pd.DataFrame",
    custom_test_values: Optional[list[str]] = None,
) -> Optional[str]:
    """Auto-detect the train/test split column in the clinical data.

    Tries each candidate column name from :data:`SPLIT_COLUMN_CANDIDATES`
    (case-insensitive) and returns the first that exists and contains
    recognisable split values.

    When *custom_test_values* are given the check is relaxed: the column
    only needs to contain at least one of the custom values (no need for
    standard train/test labels).

    Args:
        clinical_df: Clinical data DataFrame.
        custom_test_values: If supplied, accept a candidate column when
            it contains at least one of these values.

    Returns:
        Column name if found, else ``None``.
    """
    # Build a lower-case → actual-name lookup for the DataFrame columns
    col_lower_map = {c.lower().strip(): c for c in clinical_df.columns}

    custom_lower = (
        {v.lower().strip() for v in custom_test_values}
        if custom_test_values
        else None
    )

    for candidate in SPLIT_COLUMN_CANDIDATES:
        actual_col = col_lower_map.get(candidate.lower())
        if actual_col is not None:
            unique = set(clinical_df[actual_col].dropna().astype(str).unique())
            unique_lower = {v.lower().strip() for v in unique}

            if custom_lower is not None:
                # Custom mode: column must contain at least one custom value
                if unique_lower & custom_lower:
                    logger.info(
                        f"Detected split column '{actual_col}' "
                        f"(matched custom test values: "
                        f"{sorted(custom_lower & unique_lower)}). "
                        f"All values: {sorted(unique)}"
                    )
                    return actual_col
            else:
                # Standard mode: need train/test labels
                has_test = bool(unique_lower & TEST_SPLIT_VALUES)
                has_train = bool(unique_lower & TRAIN_SPLIT_VALUES)
                if has_test or has_train:
                    logger.info(
                        f"Detected split column '{actual_col}' with values: "
                        f"{sorted(unique)}"
                    )
                    return actual_col

    # No match — log available columns for debugging
    logger.debug(
        f"No split column detected. Tried candidates "
        f"{SPLIT_COLUMN_CANDIDATES} (case-insensitive). "
        f"Available columns: {list(clinical_df.columns)}"
    )
    return None


def split_train_test_patients(
    clinical_df: "pd.DataFrame",
    split_column: Optional[str] = None,
    test_split_values: Optional[list[str]] = None,
    split_csv: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> tuple[list[str], list[str]]:
    """Split patient IDs into train and test sets.

    The function tries three strategies in order of priority:

    1. **CSV file** — A dedicated CSV with ``train_split`` /
       ``test_split`` columns listing patient IDs.  This is the format
       shipped with the MAMA-MIA dataset (``train_test_splits.csv``).
       Pass the path explicitly via *split_csv*, or let auto-detection
       find it in *data_dir*.
    2. **Column + custom values** — A column in the clinical Excel is
       matched against *test_split_values* (case-insensitive).
    3. **Column + standard labels** — A column whose values contain
       ``train`` / ``test``.

    If none of the above succeeds, all patients are returned as training
    patients and the test list is empty.

    Args:
        clinical_df: Clinical data DataFrame.
        split_column: Column name containing split labels. If ``None``,
            auto-detection is attempted.
        test_split_values: Custom test-set value(s) for the split column.
        split_csv: Explicit path to a split CSV file.  When ``None``,
            the function looks for ``train_test_splits.csv`` (or similar)
            inside *data_dir*.
        data_dir: MAMA-MIA dataset root — used for auto-detecting the
            split CSV.  Ignored when *split_csv* is given explicitly.

    Returns:
        Tuple of ``(train_patient_ids, test_patient_ids)``.
    """
    # --- Strategy 1: CSV-based split ---
    csv_path = split_csv
    if csv_path is None and data_dir is not None:
        csv_path = _find_split_csv(data_dir)

    if csv_path is not None:
        try:
            csv_train, csv_test = load_split_csv(csv_path)
            # Cross-reference with clinical_df patient IDs so that we
            # only return IDs that actually exist in the data.
            valid_pids = set(clinical_df["patient_id"].astype(str).tolist())
            train_ids = [p for p in csv_train if p in valid_pids]
            test_ids = [p for p in csv_test if p in valid_pids]
            n_train_dropped = len(csv_train) - len(train_ids)
            n_test_dropped = len(csv_test) - len(test_ids)
            if n_train_dropped or n_test_dropped:
                logger.info(
                    f"Split CSV cross-ref: dropped {n_train_dropped} train "
                    f"and {n_test_dropped} test IDs not present in clinical "
                    f"data ({len(valid_pids)} patients)."
                )
            logger.info(
                f"Dataset split (CSV): {len(train_ids)} training, "
                f"{len(test_ids)} test"
            )
            return train_ids, test_ids
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(f"Failed to load split CSV: {exc}")

    # --- Strategy 2 & 3: column-based split ---
    if split_column is None and test_split_values is not None:
        split_column = detect_split_column(
            clinical_df, custom_test_values=test_split_values,
        )
    elif split_column is None:
        split_column = detect_split_column(clinical_df)

    if split_column is None:
        logger.warning(
            "No train/test split column found in clinical data. "
            "Returning all patients as training set."
        )
        return clinical_df["patient_id"].tolist(), []

    col_values = clinical_df[split_column].fillna("").astype(str)
    col_values_lower = col_values.str.lower().str.strip()

    if test_split_values is not None:
        # Custom test values mode: anything matching → test, rest → train
        custom_lower = {v.lower().strip() for v in test_split_values}
        test_mask = col_values_lower.isin(custom_lower)
        train_mask = ~test_mask
        train_ids = clinical_df.loc[train_mask, "patient_id"].tolist()
        test_ids = clinical_df.loc[test_mask, "patient_id"].tolist()
    else:
        # Default mode: look for standard train/test labels
        train_mask = col_values_lower.isin(TRAIN_SPLIT_VALUES)
        test_mask = col_values_lower.isin(TEST_SPLIT_VALUES)
        train_ids = clinical_df.loc[train_mask, "patient_id"].tolist()
        test_ids = clinical_df.loc[test_mask, "patient_id"].tolist()

        # Patients not matching either split go to training
        unassigned = ~(train_mask | test_mask)
        n_unassigned = int(unassigned.sum())
        if n_unassigned > 0:
            logger.info(
                f"{n_unassigned} patients have unrecognised split value — "
                f"assigning to training set."
            )
            train_ids.extend(
                clinical_df.loc[unassigned, "patient_id"].tolist()
            )

    logger.info(
        f"Dataset split: {len(train_ids)} training, {len(test_ids)} test "
        f"(column='{split_column}')"
    )
    return train_ids, test_ids


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _load_nifti_as_array(filepath: Path) -> NDArray[np.floating]:
    """Load a NIfTI file and return as numpy array."""
    import SimpleITK as sitk

    if not filepath.exists():
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")

    sitk_image = sitk.ReadImage(str(filepath), sitk.sitkFloat32)
    return sitk.GetArrayFromImage(sitk_image)  # already float32


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
    slice_mode: Optional[str] = None,
    n_slices: int = DEFAULT_N_SLICES,
    dual_phase: bool = False,
) -> tuple[NDArray[np.floating], list[str], list[int]]:
    """Extract radiomic features for a list of patients.

    Loads each patient's post-contrast MRI image and segmentation mask,
    then extracts radiomic features using pyradiomics. Supports both
    full 3D volume extraction and 2D slice extraction.

    When ``slice_mode`` is set, the 3D volume is first reduced to one or
    more 2D slices (with z-score normalisation) before feature extraction.
    Available modes:

    - ``"max_tumor"``: Axial slice with the largest tumour cross-section.
    - ``"center_tumor"``: Axial slice through the tumour centre of mass.
    - ``"multi_slice"``: ``n_slices`` equally-spaced slices across tumour
      extent; features are concatenated.
    - ``"all_tumor"``: Every axial slice with ≥1 tumour voxel.  Each
      slice becomes an independent sample (multiplies dataset size).
    - ``"middle"``: Middle axial slice (no mask required).
    - ``None``: Use the full 3D volume (default, backward-compatible).

    Args:
        patient_ids: List of patient identifiers.
        data_dir: Root directory of the MAMA-MIA dataset.
        images_dir: Override path to images folder.
        segmentations_dir: Override path to segmentations folder.
        phase: MRI phase to use (default: 1 = first post-contrast).
        cache_dir: Directory for caching per-patient feature files.
        n_workers: Number of parallel workers (currently sequential).
        slice_mode: 2D slice extraction strategy or None for full 3D.
        n_slices: Number of slices for ``"multi_slice"`` mode.
        dual_phase: When True, also extract features from phase 0
            (pre-contrast) and concatenate them with phase 1 features,
            doubling the feature dimension. Disabled by default.

    Returns:
        Tuple of:
            - feature_matrix: Shape (n_valid_patients, n_features).
            - valid_patient_ids: Patient IDs that were successfully processed.
            - valid_indices: Original indices of valid patients.
    """
    from eval.frd import (
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

    # Resolve slice mode
    _slice_mode: Optional[SliceMode] = None
    if slice_mode is not None:
        try:
            _slice_mode = SliceMode(slice_mode)
        except ValueError:
            raise ValueError(
                f"Invalid slice_mode '{slice_mode}'. "
                f"Choose from: {[m.value for m in SliceMode]}"
            )
        logger.info(f"2D slice extraction enabled: mode={_slice_mode.value}")

    features_list: list[NDArray[np.floating]] = []
    valid_patient_ids: list[str] = []
    valid_indices: list[int] = []
    n_cached = 0
    n_extracted = 0
    n_failed = 0

    # Cache suffix differentiates 3D vs 2D cached features
    _dp_tag = "_dualphase" if dual_phase else ""
    cache_suffix = (
        f"_phase{phase}{_dp_tag}.npy" if _slice_mode is None
        else f"_phase{phase}_{_slice_mode.value}{_dp_tag}.npy"
    )

    for i, pid in iterator:
        # Check cache first
        if cache_dir is not None:
            cache_file = cache_dir / f"{pid}{cache_suffix}"
            if cache_file.exists():
                try:
                    feat = np.load(cache_file, allow_pickle=False)
                    if feat.ndim == 2:
                        # all_tumor cache: 2-D array (n_slices, n_features)
                        for row in feat:
                            features_list.append(row)
                            valid_patient_ids.append(pid)
                            valid_indices.append(i)
                    else:
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

        # Load mask (required for mask-dependent slice modes, optional otherwise)
        _mask_dependent = _slice_mode in (
            SliceMode.MAX_TUMOR, SliceMode.CENTER_TUMOR,
            SliceMode.MULTI_SLICE, SliceMode.ALL_TUMOR,
        )
        mask_array: Optional[NDArray[np.bool_]] = None
        try:
            mask_array = _load_mask_as_array(seg_path)
            if not np.any(mask_array):
                msg = f"Segmentation mask is empty (all zeros) for {pid}: {seg_path}"
                if _mask_dependent:
                    logger.warning(
                        f"{msg}. Slice mode '{slice_mode}' requires a valid mask — "
                        f"skipping patient to avoid feature extraction from wrong region."
                    )
                    n_failed += 1
                    continue
                else:
                    logger.debug(f"{msg}, using full image for feature extraction.")
                    mask_array = None
        except FileNotFoundError:
            if _mask_dependent:
                logger.warning(
                    f"No segmentation mask found for {pid} but slice_mode='{slice_mode}' "
                    f"requires one — skipping patient. Expected: {seg_path}"
                )
                n_failed += 1
                continue
            else:
                logger.debug(f"No segmentation for {pid} ({seg_path}), using full image.")
        except Exception as e:
            if _mask_dependent:
                logger.warning(
                    f"Failed to load segmentation for {pid}: {e} — "
                    f"skipping (slice_mode='{slice_mode}' requires a mask)."
                )
                n_failed += 1
                continue
            else:
                logger.warning(
                    f"Failed to load segmentation for {pid}: {e}, using full image."
                )

        # ----- 2D slice extraction (if enabled) -----
        if _slice_mode is not None and image_array.ndim == 3:
            try:
                if _slice_mode == SliceMode.ALL_TUMOR:
                    # All-tumour: every slice with ≥1 mask voxel →
                    # each becomes an independent training sample.
                    at_imgs, at_masks, at_idxs = extract_all_tumor_slices(
                        image_array, mask=mask_array, normalize=True,
                    )
                    at_feats: list[NDArray[np.floating]] = []
                    for s_img, s_msk in zip(at_imgs, at_masks):
                        sf = extract_radiomic_features(
                            s_img, mask=s_msk,
                            feature_classes=FRD_FEATURE_CLASSES,
                            bin_width=FRD_DEFAULT_BIN_WIDTH,
                        )
                        if sf.size == 0 or np.all(sf == 0):
                            continue
                        at_feats.append(sf)
                        features_list.append(sf)
                        valid_patient_ids.append(pid)
                        valid_indices.append(i)
                    if len(at_feats) == 0:
                        logger.warning(
                            f"All tumour slices yielded empty features for "
                            f"{pid}, skipping."
                        )
                        n_failed += 1
                    else:
                        # Cache the 2-D (n_slices, n_features) array
                        if cache_dir is not None:
                            try:
                                np.save(
                                    cache_dir / f"{pid}{cache_suffix}",
                                    np.stack(at_feats, axis=0),
                                )
                            except Exception as e:
                                logger.debug(
                                    f"Failed to cache features for {pid}: {e}"
                                )
                        logger.debug(
                            f"{pid}: {len(at_feats)}/{len(at_idxs)} tumour "
                            f"slices produced valid features."
                        )
                        n_extracted += 1
                    # Skip the common append block below — already handled.
                    continue
                elif _slice_mode == SliceMode.MULTI_SLICE:
                    # Multi-slice: extract features per slice and concatenate
                    slices_2d, masks_2d, _ = extract_multi_slices(
                        image_array, mask=mask_array,
                        n_slices=n_slices, normalize=True,
                    )
                    slice_feats = []
                    for s_img, s_msk in zip(slices_2d, masks_2d):
                        sf = extract_radiomic_features(
                            s_img, mask=s_msk,
                            feature_classes=FRD_FEATURE_CLASSES,
                            bin_width=FRD_DEFAULT_BIN_WIDTH,
                        )
                        slice_feats.append(sf)
                    feat = np.concatenate(slice_feats)
                else:
                    # Single-slice modes (max_tumor, center_tumor, middle)
                    img_2d, msk_2d, slice_idx = extract_2d_slice(
                        image_array, mask=mask_array,
                        mode=_slice_mode, normalize=True,
                    )
                    feat = extract_radiomic_features(
                        img_2d, mask=msk_2d,
                        feature_classes=FRD_FEATURE_CLASSES,
                        bin_width=FRD_DEFAULT_BIN_WIDTH,
                    )
            except Exception as e:
                logger.warning(
                    f"2D slice extraction/feature extraction failed for {pid}: {e}, "
                    f"skipping."
                )
                n_failed += 1
                continue
        else:
            # ----- Full 3D feature extraction (default) -----
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

        # --- Dual-phase: extract & concatenate phase-0 features ---
        if dual_phase:
            phase0_path = _get_image_path(images_dir, pid, 0)
            try:
                phase0_array = _load_nifti_as_array(phase0_path)
            except Exception as e:
                logger.warning(
                    f"Dual-phase: failed to load phase-0 for {pid}: {e}, "
                    "skipping."
                )
                n_failed += 1
                continue
            try:
                if _slice_mode is not None and phase0_array.ndim == 3:
                    if _slice_mode in (
                        SliceMode.MAX_TUMOR, SliceMode.CENTER_TUMOR,
                        SliceMode.MIDDLE,
                    ):
                        p0_2d, p0_msk, _ = extract_2d_slice(
                            phase0_array, mask=mask_array,
                            mode=_slice_mode, normalize=True,
                        )
                        feat_p0 = extract_radiomic_features(
                            p0_2d, mask=p0_msk,
                            feature_classes=FRD_FEATURE_CLASSES,
                            bin_width=FRD_DEFAULT_BIN_WIDTH,
                        )
                    else:
                        feat_p0 = extract_radiomic_features(
                            phase0_array, mask=mask_array,
                            feature_classes=FRD_FEATURE_CLASSES,
                            bin_width=FRD_DEFAULT_BIN_WIDTH,
                        )
                else:
                    feat_p0 = extract_radiomic_features(
                        phase0_array, mask=mask_array,
                        feature_classes=FRD_FEATURE_CLASSES,
                        bin_width=FRD_DEFAULT_BIN_WIDTH,
                    )
                feat = np.concatenate([feat, feat_p0])
            except Exception as e:
                logger.warning(
                    f"Dual-phase feature extraction failed for {pid}: {e}, "
                    "using single-phase features."
                )

        # Validate feature vector
        if feat.size == 0 or np.all(feat == 0):
            logger.warning(f"Empty features for {pid}, skipping.")
            n_failed += 1
            continue

        # Cache features
        if cache_dir is not None:
            try:
                np.save(cache_dir / f"{pid}{cache_suffix}", feat)
            except Exception as e:
                logger.debug(f"Failed to cache features for {pid}: {e}")

        features_list.append(feat)
        valid_patient_ids.append(pid)
        valid_indices.append(i)
        n_extracted += 1

    n_unique_patients = len(set(valid_patient_ids))
    logger.info(
        f"Feature extraction complete: {n_extracted} extracted, "
        f"{n_cached} from cache, {n_failed} failed "
        f"({n_unique_patients}/{len(patient_ids)} patients valid, "
        f"{len(features_list)} total samples)"
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
        f"({feature_matrix.shape[0]} samples × {feature_matrix.shape[1]} features)"
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
    model_filter: Optional[str] = None,
    return_all: bool = False,
) -> tuple[Any, str, dict[str, float], Optional[list[tuple[Any, str, dict[str, float]]]]]:
    """Train multiple model configurations and select the best.

    Tries all configurations from _get_model_configs() and selects the
    model with the highest validation AUROC.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        task: Task name for logging.
        model_filter: Optional family name to restrict model configs.
        return_all: If True, additionally return all successfully trained
            models as a list of ``(model, name, metrics)`` tuples.

    Returns:
        Tuple of ``(best_model, best_config_name, best_val_metrics,
        all_models_or_None)``.  *all_models_or_None* is ``None`` when
        *return_all* is ``False``.
    """
    n_pos = max(int(y_train.sum()), 1)
    n_neg = int((1 - y_train).sum())
    spw = n_neg / n_pos
    configs = _get_model_configs(scale_pos_weight=spw, model_filter=model_filter)
    logger.info(
        f"\n{'='*60}\n"
        f"Training '{task}' classifier — trying {len(configs)} configurations\n"
        f"{'='*60}"
    )
    logger.info(
        f"Train set: {X_train.shape[0]} samples "
        f"(pos={n_pos}, neg={n_neg}, scale_pos_weight={spw:.2f})"
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
    all_trained: list[tuple[Any, str, dict[str, float]]] = []

    # Train configs concurrently using threads.  sklearn/BLAS releases the
    # GIL so multiple fits can run in parallel.  ThreadPoolExecutor avoids
    # pickling issues with lambda create_fn closures.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_threads = min(len(configs), max(1, (os.cpu_count() or 1)))

    def _fit_config(
        cfg: dict[str, Any],
    ) -> tuple[dict[str, Any], Any, dict[str, float], dict[str, float]]:
        model = cfg["create_fn"]()
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        train_m = evaluate_model(model, X_train, y_train)
        val_m = evaluate_model(model, X_val, y_val)
        logger.info(
            f"  {cfg['name']} ({elapsed:.1f}s): "
            f"train AUROC={train_m['auroc']:.4f}, "
            f"val AUROC={val_m['auroc']:.4f}"
        )
        return cfg, model, train_m, val_m

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {pool.submit(_fit_config, cfg): cfg for cfg in configs}
        for fut in as_completed(futures):
            cfg = futures[fut]
            try:
                _, model, train_m, val_m = fut.result()
            except Exception as e:
                logger.warning(f"  Failed: {cfg['name']}: {e}")
                continue

            results_log.append({
                "name": cfg["name"],
                "train_auroc": train_m["auroc"],
                "train_bal_acc": train_m["balanced_accuracy"],
                "val_auroc": val_m["auroc"],
                "val_bal_acc": val_m["balanced_accuracy"],
            })

            if return_all:
                all_trained.append((model, cfg["name"], val_m))

            val_auroc = val_m["auroc"]
            if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_model = model
                best_name = cfg["name"]
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

    return best_model, best_name, best_val_metrics, (all_trained if return_all else None)


# ---------------------------------------------------------------------------
# Cross-validation training (alternative mode)
# ---------------------------------------------------------------------------

def train_with_cross_validation(
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    task: str,
    n_folds: int = 5,
    seed: int = DEFAULT_SEED,
    model_filter: Optional[str] = None,
    return_all: bool = False,
) -> tuple[Any, str, dict[str, float], Optional[list[tuple[Any, str, dict[str, float]]]]]:
    """Train with k-fold cross-validation for model selection.

    Uses stratified k-fold CV to evaluate model configs, then retrains
    the best config on the full dataset.

    Args:
        X: Full feature matrix.
        y: Full label vector.
        task: Task name for logging.
        n_folds: Number of CV folds.
        seed: Random seed.
        model_filter: Optional family name to restrict model configs.
        return_all: If True, retrain *all* configs on the full data and
            return them in addition to the best model.

    Returns:
        Tuple of ``(best_model, best_config_name, avg_cv_metrics,
        all_models_or_None)``.
    """
    n_pos = max(int(y.sum()), 1)
    n_neg = int((1 - y).sum())
    spw = n_neg / n_pos
    configs = _get_model_configs(scale_pos_weight=spw, model_filter=model_filter)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    logger.info(
        f"\n{'='*60}\n"
        f"Cross-validation for '{task}' — {n_folds} folds, "
        f"{len(configs)} configurations\n"
        f"{'='*60}"
    )

    best_config = None
    best_avg_auroc = -1.0
    # Minimum number of folds before early-stopping can trigger.
    _MIN_FOLDS_BEFORE_STOP = 2

    for config in configs:
        fold_aurocs: list[float] = []
        fold_bal_accs: list[float] = []
        _stopped_early = False

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # --- Early stopping: if running mean is far below current best,
            # skip remaining folds to save time. ---
            if (
                fold >= _MIN_FOLDS_BEFORE_STOP
                and best_avg_auroc > 0
                and fold_aurocs
            ):
                running_mean = float(np.nanmean(fold_aurocs))
                # Stop if running mean is more than 0.10 below best so far
                if running_mean < best_avg_auroc - 0.10:
                    logger.debug(
                        f"  {config['name']}: early-stopped after fold {fold} "
                        f"(running AUROC={running_mean:.4f} vs "
                        f"best={best_avg_auroc:.4f})"
                    )
                    _stopped_early = True
                    break

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
            suffix = " [early-stopped]" if _stopped_early else ""
            logger.info(
                f"  {config['name']}: "
                f"CV AUROC={avg_auroc:.4f}±{np.nanstd(fold_aurocs):.4f}, "
                f"Bal.Acc={avg_bal_acc:.4f}{suffix}"
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

    # Optionally retrain all successful configs on full data
    all_trained: Optional[list[tuple[Any, str, dict[str, float]]]] = None
    if return_all:
        all_trained = []
        for cfg in configs:
            try:
                m = cfg["create_fn"]()
                m.fit(X, y)
                m_metrics = evaluate_model(m, X, y)
                all_trained.append((m, cfg["name"], m_metrics))
            except Exception as e:
                logger.warning(f"  Retrain {cfg['name']} failed: {e}")

    return final_model, best_config["name"], {"cv_auroc": best_avg_auroc}, all_trained


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    task: str,
    output_dir: Path,
    suffix: str = "",
) -> Path:
    """Save a trained model in the format expected by the evaluation pipeline.

    The evaluation code (RadiomicsClassifier._load_model) loads models
    using pickle.load(). This function saves the raw sklearn/xgboost
    model object (not a RadiomicsClassifier wrapper).

    Args:
        model: Trained sklearn-compatible classifier.
        task: Task name ('tnbc' or 'luminal').
        output_dir: Directory to save the model.
        suffix: Optional suffix appended before ``.pkl``
            (e.g. ``"_xgb_1"`` → ``tnbc_classifier_xgb_1.pkl``).

    Returns:
        Path to the saved model file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{task}_classifier{suffix}.pkl"

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
            "mama-synth-eval evaluation pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m eval.train_classifier \\\n"
            "      --data-dir /datasets/mama_mia_dataset \\\n"
            "      --output-dir ./trained_models\n"
            "\n"
            "After training, the models can be used for evaluation:\n"
            "  eval --clf-model-dir ./trained_models ...\n"
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
        help=(
            "Base directory for saving trained models. By default, each "
            "training run creates a versioned sub-directory "
            "(e.g. run_001_20260411_143022_cnn_tnbc). "
            "Use --flat-output to write directly into this directory."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Custom name for this training run. When set, the run "
            "directory is named run_NNN_TIMESTAMP_{run-name}. "
            "Useful for labelling experiments."
        ),
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help=(
            "Write all outputs directly into --output-dir without "
            "creating a versioned run sub-directory. Preserves the "
            "pre-v0.9.0 behaviour."
        ),
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
        "--segmentations-dir", "--masks-path",
        dest="segmentations_dir",
        type=Path,
        default=None,
        help=(
            f"Path to the segmentation masks folder. "
            f"Default: <data-dir>/{SEGMENTATIONS_SUBDIR}. "
            "Alias: --masks-path (consistent with the evaluation CLI)."
        ),
    )

    # Training arguments
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["tnbc", "luminal", "contrast", "tumor_roi"],
        default=["tnbc", "luminal"],
        help=(
            "Classification tasks to train. 'tnbc' and 'luminal' are "
            "molecular subtype tasks based on clinical labels. 'contrast' "
            "trains a pre-contrast vs post-contrast phase classifier "
            "(binary: phase 0 → 0, phase 1 → 1). 'tumor_roi' trains a "
            "tumor ROI vs contralateral mirrored ROI classifier using "
            "radiomics features. Default: tnbc luminal"
        ),
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
            "For radiomics this stores per-patient feature arrays; for "
            "CNN this stores extracted 2-D slices. "
            "The cache is shared across all versioned run directories — "
            "it is placed at <output-dir>/feature_cache (next to the "
            "run_NNN_… folders, not inside them) so features are reused "
            "across runs without duplicating them on disk."
        ),
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help=(
            "Delete cached features before extraction, forcing re-computation. "
            "Useful when masks, images, or the extraction pipeline have changed."
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel workers for feature extraction. Default: 1.",
    )

    # Quick test / subset
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help=(
            "Quick validation run with only 10 cases per task. "
            "Useful for verifying that the full training pipeline works "
            "before launching a full training run."
        ),
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=None,
        help=(
            "Limit training to the first N cases (per task). "
            "Overrides --quick-test. Default: all available cases."
        ),
    )

    # 2D Slice extraction
    slice_choices = [m.value for m in SliceMode]
    parser.add_argument(
        "--slice-mode",
        type=str,
        choices=slice_choices,
        default=None,
        help=(
            "Extract 2D slices from 3D NIfTI volumes before feature "
            "extraction.  Choices: "
            + ", ".join(f"'{c}'" for c in slice_choices)
            + ".  Default: None (use full 3D volume)."
        ),
    )
    parser.add_argument(
        "--n-slices",
        type=int,
        default=DEFAULT_N_SLICES,
        help=(
            f"Number of slices to extract in 'multi_slice' mode. "
            f"Default: {DEFAULT_N_SLICES}."
        ),
    )

    # MAMA-MIA test set evaluation
    parser.add_argument(
        "--evaluate-test-set",
        action="store_true",
        help=(
            "After training, evaluate the model on the MAMA-MIA test set "
            "split. The split is loaded from the CSV file "
            "'train_test_splits.csv' in --data-dir (auto-detected), or "
            "from the path given by --split-csv. Alternatively, a split "
            "column in the clinical Excel can be used (--split-column). "
            "Test results including confusion matrix, ROC curve, and "
            "classification report are saved alongside the training outputs."
        ),
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help=(
            "Path to a CSV file with 'train_split' and 'test_split' "
            "columns listing patient IDs.  This is the standard split "
            "format shipped with the MAMA-MIA dataset. When omitted, "
            "auto-detection looks for 'train_test_splits.csv' in "
            "--data-dir."
        ),
    )
    parser.add_argument(
        "--split-column",
        type=str,
        default=None,
        help=(
            "Column name in the clinical Excel containing train/test split "
            "labels. If not specified, auto-detection is attempted. "
            "For MAMA-MIA, consider using --split-column dataset together "
            "with --test-split-values to treat specific datasets as the "
            "test set (e.g. --test-split-values DUKE)."
        ),
    )
    parser.add_argument(
        "--test-split-values",
        nargs="+",
        type=str,
        default=None,
        help=(
            "One or more values in the split column that identify test-set "
            "patients. All other patients become the training set. "
            "Example: --split-column dataset --test-split-values DUKE ISPY1. "
            "When not specified, the default test labels ('test', 'testing') "
            "are used."
        ),
    )

    # Classifier type
    parser.add_argument(
        "--classifier-type",
        type=str,
        choices=["radiomics", "cnn"],
        default="radiomics",
        help=(
            "Type of classifier to train. 'radiomics' (default) trains "
            "sklearn-based models on radiomic features. 'cnn' trains an "
            "EfficientNet deep learning classifier on raw 2D slices. "
            "CNN mode auto-defaults --slice-mode to 'all_tumor'."
        ),
    )

    # Radiomics model family filter
    parser.add_argument(
        "--radiomics-model",
        type=str,
        choices=["all", "xgboost", "random_forest", "logistic_regression", "svm"],
        default="all",
        help=(
            "Which radiomics classifier family to train. 'all' (default) "
            "tries all available families and picks the best by AUROC. "
            "Only effective with --classifier-type radiomics."
        ),
    )

    # CNN-specific arguments
    parser.add_argument(
        "--cnn-model",
        type=str,
        default="efficientnet_b0",
        help="CNN model architecture (timm identifier). Default: efficientnet_b0.",
    )
    parser.add_argument(
        "--cnn-image-size",
        type=int,
        default=224,
        help="CNN input image size (pixels, squared). Default: 224.",
    )
    parser.add_argument(
        "--cnn-epochs",
        type=int,
        default=50,
        help="Maximum CNN training epochs. Default: 50.",
    )
    parser.add_argument(
        "--cnn-batch-size",
        type=int,
        default=32,
        help="CNN mini-batch size. Default: 32.",
    )
    parser.add_argument(
        "--cnn-lr",
        type=float,
        default=1e-4,
        help="CNN peak learning rate. Default: 1e-4.",
    )
    parser.add_argument(
        "--cnn-patience",
        type=int,
        default=10,
        help="CNN early stopping patience (epochs). Default: 10.",
    )
    parser.add_argument(
        "--cnn-num-workers",
        type=int,
        default=None,
        help=(
            "Number of DataLoader worker processes for parallel data "
            "loading during CNN training. Workers prefetch batches from "
            "the on-disk slice cache while the GPU trains on the current "
            "batch, eliminating the I/O bottleneck. Default: min(CPUs, 4) "
            "on Linux, 0 on macOS. Set to 0 to disable multi-process "
            "loading."
        ),
    )
    parser.add_argument(
        "--cnn-mask-channel",
        action="store_true",
        help=(
            "Add segmentation mask as a 4th input channel to the CNN, "
            "giving the model spatial guidance about tumour location. "
            "The first conv layer is expanded from 3→4 channels with "
            "zero-initialised mask weights so pretrained features are "
            "preserved. Only effective with --classifier-type cnn."
        ),
    )

    # Ensemble / multi-model saving
    parser.add_argument(
        "--save-all-models",
        action="store_true",
        help=(
            "Save ALL successfully trained models (not just the best). "
            "Each model is saved with a distinct filename suffix so that "
            "ensemble inference (--ensemble during evaluation) can load "
            "them all. Files are named {task}_classifier_{family}_{n}.pkl. "
            "Only effective with --classifier-type radiomics."
        ),
    )

    # Dual-phase classification
    parser.add_argument(
        "--dual-phase",
        action="store_true",
        help=(
            "Use both phase 0 (pre-contrast) and phase 1 (post-contrast) "
            "for classification. For radiomics, features from both phases "
            "are concatenated (doubling the feature dimension). For CNN, "
            "channels from both phases are stacked (6-channel input). "
            "Disabled by default because pre-contrast images are real "
            "(not synthesized) and including them dilutes evaluation of "
            "synthesis quality. Useful as an ablation baseline."
        ),
    )

    # Device / acceleration
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help=(
            "Device for CNN training and evaluation. 'auto' (default) "
            "selects CUDA if available, then MPS, then CPU. Only "
            "effective with --classifier-type cnn."
        ),
    )

    # Visualisation
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help=(
            "Disable generation of visualisation artefacts (confusion "
            "matrix, ROC curve, PR curve, feature importance, dashboard)."
        ),
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

    # Capture the base output dir *before* it is overwritten by
    # _build_run_dir — the feature/slice cache lives here so it is
    # shared across all runs rather than duplicated inside each run dir.
    base_output_dir = args.output_dir

    # Build structured run directory unless --flat-output is active.
    if not args.flat_output:
        args.output_dir = _build_run_dir(
            output_dir=args.output_dir,
            classifier_type=args.classifier_type,
            tasks=args.tasks,
            run_name=getattr(args, "run_name", None),
        )

    # Resolve --quick-test → --n-cases
    n_cases_limit: Optional[int] = args.n_cases
    if args.quick_test and n_cases_limit is None:
        n_cases_limit = 10

    logger.info("=" * 60)
    if n_cases_limit is not None:
        logger.info(f"MAMA-MIA Classifier Training Pipeline  [LIMITED: {n_cases_limit} cases]")
    else:
        logger.info("MAMA-MIA Classifier Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data directory:   {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Classifier type:  {args.classifier_type}")
    if args.classifier_type == "radiomics" and args.radiomics_model != "all":
        logger.info(f"Radiomics model:  {args.radiomics_model}")
    logger.info(f"Tasks:            {args.tasks}")
    logger.info(f"MRI phase:        {args.phase}")
    logger.info(f"Val ratio:        {args.val_ratio}")
    logger.info(f"CV folds:         {args.cv_folds}")
    logger.info(f"Seed:             {args.seed}")
    if args.slice_mode is not None:
        logger.info(f"Slice mode:       {args.slice_mode}")
        if args.slice_mode == SliceMode.MULTI_SLICE.value:
            logger.info(f"Num slices:       {args.n_slices}")
    else:
        logger.info("Slice mode:       3D (full volume)")
    if getattr(args, "dual_phase", False):
        logger.info("Dual-phase:       ENABLED (phase 0 + phase 1)")
    if args.classifier_type == "cnn":
        logger.info(f"CNN model:        {args.cnn_model}")
        logger.info(f"CNN image size:   {args.cnn_image_size}")
        logger.info(f"CNN epochs:       {args.cnn_epochs}")
        logger.info(f"CNN batch size:   {args.cnn_batch_size}")
        logger.info(f"CNN LR:           {args.cnn_lr}")
        if args.cnn_mask_channel:
            logger.info("CNN mask channel: ENABLED (+1 channel for mask)")
        # Auto-default slice mode to all_tumor for CNN
        if args.slice_mode is None:
            args.slice_mode = SliceMode.ALL_TUMOR.value
            logger.info(
                "Slice mode auto-set to 'all_tumor' for CNN training."
            )
    if args.evaluate_test_set:
        logger.info("Test-set eval:    ENABLED")
    if n_cases_limit is not None:
        logger.info(f"Case limit:       {n_cases_limit} {'(--quick-test)' if args.quick_test else ''}")

    # Set default cache dir — anchored to the *base* output dir so it is
    # shared across versioned run directories instead of being duplicated
    # inside each run folder.
    if args.cache_dir is None:
        args.cache_dir = base_output_dir / "feature_cache"

    # Clear cache if requested
    if args.clear_cache and args.cache_dir.exists():
        import shutil
        n_npy = sum(1 for _ in args.cache_dir.glob("*.npy"))
        n_npz = sum(1 for _ in args.cache_dir.glob("*.npz"))
        n_dirs = sum(
            1 for d in args.cache_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        shutil.rmtree(args.cache_dir)
        logger.info(
            f"Cleared feature cache ({n_npy} .npy, {n_npz} .npz, "
            f"{n_dirs} dirs): {args.cache_dir}"
        )

    # Create visualisation output directory
    viz_dir = args.output_dir / "visualizations"

    # 1. Load clinical data
    logger.info("\n--- Step 1: Loading clinical data ---")
    clinical_df = load_clinical_data(args.data_dir, args.clinical_data)

    # Detect test-set split if requested
    test_patient_ids_global: list[str] = []
    train_patient_ids_global: list[str] = []
    if args.evaluate_test_set:
        logger.info("\n--- Detecting MAMA-MIA train/test split ---")
        train_patient_ids_global, test_patient_ids_global = split_train_test_patients(
            clinical_df,
            split_column=args.split_column,
            test_split_values=getattr(args, "test_split_values", None),
            split_csv=getattr(args, "split_csv", None),
            data_dir=args.data_dir,
        )
        if len(test_patient_ids_global) == 0:
            logger.error(
                "=" * 60 + "\n"
                "TEST-SET EVALUATION ABORTED: No test-set patients found.\n"
                "\n"
                "No 'train_test_splits.csv' was found in --data-dir, and\n"
                "the clinical data does not contain a recognisable train/test\n"
                "split column.\n"
                "\n"
                f"CSV files tried: {SPLIT_CSV_CANDIDATES}\n"
                f"Column candidates tried: {SPLIT_COLUMN_CANDIDATES}\n"
                f"Columns in data: {list(clinical_df.columns)}\n"
                "\n"
                "Hints:\n"
                "  1. Place 'train_test_splits.csv' (with train_split and\n"
                "     test_split columns) in the --data-dir folder.\n"
                "  2. Or use --split-csv /path/to/train_test_splits.csv\n"
                "  3. Or use --split-column <col> --test-split-values <val>\n"
                "\n"
                "Training will proceed WITHOUT test-set evaluation.\n"
                + "=" * 60
            )
        else:
            logger.info(
                f"Found {len(test_patient_ids_global)} test-set patients."
            )

    # Training report to save at the end
    report: dict[str, Any] = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "classifier_type": args.classifier_type,
        "phase": args.phase,
        "val_ratio": args.val_ratio,
        "cv_folds": args.cv_folds,
        "seed": args.seed,
        "slice_mode": args.slice_mode,
        "dual_phase": getattr(args, "dual_phase", False),
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

        # If evaluating test set, exclude test patients from training
        if args.evaluate_test_set and test_patient_ids_global:
            test_set = set(test_patient_ids_global)
            train_mask = [pid not in test_set for pid in patient_ids]
            patient_ids_train = [
                pid for pid, keep in zip(patient_ids, train_mask) if keep
            ]
            labels_train = labels[train_mask]
            logger.info(
                f"Excluded {len(patient_ids) - len(patient_ids_train)} test "
                f"patients from training set."
            )
            patient_ids = patient_ids_train
            labels = labels_train

        # Apply case limit if set (--quick-test or --n-cases)
        if n_cases_limit is not None and len(patient_ids) > n_cases_limit:
            logger.info(
                f"Limiting to {n_cases_limit} cases out of {len(patient_ids)} "
                f"for task '{task}'"
            )
            patient_ids = patient_ids[:n_cases_limit]
            labels = labels[:n_cases_limit]

        # ---- CNN branch --------------------------------------------------
        if args.classifier_type == "cnn" and task != "tumor_roi":
            from eval.train_cnn_classifier import (
                DEFAULT_CNN_NUM_WORKERS,
                train_cnn_pipeline,
            )

            _cnn_nw = (
                getattr(args, "cnn_num_workers", None)
                if getattr(args, "cnn_num_workers", None) is not None
                else DEFAULT_CNN_NUM_WORKERS
            )

            # For contrast task, override dual_phase and phase so the CNN
            # pipeline extracts slices from both phases with appropriate
            # labels.  See ``train_cnn_pipeline(contrast_mode=True)`` in
            # train_cnn_classifier.py.
            _cnn_contrast = task == "contrast"

            cnn_model, cnn_name, cnn_metrics = train_cnn_pipeline(
                task=task,
                patient_ids=patient_ids,
                labels=labels,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                images_dir=args.images_dir,
                segmentations_dir=args.segmentations_dir,
                phase=args.phase,
                slice_mode=args.slice_mode or "all_tumor",
                n_slices=args.n_slices,
                val_ratio=args.val_ratio,
                cv_folds=args.cv_folds,
                seed=args.seed,
                model_name=args.cnn_model,
                image_size=args.cnn_image_size,
                num_epochs=args.cnn_epochs,
                batch_size=args.cnn_batch_size,
                learning_rate=args.cnn_lr,
                patience=args.cnn_patience,
                no_viz=args.no_viz,
                evaluate_test_set=(
                    args.evaluate_test_set and bool(test_patient_ids_global)
                ),
                test_patient_ids=test_patient_ids_global or None,
                clinical_df=clinical_df,
                use_mask_channel=args.cnn_mask_channel,
                dual_phase=getattr(args, "dual_phase", False),
                device=getattr(args, "device", None),
                cache_dir=getattr(args, "cache_dir", None),
                contrast_mode=_cnn_contrast,
                num_workers=_cnn_nw,
            )

            task_elapsed = time.time() - task_start
            cnn_report: dict[str, Any] = {
                "classifier_type": "cnn",
                "best_model": cnn_name,
                "val_metrics": cnn_metrics,
                "n_patients": len(patient_ids),
                "model_path": str(
                    args.output_dir
                    / f"{task}_classifier_cnn.pt"
                ),
                "elapsed_seconds": round(task_elapsed, 1),
            }
            # Propagate CNN test-set metrics into the report so the
            # final summary picks them up.
            if "test_auroc" in cnn_metrics:
                cnn_report["test_metrics"] = {
                    "auroc": cnn_metrics["test_auroc"],
                    "balanced_accuracy": cnn_metrics[
                        "test_balanced_accuracy"
                    ],
                    "loss": cnn_metrics.get("test_loss", float("nan")),
                }
            report["tasks"][task] = cnn_report
            logger.info(f"Task '{task}' (CNN) completed in {task_elapsed:.1f}s")
            continue
        # ---- End CNN branch ----------------------------------------------

        # 2b. Extract features (radiomics path)
        if args.classifier_type == "cnn" and task == "tumor_roi":
            logger.warning(
                "Task 'tumor_roi' does not support CNN training "
                "(radiomics-only to avoid position confound). "
                "Falling back to radiomics pipeline."
            )
        logger.info("\n--- Step 3: Extracting radiomic features ---")

        if task == "contrast":
            # Contrast task: extract from both phases and combine.
            feature_matrix, valid_labels, valid_pids, valid_indices = (
                create_contrast_dataset(
                    patient_ids=patient_ids,
                    data_dir=args.data_dir,
                    images_dir=args.images_dir,
                    segmentations_dir=args.segmentations_dir,
                    cache_dir=args.cache_dir,
                    n_workers=args.n_workers,
                    slice_mode=args.slice_mode,
                    n_slices=args.n_slices,
                )
            )
        elif task == "tumor_roi":
            # Tumor ROI task: extract tumor + mirrored ROI features.
            feature_matrix, valid_labels, valid_pids, valid_indices = (
                create_tumor_roi_dataset(
                    patient_ids=patient_ids,
                    data_dir=args.data_dir,
                    images_dir=args.images_dir,
                    segmentations_dir=args.segmentations_dir,
                    phase=args.phase,
                    cache_dir=args.cache_dir,
                    n_workers=args.n_workers,
                    slice_mode=args.slice_mode,
                    n_slices=args.n_slices,
                )
            )
        else:
            feature_matrix, valid_pids, valid_indices = extract_features_for_patients(
                patient_ids=patient_ids,
                data_dir=args.data_dir,
                images_dir=args.images_dir,
                segmentations_dir=args.segmentations_dir,
                phase=args.phase,
                cache_dir=args.cache_dir,
                n_workers=args.n_workers,
                slice_mode=args.slice_mode,
                n_slices=args.n_slices,
                dual_phase=getattr(args, "dual_phase", False),
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
        save_all = getattr(args, "save_all_models", False)
        if args.cv_folds > 0:
            # Cross-validation mode
            logger.info(f"\n--- Step 4: Training with {args.cv_folds}-fold CV ---")
            best_model, best_name, best_metrics, all_models = train_with_cross_validation(
                X=feature_matrix,
                y=valid_labels,
                task=task,
                n_folds=args.cv_folds,
                seed=args.seed,
                model_filter=args.radiomics_model,
                return_all=save_all,
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

            best_model, best_name, best_metrics, all_models = train_with_model_selection(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                task=task,
                model_filter=args.radiomics_model,
                return_all=save_all,
            )

        # 2d. Save model
        logger.info("\n--- Step 5: Saving model ---")
        model_path = save_model(best_model, task, args.output_dir)

        # Save all models when --save-all-models is active
        if save_all and all_models:
            logger.info(
                f"Saving all {len(all_models)} models for ensemble support..."
            )
            for idx, (m, m_name, m_metrics) in enumerate(all_models, start=1):
                # Sanitise config name for filename
                safe_name = (
                    m_name.lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(",", "")
                    .replace("=", "")
                )[:40]
                save_model(m, task, args.output_dir, suffix=f"_{safe_name}")

        # 2e. Generate validation visualisations
        if not args.no_viz:
            logger.info("\n--- Step 5b: Generating validation visualisations ---")
            try:
                from eval.training_visualization import TrainingVisualizer
                viz = TrainingVisualizer(output_dir=viz_dir / task)

                # Compute predictions on validation set (or full set for CV)
                if args.cv_folds > 0:
                    _viz_X, _viz_y = feature_matrix, valid_labels
                    _viz_label = "Full (CV retrained)"
                else:
                    _viz_X, _viz_y = X_val, y_val
                    _viz_label = "Validation"

                _viz_proba = best_model.predict_proba(_viz_X)
                _viz_score = (
                    _viz_proba[:, 1] if _viz_proba.ndim == 2 else _viz_proba
                )
                _viz_pred = (_viz_score >= 0.5).astype(np.int64)

                viz_paths = viz.generate_all(
                    y_true=_viz_y,
                    y_pred=_viz_pred,
                    y_score=_viz_score,
                    model=best_model,
                    task=task,
                    dataset_label=_viz_label,
                )
                logger.info(
                    f"Generated {len(viz_paths)} visualisation artefacts for "
                    f"task '{task}'"
                )
            except Exception as e:
                logger.warning(f"Visualisation generation failed: {e}")

        # 2f. Evaluate on MAMA-MIA test set (if enabled)
        test_metrics: Optional[dict[str, float]] = None
        if args.evaluate_test_set and test_patient_ids_global:
            logger.info(
                f"\n--- Step 6: Evaluating on MAMA-MIA test set "
                f"({len(test_patient_ids_global)} patients) ---"
            )
            try:
                if task == "contrast":
                    # Contrast: build test features from both phases.
                    test_features, test_labels_valid, _, _ = (
                        create_contrast_dataset(
                            patient_ids=test_patient_ids_global,
                            data_dir=args.data_dir,
                            images_dir=args.images_dir,
                            segmentations_dir=args.segmentations_dir,
                            cache_dir=args.cache_dir,
                            n_workers=args.n_workers,
                            slice_mode=args.slice_mode,
                            n_slices=args.n_slices,
                        )
                    )
                elif task == "tumor_roi":
                    # Tumor ROI: build test features from tumor + mirror.
                    test_features, test_labels_valid, _, _ = (
                        create_tumor_roi_dataset(
                            patient_ids=test_patient_ids_global,
                            data_dir=args.data_dir,
                            images_dir=args.images_dir,
                            segmentations_dir=args.segmentations_dir,
                            phase=args.phase,
                            cache_dir=args.cache_dir,
                            n_workers=args.n_workers,
                            slice_mode=args.slice_mode,
                            n_slices=args.n_slices,
                        )
                    )
                else:
                    # Subtype tasks: create labels for test patients
                    test_pids_task, test_labels_task = create_labels(
                        clinical_df[
                            clinical_df["patient_id"].isin(
                                test_patient_ids_global
                            )
                        ].copy(),
                        task,
                    )

                    if len(test_pids_task) == 0:
                        logger.warning(
                            f"No test patients with valid labels for '{task}'."
                        )
                        # Leave empty so the len < 2 check below skips.
                        test_features = np.empty((0, 0))
                        test_labels_valid = np.array([], dtype=np.int64)
                    else:
                        # Extract test features
                        test_features, test_valid_pids, test_valid_idx = (
                            extract_features_for_patients(
                                patient_ids=test_pids_task,
                                data_dir=args.data_dir,
                                images_dir=args.images_dir,
                                segmentations_dir=args.segmentations_dir,
                                phase=args.phase,
                                cache_dir=args.cache_dir,
                                n_workers=args.n_workers,
                                slice_mode=args.slice_mode,
                                n_slices=args.n_slices,
                                dual_phase=getattr(args, "dual_phase", False),
                            )
                        )
                        test_labels_valid = test_labels_task[test_valid_idx]

                if len(test_labels_valid) < 2:
                    logger.warning(
                        f"Fewer than 2 valid test patients for '{task}'."
                    )
                else:
                    # Predict on test set
                    test_metrics = evaluate_model(
                        best_model, test_features, test_labels_valid
                    )
                    logger.info(
                        f"TEST SET — {task}: "
                        f"AUROC={test_metrics['auroc']:.4f}, "
                        f"Bal.Acc={test_metrics['balanced_accuracy']:.4f}"
                    )

                    # Generate test-set visualisations
                    if not args.no_viz:
                        try:
                            test_viz = TrainingVisualizer(
                                output_dir=viz_dir / f"{task}_test"
                            )
                            test_proba = best_model.predict_proba(
                                test_features
                            )
                            test_score = (
                                test_proba[:, 1]
                                if test_proba.ndim == 2
                                else test_proba
                            )
                            test_pred = (
                                (test_score >= 0.5).astype(np.int64)
                            )

                            test_viz_paths = test_viz.generate_all(
                                y_true=test_labels_valid,
                                y_pred=test_pred,
                                y_score=test_score,
                                model=best_model,
                                task=task,
                                dataset_label="MAMA-MIA Test",
                            )
                            logger.info(
                                f"Generated {len(test_viz_paths)} test-set "
                                f"visualisations for '{task}'"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Test-set visualisation failed: {e}"
                            )

            except Exception as e:
                logger.warning(f"Test-set evaluation failed for '{task}': {e}")

        task_elapsed = time.time() - task_start
        task_report: dict[str, Any] = {
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
        if args.slice_mode is not None:
            task_report["slice_mode"] = args.slice_mode
        if test_metrics is not None:
            task_report["test_metrics"] = test_metrics

        report["tasks"][task] = task_report

        logger.info(f"Task '{task}' completed in {task_elapsed:.1f}s")

    # 3. Save training report
    save_training_report(report, args.output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Training pipeline complete!")
    logger.info(f"Models saved in: {args.output_dir}")
    if not args.no_viz:
        logger.info(f"Visualisations in: {viz_dir}")

    # Print a clear test-set summary so it can't be missed.
    if args.evaluate_test_set:
        any_test_metrics = any(
            "test_metrics" in report["tasks"].get(t, {})
            for t in args.tasks
        )
        if any_test_metrics:
            logger.info("-" * 60)
            logger.info("TEST-SET RESULTS:")
            for t in args.tasks:
                tm = report["tasks"].get(t, {}).get("test_metrics")
                if tm is not None:
                    logger.info(
                        f"  {t}: AUROC={tm['auroc']:.4f}, "
                        f"Bal.Acc={tm['balanced_accuracy']:.4f}"
                    )
                else:
                    logger.info(f"  {t}: (no test results)")
        else:
            logger.warning(
                "TEST-SET RESULTS: No test-set metrics were produced. "
                "Check warnings above."
            )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
