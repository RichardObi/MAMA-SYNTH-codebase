# mama-sia-eval

Evaluation suite for the **MAMA-SYNTH Challenge** — pre-contrast to post-contrast breast DCE-MRI synthesis. This package implements all four metric groups used for ranking participant submissions on [Grand Challenge](https://grand-challenge.org/).

## Challenge Overview

The MAMA-SYNTH challenge evaluates generative models that translate pre-contrast to post-contrast breast DCE-MRI images. Synthesized images are assessed across **four equally-weighted metric groups** (25 % each):

| Group | Metrics | Scope |
|---|---|---|
| **Classification (CLF)** | AUROC, Balanced Accuracy | Molecular subtype prediction (TNBC / Luminal) on synthesized images |
| **Segmentation (SEG)** | Dice (DSC), Hausdorff Distance 95 (HD95) | Tumor segmentation quality on synthesized images |
| **Tumor ROI (ROI)** | MSE, LPIPS, FRD | Image fidelity within dilated tumor mask |
| **Full Image (FULL)** | MSE, LPIPS, FRD | Image fidelity over the full breast region |

Rankings use **Borda-style hierarchical rank aggregation** with tie-break priority: ROI → CLF → SEG → FULL.

## What's New in v0.5.0

- **2D slice extraction** — `slice_extraction.py` with `SliceMode.MAX_TUMOR`, `CENTER_TUMOR`, `MULTI_SLICE`, and `MIDDLE` strategies for automated 2D slice extraction from 3D NIfTI volumes, with z-score normalisation
- **`--slice-mode` flag** — integrate 2D slice extraction into the classifier training pipeline (`--slice-mode max_tumor`)
- **MAMA-MIA test set evaluation** — `--evaluate-test-set` flag auto-detects the train/test split column in the clinical Excel and evaluates the trained model on left-out test patients
- **Training visualisations** — `TrainingVisualizer` class generates confusion matrices, ROC curves, precision–recall curves, feature importance plots, classification reports, and a combined dashboard figure
- **`--no-viz` flag** — disable automatic visusalisation generation during training
- **`--split-column` flag** — explicitly specify the column name containing train/test split labels
- **203+ tests** — 63 new tests for slice extraction, training visualisation, split detection, and new CLI flags

## What's New in v0.3.0

- **Dataset-level z-score normalization** — `DatasetNormalizer` fits globally on all GT images (as per challenge protocol)
- **PNG support** — 16-bit PNG slices now accepted alongside NIfTI/MHA
- **Missing prediction imputation** — worst observed score assigned per the spec
- **Batch parallel radiomics** — `ProcessPoolExecutor`-based parallel feature extraction
- **Disk caching** — SHA-256 keyed `.npz` cache for radiomic features
- **nnUNet wrapper** — `NNUNetSegmenter` for organizer-trained segmentation models
- **Pre-trained classifier loading** — load `.pkl` models from `--clf-model-dir`
- **CSV label support** — labels.csv format alongside JSON
- **tqdm progress bars** — optional, with graceful fallback
- **Visualization module** — summary tables, bar charts, radar plots, segmentation overlays
- **Web interface** — Streamlit dashboard for interactive evaluation
- **Artificial test data generator** — `generate_test_data.py` for rapid pipeline testing
- **Classifier training pipeline** — train TNBC/Luminal classifiers on the MAMA-MIA dataset with model selection
- **160+ tests** — comprehensive unit, integration, and E2E test coverage

## Metrics

### Image-to-Image Fidelity
- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **NMSE** — Normalized Mean Squared Error
- **PSNR** — Peak Signal-to-Noise Ratio (dB)
- **SSIM** — Structural Similarity Index
- **NCC** — Normalized Cross-Correlation

### Perceptual & Distributional
- **LPIPS** — Learned Perceptual Image Patch Similarity (requires `torch` + `lpips`)
- **FRD** — Fréchet Radiomics Distance using IBSI-compliant pyradiomics features (requires `pyradiomics`)

### Downstream Tasks
- **Dice / HD95** — Segmentation overlap & boundary distance
- **AUROC / Balanced Accuracy** — Classification of molecular subtypes

## Installation

### Core (image-to-image metrics only)

```bash
pip install git+https://github.com/RichardObi/mama-sia-eval
```

### With all optional dependencies

```bash
pip install "mama-sia-eval[all] @ git+https://github.com/RichardObi/mama-sia-eval"
```

### Individual extras

```bash
pip install "mama-sia-eval[frd]"              # FRD (pyradiomics)
pip install "mama-sia-eval[lpips]"             # LPIPS (torch + lpips)
pip install "mama-sia-eval[classification]"    # Classification (xgboost)
pip install "mama-sia-eval[segmentation]"      # nnUNet segmentation
pip install "mama-sia-eval[viz]"               # Visualization (matplotlib + plotly)
pip install "mama-sia-eval[web]"               # Web interface (streamlit)
pip install "mama-sia-eval[progress]"          # tqdm progress bars
```

### From source (development)

```bash
git clone https://github.com/RichardObi/mama-sia-eval.git
cd mama-sia-eval
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface

```bash
# Basic evaluation (image-to-image metrics)
python -m mama_sia_eval \
    --ground-truth-path /path/to/ground-truth \
    --predictions-path /path/to/predictions \
    --output-file metrics.json

# Full evaluation with all four metric groups
python -m mama_sia_eval \
    --ground-truth-path /path/to/ground-truth \
    --predictions-path /path/to/predictions \
    --masks-path /path/to/tumor-masks \
    --labels-path /path/to/labels.csv \
    --output-file metrics.json \
    --roi-margin-mm 10.0 \
    --seg-model-path /path/to/nnunet-model \
    --clf-model-dir /path/to/classifiers \
    --cache-dir /path/to/feature-cache
```

#### CLI Options

| Flag | Default | Description |
|---|---|---|
| `-g, --ground-truth-path` | `/opt/app/ground-truth/` | Directory with reference post-contrast images |
| `-i, --predictions-path` | `/input/` | Directory with generated post-contrast images |
| `-o, --output-file` | `/output/metrics.json` | Output JSON file path |
| `-m, --masks-path` | `None` | Tumor segmentation masks directory (enables ROI & SEG) |
| `-l, --labels-path` | `None` | JSON/CSV file with molecular subtype labels (enables CLF) |
| `--roi-margin-mm` | `10.0` | Dilation margin (mm) around tumor mask for ROI |
| `--seg-model-path` | `None` | Pre-trained nnUNet model directory for segmentation |
| `--clf-model-dir` | `None` | Directory with pre-trained `.pkl` classifier models |
| `--cache-dir` | `None` | Feature cache directory (speeds up repeated FRD runs) |
| `--disable-lpips` | `false` | Skip LPIPS (if torch unavailable) |
| `--disable-frd` | `false` | Skip FRD (if pyradiomics unavailable) |
| `--disable-segmentation` | `false` | Skip segmentation evaluation |
| `--disable-classification` | `false` | Skip classification evaluation |
| `-v, --verbose` | `false` | Verbose logging |

### Generate Test Data

```bash
# Create an artificial dataset for pipeline testing
python -m mama_sia_eval.generate_test_data \
    --output-dir ./test_data \
    --n-cases 20 \
    --shape 64 64 \
    --format nii.gz
```

### Visualize Results

```python
from mama_sia_eval.visualization import ResultVisualizer

viz = ResultVisualizer("metrics.json", output_dir="reports")
viz.generate_all()  # Creates summary tables, bar charts, radar plots, CSVs
```

### Web Interface

```bash
pip install streamlit plotly pandas
streamlit run src/mama_sia_eval/webapp.py
```

The dashboard allows participants to:
- Configure and run the evaluation pipeline interactively
- Upload existing results JSON for visualization
- View interactive metric charts (Plotly) and summary tables
- Download results and report files

### Python API

```python
from mama_sia_eval import MamaSiaEval

evaluator = MamaSiaEval(
    ground_truth_path="path/to/ground-truth",
    predictions_path="path/to/predictions",
    output_file="path/to/metrics.json",
    masks_path="path/to/tumor-masks",       # optional
    labels_path="path/to/labels.csv",       # optional (JSON or CSV)
    roi_margin_mm=10.0,
    enable_lpips=True,
    enable_frd=True,
    enable_segmentation=True,
    enable_classification=True,
    seg_model_path="path/to/nnunet-model",  # optional
    clf_model_dir="path/to/classifiers",    # optional
    cache_dir="path/to/feature-cache",      # optional
)
results = evaluator.evaluate()
```

### Using Individual Metrics

```python
import numpy as np
from mama_sia_eval import compute_ssim, compute_psnr, compute_dice, compute_hd95

prediction = np.load("prediction.npy")
ground_truth = np.load("ground_truth.npy")

ssim = compute_ssim(prediction, ground_truth)
psnr = compute_psnr(prediction, ground_truth)
print(f"SSIM: {ssim:.4f}, PSNR: {psnr:.2f} dB")

# Segmentation metrics
pred_mask = prediction > 0.5
gt_mask = ground_truth > 0.5
dice = compute_dice(pred_mask, gt_mask)
hd95 = compute_hd95(pred_mask, gt_mask, voxel_spacing=(1.0, 1.0))
print(f"Dice: {dice:.4f}, HD95: {hd95:.2f} mm")
```

### FRD (Fréchet Radiomics Distance)

```python
from mama_sia_eval.frd import compute_frd_from_features
import numpy as np

# Pre-extracted radiomics feature matrices (n_samples x n_features)
real_features = np.load("real_features.npy")
synth_features = np.load("synth_features.npy")
frd = compute_frd_from_features(real_features, synth_features)
print(f"FRD: {frd:.4f}")
```

### Ranking Submissions

```python
from mama_sia_eval.ranking import rank_submissions

submissions = {
    "team_a": {"mse_roi": 5.0, "frd_roi": 2.0, "mse_full": 4.0, ...},
    "team_b": {"mse_roi": 8.0, "frd_roi": 3.0, "mse_full": 6.0, ...},
}
rankings = rank_submissions(submissions)
for team, info in sorted(rankings.items(), key=lambda x: x[1]["overall_rank"]):
    print(f"#{info['overall_rank']}: {team}")
```

## Input Format

### Images
The evaluation accepts medical images in all formats supported by SimpleITK:
- NIfTI (`.nii`, `.nii.gz`)
- MetaImage (`.mha`, `.mhd`)
- PNG (`.png`) — 16-bit grayscale

Ground truth and prediction files are matched by filename (excluding extension).

### Tumor Masks
Binary segmentation masks in the same format as images. The mask filename must match the corresponding image filename.

### Labels File

**JSON** format:
```json
{
    "case_001": {"tnbc": 1, "luminal": 0},
    "case_002": {"tnbc": 0, "luminal": 1}
}
```

**CSV** format:
```csv
case_id,tnbc,luminal
case_001,1,0
case_002,0,1
```

## Output Format

```json
{
    "aggregate": {
        "mae": {"mean": 0.1, "std": 0.05, "min": 0.01, "max": 0.2, "n_samples": 10},
        "mse": {"mean": 0.02, "std": 0.01, ...},
        "psnr": {"mean": 30.5, "std": 2.1, ...},
        "ssim": {"mean": 0.95, "std": 0.02, ...},
        ...
    },
    "full_image": {
        "mse": {"mean": ..., "std": ...},
        "lpips": {"mean": ..., "std": ...},
        "frd": 12.34
    },
    "roi": {
        "mse": {"mean": ..., "std": ...},
        "lpips": {"mean": ..., "std": ...},
        "frd": 8.91
    },
    "segmentation": {
        "dice": {"mean": 0.82, "std": 0.08, ...},
        "hd95": {"mean": 3.14, "std": 1.2, ...}
    },
    "classification": {
        "auroc_tnbc": 0.85,
        "auroc_luminal": 0.79,
        "balanced_accuracy_tnbc": 0.81,
        "balanced_accuracy_luminal": 0.78
    },
    "missing_predictions": ["case_005"],
    "cases": [
        {"case_id": "case001", "mae": 0.1, "mse": 0.02, ...},
        {"case_id": "case005", "mae": 0.2, "_imputed": true, ...}
    ]
}
```

## Classifier Training

Train molecular subtype classifiers (TNBC and Luminal) on the [MAMA-MIA](https://github.com/RichardObi/MAMA-MIA) dataset. Trained models are saved as `.pkl` files directly compatible with the evaluation pipeline.

### Prerequisites

1. Download the [MAMA-MIA dataset](https://www.synapse.org/Synapse:syn60868042) (1,506 breast cancer DCE-MRI cases)
2. Install with training dependencies:

```bash
pip install "mama-sia-eval[training] @ git+https://github.com/RichardObi/mama-sia-eval"
```

### Expected Dataset Layout

```
mama-mia-dataset/
├── clinical_and_imaging_info.xlsx   # Clinical metadata (sheet: dataset_info)
├── images/
│   ├── ISPY1_1001/
│   │   ├── ISPY1_1001_0000.nii.gz   # Pre-contrast (phase 0)
│   │   └── ISPY1_1001_0001.nii.gz   # First post-contrast (phase 1)
│   ├── ISPY2_100001/
│   │   ├── ISPY2_100001_0000.nii.gz
│   │   └── ISPY2_100001_0001.nii.gz
│   └── ...
└── segmentations/
    ├── ISPY1_1001.nii.gz
    ├── ISPY2_100001.nii.gz
    └── ...
```

### Quick Start (Single Command)

```bash
# Train both classifiers (TNBC and Luminal)
python -m mama_sia_eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir /path/to/trained-models
```

Or using the installed console script:

```bash
mamasia-train \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir /path/to/trained-models
```

### Training CLI Options

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | *required* | Root directory of the MAMA-MIA dataset |
| `--output-dir` | *required* | Where to save trained model `.pkl` files |
| `--clinical-data` | `None` | Path to Excel file (auto-detected in `--data-dir` if omitted) |
| `--images-dir` | `None` | Images directory (default: `<data-dir>/images`) |
| `--segmentations-dir` | `None` | Segmentations directory (default: `<data-dir>/segmentations`) |
| `--tasks` | `tnbc,luminal` | Comma-separated list of tasks to train |
| `--phase` | `1` | DCE-MRI phase to use (0=pre-contrast, 1=first post-contrast) |
| `--val-ratio` | `0.2` | Fraction of data for validation (holdout mode) |
| `--cv-folds` | `0` | Number of cross-validation folds (0 = holdout mode) |
| `--seed` | `42` | Random seed for reproducibility |
| `--cache-dir` | `None` | Cache directory for extracted features |
| `--n-workers` | `1` | Number of parallel workers for feature extraction |
| `--slice-mode` | `None` | 2D extraction strategy: `max_tumor`, `center_tumor`, `multi_slice`, `middle` |
| `--n-slices` | `5` | Number of slices for `multi_slice` mode |
| `--evaluate-test-set` | `false` | Evaluate on MAMA-MIA test split after training |
| `--split-column` | `None` | Column name in clinical Excel for train/test split (auto-detected) |
| `--no-viz` | `false` | Skip generation of visualisation artefacts |
| `--quick-test` | `false` | Quick validation run with 10 cases per task |
| `--n-cases` | `None` | Limit training to first N cases per task |
| `-v, --verbose` | `false` | Verbose logging |

### Training Modes

**Holdout mode** (default): Splits data into train/val sets. Best model is selected by validation AUROC.

```bash
python -m mama_sia_eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --val-ratio 0.2
```

**Cross-validation mode**: Uses k-fold CV to select the best hyperparameters, then retrains on the full dataset.

```bash
python -m mama_sia_eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --cv-folds 5
```

**2D slice extraction**: Extract the most informative 2D slice from each 3D MRI volume before feature extraction. Recommended for classification tasks where a single representative slice captures the essential tumour characteristics.

```bash
# Use the slice with the largest tumour cross-section
python -m mama_sia_eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --slice-mode max_tumor

# Multi-slice feature extraction (concatenate features from 5 slices)
python -m mama_sia_eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --slice-mode multi_slice \
    --n-slices 5
```

**Test-set evaluation**: Train on the MAMA-MIA training split and automatically evaluate on the test split.

```bash
python -m mama_sia_eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --evaluate-test-set
```

### 2D Slice Extraction (Python API)

```python
from mama_sia_eval.slice_extraction import extract_2d_slice, SliceMode

# Load a 3D NIfTI volume (D, H, W)
import SimpleITK as sitk
import numpy as np

vol = sitk.GetArrayFromImage(sitk.ReadImage("path/to/volume.nii.gz"))
mask = sitk.GetArrayFromImage(sitk.ReadImage("path/to/mask.nii.gz")).astype(bool)

# Extract the slice with the largest tumour area
img_2d, mask_2d, slice_idx = extract_2d_slice(
    vol, mask, mode=SliceMode.MAX_TUMOR, normalize=True
)
print(f"Selected slice {slice_idx}, shape: {img_2d.shape}")
```

### Training Visualisations (Python API)

```python
from mama_sia_eval.training_visualization import TrainingVisualizer

viz = TrainingVisualizer(output_dir="./reports")
viz.generate_all(
    y_true=y_true, y_pred=y_pred, y_score=y_score,
    model=trained_model, task="tnbc",
    dataset_label="Validation",
)
# Creates: confusion_matrix_tnbc.png, roc_curve_tnbc.png,
#          pr_curve_tnbc.png, feature_importance_tnbc.png,
#          dashboard_tnbc.png, classification_report_tnbc.{txt,json},
#          confusion_matrix_tnbc.json
```

### Training Output

After training, the output directory contains:

```
trained-models/
├── tnbc_classifier.pkl              # TNBC classifier (pickle format)
├── luminal_classifier.pkl           # Luminal classifier (pickle format)
├── training_report.json             # Training metadata, metrics, and config
├── feature_cache/                   # Cached feature vectors (per patient)
└── visualizations/
    ├── tnbc/                        # Validation-set visualisations
    │   ├── confusion_matrix_tnbc.png
    │   ├── confusion_matrix_tnbc.json
    │   ├── roc_curve_tnbc.png
    │   ├── pr_curve_tnbc.png
    │   ├── feature_importance_tnbc.png
    │   ├── classification_report_tnbc.txt
    │   ├── classification_report_tnbc.json
    │   └── dashboard_tnbc.png       # Combined 2×2 dashboard
    ├── tnbc_test/                   # Test-set visualisations (if --evaluate-test-set)
    │   └── ...
    ├── luminal/
    │   └── ...
    └── luminal_test/
        └── ...
```

The `.pkl` files are directly usable with the evaluation pipeline:

```bash
python -m mama_sia_eval \
    --ground-truth-path /path/to/ground-truth \
    --predictions-path /path/to/predictions \
    --clf-model-dir /path/to/trained-models \
    --labels-path /path/to/labels.csv \
    --output-file metrics.json
```

### Model Selection

The training pipeline tries multiple model configurations:
- **XGBoost** (5 configs): varying `n_estimators`, `max_depth`, `learning_rate`, `subsample`
- **Random Forest** (3 configs): varying `n_estimators`, `max_depth`, `min_samples_split`

If XGBoost is not installed, only Random Forest configurations are tried. The best model is selected by validation AUROC (holdout) or mean cross-validation AUROC.

### Feature Extraction

Training uses the same radiomic feature extraction pipeline as evaluation (`extract_radiomic_features` from `frd.py`), ensuring consistency between training and inference. Features are extracted from the first post-contrast phase (phase 1) by default, using the tumor segmentation mask.

Use `--cache-dir` to cache extracted features across runs:

```bash
python -m mama_sia_eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --cache-dir ./feature-cache
```

## Docker

### Build

```bash
docker build -t mama-sia-eval .
```

### Run

```bash
docker run --rm \
    -v /path/to/ground-truth:/opt/app/ground-truth:ro \
    -v /path/to/predictions:/input:ro \
    -v /path/to/output:/output \
    mama-sia-eval
```

### Run with all inputs

```bash
docker run --rm \
    -v /path/to/ground-truth:/opt/app/ground-truth:ro \
    -v /path/to/predictions:/input:ro \
    -v /path/to/masks:/opt/app/masks:ro \
    -v /path/to/labels.csv:/opt/app/labels.csv:ro \
    -v /path/to/output:/output \
    mama-sia-eval \
    --masks-path /opt/app/masks \
    --labels-path /opt/app/labels.csv
```

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=mama_sia_eval --cov-report=term-missing

# E2E integration tests only
pytest tests/test_e2e.py -v

# Specific module
pytest tests/test_frd.py -v
```

### Code Quality

```bash
ruff check src/ tests/
mypy src/
```

## Project Structure

```
mama-sia-eval/
├── src/mama_sia_eval/
│   ├── __init__.py              # Package exports (v0.5.0)
│   ├── __main__.py              # CLI entry point
│   ├── evaluation.py            # Main evaluation pipeline (MamaSiaEval, DatasetNormalizer)
│   ├── metrics.py               # Image-to-image & segmentation metrics
│   ├── frd.py                   # Fréchet Radiomics Distance (batch, cached)
│   ├── classification.py        # Molecular subtype classification
│   ├── train_classifier.py      # Classifier training on MAMA-MIA dataset
│   ├── slice_extraction.py      # 2D slice extraction from 3D NIfTI volumes
│   ├── training_visualization.py # Confusion matrix, ROC, PR, dashboards
│   ├── segmentation.py          # Tumor segmentation (ThresholdSegmenter, NNUNetSegmenter)
│   ├── roi_utils.py             # Tumor ROI extraction & mask dilation
│   ├── ranking.py               # Borda-style rank aggregation
│   ├── visualization.py         # Result visualization (tables, charts, overlays)
│   ├── webapp.py                # Streamlit web interface
│   └── generate_test_data.py    # Artificial test data generator
├── tests/                       # 200+ tests (unit + integration + E2E)
│   ├── conftest.py              # Shared fixtures
│   ├── test_evaluation.py       # Evaluation & normalizer tests
│   ├── test_metrics.py          # Metric tests
│   ├── test_frd.py              # FRD tests
│   ├── test_classification.py   # Classification tests
│   ├── test_segmentation.py     # Segmentation tests
│   ├── test_roi_utils.py        # ROI utility tests
│   ├── test_ranking.py          # Ranking tests
│   ├── test_e2e.py              # End-to-end integration tests
│   ├── test_train_classifier.py # Classifier training tests
│   ├── test_slice_extraction.py # 2D slice extraction tests
│   └── test_training_visualization.py # Training visualisation tests
├── PDF_ANALYSIS.md              # Contradictions & uncertainties analysis
├── Dockerfile                   # Grand Challenge container
├── pyproject.toml               # Package configuration
├── requirements.txt             # Core dependencies
└── requirements_dev.txt         # Development dependencies
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Citation

If you use this evaluation code in your research, please cite the MAMA-SYNTH challenge:

```bibtex
@misc{mama_synth_2025,
    title={MAMA-SYNTH: Pre-contrast to Post-contrast Breast DCE-MRI Synthesis Challenge},
    year={2025},
    url={https://github.com/RichardObi/mama-sia-eval}
}
```
