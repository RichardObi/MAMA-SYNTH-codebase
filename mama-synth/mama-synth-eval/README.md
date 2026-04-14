# mama-synth-eval

Evaluation suite for the **MAMA-SYNTH Challenge** — pre-contrast to post-contrast breast DCE-MRI synthesis. This package implements all four metric groups used for ranking participant submissions on [Grand Challenge](https://grand-challenge.org/).

## Challenge Overview

The MAMA-SYNTH challenge evaluates generative models that translate pre-contrast to post-contrast breast DCE-MRI images. Synthesized images are assessed across **four equally-weighted metric groups** (25 % each):

| Group | Metrics | Scope |
|---|---|---|
| **Full Image (FULL)** | MSE, LPIPS | Global image fidelity over the full breast region |
| **Tumor ROI (ROI)** | SSIM, FRD | Image fidelity within a dilated tumor mask |
| **Classification (CLF)** | AUROC luminal, AUROC TNBC | Molecular subtype prediction on synthesized images |
| **Segmentation (SEG)** | Dice, HD95 | Tumor segmentation quality on synthesized images |

Rankings use **Borda-style hierarchical rank aggregation** with tie-break priority: ROI → CLF → SEG → FULL.

## What's New in v0.9.0

- **Structured output folders** (`--run-name`, `--flat-output`) — training runs are now organised into versioned sub-directories under the output path:
  ```
  trained_models/
  ├── run_001_20250101_120000_radiomics_tnbc_luminal/
  ├── run_002_20250102_093000_cnn_contrast/
  └── latest -> run_002_...
  ```
  Each run directory is auto-numbered and timestamped. A `latest` symlink always points to the most recent run. Use `--run-name my_experiment` to add a custom label, or `--flat-output` to disable versioned directories and write files directly to `--output-dir` (legacy behaviour).

- **Contrast classification** (`--tasks contrast`) — new binary classification task that distinguishes pre-contrast (phase 0) from post-contrast (phase 1) images. Works with both radiomics (`--classifier-type radiomics`) and CNN (`--classifier-type cnn`) backends. For radiomics, features are extracted independently from each phase and combined. For CNN, slices from both phases are extracted with appropriate labels and patient-aware splitting prevents leakage.
  ```bash
  mamasia-train \
      --data-dir /path/to/mama-mia-dataset \
      --output-dir ./trained_models \
      --tasks contrast \
      --classifier-type cnn
  ```

- **Incremental slice caching** — CNN slice extraction now saves each slice individually as a `.npy` file inside a per-patient directory, so even if the process is killed mid-extraction the already-saved slices survive on disk. A `_done` marker file indicates the patient was fully extracted. Incomplete extractions are automatically re-started on the next run. Legacy `.npz` caches are still supported for backwards compatibility.
  ```
  cache_dir/
  ├── ISPY1_1001_ph1_all_tumor_n5_dp0_msk0/
  │   ├── slice_0.npy
  │   ├── slice_1.npy
  │   ├── mask_0.npy
  │   ├── mask_1.npy
  │   └── _done
  └── ISPY1_1001_ph1_all_tumor_n5_dp0_msk0.npz   # legacy (still readable)
  ```

- **Bug fixes**:
  - Fixed version mismatch between `pyproject.toml` and README (now consistently v0.9.0).
  - Fixed `--clear-cache` only removing `.npy` files — now also removes `.npz` files and per-slice cache directories.
  - Removed unused `import hashlib` from `train_cnn_classifier.py`.
  - Removed unused `import sys` from `train_classifier.py`.
  - Removed unused `batch_size_extract` parameter from `extract_slices_for_cnn()`.
  - Fixed hardcoded "4-channel" in mask channel log message — now reflects actual channel count.
  - Fixed inaccurate docstrings for channel counts in `MRISliceDataset` and `train_cnn` (now document both single-phase and dual-phase cases).

## What's New in v0.8.0

- **CSV-based train/test split** — the MAMA-MIA dataset ships a `train_test_splits.csv` file with `train_split` and `test_split` columns listing patient IDs. This is now auto-detected in `--data-dir` and used as the primary split mechanism. Use `--split-csv /path/to/file.csv` to point to a custom location. Falls back to column-based detection when no CSV is found.
- **CNN slice caching** (`--cache-dir`) — extracted 2-D slices are written to disk on the first run and re-loaded on subsequent runs, avoiding repeated NIfTI I/O. Each patient's slices are stored as individual `.npy` files inside a per-patient directory (upgraded from single `.npz` files in v0.9.0), keyed by extraction parameters (phase, slice mode, dual-phase, mask channel), so different configurations use separate caches.
- **GPU / device selection** (`--device`) — new CLI flag for CNN training and evaluation: `auto` (default, selects CUDA → MPS → CPU), `cpu`, `cuda`, or `mps`. Previously the device was always auto-detected with no way to override.
- **MAMA-MIA test split by dataset** (`--test-split-values`) — use arbitrary column values as the test set when no CSV is available. With `--split-column dataset --test-split-values DUKE`, all DUKE patients become the test set.
- **Bug fixes**:
  - Fixed `NameError` in `train_cnn()` — `dual_phase` parameter was referenced but not declared in the function signature.
  - Fixed `evaluate_cnn()` crashing on GPU/MPS — `.cpu()` was missing before `.numpy()` on tensors.
  - Fixed dual-phase evaluation crash when pre-contrast files are partially missing — zero-padding is now applied to maintain consistent feature widths.

## What's New in v0.7.0

- **Dual-phase classification** (`--dual-phase`) — optionally use both pre-contrast (phase 0) and post-contrast (phase 1) images for molecular subtype classification. For radiomics, features from both phases are concatenated (doubling the feature dimension). For CNN, channels from both phases are stacked into a 6-channel input. Disabled by default to preserve challenge emphasis on post-contrast synthesis quality; useful as an ablation baseline. Use `--precontrast-path` to specify the pre-contrast image directory during evaluation.
- **CNN mask channel** (`--cnn-mask-channel`) — add the tumor segmentation mask as a 4th input channel to the CNN classifier, providing spatial guidance about tumor location. The first convolutional layer is expanded from 3→4 channels with zero-initialised mask weights so pretrained ImageNet features are preserved.
- **Radiomics model selection** (`--radiomics-model`) — filter the radiomics classifier family during training: `random_forest`, `logistic_regression`, `svm`, `xgboost`, or `all` (default). Reduces training time when only a specific model family is needed.
- **Ensemble inference** (`--ensemble`, `--save-all-models`) — average predicted probabilities across multiple trained classifiers. During training, `--save-all-models` persists every successfully trained model (not just the best). During evaluation, `--ensemble` loads all model files matching `{task}_classifier*.pkl` and `{task}_classifier*.pt` from `--clf-model-dir` and ensembles their predictions.
- **EnsembleClassifier** — new class in `classification.py` that combines `RadiomicsClassifier` and/or `CNNClassifier` instances, with auto-discovery of model files via `EnsembleClassifier.discover_models()`.
- **300+ tests** — 34 new tests covering dual-phase feature extraction (radiomics + CNN), mask channel, radiomics model selection, ensemble inference, and CLI integration.

## What's New in v0.6.0

- **EfficientNet CNN classifier** — `--classifier-type cnn` trains an EfficientNet-B0 deep learning classifier as a higher-capacity alternative to the radiomics-based models (~0.6 AUROC). Includes ImageNet pre-training, cosine LR scheduling, class-weighted loss, early stopping, and gradient clipping.
- **Synthesis pipeline** — two new CLI commands (`mamasynth-synthesize` and `mamasynth-synthesize-and-evaluate`) wrap medigan's Pix2PixHD model for one-command synthesis and evaluation. Participants can also point `--predictions-dir` at their own model outputs.
- **New optional extras** — `[cnn]` (torch + timm), `[synthesis]` (medigan), `[full]` (everything)
- **Improved radiomics classifiers** — all models now use `Pipeline(StandardScaler → VarianceThreshold → Classifier)` with class-balanced weights; added Logistic Regression (4 configs) and SVM-RBF (3 configs) alongside XGBoost and Random Forest (13 total configs)
- **Radiomics feature caching improvements** — `all_tumor` mode now caches correctly; added `--clear-cache` CLI flag and `allow_pickle=False` security
- **282 tests** — 36 new tests for CNN training, synthesis pipeline, and CLI integration

## What's New in v0.5.0

- **2D slice extraction** — `slice_extraction.py` with `SliceMode.MAX_TUMOR`, `CENTER_TUMOR`, `MULTI_SLICE`, `ALL_TUMOR`, and `MIDDLE` strategies for automated 2D slice extraction from 3D NIfTI volumes, with z-score normalisation
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

Eight equally-weighted metrics — two per task — determine the final ranking:

| Task | Metric | Key in `metrics.json` |
|---|---|---|
| **Full Image** | MSE — Mean Squared Error | `mse_full_image` |
| **Full Image** | LPIPS — Perceptual similarity (requires `torch` + `lpips`) | `lpips_full_image` |
| **Tumor ROI** | SSIM — Structural Similarity Index | `ssim_roi` |
| **Tumor ROI** | FRD — Fréchet Radiomics Distance (requires `pyradiomics`) | `frd_roi` |
| **Classification** | AUROC luminal — Area under ROC, Luminal subtype | `auroc_luminal` |
| **Classification** | AUROC TNBC — Area under ROC, triple-negative subtype | `auroc_tnbc` |
| **Segmentation** | Dice — Overlap coefficient | `dice` |
| **Segmentation** | HD95 — 95th-percentile Hausdorff distance | `hausdorff95` |

All image-based metrics are computed on **dataset-level z-score-normalized** images. Additional per-case statistics (MAE, NMSE, PSNR, NCC) are stored in the `results` array of the output JSON for reference but are **not used for ranking**.

## Installation

### Core (image-to-image metrics only)

```bash
pip install git+https://github.com/RichardObi/MAMA-SYNTH-codebase/tree/main/mama-synth/mama-synth-eval
```

### With all optional dependencies

```bash
pip install "eval[all] @ git+https://github.com/RichardObi/MAMA-SYNTH-codebase/tree/main/mama-synth/mama-synth-eval"
```

### Individual extras

```bash
pip install "eval[frd]"              # FRD (pyradiomics)
pip install "eval[lpips]"             # LPIPS (torch + lpips)
pip install "eval[classification]"    # Classification (xgboost)
pip install "eval[segmentation]"      # nnUNet segmentation
pip install "eval[viz]"               # Visualization (matplotlib + plotly)
pip install "eval[web]"               # Web interface (streamlit)
pip install "eval[progress]"          # tqdm progress bars
pip install "eval[cnn]"              # CNN classifier (torch + timm)
pip install "eval[synthesis]"         # Synthesis pipeline (medigan)
pip install "eval[full]"              # All dependencies
```

### From source (development)

```bash
git clone https://github.com/RichardObi/MAMA-SYNTH-codebase.git
cd MAMA-SYNTH-codebase
cd mama-synth
cd mama-synth-eval
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface

```bash
# Basic evaluation (image-to-image metrics)
python -m eval \
    --ground-truth-path /path/to/ground-truth \
    --predictions-path /path/to/predictions \
    --output-file metrics.json

# Full evaluation with all four metric groups
python -m eval \
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
| `--ensemble` | `false` | Average probabilities across all models in `--clf-model-dir` |
| `--dual-phase` | `false` | Concatenate pre-contrast features for classification |
| `--precontrast-path` | `None` | Pre-contrast images directory for `--dual-phase` |
| `--cache-dir` | `None` | Feature cache directory (speeds up repeated FRD runs) |
| `--disable-lpips` | `false` | Skip LPIPS (if torch unavailable) |
| `--disable-frd` | `false` | Skip FRD (if pyradiomics unavailable) |
| `--disable-segmentation` | `false` | Skip segmentation evaluation |
| `--disable-classification` | `false` | Skip classification evaluation |
| `-v, --verbose` | `false` | Verbose logging |

### Generate Test Data

```bash
# Create an artificial dataset for pipeline testing
python -m eval.generate_test_data \
    --output-dir ./test_data \
    --n-cases 20 \
    --shape 64 64 \
    --format nii.gz
```

### Visualize Results

```python
from eval.visualization import ResultVisualizer

viz = ResultVisualizer("metrics.json", output_dir="reports")
viz.generate_all()  # Creates summary tables, bar charts, radar plots, CSVs
```

### Web Interface

```bash
pip install streamlit plotly pandas
streamlit run src/eval/webapp.py
```

The dashboard allows participants to:
- Configure and run the evaluation pipeline interactively
- Upload existing results JSON for visualization
- View interactive metric charts (Plotly) and summary tables
- Download results and report files

### Python API

```python
from eval import MamaSynthEval

evaluator = MamaSynthEval(
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
from eval import compute_ssim, compute_psnr, compute_dice, compute_hd95

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
from eval.frd import compute_frd_from_features
import numpy as np

# Pre-extracted radiomics feature matrices (n_samples x n_features)
real_features = np.load("real_features.npy")
synth_features = np.load("synth_features.npy")
frd = compute_frd_from_features(real_features, synth_features)
print(f"FRD: {frd:.4f}")
```

### Ranking Submissions

```python
from eval.ranking import rank_submissions

submissions = {
    "team_a": {"mse_full_image": 5.0, "lpips_full_image": 0.12, "ssim_roi": 0.88, "frd_roi": 2.0, "auroc_luminal": 0.82, "auroc_tnbc": 0.79, "dice": 0.84, "hausdorff95": 3.1},
    "team_b": {"mse_full_image": 8.0, "lpips_full_image": 0.21, "ssim_roi": 0.75, "frd_roi": 3.5, "auroc_luminal": 0.71, "auroc_tnbc": 0.68, "dice": 0.77, "hausdorff95": 5.4},
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
    "aggregates": {
        "mse_full_image":   {"mean": 0.02, "std": 0.01},
        "lpips_full_image": {"mean": 0.15, "std": 0.05},
        "ssim_roi":         {"mean": 0.87, "std": 0.06},
        "frd_roi":          8.91,
        "auroc_luminal":    0.79,
        "auroc_tnbc":       0.85,
        "dice":             {"mean": 0.82, "std": 0.08},
        "hausdorff95":      {"mean": 3.14, "std": 1.20}
    },
    "results": [
        {"case_id": "case001", "mse_full_image": 0.02, "ssim_roi": 0.88, "dice": 0.84, ...},
        {"case_id": "case005", "_imputed": true, ...}
    ]
}
```

## Classifier Training

Train molecular subtype classifiers (TNBC and Luminal) on the [MAMA-MIA](https://github.com/LidiaGarrucho/MAMA-MIA) dataset. Trained models are saved as `.pkl` files directly compatible with the evaluation pipeline.

### Prerequisites

1. Download the [MAMA-MIA dataset](https://www.synapse.org/Synapse:syn60868042) (1,506 breast cancer DCE-MRI cases)
2. Install with training dependencies:

```bash
pip install "eval[training] @ git+https://github.com/RichardObi/MAMA-SYNTH-codebase/tree/main/mama-synth/mama-synth-eval/"
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
python -m eval.train_classifier \
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
| `--segmentations-dir` / `--masks-path` | `None` | Path to segmentation masks folder (default: `<data-dir>/segmentations`). Use this when masks are not in the default location. |
| `--tasks` | `tnbc,luminal` | Comma-separated list of tasks: `tnbc`, `luminal`, `contrast` |
| `--phase` | `1` | DCE-MRI phase to use (0=pre-contrast, 1=first post-contrast) |
| `--val-ratio` | `0.2` | Fraction of data for validation (holdout mode) |
| `--cv-folds` | `0` | Number of cross-validation folds (0 = holdout mode) |
| `--seed` | `42` | Random seed for reproducibility |
| `--cache-dir` | `None` | Cache directory for extracted features |
| `--n-workers` | `1` | Number of parallel workers for feature extraction |
| `--slice-mode` | `None` | 2D extraction strategy: `max_tumor`, `center_tumor`, `multi_slice`, `all_tumor`, `middle` |
| `--n-slices` | `5` | Number of slices for `multi_slice` mode |
| `--evaluate-test-set` | `false` | Evaluate on MAMA-MIA test split after training |
| `--split-csv` | `None` | Path to `train_test_splits.csv` (auto-detected in `--data-dir`) |
| `--split-column` | `None` | Column name in clinical Excel for train/test split (auto-detected) |
| `--test-split-values` | `None` | Custom test-set values for `--split-column` (e.g. `DUKE ISPY1`) |
| `--no-viz` | `false` | Skip generation of visualisation artefacts |
| `--quick-test` | `false` | Quick validation run with 10 cases per task |
| `--n-cases` | `None` | Limit training to first N cases per task |
| `--clear-cache` | `false` | Delete cached features before training |
| `--classifier-type` | `radiomics` | Classifier backend: `radiomics` or `cnn` |
| `--cnn-model` | `efficientnet_b0` | timm model name (CNN mode only) |
| `--cnn-image-size` | `224` | Input image size in pixels (CNN mode only) |
| `--cnn-epochs` | `50` | Maximum training epochs (CNN mode only) |
| `--cnn-batch-size` | `32` | Batch size (CNN mode only) |
| `--cnn-lr` | `1e-4` | Initial learning rate (CNN mode only) |
| `--cnn-patience` | `10` | Early-stopping patience in epochs (CNN mode only) |
| `--cnn-mask-channel` | `false` | Add tumor mask as extra CNN input channel |
| `--radiomics-model` | `all` | Radiomics family filter: `all`, `xgboost`, `random_forest`, `logistic_regression`, `svm` |
| `--save-all-models` | `false` | Save all trained models (not just best) for ensemble |
| `--dual-phase` | `false` | Use both phase 0 + phase 1 for classification |
| `--device` | `auto` | Device for CNN training: `auto`, `cpu`, `cuda`, `mps` |
| `--run-name` | `None` | Custom label appended to the versioned run directory name |
| `--flat-output` | `false` | Disable versioned run directories; write directly to `--output-dir` |
| `-v, --verbose` | `false` | Verbose logging |

### Training Modes

**Holdout mode** (default): Splits data into train/val sets. Best model is selected by validation AUROC.

```bash
python -m eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --val-ratio 0.2
```

**Cross-validation mode**: Uses k-fold CV to select the best hyperparameters, then retrains on the full dataset.

```bash
python -m eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --cv-folds 5
```

**2D slice extraction**: Extract the most informative 2D slice from each 3D MRI volume before feature extraction. Recommended for classification tasks where a single representative slice captures the essential tumour characteristics.

```bash
# Use the slice with the largest tumour cross-section
python -m eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --slice-mode max_tumor

# Multi-slice feature extraction (concatenate features from 5 slices)
python -m eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --slice-mode multi_slice \
    --n-slices 5

# All tumour slices: every slice with ≥1 mask voxel becomes a training sample
python -m eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --slice-mode all_tumor
```

**CNN classifier** (EfficientNet): Train a deep learning classifier instead of the radiomics-based models. Automatically defaults to `all_tumor` slice mode.

```bash
# Train an EfficientNet-B0 CNN classifier
mamasia-train \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --classifier-type cnn

# Customise CNN hyperparameters
mamasia-train \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --classifier-type cnn \
    --cnn-model efficientnet_b1 \
    --cnn-epochs 100 \
    --cnn-batch-size 16 \
    --cnn-lr 5e-5 \
    --cnn-patience 15
```

Requires the `[cnn]` extra: `pip install "eval[cnn]"`
**Model selection by family**: Train only a specific classifier family.

```bash
# Train only Random Forest classifiers
mamasia-train \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --radiomics-model random_forest
```

**CNN with mask channel**: Add tumor mask as spatial guidance.

```bash
mamasia-train \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --classifier-type cnn \
    --cnn-mask-channel
```

**Save all models for ensemble**: Persist every trained model.

```bash
mamasia-train \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --save-all-models
```

**Dual-phase classification**: Use both pre-contrast and post-contrast images.

```bash
mamasia-train \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --dual-phase
```

**Contrast classification**: Train a binary classifier to distinguish pre-contrast from post-contrast images (useful as the main MAMA-SYNTH challenge task).

```bash
# Radiomics-based contrast classifier
mamasia-train \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./trained_models \
    --tasks contrast

# CNN-based contrast classifier
mamasia-train \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./trained_models \
    --tasks contrast \
    --classifier-type cnn \
    --cache-dir ./slice_cache
```

**Ensemble evaluation**: Average predictions across all saved models.

```bash
python -m eval \
    --ground-truth-path /path/to/ground-truth \
    --predictions-path /path/to/predictions \
    --clf-model-dir /path/to/trained-models \
    --labels-path /path/to/labels.csv \
    --ensemble \
    --output-file metrics.json
```

**Dual-phase evaluation**: Include pre-contrast features during evaluation.

```bash
python -m eval \
    --ground-truth-path /path/to/ground-truth \
    --predictions-path /path/to/predictions \
    --clf-model-dir /path/to/trained-models \
    --labels-path /path/to/labels.csv \
    --dual-phase \
    --precontrast-path /path/to/precontrast-images \
    --output-file metrics.json
```


**Test-set evaluation**: Train on the MAMA-MIA training split and automatically evaluate on the test split. The split is loaded from `train_test_splits.csv` (auto-detected in `--data-dir`).

```bash
python -m eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --evaluate-test-set
```

**Test-set evaluation with explicit split CSV**: Point to a specific split file.

```bash
python -m eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --evaluate-test-set \
    --split-csv /path/to/train_test_splits.csv
```

**Test-set evaluation by dataset source**: Use specific source datasets (e.g. DUKE) as test set (fallback when no split CSV is available).

```bash
python -m eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --evaluate-test-set \
    --split-column dataset \
    --test-split-values DUKE
```

**GPU training with slice caching**: Train CNN on GPU and cache extracted slices.

```bash
python -m eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --classifier-type cnn \
    --device cuda \
    --cache-dir ./slice_cache
```

### 2D Slice Extraction (Python API)

```python
from eval.slice_extraction import extract_2d_slice, SliceMode

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
from eval.training_visualization import TrainingVisualizer

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

After training, the output directory contains versioned run sub-directories (unless `--flat-output` is used):

```
trained-models/
├── feature_cache/                   # Shared across ALL runs — features are
│   └── ...                          # reused and never duplicated per run
├── latest -> run_002_20250102_093000_cnn_contrast/     # symlink
├── run_001_20250101_120000_radiomics_tnbc_luminal/
│   ├── tnbc_classifier.pkl              # TNBC classifier (radiomics mode)
│   ├── tnbc_classifier_RF_0.pkl         # Additional models (--save-all-models)
│   ├── luminal_classifier.pkl           # Luminal classifier (radiomics mode)
│   ├── training_report.json             # Training metadata, metrics, and config
│   └── visualizations/
│       ├── tnbc/
│       │   ├── confusion_matrix_tnbc.png
│       │   ├── roc_curve_tnbc.png
│       │   ├── pr_curve_tnbc.png
│       │   ├── feature_importance_tnbc.png
│       │   ├── classification_report_tnbc.{txt,json}
│       │   └── dashboard_tnbc.png
│       ├── tnbc_test/                   # Test-set visualisations (--evaluate-test-set)
│       │   └── ...
│       └── luminal/
│           └── ...
└── run_002_20250102_093000_cnn_contrast/
    ├── contrast_classifier_cnn.pt       # Contrast classifier (CNN mode)
    ├── training_report.json
    └── visualizations/
        └── contrast/
            └── ...
```

The `.pkl` files are directly usable with the evaluation pipeline:

```bash
python -m eval \
    --ground-truth-path /path/to/ground-truth \
    --predictions-path /path/to/predictions \
    --clf-model-dir /path/to/trained-models \
    --labels-path /path/to/labels.csv \
    --output-file metrics.json
```

### Model Selection

**Radiomics mode** (`--classifier-type radiomics`, default): The pipeline tries 13 model configurations, each wrapped in a `Pipeline(StandardScaler → VarianceThreshold → Classifier)` with class-balanced weights:
- **XGBoost** (3 configs): varying `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`
- **Random Forest** (3 configs): varying `n_estimators`, `max_depth`, `class_weight=balanced`
- **Logistic Regression** (4 configs): L1/L2 regularisation with `class_weight=balanced`
- **SVM-RBF** (3 configs): varying `C` and `gamma`, probability calibration, `class_weight=balanced`

If XGBoost is not installed, only the remaining configurations are tried. The best model is selected by validation AUROC (holdout) or mean cross-validation AUROC.

**CNN mode** (`--classifier-type cnn`): Trains an EfficientNet-B0 (or other timm model) end-to-end on 2D MRI slices. Uses AdamW optimiser with linear warmup + cosine decay, BCEWithLogitsLoss with class-weighted `pos_weight`, gradient clipping, and early stopping on validation AUROC. The trained model is saved as a `.pt` checkpoint.

### Feature Extraction

Training uses the same radiomic feature extraction pipeline as evaluation (`extract_radiomic_features` from `frd.py`), ensuring consistency between training and inference. Features are extracted from the first post-contrast phase (phase 1) by default, using the tumor segmentation mask.

Use `--cache-dir` to cache extracted features across runs:

```bash
python -m eval.train_classifier \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./models \
    --cache-dir ./feature-cache
```

## Synthesis Pipeline

Generate post-contrast DCE-MRI images from pre-contrast inputs using medigan's Pix2PixHD baseline model, and optionally evaluate the results in a single command.

### Prerequisites

```bash
pip install "eval[synthesis]"  # Installs medigan
```

### Synthesize Only

```bash
# Generate post-contrast images from MAMA-MIA pre-contrast data
mamasynth-synthesize \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./synthesized_images

# Or specify input directory explicitly
mamasynth-synthesize \
    --input-dir /path/to/pre-contrast-images \
    --output-dir ./synthesized_images
```

### Synthesize and Evaluate (One Command)

```bash
# Full pipeline: synthesize + evaluate
mamasynth-synthesize-and-evaluate \
    --data-dir /path/to/mama-mia-dataset \
    --output-dir ./synthesized_images \
    --output-file metrics.json

# Skip synthesis and evaluate existing predictions
mamasynth-synthesize-and-evaluate \
    --predictions-dir ./my_model_outputs \
    --ground-truth-path /path/to/ground-truth \
    --output-file metrics.json \
    --masks-path /path/to/masks \
    --labels-path /path/to/labels.csv \
    --clf-model-dir /path/to/classifiers
```

### Using Your Own Model

Participants can skip the built-in synthesis and point directly at their own model outputs:

```bash
# Just evaluate your own predictions
mamasynth-synthesize-and-evaluate \
    --predictions-dir /path/to/your-predictions \
    --ground-truth-path /path/to/ground-truth \
    --masks-path /path/to/masks \
    --labels-path /path/to/labels.csv \
    --output-file metrics.json
```

### Synthesis CLI Options

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `None` | MAMA-MIA dataset root (auto-resolves images/GT paths) |
| `--input-dir` | `None` | Directory with pre-contrast images (overrides `--data-dir`) |
| `--output-dir` | `./synthesized_output` | Where to save generated images |
| `--model` | `medigan` | Synthesis model name |
| `--model-id` | `00023_PIX2PIXHD_BREAST_DCEMRI` | medigan model ID |
| `--phase` | `0` | DCE-MRI phase of input images (0=pre-contrast) |
| `--gpu-id` | `0` | GPU device — bare int (`0`→`cuda:0`, `-1`→CPU) or full string (`cuda:0`, `cpu`) |
| `--image-size` | `512` | Spatial resolution for the model |
| `--keep-work-dir` | `false` | Keep intermediate staging directories (PNG slices) for debugging |
| `--predictions-dir` | `None` | Skip synthesis, evaluate existing predictions |
| `--ground-truth-path` | `None` | Ground-truth directory for evaluation |
| `--output-file` | `metrics.json` | Evaluation output JSON path |
| `--masks-path` | `None` | Tumor masks directory |
| `--labels-path` | `None` | Molecular subtype labels (JSON/CSV) |
| `--clf-model-dir` | `None` | Pre-trained classifier directory |
| `--seg-model-path` | `None` | Pre-trained nnUNet model directory |
| `--skip-synthesis` | `false` | Skip synthesis step in combined command |

## Docker

### Build

```bash
docker build -t eval .
```

### Run

```bash
docker run --rm \
    -v /path/to/ground-truth:/opt/app/ground-truth:ro \
    -v /path/to/predictions:/input:ro \
    -v /path/to/output:/output \
    eval
```

### Run with all inputs

```bash
docker run --rm \
    -v /path/to/ground-truth:/opt/app/ground-truth:ro \
    -v /path/to/predictions:/input:ro \
    -v /path/to/masks:/opt/app/masks:ro \
    -v /path/to/labels.csv:/opt/app/labels.csv:ro \
    -v /path/to/output:/output \
    eval \
    --masks-path /opt/app/masks \
    --labels-path /opt/app/labels.csv
```

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=eval --cov-report=term-missing

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
mama-synth-eval
├── src/eval/
│   ├── __init__.py              # Package exports (v0.6.0)
│   ├── __main__.py              # CLI entry point
│   ├── evaluation.py            # Main evaluation pipeline (MamaSynthEval, DatasetNormalizer)
│   ├── metrics.py               # Image-to-image & segmentation metrics
│   ├── frd.py                   # Fréchet Radiomics Distance (batch, cached)
│   ├── classification.py        # Molecular subtype classification (RadiomicsClassifier, CNNClassifier, EnsembleClassifier)
│   ├── train_classifier.py      # Classifier training on MAMA-MIA dataset
│   ├── train_cnn_classifier.py  # EfficientNet CNN training pipeline
│   ├── slice_extraction.py      # 2D slice extraction from 3D NIfTI volumes
│   ├── training_visualization.py # Confusion matrix, ROC, PR, dashboards
│   ├── segmentation.py          # Tumor segmentation (ThresholdSegmenter, NNUNetSegmenter)
│   ├── roi_utils.py             # Tumor ROI extraction & mask dilation
│   ├── ranking.py               # Borda-style rank aggregation
│   ├── visualization.py         # Result visualization (tables, charts, overlays)
│   ├── synthesize.py            # Synthesis pipeline (medigan wrapper + CLI)
│   ├── webapp.py                # Streamlit web interface
│   └── generate_test_data.py    # Artificial test data generator
├── tests/                       # 300+ tests (unit + integration + E2E)
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
│   ├── test_train_cnn_classifier.py # CNN classifier training tests
│   ├── test_v070_features.py    # v0.7.0 feature tests (dual-phase, ensemble, etc.)
│   ├── test_v090_features.py    # v0.9.0 feature tests (run dirs, contrast, caching)
│   ├── test_synthesize.py       # Synthesis pipeline tests
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
    url={https://github.com/RichardObi/MAMA-SYNTH-codebase}
}
```
