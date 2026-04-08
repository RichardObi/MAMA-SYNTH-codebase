# MAMA-SYNTH Challenge PDF Analysis

## Contradictions & Uncertainties Identified

### 1. CRITICAL: Normalization Scope — Dataset-Level vs Per-Image

**PDF (§3.2 Preprocessing):** _"code will be provided adapted to the training dataset to pass the images to the target **16-bit PNG slices (512×512) with z-score normalization applied at the dataset level**"_

**Current implementation:** `normalize_intensity()` computes z-score **per image** (using each image's own mean/std).

**Impact:** Dataset-level z-score normalizes all images using the **global** mean and std computed across the entire dataset. This is fundamentally different from per-image normalization and affects all pixel-level metrics (MSE, LPIPS, FRD).

**Resolution:** Implemented `DatasetNormalizer` class that first computes global statistics, then normalizes each image consistently.

---

### 2. IMPORTANT: Image Format — NIfTI vs 16-bit PNG

**PDF (§3.2):** _"target 16-bit PNG slices (512×512)"_

**PDF (§3.1):** Training data is provided in NIfTI format from Radboud/Fleming datasets.

**Current implementation:** Only supports `.nii`, `.nii.gz`, `.mha`, `.mhd` — **missing `.png`**.

**Impact:** Participants submit 16-bit PNG slices, but the evaluator cannot load them.

**Resolution:** Added `.png` to `SUPPORTED_EXTENSIONS` with a PNG loader path.

---

### 3. IMPORTANT: 2D Slices vs 3D Volumes

**PDF (§3.3):** _"participants are provided only with the **2D pre-contrast slice** and are tasked with synthesizing the corresponding 2D post-contrast slice"_

**PDF (§3.4 Assessment — Full Image):** _"Computed over the **full breast image** volume comparing real and synthesized images"_

**Ambiguity:** The word "volume" in metric descriptions is misleading since participants submit **2D slices** (512×512 PNGs). The evaluation operates on individual 2D slices, not 3D volumes. "Volume" likely refers to "across all slices in the test set" for FRD (which is inherently a dataset-level metric).

**Resolution:** Evaluation pipeline treats each case as a 2D slice. FRD aggregates features across all slices.

---

### 4. IMPORTANT: FRD Is Dataset-Level, Not Per-Case

**PDF (§3.4 Ranking):** _"Compute metric value per test case → Average across test cases"_

**But:** FRD (Fréchet Radiomics Distance) is inherently a **distribution-level** metric. It compares feature distributions across entire image sets, producing a single scalar — not a per-case value.

**Contradiction:** The ranking procedure says "average per test case" but FRD has no per-case interpretation.

**Resolution:** FRD is computed once per submission (dataset-level) and used directly as the submission's FRD score. It is NOT averaged across test cases. This is the only mathematically valid interpretation.

---

### 5. MODERATE: Missing Data Handling

**PDF (§3.4):** _"Missing outputs for individual cases are assigned the **worst observed score** for that metric prior to aggregation."_

**Current implementation:** Missing predictions are simply skipped (logged as warnings).

**Impact:** A submission that omits difficult cases would not be penalized.

**Resolution:** Implemented worst-score imputation for missing cases.

---

### 6. MODERATE: Tumor ROI Margin Not Specified

**PDF (§3.4 ROI):** _"tumor mask with a **fixed margin**"_

**Uncertainty:** The exact margin value is not specified in the PDF. Common values in breast MRI research range from 5–20 mm.

**Resolution:** Default set to 10 mm (configurable via `--roi-margin-mm`). Flagged for organizer confirmation.

---

### 7. MODERATE: Pre-Trained Evaluator Models Not Provided

**PDF (§3.4):** _"Classification is performed using **fixed, pre-trained evaluators**"_ and _"Segmentation performance is measured by applying a **fixed tumor segmentation model**"_

**PDF (§3.1):** _"a baseline nnUNet was used to provide initial automatic segmentation"_

**Impact:** The challenge evaluation requires specific pre-trained models that organizers maintain. These are not included in the evaluation package.

**Resolution:** Implemented `NNUNetSegmenter` wrapper and `RadiomicsClassifier` with model save/load capability. Organizers provide `.pkl`/checkpoint files at evaluation time. `ThresholdSegmenter` serves as a local testing baseline.

---

### 8. MINOR: Statistical Analyses Not Yet Implemented

**PDF (§3.4):** Describes several statistical tests:
- Bootstrap confidence intervals (1000 resamples)
- Wilcoxon signed-rank tests (pairwise between submissions)
- Significance at α = 0.05 with Bonferroni correction

**Current implementation:** Not present yet. These are post-hoc analyses run by organizers after all submissions are collected, not per-submission evaluation.

**Resolution:** Added `StatisticalAnalysis` class in the ranking module for organizer-side use.

---

### 9. MINOR: Baseline Model Reference

**PDF (§3.5):** _"Baseline method is Pix2PixHD adapted for DCE-MRI, sourced from medigan"_

**Impact:** No code impact — this describes the challenge's reference model, not the evaluator.

---

### 10. INFO: Data Sources and Split

- **Radboud University Medical Center:** ~200 cases (train + validation)
- **Fleming Institute, Greece:** ~100 cases (external test set)
- Split: ~60% train / ~20% validation / ~20% test
- Annotations: Expert radiologist tumor delineation, nnUNet-assisted initial segmentation
- Molecular subtypes: TNBC, Luminal A/B from clinical pathology reports

---

## Summary of Actions Taken

| Issue | Severity | Status |
|---|---|---|
| Dataset-level z-score | Critical | Fixed |
| PNG format support | Important | Fixed |
| 2D vs 3D ambiguity | Important | Clarified (2D) |
| FRD dataset-level | Important | Documented |
| Missing data penalty | Moderate | Implemented |
| ROI margin unspecified | Moderate | Default 10mm, flagged |
| Pre-trained models | Moderate | Wrapper classes added |
| Statistical analyses | Minor | Added |
