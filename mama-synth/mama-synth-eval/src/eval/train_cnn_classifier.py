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
CNN classifier training pipeline for MAMA-MIA dataset.

Trains EfficientNet-based deep learning classifiers for TNBC and Luminal
molecular subtype prediction using 2D DCE-MRI slices. The CNN approach
serves as a higher-capacity alternative to the radiomics-based classifiers.

Key design choices
------------------
- **EfficientNet-B0** (via ``timm``) initialised with ImageNet pretrained
  weights — strong transfer-learning baseline for medical images.
- **Single-channel MRI → 3-channel input**: grayscale slices are replicated
  to 3 channels so that the pretrained convolution filters can be reused.
- **Aggressive data augmentation**: random horizontal/vertical flip,
  rotation, affine transforms, and Gaussian noise to combat overfitting
  on the relatively small MAMA-MIA dataset.
- **Class-imbalance handling**: ``BCEWithLogitsLoss`` with ``pos_weight``
  computed from the training label distribution.
- **Cosine-annealing LR schedule** with linear warmup.
- **Early stopping** on validation AUROC.

Trained models are saved as ``{task}_classifier_cnn.pt`` and are
compatible with the evaluation pipeline via :class:`CNNClassifier` in
``classification.py``.

Usage (integrated via ``--classifier-type cnn`` in the training CLI)::

    python -m eval.train_classifier \\
        --data-dir /path/to/mama-mia \\
        --output-dir ./models \\
        --classifier-type cnn
"""

import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CNN_IMAGE_SIZE = 224
DEFAULT_CNN_BATCH_SIZE = 32
DEFAULT_CNN_NUM_EPOCHS = 50
DEFAULT_CNN_LEARNING_RATE = 1e-4
DEFAULT_CNN_WEIGHT_DECAY = 1e-4
DEFAULT_CNN_PATIENCE = 10
DEFAULT_CNN_MODEL_NAME = "efficientnet_b0"
CNN_MODEL_SUFFIX = "_classifier_cnn.pt"

# Re-use constants from the main training module
DEFAULT_PHASE = 1
DEFAULT_N_SLICES = 5

# Default seed
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def _check_cnn_dependencies() -> None:
    """Verify that CNN training dependencies are available."""
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import torchvision  # noqa: F401
    except ImportError:
        missing.append("torchvision")
    try:
        import timm  # noqa: F401
    except ImportError:
        missing.append("timm")
    if missing:
        raise ImportError(
            f"CNN training requires: {', '.join(missing)}. "
            "Install with: pip install 'mama-synth-eval[cnn]'"
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MRISliceDataset:
    """PyTorch-compatible dataset for 2D MRI slices.

    Each item returns a ``(tensor, label)`` pair where *tensor* has shape
    ``(C, image_size, image_size)`` and *label* is a scalar ``float32``
    (0.0 or 1.0).

    When *masks* is ``None``, the output has 3 channels (replicated
    grayscale for pretrained backbones) or 6 channels for dual-phase
    input.  When *masks* is provided an extra channel is appended
    (4 channels single-phase, 7 channels dual-phase).

    Parameters
    ----------
    slices : list[NDArray]
        List of 2D numpy arrays with arbitrary (H, W) shapes.
    labels : NDArray
        1-D binary label array (one entry per slice).
    image_size : int
        Output spatial resolution (squared).
    augment : bool
        Whether to apply training-time data augmentation.
    masks : list[NDArray] | None
        Optional list of 2D binary mask arrays aligned with *slices*.
        When given, a 4th channel is appended to each sample.
    """

    def __init__(
        self,
        slices: list[NDArray],
        labels: NDArray,
        image_size: int = DEFAULT_CNN_IMAGE_SIZE,
        augment: bool = False,
        masks: Optional[list[NDArray]] = None,
    ) -> None:
        import torch  # noqa: F401 — lazy import (availability check)

        self.slices = slices
        self.labels = np.asarray(labels, dtype=np.float32)
        self.image_size = image_size
        self.augment = augment
        self.masks = masks

        # Build augmentation pipeline (operates on C-channel tensors)
        if augment:
            from torchvision import transforms

            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1),
                ),
            ])
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int):
        import torch
        import torch.nn.functional as F

        raw = self.slices[idx]
        label = self.labels[idx]

        # --- Dual-phase input: (2, H, W) → (6, H, W) ---------------------
        if raw.ndim == 3 and raw.shape[0] == 2:
            channels: list[torch.Tensor] = []
            for ch_idx in range(2):
                ch = raw[ch_idx].astype(np.float32)
                vmin, vmax = float(ch.min()), float(ch.max())
                if vmax - vmin > 1e-8:
                    ch = (ch - vmin) / (vmax - vmin)
                else:
                    ch = np.zeros_like(ch)
                t = torch.from_numpy(ch).unsqueeze(0)  # (1, H, W)
                t = F.interpolate(
                    t.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)  # (1, H, W)
                channels.append(t.repeat(3, 1, 1))  # (3, H, W)
            tensor = torch.cat(channels, dim=0)  # (6, H, W)

            # Optional mask channel for dual-phase
            if self.masks is not None:
                mask_arr = self.masks[idx].astype(np.float32)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[0]
                mask_t = torch.from_numpy(mask_arr).unsqueeze(0)  # (1, H, W)
                mask_t = F.interpolate(
                    mask_t.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="nearest",
                ).squeeze(0)
                tensor = torch.cat([tensor, mask_t], dim=0)  # (7, H, W)

        else:
            # --- Single-phase input (original flow) -----------------------
            img = raw.astype(np.float32)

            # Per-slice min-max normalisation → [0, 1]
            vmin, vmax = float(img.min()), float(img.max())
            if vmax - vmin > 1e-8:
                img = (img - vmin) / (vmax - vmin)
            else:
                img = np.zeros_like(img)

            # → (1, H, W) tensor
            tensor = torch.from_numpy(img).unsqueeze(0)

            # Resize to (image_size, image_size)
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            # Replicate to 3 channels for pretrained backbone
            tensor = tensor.repeat(3, 1, 1)

            # Optionally add mask as 4th channel
            if self.masks is not None:
                mask_arr = self.masks[idx].astype(np.float32)
                mask_t = torch.from_numpy(mask_arr).unsqueeze(0)  # (1, H, W)
                mask_t = F.interpolate(
                    mask_t.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="nearest",
                ).squeeze(0)  # (1, image_size, image_size)
                tensor = torch.cat([tensor, mask_t], dim=0)  # (4, H, W)

        # Data augmentation (training only)
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_cnn_model(
    model_name: str = DEFAULT_CNN_MODEL_NAME,
    pretrained: bool = True,
    num_classes: int = 1,
    in_chans: int = 3,
):
    """Create a ``timm`` CNN model.

    Parameters
    ----------
    model_name : str
        Any model identifier supported by ``timm.create_model``.
    pretrained : bool
        Load ImageNet-pretrained weights.
    num_classes : int
        Number of output units (1 for binary classification with
        BCEWithLogitsLoss).
    in_chans : int
        Number of input channels.  When set to 4 (mask-channel mode) and
        *pretrained* is ``True``, the first convolutional layer is
        expanded by copying the pretrained 3-channel weights and
        zero-initialising the 4th channel so that the model starts from
        a sensible state.

    Returns
    -------
    model : torch.nn.Module
    """
    import timm
    import torch

    if in_chans == 3 or not pretrained:
        # Standard case — timm handles in_chans natively
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
        )
        return model

    # Pretrained + non-standard in_chans: load 3-channel pretrained first,
    # then manually extend the first conv layer.
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        in_chans=3,
    )

    # Find the first Conv2d layer
    first_conv = None
    first_conv_name = ""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv = module
            first_conv_name = name
            break

    if first_conv is not None and first_conv.in_channels == 3:
        old_weight = first_conv.weight.data  # (out, 3, kH, kW)
        new_in = in_chans
        new_conv = torch.nn.Conv2d(
            new_in,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
            padding_mode=first_conv.padding_mode,
        )
        # Copy pretrained weights for first 3 channels, zero-init extra
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_weight
            new_conv.weight[:, 3:, :, :] = 0.0
            if first_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

        # Replace the layer in the model
        parts = first_conv_name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_conv)

        logger.info(
            f"Expanded first conv '{first_conv_name}' from 3 → {new_in} "
            f"input channels (extra channels zero-initialised)."
        )
    else:
        logger.warning(
            f"Could not find 3-channel Conv2d to expand. Using timm "
            f"default in_chans={in_chans} handling."
        )
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
        )

    return model


# ---------------------------------------------------------------------------
# Slice extraction for CNN (returns raw pixel arrays, not radiomic features)
# ---------------------------------------------------------------------------

def extract_slices_for_cnn(
    patient_ids: list[str],
    labels: NDArray,
    data_dir: Path,
    images_dir: Optional[Path] = None,
    segmentations_dir: Optional[Path] = None,
    phase: int = DEFAULT_PHASE,
    slice_mode: str = "all_tumor",
    n_slices: int = DEFAULT_N_SLICES,
    return_masks: bool = False,
    dual_phase: bool = False,
    cache_dir: Optional[Path] = None,
) -> tuple[list[NDArray], NDArray, list[str], Optional[list[NDArray]]]:
    """Extract raw 2D slices for CNN training.

    Similar to :func:`extract_features_for_patients` in
    ``train_classifier.py`` but returns raw pixel arrays instead of
    radiomic feature vectors.

    When *cache_dir* is provided, extracted slices are saved per-patient
    as ``.npz`` files and reused on subsequent runs, avoiding repeated
    NIfTI loading and slice extraction.  Patients are processed in
    batches of *batch_size_extract* to limit peak CPU/memory usage.

    Parameters
    ----------
    patient_ids : list[str]
        Patient identifiers.
    labels : NDArray
        Binary labels aligned with *patient_ids*.
    data_dir : Path
        Root MAMA-MIA dataset directory.
    images_dir, segmentations_dir : Path, optional
        Override default subfolder paths.
    phase : int
        MRI phase index (0=pre-contrast, 1=first post-contrast).
    slice_mode : str
        Slice extraction strategy (see :class:`SliceMode`).
    n_slices : int
        Number of slices for ``multi_slice`` mode.
    return_masks : bool
        If ``True``, also return the corresponding 2D mask slices
        (for mask-channel mode).  When a mask is unavailable for a
        particular slice, a zero-filled array of the same shape is used.
    dual_phase : bool
        When ``True``, stack phase-0 and phase-1 slices into a 2-channel
        array per slice.
    cache_dir : Path | None
        Directory for per-patient slice caches.  Each patient's slices
        are stored as individual ``.npy`` files inside a patient
        sub-directory.  A ``_done`` marker indicates the extraction
        was fully completed.  If ``None``, caching is disabled.

    Returns
    -------
    slices : list[NDArray]
        Raw 2D numpy arrays (one per extracted slice).
    slice_labels : NDArray
        Corresponding binary labels (may be longer than *patient_ids*
        when ``all_tumor`` mode produces multiple slices per patient).
    slice_pids : list[str]
        Patient ID for each slice.
    slice_masks : list[NDArray] | None
        Corresponding 2D mask arrays (only when *return_masks* is
        ``True``; otherwise ``None``).
    """
    from eval.slice_extraction import (
        SliceMode,
        extract_2d_slice,
        extract_all_tumor_slices,
        extract_multi_slices,
    )
    from eval.train_classifier import (
        IMAGES_SUBDIR,
        SEGMENTATIONS_SUBDIR,
        _get_image_path,
        _get_segmentation_path,
        _load_mask_as_array,
        _load_nifti_as_array,
    )

    if images_dir is None:
        images_dir = data_dir / IMAGES_SUBDIR
    if segmentations_dir is None:
        segmentations_dir = data_dir / SEGMENTATIONS_SUBDIR

    _slice_mode = SliceMode(slice_mode)

    # --- Caching helpers -------------------------------------------------
    # Build a tag that uniquely identifies extraction parameters so cache
    # files are invalidated when settings change.
    _cache_tag = (
        f"ph{phase}_{slice_mode}_n{n_slices}"
        f"_dp{int(dual_phase)}_msk{int(return_masks)}"
    )

    def _patient_cache_dir(pid: str) -> Optional[Path]:
        if cache_dir is None:
            return None
        return cache_dir / f"{pid}_{_cache_tag}"

    def _load_from_cache(pid: str) -> Optional[dict]:
        """Try to load cached slices for a patient.

        Uses per-slice ``.npy`` files inside a patient directory so that
        partial extractions (e.g. process killed mid-patient) still
        survive on disk and are usable.  A ``_done`` marker file
        indicates the patient was fully extracted.
        """
        pdir = _patient_cache_dir(pid)
        if pdir is None or not pdir.exists():
            # Fall back to legacy single-file `.npz` cache.
            _npz = cache_dir / f"{pid}_{_cache_tag}.npz" if cache_dir else None
            if _npz is not None and _npz.exists():
                try:
                    data = np.load(str(_npz), allow_pickle=True)
                    return {
                        "slices": list(data["slices"]),
                        "masks": list(data["masks"]) if "masks" in data else None,
                        "n_slices": int(data["n_slices"]),
                    }
                except Exception:
                    pass
            return None

        # New per-slice cache: enumerate slice_{i}.npy files.
        done_marker = pdir / "_done"
        if not done_marker.exists():
            # Patient extraction was interrupted — ignore partial cache
            # so we re-extract cleanly.
            return None

        slices: list[NDArray] = []
        masks: list[NDArray] = []
        i = 0
        while True:
            sf = pdir / f"slice_{i}.npy"
            if not sf.exists():
                break
            try:
                slices.append(np.load(str(sf), allow_pickle=False))
            except Exception:
                break
            mf = pdir / f"mask_{i}.npy"
            if mf.exists():
                try:
                    masks.append(np.load(str(mf), allow_pickle=False))
                except Exception:
                    masks.append(np.zeros_like(slices[-1]))
            i += 1

        if len(slices) == 0:
            return None

        return {
            "slices": slices,
            "masks": masks if masks else None,
            "n_slices": len(slices),
        }

    def _save_slice_to_cache(
        pid: str,
        idx: int,
        slice_arr: NDArray,
        mask_arr: Optional[NDArray],
    ) -> None:
        """Save a single slice to the per-patient cache directory."""
        pdir = _patient_cache_dir(pid)
        if pdir is None:
            return
        try:
            pdir.mkdir(parents=True, exist_ok=True)
            np.save(str(pdir / f"slice_{idx}.npy"), slice_arr)
            if mask_arr is not None:
                np.save(str(pdir / f"mask_{idx}.npy"), mask_arr)
        except Exception as e:
            logger.warning(
                f"Failed to write cache slice {idx} for {pid}: {e}  "
                f"Caching disabled for this slice — will re-extract on next run."
            )

    def _mark_patient_done(pid: str) -> None:
        """Write a marker file indicating all slices have been cached."""
        pdir = _patient_cache_dir(pid)
        if pdir is None:
            return
        try:
            (pdir / "_done").touch()
        except Exception as e:
            logger.warning(
                f"Could not write cache completion marker for {pid}: {e}  "
                f"This patient will be re-extracted on the next run."
            )

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Pre-scan: count how many patients already have a complete cache
        # so the user can see upfront whether extraction will happen.
        n_already_cached = sum(
            1 for pid in patient_ids
            if (d := _patient_cache_dir(pid)) is not None
            and (d / "_done").exists()
        )
        n_need_extract = len(patient_ids) - n_already_cached
        if n_already_cached == len(patient_ids):
            logger.info(
                f"CNN slice cache: {cache_dir}  (tag='{_cache_tag}')  "
                f"— ALL {len(patient_ids)} patients already cached, "
                f"loading from disk (no NIfTI I/O)."
            )
        elif n_already_cached > 0:
            logger.info(
                f"CNN slice cache: {cache_dir}  (tag='{_cache_tag}')  "
                f"— {n_already_cached}/{len(patient_ids)} patients cached; "
                f"will extract {n_need_extract} missing patients from NIfTI."
            )
        else:
            logger.info(
                f"CNN slice cache: {cache_dir}  (tag='{_cache_tag}')  "
                f"— cache empty, extracting all {len(patient_ids)} patients "
                f"from NIfTI and writing to cache."
            )
    else:
        logger.info(
            "CNN slice cache: DISABLED (no --cache-dir supplied).  "
            "Pass --cache-dir to avoid re-extracting slices on every run."
        )

    # Progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(
            enumerate(patient_ids),
            total=len(patient_ids),
            desc="Extracting slices (CNN)",
            unit="patient",
        )
    except ImportError:
        iterator = enumerate(patient_ids)

    all_slices: list[NDArray] = []
    all_labels: list[float] = []
    all_pids: list[str] = []
    all_masks: list[NDArray] = []  # only populated when return_masks=True
    n_failed = 0

    def _maybe_stack_dual(
        s: NDArray, vol0: "Optional[NDArray]", si: int,
    ) -> NDArray:
        """If dual_phase, stack phase-0 slice under the phase-1 slice."""
        if not dual_phase or vol0 is None:
            return s
        if vol0.ndim == 3 and 0 <= si < vol0.shape[0]:
            p0 = vol0[si].astype(np.float32)
        elif vol0.ndim == 2:
            p0 = vol0.astype(np.float32)
        else:
            p0 = np.zeros_like(s, dtype=np.float32)
        return np.stack([s, p0], axis=0)  # (2, H, W)

    n_cache_hits = 0

    for i, pid in iterator:
        try:
            label = float(labels[i])

            # --- Try cache first ------------------------------------------
            cached = _load_from_cache(pid)
            if cached is not None:
                logger.debug(
                    f"Cache hit: {pid} → {len(cached['slices'])} slices "
                    f"loaded from disk."
                )
                for j, s in enumerate(cached["slices"]):
                    all_slices.append(s)
                    all_labels.append(label)
                    all_pids.append(pid)
                    if return_masks and cached["masks"] is not None:
                        all_masks.append(cached["masks"][j])
                    elif return_masks:
                        all_masks.append(np.zeros_like(s[-1] if s.ndim == 3 else s, dtype=np.float32))
                n_cache_hits += 1
                continue

            # --- Extract from NIfTI (cache miss) --------------------------
            logger.debug(f"Cache miss: {pid} — loading NIfTI and extracting slices.")
            img_path = _get_image_path(images_dir, pid, phase)
            volume = _load_nifti_as_array(img_path)

            # Dual-phase: also load pre-contrast volume
            phase0_volume: Optional[NDArray] = None
            if dual_phase:
                try:
                    phase0_path = _get_image_path(images_dir, pid, 0)
                    phase0_volume = _load_nifti_as_array(phase0_path)
                except Exception as e:
                    logger.warning(
                        f"Dual-phase: failed to load phase-0 for "
                        f"{pid}: {e}, skipping patient."
                    )
                    continue

            # Load mask (optional for MIDDLE mode)
            mask: Optional[NDArray] = None
            if _slice_mode != SliceMode.MIDDLE:
                seg_path = _get_segmentation_path(segmentations_dir, pid)
                try:
                    mask = _load_mask_as_array(seg_path)
                except FileNotFoundError:
                    if _slice_mode in (
                        SliceMode.MAX_TUMOR,
                        SliceMode.CENTER_TUMOR,
                        SliceMode.ALL_TUMOR,
                    ):
                        logger.warning(
                            f"Mask not found for {pid}, skipping (required "
                            f"for {_slice_mode.value} mode)."
                        )
                        n_failed += 1
                        continue
                    # MULTI_SLICE can fall back to evenly spaced
                    mask = None

            if _slice_mode == SliceMode.ALL_TUMOR:
                slices_2d, masks_2d, indices = extract_all_tumor_slices(
                    volume, mask,
                )
                patient_slices: list[NDArray] = []
                patient_masks: list[NDArray] = []
                for j, s in enumerate(slices_2d):
                    stacked = _maybe_stack_dual(s, phase0_volume, indices[j])
                    patient_slices.append(stacked)
                    all_slices.append(stacked)
                    all_labels.append(label)
                    all_pids.append(pid)
                    m2d_val: Optional[NDArray] = None
                    if return_masks:
                        m2d = masks_2d[j] if masks_2d is not None and j < len(masks_2d) else np.zeros_like(s)
                        m2d = m2d.astype(np.float32)
                        patient_masks.append(m2d)
                        all_masks.append(m2d)
                        m2d_val = m2d
                    _save_slice_to_cache(pid, j, stacked, m2d_val)
                _mark_patient_done(pid)

            elif _slice_mode == SliceMode.MULTI_SLICE:
                slices_2d, masks_2d, indices = extract_multi_slices(
                    volume, mask, n_slices=n_slices,
                )
                patient_slices = []
                patient_masks = []
                for j, s in enumerate(slices_2d):
                    stacked = _maybe_stack_dual(s, phase0_volume, indices[j])
                    patient_slices.append(stacked)
                    all_slices.append(stacked)
                    all_labels.append(label)
                    all_pids.append(pid)
                    m2d_val = None
                    if return_masks:
                        m2d = masks_2d[j] if masks_2d is not None and j < len(masks_2d) else np.zeros_like(s)
                        m2d = m2d.astype(np.float32)
                        patient_masks.append(m2d)
                        all_masks.append(m2d)
                        m2d_val = m2d
                    _save_slice_to_cache(pid, j, stacked, m2d_val)
                _mark_patient_done(pid)

            else:
                # Single-slice modes
                slice_2d, mask_2d, idx = extract_2d_slice(
                    volume, mask, mode=_slice_mode,
                )
                stacked = _maybe_stack_dual(slice_2d, phase0_volume, idx)
                all_slices.append(stacked)
                all_labels.append(label)
                all_pids.append(pid)
                m2d_single: Optional[NDArray] = None
                if return_masks:
                    m2d = mask_2d if mask_2d is not None else np.zeros_like(slice_2d)
                    m2d = m2d.astype(np.float32)
                    all_masks.append(m2d)
                    m2d_single = m2d
                _save_slice_to_cache(pid, 0, stacked, m2d_single)
                _mark_patient_done(pid)

        except Exception as e:
            logger.warning(f"Failed to extract slices for {pid}: {e}")
            n_failed += 1
            continue

    if len(all_slices) == 0:
        raise RuntimeError(
            "No slices could be extracted for CNN training. "
            "Check image/mask paths and data integrity."
        )

    logger.info(
        f"CNN slice extraction: {len(all_slices)} slices from "
        f"{len(patient_ids) - n_failed}/{len(patient_ids)} patients "
        f"({n_failed} failed, {n_cache_hits} from cache)"
    )

    return (
        all_slices,
        np.array(all_labels, dtype=np.float32),
        all_pids,
        all_masks if return_masks else None,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_cnn(
    train_slices: list[NDArray],
    train_labels: NDArray,
    val_slices: list[NDArray],
    val_labels: NDArray,
    task: str,
    output_dir: Path,
    model_name: str = DEFAULT_CNN_MODEL_NAME,
    image_size: int = DEFAULT_CNN_IMAGE_SIZE,
    num_epochs: int = DEFAULT_CNN_NUM_EPOCHS,
    batch_size: int = DEFAULT_CNN_BATCH_SIZE,
    learning_rate: float = DEFAULT_CNN_LEARNING_RATE,
    weight_decay: float = DEFAULT_CNN_WEIGHT_DECAY,
    patience: int = DEFAULT_CNN_PATIENCE,
    seed: int = DEFAULT_SEED,
    train_masks: Optional[list[NDArray]] = None,
    val_masks: Optional[list[NDArray]] = None,
    use_mask_channel: bool = False,
    dual_phase: bool = False,
    device: Optional[str] = None,
) -> tuple[Any, dict[str, float]]:
    """Train a CNN classifier on 2D MRI slices.

    Parameters
    ----------
    train_slices, val_slices : list[NDArray]
        Raw 2D slices for training and validation.
    train_labels, val_labels : NDArray
        Binary labels.
    task : str
        ``"tnbc"`` or ``"luminal"``.
    output_dir : Path
        Directory for saving the best model checkpoint.
    model_name : str
        ``timm`` model identifier (default: ``"efficientnet_b0"``).
    image_size : int
        Input image resolution (pixels, squared).
    num_epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Peak learning rate.
    weight_decay : float
        AdamW weight decay.
    patience : int
        Early stopping patience (epochs without AUROC improvement).
    seed : int
        Random seed.
    train_masks, val_masks : list[NDArray] | None
        Optional 2D mask arrays (one per slice).  Required when
        *use_mask_channel* is ``True``.
    use_mask_channel : bool
        When ``True``, an extra mask channel is appended to the input
        (4 channels single-phase, 7 channels dual-phase).
    dual_phase : bool
        When ``True``, input slices have two stacked phases (6 base
        channels instead of 3).
    device : str | None
        Target device (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``).
        When ``None`` or ``"auto"``, the best available device is
        auto-detected.

    Returns
    -------
    best_model : torch.nn.Module
        Trained model (on CPU).
    best_metrics : dict[str, float]
        Validation metrics for the best epoch.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from torch.utils.data import DataLoader

    _check_cnn_dependencies()

    # Seed everything
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine device
    if device is not None and device != "auto":
        device = torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"CNN training device: {device}")

    # --- Datasets & Loaders -----------------------------------------------
    train_ds = MRISliceDataset(
        train_slices, train_labels,
        image_size=image_size, augment=True,
        masks=train_masks if use_mask_channel else None,
    )
    val_ds = MRISliceDataset(
        val_slices, val_labels,
        image_size=image_size, augment=False,
        masks=val_masks if use_mask_channel else None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
        drop_last=len(train_ds) > batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # --- Model ------------------------------------------------------------
    base_chans = 6 if dual_phase else 3
    in_chans = base_chans + 1 if use_mask_channel else base_chans
    model = create_cnn_model(
        model_name=model_name, pretrained=True, num_classes=1,
        in_chans=in_chans,
    )
    model = model.to(device)

    # --- Loss (class-weighted) --------------------------------------------
    n_pos = float(train_labels.sum())
    n_neg = float(len(train_labels) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(
        f"Class balance: {int(n_pos)} positive, {int(n_neg)} negative "
        f"(pos_weight={pos_weight.item():.2f})"
    )

    # --- Optimiser --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )

    # --- LR schedule: linear warmup + cosine decay ------------------------
    warmup_epochs = min(5, num_epochs // 5)
    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(float(warmup_steps), 1.0)
        progress = float(step - warmup_steps) / max(
            float(total_steps - warmup_steps), 1.0
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training loop ----------------------------------------------------
    best_val_auroc = -1.0
    best_epoch = -1
    best_state: Optional[dict] = None
    best_metrics: dict[str, float] = {}
    epochs_no_improve = 0

    logger.info(
        f"\n{'='*60}\n"
        f"CNN Training — '{task}' ({model_name})\n"
        f"  Train: {len(train_ds)} slices | Val: {len(val_ds)} slices\n"
        f"  Epochs: {num_epochs} | Batch: {batch_size} | "
        f"LR: {learning_rate} | Patience: {patience}\n"
        f"{'='*60}"
    )

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # --- Train --------------------------------------------------------
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images).squeeze(-1)
            loss = criterion(logits, targets)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss / max(n_train_batches, 1)

        # --- Validate -----------------------------------------------------
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        all_val_probs: list[float] = []
        all_val_targets: list[float] = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits = model(images).squeeze(-1)
                loss = criterion(logits, targets)

                val_loss += loss.item()
                n_val_batches += 1

                probs = torch.sigmoid(logits).cpu().numpy()
                all_val_probs.extend(probs.tolist())
                all_val_targets.extend(targets.cpu().numpy().tolist())

        avg_val_loss = val_loss / max(n_val_batches, 1)

        # Metrics
        y_true = np.array(all_val_targets)
        y_score = np.array(all_val_probs)
        y_pred = (y_score >= 0.5).astype(int)

        if len(np.unique(y_true)) >= 2:
            val_auroc = float(roc_auc_score(y_true, y_score))
        else:
            val_auroc = 0.5

        val_bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch+1:3d}/{num_epochs} — "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"val_AUROC={val_auroc:.4f}  "
            f"val_BalAcc={val_bal_acc:.4f}  "
            f"lr={current_lr:.2e}  "
            f"({elapsed:.1f}s)"
        )

        # Early stopping check
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            best_metrics = {
                "auroc": val_auroc,
                "balanced_accuracy": val_bal_acc,
                "val_loss": avg_val_loss,
                "train_loss": avg_train_loss,
                "epoch": epoch + 1,
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(best AUROC {best_val_auroc:.4f} at epoch {best_epoch})"
                )
                break

    # --- Save best model --------------------------------------------------
    if best_state is None:
        logger.warning("No valid model was found during training.")
        # Use final model state as fallback
        best_state = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }
        best_metrics = {
            "auroc": 0.0,
            "balanced_accuracy": 0.0,
            "epoch": num_epochs,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{task}{CNN_MODEL_SUFFIX}"

    checkpoint = {
        "model_state_dict": best_state,
        "model_name": model_name,
        "num_classes": 1,
        "image_size": image_size,
        "in_chans": in_chans,
        "use_mask_channel": use_mask_channel,
        "task": task,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
    }

    import torch as _torch
    _torch.save(checkpoint, model_path)
    logger.info(
        f"Best CNN model saved → {model_path}  "
        f"(epoch {best_epoch}, AUROC={best_val_auroc:.4f})"
    )

    # Move model back to CPU with best weights for return
    model.cpu()
    model.load_state_dict(best_state)

    return model, best_metrics


# ---------------------------------------------------------------------------
# CNN evaluation helper
# ---------------------------------------------------------------------------

def evaluate_cnn(
    model,
    slices: list[NDArray],
    labels: NDArray,
    image_size: int = DEFAULT_CNN_IMAGE_SIZE,
    batch_size: int = DEFAULT_CNN_BATCH_SIZE,
    masks: Optional[list[NDArray]] = None,
    device: Optional[str] = None,
) -> dict[str, float]:
    """Evaluate a trained CNN on a set of slices.

    Parameters
    ----------
    model : torch.nn.Module
        Trained CNN (on CPU).
    slices : list[NDArray]
        Raw 2D slice arrays.
    labels : NDArray
        Binary labels.
    image_size : int
        Expected input resolution.
    batch_size : int
        Inference batch size.
    masks : list[NDArray] | None
        Optional 2D mask arrays (for mask-channel models).
    device : str | None
        Target device. If None or "auto", auto-detects best device.

    Returns
    -------
    dict with ``auroc``, ``balanced_accuracy``, and ``loss``.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from torch.utils.data import DataLoader

    if device is not None and device != "auto":
        device = torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    ds = MRISliceDataset(
        slices, labels, image_size=image_size, augment=False,
        masks=masks,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()
    all_probs: list[float] = []
    all_targets: list[float] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images).squeeze(-1)
            total_loss += criterion(logits, targets).item()
            n_batches += 1
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_score = np.array(all_probs)
    y_pred = (y_score >= 0.5).astype(int)

    auroc = float("nan")
    if len(np.unique(y_true)) >= 2:
        auroc = float(roc_auc_score(y_true, y_score))

    return {
        "auroc": auroc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "loss": total_loss / max(n_batches, 1),
    }


# ---------------------------------------------------------------------------
# CNN model persistence
# ---------------------------------------------------------------------------

def load_cnn_model(
    path: Path,
    device: str = "cpu",
):
    """Load a saved CNN checkpoint.

    Parameters
    ----------
    path : Path
        ``{task}_classifier_cnn.pt`` file.
    device : str
        Target device (``"cpu"``, ``"cuda"``, ``"mps"``).

    Returns
    -------
    model : torch.nn.Module
        Model loaded with trained weights, in eval mode.
    config : dict
        Checkpoint metadata (``model_name``, ``image_size``, ``task``, …).
    """
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = create_cnn_model(
        model_name=checkpoint["model_name"],
        pretrained=False,
        num_classes=checkpoint.get("num_classes", 1),
        in_chans=checkpoint.get("in_chans", 3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    config = {
        k: v for k, v in checkpoint.items() if k != "model_state_dict"
    }
    return model, config


# ---------------------------------------------------------------------------
# CNN pipeline entry point (called from train_classifier.main)
# ---------------------------------------------------------------------------

def train_cnn_pipeline(
    task: str,
    patient_ids: list[str],
    labels: NDArray,
    data_dir: Path,
    output_dir: Path,
    images_dir: Optional[Path] = None,
    segmentations_dir: Optional[Path] = None,
    phase: int = DEFAULT_PHASE,
    slice_mode: str = "all_tumor",
    n_slices: int = DEFAULT_N_SLICES,
    val_ratio: float = 0.2,
    cv_folds: int = 0,
    seed: int = DEFAULT_SEED,
    model_name: str = DEFAULT_CNN_MODEL_NAME,
    image_size: int = DEFAULT_CNN_IMAGE_SIZE,
    num_epochs: int = DEFAULT_CNN_NUM_EPOCHS,
    batch_size: int = DEFAULT_CNN_BATCH_SIZE,
    learning_rate: float = DEFAULT_CNN_LEARNING_RATE,
    patience: int = DEFAULT_CNN_PATIENCE,
    no_viz: bool = False,
    evaluate_test_set: bool = False,
    test_patient_ids: Optional[list[str]] = None,
    clinical_df: Optional[Any] = None,
    use_mask_channel: bool = False,
    dual_phase: bool = False,
    device: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    contrast_mode: bool = False,
) -> tuple[Any, str, dict[str, float]]:
    """Full CNN training pipeline for a single task.

    Extracts 2D slices, splits into train/val, trains the CNN,
    evaluates, saves the model, and optionally evaluates on the test set.

    Returns
    -------
    model : torch.nn.Module
        Best trained model.
    model_name_str : str
        Human-readable model description.
    best_metrics : dict
        Validation metrics for the best epoch.
    """
    from sklearn.model_selection import train_test_split

    _check_cnn_dependencies()

    logger.info(f"\n--- CNN: Extracting 2D slices for task '{task}' ---")
    if use_mask_channel:
        logger.info("Mask-channel mode: masks will be added as 4th input channel.")
    if contrast_mode:
        logger.info("Contrast mode: extracting phase 0 + phase 1 slices.")

    # 1. Extract slices
    if contrast_mode:
        # Contrast classification: extract from phase 0 (label=0) and
        # phase 1 (label=1) separately, then combine.
        p0_slices, p0_labels, p0_pids, p0_masks = extract_slices_for_cnn(
            patient_ids=patient_ids,
            labels=np.zeros(len(patient_ids), dtype=np.float32),
            data_dir=data_dir,
            images_dir=images_dir,
            segmentations_dir=segmentations_dir,
            phase=0,
            slice_mode=slice_mode,
            n_slices=n_slices,
            return_masks=use_mask_channel,
            dual_phase=False,
            cache_dir=cache_dir,
        )
        p1_slices, p1_labels, p1_pids, p1_masks = extract_slices_for_cnn(
            patient_ids=patient_ids,
            labels=np.ones(len(patient_ids), dtype=np.float32),
            data_dir=data_dir,
            images_dir=images_dir,
            segmentations_dir=segmentations_dir,
            phase=1,
            slice_mode=slice_mode,
            n_slices=n_slices,
            return_masks=use_mask_channel,
            dual_phase=False,
            cache_dir=cache_dir,
        )
        all_slices = p0_slices + p1_slices
        slice_labels = np.concatenate([p0_labels, p1_labels])
        slice_pids = [f"{p}_ph0" for p in p0_pids] + [f"{p}_ph1" for p in p1_pids]
        all_masks: Optional[list[NDArray]] = None
        if p0_masks is not None and p1_masks is not None:
            all_masks = p0_masks + p1_masks
        logger.info(
            f"Contrast slices: {len(p0_slices)} pre-contrast + "
            f"{len(p1_slices)} post-contrast = {len(all_slices)} total."
        )
    else:
        all_slices, slice_labels, slice_pids, all_masks = extract_slices_for_cnn(
            patient_ids=patient_ids,
            labels=labels,
            data_dir=data_dir,
            images_dir=images_dir,
            segmentations_dir=segmentations_dir,
            phase=phase,
            slice_mode=slice_mode,
            n_slices=n_slices,
            return_masks=use_mask_channel,
            dual_phase=dual_phase,
            cache_dir=cache_dir,
        )

    # 2. Patient-aware train/val split
    #    Split by patient, not by slice, to avoid data leakage.
    unique_pids = sorted(set(slice_pids))
    pid_labels = {}
    for pid, lbl in zip(slice_pids, slice_labels):
        pid_labels[pid] = lbl

    pid_array = np.array(unique_pids)
    pid_label_array = np.array([pid_labels[p] for p in unique_pids])

    if cv_folds > 0:
        # For CV, we use all data for training, but with cross-validation
        # For simplicity, we use 80/20 split for CNN (CV is less common for DL)
        logger.info(
            f"Note: CV mode ({cv_folds} folds) is not standard for CNN "
            "training. Using a single train/val split instead."
        )

    train_pid_arr, val_pid_arr = train_test_split(
        pid_array,
        test_size=val_ratio,
        random_state=seed,
        stratify=pid_label_array,
    )
    train_pid_set = set(train_pid_arr.tolist())
    val_pid_set = set(val_pid_arr.tolist())

    train_slices = [s for s, p in zip(all_slices, slice_pids) if p in train_pid_set]
    train_labels_arr = np.array(
        [l for l, p in zip(slice_labels, slice_pids) if p in train_pid_set],
        dtype=np.float32,
    )
    val_slices = [s for s, p in zip(all_slices, slice_pids) if p in val_pid_set]
    val_labels_arr = np.array(
        [l for l, p in zip(slice_labels, slice_pids) if p in val_pid_set],
        dtype=np.float32,
    )

    # Split masks if mask-channel mode
    train_masks_list: Optional[list[NDArray]] = None
    val_masks_list: Optional[list[NDArray]] = None
    if use_mask_channel and all_masks is not None:
        train_masks_list = [m for m, p in zip(all_masks, slice_pids) if p in train_pid_set]
        val_masks_list = [m for m, p in zip(all_masks, slice_pids) if p in val_pid_set]

    logger.info(
        f"Patient-aware split: {len(train_pid_set)} train patients "
        f"({len(train_slices)} slices), {len(val_pid_set)} val patients "
        f"({len(val_slices)} slices)"
    )

    # 3. Train
    best_model, best_metrics = train_cnn(
        train_slices=train_slices,
        train_labels=train_labels_arr,
        val_slices=val_slices,
        val_labels=val_labels_arr,
        task=task,
        output_dir=output_dir,
        model_name=model_name,
        image_size=image_size,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        seed=seed,
        train_masks=train_masks_list,
        val_masks=val_masks_list,
        use_mask_channel=use_mask_channel,
        dual_phase=dual_phase,
        device=device,
    )

    mask_tag = ", mask_ch" if use_mask_channel else ""
    model_desc = f"CNN({model_name}, img={image_size}{mask_tag})"

    # 4. Test-set evaluation
    if evaluate_test_set and test_patient_ids and clinical_df is not None:
        logger.info(
            f"\n--- CNN: Evaluating on test set "
            f"({len(test_patient_ids)} patients) ---"
        )
        try:
            from eval.train_classifier import create_labels

            test_clinical = clinical_df[
                clinical_df["patient_id"].isin(test_patient_ids)
            ].copy()
            test_pids_task, test_labels_task = create_labels(
                test_clinical, task,
            )

            if len(test_pids_task) > 0:
                if contrast_mode:
                    # Contrast: extract from both phases for test set
                    tp0_sl, tp0_lb, tp0_pid, tp0_msk = extract_slices_for_cnn(
                        patient_ids=test_pids_task,
                        labels=np.zeros(len(test_pids_task), dtype=np.float32),
                        data_dir=data_dir,
                        images_dir=images_dir,
                        segmentations_dir=segmentations_dir,
                        phase=0, slice_mode=slice_mode, n_slices=n_slices,
                        return_masks=use_mask_channel, dual_phase=False,
                        cache_dir=cache_dir,
                    )
                    tp1_sl, tp1_lb, tp1_pid, tp1_msk = extract_slices_for_cnn(
                        patient_ids=test_pids_task,
                        labels=np.ones(len(test_pids_task), dtype=np.float32),
                        data_dir=data_dir,
                        images_dir=images_dir,
                        segmentations_dir=segmentations_dir,
                        phase=1, slice_mode=slice_mode, n_slices=n_slices,
                        return_masks=use_mask_channel, dual_phase=False,
                        cache_dir=cache_dir,
                    )
                    test_slices = tp0_sl + tp1_sl
                    test_lbl = np.concatenate([tp0_lb, tp1_lb])
                    test_masks_ret: Optional[list[NDArray]] = None
                    if tp0_msk is not None and tp1_msk is not None:
                        test_masks_ret = tp0_msk + tp1_msk
                else:
                    test_slices, test_lbl, test_spids, test_masks_ret = extract_slices_for_cnn(
                        patient_ids=test_pids_task,
                        labels=test_labels_task,
                        data_dir=data_dir,
                        images_dir=images_dir,
                        segmentations_dir=segmentations_dir,
                        phase=phase,
                        slice_mode=slice_mode,
                        n_slices=n_slices,
                        return_masks=use_mask_channel,
                        dual_phase=dual_phase,
                        cache_dir=cache_dir,
                    )

                # Patient-level aggregation for test: average per-slice probs
                test_metrics = evaluate_cnn(
                    best_model, test_slices, test_lbl,
                    image_size=image_size,
                    masks=test_masks_ret,
                    device=device,
                )
                logger.info(
                    f"CNN TEST SET — {task}: "
                    f"AUROC={test_metrics['auroc']:.4f}, "
                    f"Bal.Acc={test_metrics['balanced_accuracy']:.4f}, "
                    f"Loss={test_metrics['loss']:.4f}"
                )
                best_metrics["test_auroc"] = test_metrics["auroc"]
                best_metrics["test_balanced_accuracy"] = (
                    test_metrics["balanced_accuracy"]
                )
                best_metrics["test_loss"] = test_metrics["loss"]
        except Exception as e:
            logger.warning(f"CNN test-set evaluation failed: {e}")

    return best_model, model_desc, best_metrics
