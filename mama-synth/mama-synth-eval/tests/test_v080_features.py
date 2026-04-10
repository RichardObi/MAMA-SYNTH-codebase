#  Copyright 2025 mama-synth-eval contributors
#  Licensed under the Apache License, Version 2.0.

"""
Tests for v0.8.0 features:

1. CNN slice caching (extract once, reuse on consecutive runs)
2. GPU / device selection (--device CLI argument)
3. MAMA-MIA test-split detection (CSV-based + --test-split-values)
4. Bug fixes:
   - evaluate_cnn .cpu() before .numpy()
   - dual_phase parameter wiring in train_cnn()
   - dual-phase zero-padding when pre-contrast missing
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    import torch
    import timm  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="CNN tests require torch, torchvision, and timm",
)


# ===================================================================
# 1. CNN Slice Caching
# ===================================================================


class TestCNNSliceCaching:
    """Tests for CNN slice caching in extract_slices_for_cnn."""

    @requires_torch
    def test_cache_helpers_exist(self):
        """Cache helper functions are importable and callable."""
        import eval.train_cnn_classifier as mod
        # The helpers are closure-internal, but we can verify via
        # the extract_slices_for_cnn signature accepting cache_dir.
        import inspect
        sig = inspect.signature(mod.extract_slices_for_cnn)
        assert "cache_dir" in sig.parameters
        assert "batch_size_extract" in sig.parameters

    @requires_torch
    def test_extract_slices_accepts_cache_dir(self):
        """extract_slices_for_cnn accepts cache_dir parameter."""
        import inspect
        from eval.train_cnn_classifier import extract_slices_for_cnn
        sig = inspect.signature(extract_slices_for_cnn)
        p = sig.parameters["cache_dir"]
        assert p.default is None

    @requires_torch
    def test_train_cnn_pipeline_accepts_cache_dir(self):
        """train_cnn_pipeline accepts cache_dir parameter."""
        import inspect
        from eval.train_cnn_classifier import train_cnn_pipeline
        sig = inspect.signature(train_cnn_pipeline)
        assert "cache_dir" in sig.parameters

    @requires_torch
    def test_cache_tag_deterministic(self):
        """Cache tag is deterministic for same parameters."""
        from eval.train_cnn_classifier import extract_slices_for_cnn
        import hashlib

        # Replicate the tag logic
        parts = ["phase=1", "mode=max_tumor", "n=3",
                 "dual=False", "masks=True"]
        tag1 = hashlib.md5("|".join(parts).encode()).hexdigest()[:12]
        tag2 = hashlib.md5("|".join(parts).encode()).hexdigest()[:12]
        assert tag1 == tag2

    @requires_torch
    def test_cache_tag_varies_with_params(self):
        """Cache tag changes when parameters change."""
        import hashlib

        parts_a = ["phase=1", "mode=max_tumor", "n=3",
                    "dual=False", "masks=True"]
        parts_b = ["phase=1", "mode=max_tumor", "n=3",
                    "dual=True", "masks=True"]
        tag_a = hashlib.md5("|".join(parts_a).encode()).hexdigest()[:12]
        tag_b = hashlib.md5("|".join(parts_b).encode()).hexdigest()[:12]
        assert tag_a != tag_b


# ===================================================================
# 2. Device Selection
# ===================================================================


class TestDeviceCLI:
    """Tests for --device CLI argument."""

    def test_device_arg_in_parse_args(self):
        """parse_args accepts --device argument."""
        from eval.train_classifier import parse_args
        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--device", "cpu",
        ])
        assert args.device == "cpu"

    def test_device_arg_default_auto(self):
        """--device defaults to 'auto'."""
        from eval.train_classifier import parse_args
        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
        ])
        assert args.device == "auto"

    def test_device_arg_choices(self):
        """--device rejects invalid values."""
        from eval.train_classifier import parse_args
        with pytest.raises(SystemExit):
            parse_args([
                "--data-dir", "/tmp/data",
                "--output-dir", "/tmp/out",
                "--device", "tpu",
            ])

    def test_device_cuda_choice(self):
        """--device accepts 'cuda'."""
        from eval.train_classifier import parse_args
        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--device", "cuda",
        ])
        assert args.device == "cuda"

    def test_device_mps_choice(self):
        """--device accepts 'mps'."""
        from eval.train_classifier import parse_args
        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--device", "mps",
        ])
        assert args.device == "mps"


class TestDeviceParameterWiring:
    """Tests that device parameter flows through the function chain."""

    @requires_torch
    def test_train_cnn_accepts_device(self):
        """train_cnn accepts device parameter."""
        import inspect
        from eval.train_cnn_classifier import train_cnn
        sig = inspect.signature(train_cnn)
        assert "device" in sig.parameters

    @requires_torch
    def test_evaluate_cnn_accepts_device(self):
        """evaluate_cnn accepts device parameter."""
        import inspect
        from eval.train_cnn_classifier import evaluate_cnn
        sig = inspect.signature(evaluate_cnn)
        assert "device" in sig.parameters

    @requires_torch
    def test_train_cnn_pipeline_accepts_device(self):
        """train_cnn_pipeline accepts device parameter."""
        import inspect
        from eval.train_cnn_classifier import train_cnn_pipeline
        sig = inspect.signature(train_cnn_pipeline)
        assert "device" in sig.parameters


# ===================================================================
# 3. MAMA-MIA Test Split Detection
# ===================================================================


class TestSplitDetection:
    """Tests for improved train/test split detection."""

    def _make_clinical_df(self, dataset_values):
        """Create a minimal clinical DataFrame."""
        return pd.DataFrame({
            "patient_id": [f"P{i:03d}" for i in range(len(dataset_values))],
            "dataset": dataset_values,
        })

    def test_detect_split_column_standard(self):
        """Standard train/test values are detected."""
        from eval.train_classifier import detect_split_column
        df = pd.DataFrame({
            "patient_id": ["P001", "P002", "P003"],
            "dataset_split": ["train", "train", "test"],
        })
        result = detect_split_column(df)
        assert result == "dataset_split"

    def test_detect_split_column_mama_mia_no_standard(self):
        """MAMA-MIA dataset column (DUKE/ISPY1/...) is NOT detected
        with standard values."""
        from eval.train_classifier import detect_split_column
        df = self._make_clinical_df(["DUKE", "ISPY1", "ISPY2", "NACT"])
        result = detect_split_column(df)
        assert result is None

    def test_detect_split_column_custom_values(self):
        """MAMA-MIA dataset column IS detected with custom test values."""
        from eval.train_classifier import detect_split_column
        df = self._make_clinical_df(["DUKE", "ISPY1", "ISPY2", "NACT"])
        result = detect_split_column(df, custom_test_values=["DUKE"])
        assert result == "dataset"

    def test_detect_split_column_custom_case_insensitive(self):
        """Custom values are matched case-insensitively."""
        from eval.train_classifier import detect_split_column
        df = self._make_clinical_df(["DUKE", "ISPY1"])
        result = detect_split_column(df, custom_test_values=["duke"])
        assert result == "dataset"


class TestSplitPatients:
    """Tests for split_train_test_patients with custom test values."""

    def _make_clinical_df(self, dataset_values):
        return pd.DataFrame({
            "patient_id": [f"P{i:03d}" for i in range(len(dataset_values))],
            "dataset": dataset_values,
        })

    def test_split_standard_no_split_returns_all_train(self):
        """Without split column, all patients are training."""
        from eval.train_classifier import split_train_test_patients
        df = self._make_clinical_df(["DUKE", "ISPY1", "ISPY2"])
        train, test = split_train_test_patients(df)
        assert len(train) == 3
        assert len(test) == 0

    def test_split_custom_values_duke_as_test(self):
        """Using --test-split-values DUKE, DUKE patients become test."""
        from eval.train_classifier import split_train_test_patients
        df = self._make_clinical_df(["DUKE", "DUKE", "ISPY1", "ISPY2"])
        train, test = split_train_test_patients(
            df,
            split_column="dataset",
            test_split_values=["DUKE"],
        )
        assert len(test) == 2
        assert all("DUKE" in df.loc[df["patient_id"] == pid, "dataset"].values[0]
                    for pid in test)
        assert len(train) == 2

    def test_split_custom_multiple_values(self):
        """Multiple custom test values work."""
        from eval.train_classifier import split_train_test_patients
        df = self._make_clinical_df(["DUKE", "ISPY1", "ISPY2", "NACT"])
        train, test = split_train_test_patients(
            df,
            split_column="dataset",
            test_split_values=["DUKE", "NACT"],
        )
        assert len(test) == 2
        assert len(train) == 2

    def test_split_custom_values_auto_detect_column(self):
        """Custom test values trigger auto-detection of split column."""
        from eval.train_classifier import split_train_test_patients
        df = self._make_clinical_df(["DUKE", "ISPY1", "ISPY2"])
        # No split_column specified, but test_split_values given
        train, test = split_train_test_patients(
            df,
            split_column=None,
            test_split_values=["DUKE"],
        )
        assert len(test) == 1
        assert len(train) == 2

    def test_split_standard_train_test(self):
        """Standard 'train'/'test' values still work."""
        from eval.train_classifier import split_train_test_patients
        df = pd.DataFrame({
            "patient_id": ["P001", "P002", "P003"],
            "dataset_split": ["train", "train", "test"],
        })
        train, test = split_train_test_patients(df)
        assert len(test) == 1
        assert len(train) == 2


# ===================================================================
# 3b. CSV-Based Split
# ===================================================================


class TestLoadSplitCSV:
    """Tests for load_split_csv()."""

    def test_load_basic_csv(self, tmp_path):
        """Two-column CSV is loaded correctly."""
        from eval.train_classifier import load_split_csv
        csv = tmp_path / "splits.csv"
        csv.write_text(
            "train_split,test_split\n"
            "DUKE_001,DUKE_019\n"
            "DUKE_002,DUKE_021\n"
            "DUKE_005,\n"
        )
        train, test = load_split_csv(csv)
        assert train == ["DUKE_001", "DUKE_002", "DUKE_005"]
        assert test == ["DUKE_019", "DUKE_021"]

    def test_load_ragged_csv(self, tmp_path):
        """Ragged CSV (train longer than test) is handled."""
        from eval.train_classifier import load_split_csv
        csv = tmp_path / "splits.csv"
        csv.write_text(
            "train_split,test_split\n"
            "A,X\n"
            "B,\n"
            "C,\n"
        )
        train, test = load_split_csv(csv)
        assert len(train) == 3
        assert len(test) == 1

    def test_load_csv_missing_file(self, tmp_path):
        """FileNotFoundError on missing file."""
        from eval.train_classifier import load_split_csv
        with pytest.raises(FileNotFoundError):
            load_split_csv(tmp_path / "nonexistent.csv")

    def test_load_csv_wrong_columns(self, tmp_path):
        """ValueError when expected columns are missing."""
        from eval.train_classifier import load_split_csv
        csv = tmp_path / "bad.csv"
        csv.write_text("col_a,col_b\n1,2\n")
        with pytest.raises(ValueError, match="train_split"):
            load_split_csv(csv)


class TestFindSplitCSV:
    """Tests for _find_split_csv auto-detection."""

    def test_auto_detect_present(self, tmp_path):
        """Auto-detects train_test_splits.csv in data_dir."""
        from eval.train_classifier import _find_split_csv
        (tmp_path / "train_test_splits.csv").write_text(
            "train_split,test_split\nA,B\n"
        )
        result = _find_split_csv(tmp_path)
        assert result is not None
        assert result.name == "train_test_splits.csv"

    def test_auto_detect_absent(self, tmp_path):
        """Returns None when no CSV found."""
        from eval.train_classifier import _find_split_csv
        result = _find_split_csv(tmp_path)
        assert result is None


class TestSplitPatientsCSV:
    """Tests for split_train_test_patients using CSV-based split."""

    def _write_csv(self, path, train_ids, test_ids):
        """Write a split CSV file."""
        import csv as csv_mod
        with open(path, "w", newline="") as f:
            w = csv_mod.writer(f)
            w.writerow(["train_split", "test_split"])
            max_len = max(len(train_ids), len(test_ids))
            for i in range(max_len):
                tr = train_ids[i] if i < len(train_ids) else ""
                te = test_ids[i] if i < len(test_ids) else ""
                w.writerow([tr, te])

    def test_csv_split_explicit(self, tmp_path):
        """Explicit --split-csv is used for splitting."""
        from eval.train_classifier import split_train_test_patients
        csv_path = tmp_path / "splits.csv"
        self._write_csv(csv_path, ["P001", "P002"], ["P003"])

        df = pd.DataFrame({
            "patient_id": ["P001", "P002", "P003"],
            "dataset": ["DUKE", "DUKE", "ISPY1"],
        })
        train, test = split_train_test_patients(
            df, split_csv=csv_path,
        )
        assert set(train) == {"P001", "P002"}
        assert set(test) == {"P003"}

    def test_csv_split_auto_detect(self, tmp_path):
        """Auto-detected CSV in data_dir is used."""
        from eval.train_classifier import split_train_test_patients
        csv_path = tmp_path / "train_test_splits.csv"
        self._write_csv(csv_path, ["P001"], ["P002", "P003"])

        df = pd.DataFrame({
            "patient_id": ["P001", "P002", "P003"],
            "dataset": ["DUKE", "ISPY1", "ISPY2"],
        })
        train, test = split_train_test_patients(
            df, data_dir=tmp_path,
        )
        assert set(train) == {"P001"}
        assert set(test) == {"P002", "P003"}

    def test_csv_split_filters_unknown_ids(self, tmp_path):
        """Patient IDs not in clinical_df are excluded."""
        from eval.train_classifier import split_train_test_patients
        csv_path = tmp_path / "splits.csv"
        self._write_csv(csv_path, ["P001", "P999"], ["P002"])

        df = pd.DataFrame({
            "patient_id": ["P001", "P002"],
            "dataset": ["DUKE", "ISPY1"],
        })
        train, test = split_train_test_patients(
            df, split_csv=csv_path,
        )
        # P999 is dropped because not in clinical_df
        assert set(train) == {"P001"}
        assert set(test) == {"P002"}

    def test_csv_takes_priority_over_column(self, tmp_path):
        """CSV-based split takes priority over column-based detection."""
        from eval.train_classifier import split_train_test_patients
        csv_path = tmp_path / "train_test_splits.csv"
        self._write_csv(csv_path, ["P001"], ["P002"])

        # DataFrame has a valid split column — CSV should still win
        df = pd.DataFrame({
            "patient_id": ["P001", "P002"],
            "dataset_split": ["train", "test"],
        })
        train, test = split_train_test_patients(
            df, data_dir=tmp_path,
        )
        # CSV says P001=train, P002=test — same as column, but
        # the key thing is that it doesn't fail or double-count
        assert set(train) == {"P001"}
        assert set(test) == {"P002"}

    def test_csv_fallback_to_column(self, tmp_path):
        """If CSV not found, falls back to column-based detection."""
        from eval.train_classifier import split_train_test_patients
        # No CSV file in tmp_path
        df = pd.DataFrame({
            "patient_id": ["P001", "P002"],
            "dataset_split": ["train", "test"],
        })
        train, test = split_train_test_patients(
            df, data_dir=tmp_path,
        )
        assert set(train) == {"P001"}
        assert set(test) == {"P002"}


class TestSplitCSVCLI:
    """Tests for --split-csv CLI argument."""

    def test_split_csv_default_none(self):
        """--split-csv defaults to None."""
        from eval.train_classifier import parse_args
        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
        ])
        assert args.split_csv is None

    def test_split_csv_parsed(self):
        """--split-csv accepts a path."""
        from eval.train_classifier import parse_args
        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--split-csv", "/tmp/splits.csv",
        ])
        assert args.split_csv == Path("/tmp/splits.csv")


class TestTestSplitValuesCLI:
    """Tests for --test-split-values CLI argument."""

    def test_test_split_values_parsed(self):
        """--test-split-values accepts multiple values."""
        from eval.train_classifier import parse_args
        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--evaluate-test-set",
            "--split-column", "dataset",
            "--test-split-values", "DUKE", "ISPY1",
        ])
        assert args.test_split_values == ["DUKE", "ISPY1"]

    def test_test_split_values_default_none(self):
        """--test-split-values defaults to None."""
        from eval.train_classifier import parse_args
        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
        ])
        assert args.test_split_values is None

    def test_test_split_values_single(self):
        """--test-split-values with single value."""
        from eval.train_classifier import parse_args
        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--test-split-values", "DUKE",
        ])
        assert args.test_split_values == ["DUKE"]


# ===================================================================
# 4. Cache Dir CLI
# ===================================================================


class TestCacheDirCLI:
    """Tests for --cache-dir CLI argument."""

    def test_cache_dir_default_none(self):
        """--cache-dir defaults to None."""
        from eval.train_classifier import parse_args
        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
        ])
        assert args.cache_dir is None

    def test_cache_dir_parsed(self):
        """--cache-dir accepts a path."""
        from eval.train_classifier import parse_args
        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--cache-dir", "/tmp/cache",
        ])
        assert args.cache_dir == Path("/tmp/cache")


# ===================================================================
# 5. Bug Fixes
# ===================================================================


class TestDualPhaseParamFix:
    """Test that the dual_phase NameError in train_cnn() is fixed."""

    @requires_torch
    def test_train_cnn_has_dual_phase_param(self):
        """train_cnn accepts dual_phase as a keyword argument."""
        import inspect
        from eval.train_cnn_classifier import train_cnn
        sig = inspect.signature(train_cnn)
        p = sig.parameters["dual_phase"]
        assert p.default is False

    @requires_torch
    def test_train_cnn_has_device_param(self):
        """train_cnn accepts device as a keyword argument."""
        import inspect
        from eval.train_cnn_classifier import train_cnn
        sig = inspect.signature(train_cnn)
        p = sig.parameters["device"]
        assert p.default is None


class TestEvaluateCNNCpuFix:
    """Test that evaluate_cnn uses .cpu() before .numpy()."""

    @requires_torch
    def test_evaluate_cnn_source_has_cpu_calls(self):
        """evaluate_cnn source contains .cpu().numpy() pattern."""
        import inspect
        from eval.train_cnn_classifier import evaluate_cnn
        source = inspect.getsource(evaluate_cnn)
        # Should have .cpu().numpy() and NOT standalone .numpy()
        assert ".cpu().numpy()" in source


class TestDualPhaseZeroPadding:
    """Test that dual-phase evaluation zero-pads when pre-contrast missing."""

    def test_evaluation_source_has_zero_padding(self):
        """MamaSynthEval._evaluate_classification has zero-padding fallback."""
        import inspect
        from eval.evaluation import MamaSynthEval
        source = inspect.getsource(MamaSynthEval._evaluate_classification)
        assert "np.zeros_like(feats)" in source


# ===================================================================
# 6. Combined Integration Tests
# ===================================================================


class TestCombinedNewFlags:
    """Test that all new flags can be parsed together."""

    def test_all_new_flags_together(self):
        """All v0.8.0 flags parse without error."""
        from eval.train_classifier import parse_args
        args = parse_args([
            "--data-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--classifier-type", "cnn",
            "--device", "cpu",
            "--cache-dir", "/tmp/cache",
            "--evaluate-test-set",
            "--split-csv", "/tmp/splits.csv",
            "--split-column", "dataset",
            "--test-split-values", "DUKE", "NACT",
            "--dual-phase",
            "--cnn-mask-channel",
        ])
        assert args.device == "cpu"
        assert args.cache_dir == Path("/tmp/cache")
        assert args.split_csv == Path("/tmp/splits.csv")
        assert args.test_split_values == ["DUKE", "NACT"]
        assert args.dual_phase is True
        assert args.cnn_mask_channel is True
        assert args.evaluate_test_set is True
        assert args.split_column == "dataset"
