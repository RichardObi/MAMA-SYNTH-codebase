#  Copyright 2025 mama-synth-eval contributors
#  Licensed under the Apache License, Version 2.0.

"""Tests for the synthesis module (synthesize.py)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: create a dummy NIfTI-like file (just needs to exist for path tests)
# ---------------------------------------------------------------------------

def _create_dummy_files(
    root: Path,
    patient_ids: list[str],
    phase: int = 0,
    nested: bool = True,
) -> list[Path]:
    """Create dummy directories and files matching MAMA-MIA layout."""
    files = []
    for pid in patient_ids:
        if nested:
            d = root / pid
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"{pid}_{phase:04d}.nii.gz"
        else:
            root.mkdir(parents=True, exist_ok=True)
            f = root / f"{pid}_{phase:04d}.nii.gz"
        f.touch()
        files.append(f)
    return files


# ---------------------------------------------------------------------------
# _extract_patient_id
# ---------------------------------------------------------------------------


class TestExtractPatientId:
    """Tests for _extract_patient_id helper."""

    def test_standard_nifti_name(self):
        from eval.synthesize import _extract_patient_id

        path = Path("/data/ISPY1_1001_0001.nii.gz")
        assert _extract_patient_id(path) == "ISPY1_1001"

    def test_pre_contrast(self):
        from eval.synthesize import _extract_patient_id

        path = Path("/data/DUKE_0042_0000.nii.gz")
        assert _extract_patient_id(path) == "DUKE_0042"

    def test_mha_format(self):
        from eval.synthesize import _extract_patient_id

        path = Path("patient_ABC_0002.mha")
        assert _extract_patient_id(path) == "patient_ABC"

    def test_no_phase_suffix(self):
        from eval.synthesize import _extract_patient_id

        path = Path("image_without_phase.nii.gz")
        assert _extract_patient_id(path) == "image_without_phase"


# ---------------------------------------------------------------------------
# _discover_input_images
# ---------------------------------------------------------------------------


class TestDiscoverInputImages:
    """Tests for input image discovery."""

    def test_nested_layout(self):
        from eval.synthesize import _discover_input_images

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_dummy_files(root, ["P001", "P002", "P003"], phase=0, nested=True)

            images = _discover_input_images(root, phase=0)

        assert len(images) == 3

    def test_flat_layout(self):
        from eval.synthesize import _discover_input_images

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_dummy_files(root, ["P001", "P002"], phase=0, nested=False)

            images = _discover_input_images(root, phase=0)

        assert len(images) == 2

    def test_wrong_phase_returns_empty(self):
        from eval.synthesize import _discover_input_images

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _create_dummy_files(root, ["P001"], phase=0, nested=True)

            # Looking for phase=1 should find nothing
            images = _discover_input_images(root, phase=1)

        assert len(images) == 0

    def test_empty_dir_returns_empty(self):
        from eval.synthesize import _discover_input_images

        with tempfile.TemporaryDirectory() as tmp:
            images = _discover_input_images(Path(tmp), phase=0)

        assert len(images) == 0


# ---------------------------------------------------------------------------
# CLI argument parsing — synthesize
# ---------------------------------------------------------------------------


class TestSynthesizeArgs:
    """Tests for parse_synthesize_args."""

    def test_minimal_args_with_data_dir(self):
        from eval.synthesize import parse_synthesize_args

        args = parse_synthesize_args([
            "--data-dir", "/path/to/mama-mia",
            "--output-dir", "/path/to/output",
        ])
        assert args.data_dir == Path("/path/to/mama-mia")
        assert args.output_dir == Path("/path/to/output")
        assert args.input_dir == Path("/path/to/mama-mia/images")
        assert args.model == "medigan"

    def test_input_dir_override(self):
        from eval.synthesize import parse_synthesize_args

        args = parse_synthesize_args([
            "--input-dir", "/custom/inputs",
            "--output-dir", "/path/to/output",
        ])
        assert args.input_dir == Path("/custom/inputs")

    def test_missing_both_dirs_fails(self):
        from eval.synthesize import parse_synthesize_args

        with pytest.raises(SystemExit):
            parse_synthesize_args(["--output-dir", "/tmp/out"])


# ---------------------------------------------------------------------------
# CLI argument parsing — synthesize-and-evaluate
# ---------------------------------------------------------------------------


class TestSynthesizeAndEvaluateArgs:
    """Tests for parse_synthesize_and_evaluate_args."""

    def test_predictions_dir_skips_synthesis(self):
        from eval.synthesize import parse_synthesize_and_evaluate_args

        args = parse_synthesize_and_evaluate_args([
            "--predictions-dir", "/path/to/preds",
            "--ground-truth-path", "/path/to/gt",
            "--output-file", "metrics.json",
        ])
        assert args._skip_synthesis is True
        assert args.predictions_dir == Path("/path/to/preds")

    def test_data_dir_defaults(self):
        from eval.synthesize import parse_synthesize_and_evaluate_args

        with tempfile.TemporaryDirectory() as tmp:
            # Create segmentations dir so it auto-detects
            seg = Path(tmp) / "segmentations"
            seg.mkdir()

            args = parse_synthesize_and_evaluate_args([
                "--data-dir", tmp,
                "--output-dir", "/tmp/out",
                "--output-file", "m.json",
            ])

        assert args.ground_truth_path == Path(tmp) / "images"
        assert args.masks_path == Path(tmp) / "segmentations"
        assert args._skip_synthesis is False

    def test_synthesis_mode_requires_output_dir(self):
        from eval.synthesize import parse_synthesize_and_evaluate_args

        with pytest.raises(SystemExit):
            parse_synthesize_and_evaluate_args([
                "--data-dir", "/path/to/data",
                "--output-file", "m.json",
            ])

    def test_evaluation_options_parsed(self):
        from eval.synthesize import parse_synthesize_and_evaluate_args

        args = parse_synthesize_and_evaluate_args([
            "--predictions-dir", "/preds",
            "--ground-truth-path", "/gt",
            "--output-file", "out.json",
            "--labels-path", "/labels.csv",
            "--clf-model-dir", "/models",
            "--disable-lpips",
            "--disable-segmentation",
        ])
        assert args.labels_path == Path("/labels.csv")
        assert args.clf_model_dir == Path("/models")
        assert args.disable_lpips is True
        assert args.disable_segmentation is True
        assert args.disable_frd is False


# ---------------------------------------------------------------------------
# run_evaluation (smoke test with mocked MamaSynthEval)
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    """Tests for the run_evaluation wrapper."""

    def test_delegates_to_evaluator(self):
        """run_evaluation should instantiate MamaSynthEval and call evaluate()."""
        from unittest.mock import patch, MagicMock

        mock_eval = MagicMock()
        mock_eval.evaluate.return_value = {
            "aggregates": {"mse_full_image": {"mean": 0.01, "std": 0.005}},
            "results": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "metrics.json"

            with patch(
                "eval.evaluation.MamaSynthEval", return_value=mock_eval,
            ) as mock_cls:
                from eval.synthesize import run_evaluation

                result = run_evaluation(
                    predictions_dir=Path("/preds"),
                    ground_truth_dir=Path("/gt"),
                    output_file=out_file,
                )

        mock_cls.assert_called_once()
        mock_eval.evaluate.assert_called_once()
        assert "aggregates" in result
