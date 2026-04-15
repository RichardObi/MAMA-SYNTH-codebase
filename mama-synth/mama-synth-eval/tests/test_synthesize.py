#  Copyright 2025 mama-synth-eval contributors
#  Licensed under the Apache License, Version 2.0.

"""Tests for the synthesis module (synthesize.py)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# _ensure_medigan_importable
# ---------------------------------------------------------------------------


class TestEnsureMediganImportable:
    """Tests for _ensure_medigan_importable helper."""

    def test_adds_cwd_to_sys_path(self):
        from eval.synthesize import _ensure_medigan_importable

        with tempfile.TemporaryDirectory() as tmp:
            real_tmp = os.path.realpath(tmp)
            original_cwd = os.getcwd()
            original_path = list(sys.path)
            try:
                os.chdir(real_tmp)
                # Remove it from sys.path if already there
                sys.path = [p for p in sys.path if p != real_tmp]

                _ensure_medigan_importable()

                assert real_tmp in sys.path
            finally:
                os.chdir(original_cwd)
                sys.path = original_path

    def test_creates_models_init_py(self):
        from eval.synthesize import _ensure_medigan_importable

        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            original_path = list(sys.path)
            try:
                os.chdir(tmp)
                _ensure_medigan_importable()

                assert (Path(tmp) / "models" / "__init__.py").exists()
            finally:
                os.chdir(original_cwd)
                sys.path = original_path

    def test_idempotent(self):
        from eval.synthesize import _ensure_medigan_importable

        with tempfile.TemporaryDirectory() as tmp:
            real_tmp = os.path.realpath(tmp)
            original_cwd = os.getcwd()
            original_path = list(sys.path)
            try:
                os.chdir(real_tmp)
                _ensure_medigan_importable()
                _ensure_medigan_importable()  # second call should not error

                assert (Path(real_tmp) / "models" / "__init__.py").exists()
                assert sys.path.count(real_tmp) == 1
            finally:
                os.chdir(original_cwd)
                sys.path = original_path


# ---------------------------------------------------------------------------
# _nifti_to_png_slices / _png_slices_to_nifti roundtrip
# ---------------------------------------------------------------------------

# Import sys at module level for the path cleanup in tests
import sys


def _make_test_nifti(path: Path, shape: tuple = (4, 8, 8)) -> np.ndarray:
    """Create a small NIfTI file and return the underlying array."""
    import SimpleITK as sitk

    arr = np.random.RandomState(42).rand(*shape).astype(np.float32) * 1000
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))
    return arr


class TestNiftiPngRoundtrip:
    """Tests for _nifti_to_png_slices and _png_slices_to_nifti."""

    @pytest.fixture(autouse=True)
    def _check_sitk(self):
        pytest.importorskip("SimpleITK")
        pytest.importorskip("PIL")

    def test_roundtrip_preserves_shape(self):
        from eval.synthesize import _nifti_to_png_slices, _png_slices_to_nifti

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            nifti_path = tmp / "test.nii.gz"
            png_dir = tmp / "slices"
            out_path = tmp / "roundtrip.nii.gz"

            original = _make_test_nifti(nifti_path, shape=(4, 16, 16))
            meta = _nifti_to_png_slices(nifti_path, png_dir)

            assert meta["num_slices"] == 4
            assert meta["shape"] == (4, 16, 16)
            assert len(list(png_dir.glob("*.png"))) == 4

            _png_slices_to_nifti(png_dir, out_path, meta)

            import SimpleITK as sitk

            result = sitk.GetArrayFromImage(
                sitk.ReadImage(str(out_path))
            )
            assert result.shape == original.shape

    def test_roundtrip_preserves_intensity_range(self):
        from eval.synthesize import _nifti_to_png_slices, _png_slices_to_nifti

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            nifti_path = tmp / "test.nii.gz"
            png_dir = tmp / "slices"
            out_path = tmp / "roundtrip.nii.gz"

            original = _make_test_nifti(nifti_path, shape=(3, 10, 10))
            meta = _nifti_to_png_slices(nifti_path, png_dir)
            _png_slices_to_nifti(png_dir, out_path, meta)

            import SimpleITK as sitk

            result = sitk.GetArrayFromImage(
                sitk.ReadImage(str(out_path))
            ).astype(np.float32)

            # Intensity range should be approximately preserved
            # (quantisation to uint8 introduces some error)
            assert abs(result.min() - original.min()) < (
                original.max() - original.min()
            ) * 0.05
            assert abs(result.max() - original.max()) < (
                original.max() - original.min()
            ) * 0.05

    def test_roundtrip_preserves_spacing(self):
        from eval.synthesize import _nifti_to_png_slices, _png_slices_to_nifti

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            nifti_path = tmp / "test.nii.gz"
            png_dir = tmp / "slices"
            out_path = tmp / "roundtrip.nii.gz"

            _make_test_nifti(nifti_path)
            meta = _nifti_to_png_slices(nifti_path, png_dir)
            _png_slices_to_nifti(png_dir, out_path, meta)

            import SimpleITK as sitk

            result_img = sitk.ReadImage(str(out_path))
            assert result_img.GetSpacing() == pytest.approx((1.0, 1.0, 1.0))

    def test_nifti_to_png_invalid_dim_raises(self):
        from eval.synthesize import _nifti_to_png_slices

        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            nifti_path = tmp / "flat.nii.gz"
            arr = np.zeros((8, 8), dtype=np.float32)
            img = sitk.GetImageFromArray(arr)
            sitk.WriteImage(img, str(nifti_path))

            with pytest.raises(ValueError, match="Expected 3D or 4D"):
                _nifti_to_png_slices(nifti_path, tmp / "out")

    def test_png_to_nifti_empty_dir_raises(self):
        from eval.synthesize import _png_slices_to_nifti

        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with pytest.raises(FileNotFoundError, match="No image files"):
                _png_slices_to_nifti(
                    empty,
                    Path(tmp) / "out.nii.gz",
                    {"shape": (1, 8, 8), "min_val": 0, "max_val": 1,
                     "spacing": (1, 1, 1), "origin": (0, 0, 0),
                     "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1)},
                )

    def test_png_to_nifti_finds_subdir(self):
        """_png_slices_to_nifti should find PNGs in subdirectories like batch_0/."""
        from eval.synthesize import _nifti_to_png_slices, _png_slices_to_nifti

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            nifti_path = tmp / "test.nii.gz"
            png_dir = tmp / "slices"
            out_root = tmp / "output"
            out_path = tmp / "reassembled.nii.gz"

            _make_test_nifti(nifti_path, shape=(3, 8, 8))
            meta = _nifti_to_png_slices(nifti_path, png_dir)

            # Simulate medigan saving into a batch_0/ subdirectory
            batch_dir = out_root / "batch_0"
            batch_dir.mkdir(parents=True)
            import shutil
            for png in png_dir.glob("*.png"):
                shutil.copy(png, batch_dir / png.name)

            # Pass the parent — the function should find batch_0/
            _png_slices_to_nifti(out_root, out_path, meta)

            import SimpleITK as sitk
            result = sitk.GetArrayFromImage(sitk.ReadImage(str(out_path)))
            assert result.shape == (3, 8, 8)


# ---------------------------------------------------------------------------
# _find_generated_images
# ---------------------------------------------------------------------------


class TestFindGeneratedImages:
    """Tests for _find_generated_images helper."""

    def test_root_has_pngs(self):
        from eval.synthesize import _find_generated_images

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.png").touch()
            (root / "b.png").touch()
            result = _find_generated_images(root)
            assert len(result) == 2
            assert all(p.suffix == ".png" for p in result)

    def test_finds_in_subdirectory(self):
        from eval.synthesize import _find_generated_images

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "batch_0"
            sub.mkdir()
            (sub / "out.png").touch()
            result = _find_generated_images(root)
            assert len(result) == 1

    def test_multiple_image_formats(self):
        from eval.synthesize import _find_generated_images

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
                (root / f"img{ext}").touch()
            result = _find_generated_images(root)
            assert len(result) == 5

    def test_ignores_non_image_files(self):
        from eval.synthesize import _find_generated_images

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "readme.txt").touch()
            (root / "data.csv").touch()
            (root / "image.png").touch()
            result = _find_generated_images(root)
            assert len(result) == 1
            assert result[0].name == "image.png"

    def test_sorted_output(self):
        from eval.synthesize import _find_generated_images

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("c.png", "a.png", "b.png"):
                (root / name).touch()
            result = _find_generated_images(root)
            assert [p.name for p in result] == ["a.png", "b.png", "c.png"]

    def test_empty_dir_raises(self):
        from eval.synthesize import _find_generated_images

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError, match="No image files"):
                _find_generated_images(Path(tmp))


# ---------------------------------------------------------------------------
# _find_png_output_dir
# ---------------------------------------------------------------------------


class TestFindPngOutputDir:
    """Tests for _find_png_output_dir helper."""

    def test_root_has_pngs(self):
        from eval.synthesize import _find_png_output_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.png").touch()
            assert _find_png_output_dir(root) == root

    def test_subdir_has_pngs(self):
        from eval.synthesize import _find_png_output_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "batch_0"
            sub.mkdir()
            (sub / "out.png").touch()
            assert _find_png_output_dir(root) == sub

    def test_nested_subdir(self):
        from eval.synthesize import _find_png_output_dir

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deep = root / "a" / "b"
            deep.mkdir(parents=True)
            (deep / "img.png").touch()
            assert _find_png_output_dir(root) == deep

    def test_empty_dir_raises(self):
        from eval.synthesize import _find_png_output_dir

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError, match="No PNG files"):
                _find_png_output_dir(Path(tmp))


# ---------------------------------------------------------------------------
# synthesize_with_medigan (mocked medigan)
# ---------------------------------------------------------------------------


class TestSynthesizeWithMedigan:
    """Tests for the synthesize_with_medigan pipeline (medigan is mocked)."""

    @pytest.fixture(autouse=True)
    def _check_sitk(self):
        pytest.importorskip("SimpleITK")
        pytest.importorskip("PIL")

    def test_calls_medigan_with_png_dirs(self):
        """synthesize_with_medigan should pass temporary PNG directories
        to generators.generate(), not raw NIfTI paths."""
        from eval.synthesize import synthesize_with_medigan

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            input_dir = tmp / "input" / "P001"
            input_dir.mkdir(parents=True)
            output_dir = tmp / "output"

            _make_test_nifti(
                input_dir / "P001_0000.nii.gz", shape=(3, 8, 8)
            )

            mock_gen_instance = MagicMock()

            # Make generate() side-effect: copy input PNGs to output dir
            # in a batch_0/ subdirectory (matching real medigan behaviour)
            def fake_generate(**kwargs):
                in_dir = Path(kwargs["input_path"])
                out_dir = Path(kwargs["output_path"]) / "batch_0"
                out_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                for png in sorted(in_dir.glob("*.png")):
                    shutil.copy(png, out_dir / png.name)

            mock_gen_instance.generate.side_effect = fake_generate

            # Build a fake medigan module with a Generators class
            mock_medigan = MagicMock()
            mock_medigan.Generators.return_value = mock_gen_instance

            original_cwd = os.getcwd()
            original_path = list(sys.path)
            try:
                os.chdir(tmp)
                with patch.dict(
                    "sys.modules",
                    {"medigan": mock_medigan},
                ):
                    with patch(
                        "eval.synthesize._ensure_medigan_importable"
                    ):
                        result = synthesize_with_medigan(
                            input_dir=tmp / "input",
                            output_dir=output_dir,
                            gpu_id="-1",
                            image_size=256,
                        )
            finally:
                os.chdir(original_cwd)
                sys.path = original_path

            # Verify generate was called with directory paths (not NIfTI)
            call_kwargs = mock_gen_instance.generate.call_args
            assert call_kwargs is not None
            kw = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
            assert not kw["input_path"].endswith(".nii.gz")
            assert kw["image_size"] == "256"
            assert kw["gpu_id"] == "cpu"

            # Output PNG should have been written (no longer NIfTI)
            assert len(result) == 1
            assert result[0].exists()
            assert result[0].suffix == ".png"

            # num_samples should equal the number of extracted slices
            kw = (
                mock_gen_instance.generate.call_args.kwargs
                if mock_gen_instance.generate.call_args.kwargs
                else mock_gen_instance.generate.call_args[1]
            )
            assert kw["num_samples"] == 1  # single slice (no mask → middle)

    def test_num_samples_matches_slice_count(self):
        """num_samples passed to medigan must equal the extracted slice count."""
        pytest.importorskip("SimpleITK")
        from eval.synthesize import synthesize_with_medigan

        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            input_dir = tmp / "input" / "P001"
            input_dir.mkdir(parents=True)
            output_dir = tmp / "output"
            masks_dir = tmp / "masks"
            masks_dir.mkdir()

            _make_test_nifti(
                input_dir / "P001_0000.nii.gz", shape=(10, 8, 8),
            )

            # Create a mask with tumour in 3 slices
            mask = np.zeros((10, 8, 8), dtype=np.uint8)
            mask[2, 3:5, 3:5] = 1
            mask[5, 3:5, 3:5] = 1
            mask[8, 3:5, 3:5] = 1
            img = sitk.GetImageFromArray(mask)
            sitk.WriteImage(img, str(masks_dir / "P001_0000.nii.gz"))

            mock_gen_instance = MagicMock()

            def fake_generate(**kwargs):
                in_dir = Path(kwargs["input_path"])
                out_dir = Path(kwargs["output_path"]) / "batch_0"
                out_dir.mkdir(parents=True, exist_ok=True)
                import shutil as _sh
                for png in sorted(in_dir.glob("*.png")):
                    _sh.copy(png, out_dir / png.name)

            mock_gen_instance.generate.side_effect = fake_generate
            mock_medigan = MagicMock()
            mock_medigan.Generators.return_value = mock_gen_instance

            original_cwd = os.getcwd()
            original_path = list(sys.path)
            try:
                os.chdir(tmp)
                with patch.dict("sys.modules", {"medigan": mock_medigan}):
                    with patch("eval.synthesize._ensure_medigan_importable"):
                        result = synthesize_with_medigan(
                            input_dir=tmp / "input",
                            output_dir=output_dir,
                            gpu_id="-1",
                            image_size=256,
                            masks_dir=masks_dir,
                            slice_mode="all_tumor",
                        )
            finally:
                os.chdir(original_cwd)
                sys.path = original_path

            kw = (
                mock_gen_instance.generate.call_args.kwargs
                if mock_gen_instance.generate.call_args.kwargs
                else mock_gen_instance.generate.call_args[1]
            )
            assert kw["num_samples"] == 3, (
                f"Expected num_samples=3 (3 tumour slices), got {kw['num_samples']}"
            )
            assert len(result) == 3
        """Should raise FileNotFoundError when input dir is empty."""
        from eval.synthesize import synthesize_with_medigan

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            (tmp / "input").mkdir()

            mock_medigan = MagicMock()

            original_cwd = os.getcwd()
            original_path = list(sys.path)
            try:
                os.chdir(tmp)
                with patch.dict(
                    "sys.modules",
                    {"medigan": mock_medigan},
                ):
                    with patch(
                        "eval.synthesize._ensure_medigan_importable"
                    ):
                        with pytest.raises(FileNotFoundError, match="No pre-contrast"):
                            synthesize_with_medigan(
                                input_dir=tmp / "input",
                                output_dir=tmp / "output",
                            )
            finally:
                os.chdir(original_cwd)
                sys.path = original_path


# ---------------------------------------------------------------------------
# CLI argument parsing — new options
# ---------------------------------------------------------------------------


class TestNewCLIOptions:
    """Tests for --gpu-id and --image-size CLI options."""

    def test_synthesize_gpu_id_default(self):
        from eval.synthesize import parse_synthesize_args

        args = parse_synthesize_args([
            "--data-dir", "/data",
            "--output-dir", "/out",
        ])
        assert args.gpu_id == "0"

    def test_synthesize_gpu_id_custom(self):
        from eval.synthesize import parse_synthesize_args

        args = parse_synthesize_args([
            "--data-dir", "/data",
            "--output-dir", "/out",
            "--gpu-id", "-1",
        ])
        assert args.gpu_id == "-1"

    def test_synthesize_image_size_default(self):
        from eval.synthesize import parse_synthesize_args

        args = parse_synthesize_args([
            "--data-dir", "/data",
            "--output-dir", "/out",
        ])
        assert args.image_size == 512

    def test_synthesize_image_size_custom(self):
        from eval.synthesize import parse_synthesize_args

        args = parse_synthesize_args([
            "--data-dir", "/data",
            "--output-dir", "/out",
            "--image-size", "256",
        ])
        assert args.image_size == 256

    def test_synth_and_eval_gpu_id(self):
        from eval.synthesize import parse_synthesize_and_evaluate_args

        args = parse_synthesize_and_evaluate_args([
            "--predictions-dir", "/preds",
            "--ground-truth-path", "/gt",
            "--output-file", "m.json",
            "--gpu-id", "2",
        ])
        assert args.gpu_id == "2"

    def test_synth_and_eval_image_size(self):
        from eval.synthesize import parse_synthesize_and_evaluate_args

        args = parse_synthesize_and_evaluate_args([
            "--predictions-dir", "/preds",
            "--ground-truth-path", "/gt",
            "--output-file", "m.json",
            "--image-size", "1024",
        ])
        assert args.image_size == 1024

    def test_synthesize_keep_work_dir_default(self):
        from eval.synthesize import parse_synthesize_args

        args = parse_synthesize_args([
            "--data-dir", "/data",
            "--output-dir", "/out",
        ])
        assert args.keep_work_dir is False

    def test_synthesize_keep_work_dir_flag(self):
        from eval.synthesize import parse_synthesize_args

        args = parse_synthesize_args([
            "--data-dir", "/data",
            "--output-dir", "/out",
            "--keep-work-dir",
        ])
        assert args.keep_work_dir is True

    def test_synth_and_eval_keep_work_dir_default(self):
        from eval.synthesize import parse_synthesize_and_evaluate_args

        args = parse_synthesize_and_evaluate_args([
            "--predictions-dir", "/preds",
            "--ground-truth-path", "/gt",
            "--output-file", "m.json",
        ])
        assert args.keep_work_dir is False

    def test_synth_and_eval_keep_work_dir_flag(self):
        from eval.synthesize import parse_synthesize_and_evaluate_args

        args = parse_synthesize_and_evaluate_args([
            "--predictions-dir", "/preds",
            "--ground-truth-path", "/gt",
            "--output-file", "m.json",
            "--keep-work-dir",
        ])
        assert args.keep_work_dir is True


# ---------------------------------------------------------------------------
# Staging directory behaviour (synthesize_with_medigan)
# ---------------------------------------------------------------------------


class TestStagingDirectoryBehaviour:
    """Tests that synthesize_with_medigan uses in-project staging dirs."""

    @pytest.fixture(autouse=True)
    def _check_sitk(self):
        pytest.importorskip("SimpleITK")
        pytest.importorskip("PIL")

    def _run_synthesis(self, tmp, *, keep_work_dir=False, fail=False):
        """Helper: run synthesize_with_medigan with mocked medigan."""
        from eval.synthesize import synthesize_with_medigan, WORK_SUBDIR

        input_dir = tmp / "input" / "P001"
        input_dir.mkdir(parents=True)
        output_dir = tmp / "output"

        _make_test_nifti(
            input_dir / "P001_0000.nii.gz", shape=(3, 8, 8)
        )

        mock_gen_instance = MagicMock()

        def fake_generate(**kwargs):
            if fail:
                raise RuntimeError("Simulated medigan failure")
            in_dir = Path(kwargs["input_path"])
            out_dir = Path(kwargs["output_path"]) / "batch_0"
            out_dir.mkdir(parents=True, exist_ok=True)
            import shutil as _shutil
            for png in sorted(in_dir.glob("*.png")):
                _shutil.copy(png, out_dir / png.name)

        mock_gen_instance.generate.side_effect = fake_generate

        mock_medigan = MagicMock()
        mock_medigan.Generators.return_value = mock_gen_instance

        original_cwd = os.getcwd()
        original_path = list(sys.path)
        try:
            os.chdir(tmp)
            with patch.dict("sys.modules", {"medigan": mock_medigan}):
                with patch("eval.synthesize._ensure_medigan_importable"):
                    result = synthesize_with_medigan(
                        input_dir=tmp / "input",
                        output_dir=output_dir,
                        gpu_id="-1",
                        image_size=256,
                        keep_work_dir=keep_work_dir,
                    )
        finally:
            os.chdir(original_cwd)
            sys.path = original_path

        return result, output_dir, output_dir / WORK_SUBDIR

    def test_staging_dir_inside_output_dir(self):
        """Staging directory should be created inside output_dir."""
        from eval.synthesize import WORK_SUBDIR

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            result, output_dir, work_root = self._run_synthesis(
                tmp, keep_work_dir=True
            )

            # Work dir exists inside output_dir
            assert work_root.parent == output_dir
            assert work_root.name == WORK_SUBDIR
            assert work_root.exists()

    def test_work_dir_cleaned_on_success(self):
        """Staging dirs should be removed after successful synthesis."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            result, output_dir, work_root = self._run_synthesis(tmp)

            # Work dir should be removed (or at least empty)
            assert not work_root.exists() or not list(work_root.iterdir())
            # But output PNG should exist
            assert len(result) == 1
            assert result[0].exists()
            assert result[0].suffix == ".png"

    def test_keep_work_dir_preserves_staging(self):
        """With keep_work_dir=True, staging dirs should persist."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            result, output_dir, work_root = self._run_synthesis(
                tmp, keep_work_dir=True
            )

            assert work_root.exists()
            # Patient dir should still be there
            patient_dirs = list(work_root.iterdir())
            assert len(patient_dirs) == 1
            assert patient_dirs[0].name == "P001"

    def test_work_dir_retained_on_failure(self):
        """On synthesis failure, staging dirs should be kept for debugging."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            result, output_dir, work_root = self._run_synthesis(
                tmp, fail=True
            )

            # Should have 0 successful results
            assert len(result) == 0
            # But work dir should still exist with patient subdirectory
            assert work_root.exists()
            patient_dirs = list(work_root.iterdir())
            assert len(patient_dirs) == 1
            assert patient_dirs[0].name == "P001"


# ---------------------------------------------------------------------------
# _select_slices
# ---------------------------------------------------------------------------


class TestSelectSlices:
    """Tests for _select_slices helper."""

    def test_max_tumor_returns_single_index(self):
        from eval.synthesize import _select_slices

        volume = np.zeros((10, 8, 8), dtype=np.float32)
        mask = np.zeros((10, 8, 8), dtype=bool)
        mask[7, 2:6, 2:6] = True  # large area at slice 7

        result = _select_slices(volume, mask, "max_tumor")
        assert result == [7]

    def test_center_tumor_returns_centroid_slice(self):
        from eval.synthesize import _select_slices

        volume = np.zeros((10, 8, 8), dtype=np.float32)
        mask = np.zeros((10, 8, 8), dtype=bool)
        mask[3:7, 3:5, 3:5] = True  # span slices 3–6, centroid at ~4.5

        result = _select_slices(volume, mask, "center_tumor")
        assert len(result) == 1
        assert 3 <= result[0] <= 6  # within tumour extent

    def test_all_tumor_returns_all_foreground_slices(self):
        from eval.synthesize import _select_slices

        volume = np.zeros((10, 8, 8), dtype=np.float32)
        mask = np.zeros((10, 8, 8), dtype=bool)
        mask[2, 4, 4] = True
        mask[5, 3, 3] = True
        mask[8, 2, 2] = True

        result = _select_slices(volume, mask, "all_tumor")
        assert result == [2, 5, 8]

    def test_max_tumor_no_mask_falls_back_to_middle(self):
        from eval.synthesize import _select_slices

        volume = np.zeros((10, 8, 8), dtype=np.float32)
        result = _select_slices(volume, None, "max_tumor")
        assert result == [5]

    def test_all_tumor_no_mask_raises(self):
        from eval.synthesize import _select_slices

        volume = np.zeros((10, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="requires a segmentation mask"):
            _select_slices(volume, None, "all_tumor")

    def test_unknown_mode_raises(self):
        from eval.synthesize import _select_slices

        volume = np.zeros((10, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown slice_mode"):
            _select_slices(volume, None, "invalid_mode")


# ---------------------------------------------------------------------------
# _load_mask_for_patient
# ---------------------------------------------------------------------------


class TestLoadMaskForPatient:
    """Tests for _load_mask_for_patient helper."""

    @pytest.fixture(autouse=True)
    def _check_sitk(self):
        pytest.importorskip("SimpleITK")

    def test_flat_layout(self):
        from eval.synthesize import _load_mask_for_patient

        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as tmp:
            masks_dir = Path(tmp)
            # Create mask file {pid}_0000.nii.gz
            arr = np.ones((4, 8, 8), dtype=np.uint8)
            img = sitk.GetImageFromArray(arr)
            sitk.WriteImage(img, str(masks_dir / "P001_0000.nii.gz"))

            result = _load_mask_for_patient("P001", masks_dir)
            assert result is not None
            assert result.dtype == bool
            assert result.shape == (4, 8, 8)

    def test_nested_layout(self):
        from eval.synthesize import _load_mask_for_patient

        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as tmp:
            masks_dir = Path(tmp)
            nested = masks_dir / "P001"
            nested.mkdir()
            arr = np.ones((4, 8, 8), dtype=np.uint8)
            img = sitk.GetImageFromArray(arr)
            sitk.WriteImage(img, str(nested / "P001_0000.nii.gz"))

            result = _load_mask_for_patient("P001", masks_dir)
            assert result is not None

    def test_no_masks_dir_returns_none(self):
        from eval.synthesize import _load_mask_for_patient

        assert _load_mask_for_patient("P001", None) is None

    def test_no_matching_file_returns_none(self):
        from eval.synthesize import _load_mask_for_patient

        with tempfile.TemporaryDirectory() as tmp:
            assert _load_mask_for_patient("P999", Path(tmp)) is None

    def test_mask_without_phase_suffix_4digit_pid(self):
        """Regression: mask named {pid}.nii.gz where pid ends in 4 digits.

        _extract_patient_id previously stripped the trailing 4-digit part
        from the mask filename, causing a mismatch when the patient ID
        itself ended in 4 digits (e.g. ISPY1_1001.nii.gz -> ISPY1).
        """
        from eval.synthesize import _load_mask_for_patient

        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as tmp:
            masks_dir = Path(tmp)
            # Flat mask WITHOUT phase suffix, pid ends in 4 digits
            arr = np.ones((4, 8, 8), dtype=np.uint8)
            img = sitk.GetImageFromArray(arr)
            sitk.WriteImage(img, str(masks_dir / "ISPY1_1001.nii.gz"))

            result = _load_mask_for_patient("ISPY1_1001", masks_dir)
            assert result is not None, (
                "Mask should be found for 4-digit patient ID without phase suffix"
            )
            assert result.dtype == bool

    def test_nested_mask_without_phase_suffix_4digit_pid(self):
        """Regression: nested mask named {pid}/{pid}.nii.gz, 4-digit pid."""
        from eval.synthesize import _load_mask_for_patient

        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as tmp:
            masks_dir = Path(tmp)
            nested = masks_dir / "ISPY1_1001"
            nested.mkdir()
            arr = np.ones((4, 8, 8), dtype=np.uint8)
            img = sitk.GetImageFromArray(arr)
            sitk.WriteImage(img, str(nested / "ISPY1_1001.nii.gz"))

            result = _load_mask_for_patient("ISPY1_1001", masks_dir)
            assert result is not None, (
                "Nested mask should be found for 4-digit patient ID"
            )


# ---------------------------------------------------------------------------
# _stem_matches_patient
# ---------------------------------------------------------------------------


class TestStemMatchesPatient:
    """Tests for the _stem_matches_patient helper."""

    def test_exact_match_no_suffix(self):
        from eval.synthesize import _stem_matches_patient

        assert _stem_matches_patient("DUKE_055.nii.gz", "DUKE_055")

    def test_match_with_phase_suffix(self):
        from eval.synthesize import _stem_matches_patient

        assert _stem_matches_patient("DUKE_055_0000.nii.gz", "DUKE_055")

    def test_no_match_different_patient(self):
        from eval.synthesize import _stem_matches_patient

        assert not _stem_matches_patient("DUKE_056.nii.gz", "DUKE_055")

    def test_no_false_positive_prefix(self):
        """DUKE_05 must not match DUKE_055."""
        from eval.synthesize import _stem_matches_patient

        assert not _stem_matches_patient("DUKE_055.nii.gz", "DUKE_05")
        assert not _stem_matches_patient("DUKE_055_0000.nii.gz", "DUKE_05")

    def test_4digit_pid_exact(self):
        """Patient ID ends in 4 digits — still a valid exact match."""
        from eval.synthesize import _stem_matches_patient

        assert _stem_matches_patient("ISPY1_1001.nii.gz", "ISPY1_1001")

    def test_4digit_pid_with_phase(self):
        from eval.synthesize import _stem_matches_patient

        assert _stem_matches_patient("ISPY1_1001_0000.nii.gz", "ISPY1_1001")

    def test_png_extension(self):
        from eval.synthesize import _stem_matches_patient

        assert _stem_matches_patient("DUKE_055.png", "DUKE_055")

    def test_mha_extension(self):
        from eval.synthesize import _stem_matches_patient

        assert _stem_matches_patient("DUKE_055_0000.mha", "DUKE_055")

    def test_no_match_non_phase_suffix(self):
        """DUKE_055_seg.nii.gz should NOT match (suffix 'seg' is not 4 digits)."""
        from eval.synthesize import _stem_matches_patient

        assert not _stem_matches_patient("DUKE_055_seg.nii.gz", "DUKE_055")


# ---------------------------------------------------------------------------
# _extract_and_save_slices
# ---------------------------------------------------------------------------


class TestExtractAndSaveSlices:
    """Tests for _extract_and_save_slices helper."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        pytest.importorskip("SimpleITK")
        pytest.importorskip("PIL")

    def test_single_slice_patient_naming(self):
        from eval.synthesize import _extract_and_save_slices

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            nifti = tmp / "test.nii.gz"
            out = tmp / "slices"
            _make_test_nifti(nifti, shape=(4, 8, 8))

            result = _extract_and_save_slices(
                nifti, out, "P001", slice_mode="max_tumor",
            )
            assert len(result) == 1
            assert result[0][0].name == "P001.png"
            assert result[0][0].exists()

    def test_sequential_naming(self):
        from eval.synthesize import _extract_and_save_slices

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            nifti = tmp / "test.nii.gz"
            out = tmp / "slices"
            _make_test_nifti(nifti, shape=(4, 8, 8))

            result = _extract_and_save_slices(
                nifti, out, "P001", slice_mode="max_tumor",
                sequential_naming=True,
            )
            assert len(result) == 1
            assert result[0][0].name == "slice_0000.png"

    def test_all_tumor_multi_slice_naming(self):
        from eval.synthesize import _extract_and_save_slices

        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            nifti = tmp / "test.nii.gz"
            out = tmp / "slices"
            _make_test_nifti(nifti, shape=(10, 8, 8))

            # Create mask with tumour in slices 2 and 7
            mask = np.zeros((10, 8, 8), dtype=bool)
            mask[2, 3:5, 3:5] = True
            mask[7, 2:6, 2:6] = True

            result = _extract_and_save_slices(
                nifti, out, "P001", mask_arr=mask,
                slice_mode="all_tumor",
            )
            assert len(result) == 2
            names = [r[0].name for r in result]
            assert "P001_s0002.png" in names
            assert "P001_s0007.png" in names

    def test_output_is_grayscale_png(self):
        from eval.synthesize import _extract_and_save_slices
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            nifti = tmp / "test.nii.gz"
            out = tmp / "slices"
            _make_test_nifti(nifti, shape=(4, 16, 16))

            result = _extract_and_save_slices(
                nifti, out, "P001", slice_mode="max_tumor",
            )
            img = Image.open(result[0][0])
            assert img.mode == "L"
            assert img.size == (16, 16)


# ---------------------------------------------------------------------------
# extract_ground_truth_slices
# ---------------------------------------------------------------------------


class TestExtractGroundTruthSlices:
    """Tests for extract_ground_truth_slices."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        pytest.importorskip("SimpleITK")
        pytest.importorskip("PIL")

    def test_extracts_gt_slices_as_png(self):
        from eval.synthesize import extract_ground_truth_slices

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Create GT dir with post-contrast NIfTI
            gt_dir = tmp / "gt" / "P001"
            gt_dir.mkdir(parents=True)
            _make_test_nifti(gt_dir / "P001_0001.nii.gz", shape=(4, 8, 8))

            out = tmp / "gt_slices"
            result = extract_ground_truth_slices(
                gt_dir=tmp / "gt",
                masks_dir=None,
                output_dir=out,
                slice_mode="max_tumor",
                phase=1,
            )
            assert len(result) == 1
            assert result[0].suffix == ".png"
            assert result[0].exists()

    def test_extracts_mask_slices_when_requested(self):
        from eval.synthesize import extract_ground_truth_slices

        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # Create GT
            gt_dir = tmp / "gt" / "P001"
            gt_dir.mkdir(parents=True)
            _make_test_nifti(gt_dir / "P001_0001.nii.gz", shape=(4, 8, 8))
            # Create mask
            masks_dir = tmp / "masks"
            masks_dir.mkdir()
            mask_arr = np.zeros((4, 8, 8), dtype=np.uint8)
            mask_arr[2, 3:5, 3:5] = 1
            img = sitk.GetImageFromArray(mask_arr)
            sitk.WriteImage(img, str(masks_dir / "P001_0000.nii.gz"))

            out = tmp / "gt_slices"
            mask_out = tmp / "mask_slices"

            result = extract_ground_truth_slices(
                gt_dir=tmp / "gt",
                masks_dir=masks_dir,
                output_dir=out,
                slice_mode="max_tumor",
                phase=1,
                masks_output_dir=mask_out,
            )
            assert len(result) == 1
            # Mask slice should also exist
            mask_files = list(mask_out.glob("*.png"))
            assert len(mask_files) == 1


# ---------------------------------------------------------------------------
# CLI argument parsing — slice mode and masks
# ---------------------------------------------------------------------------


class TestSliceModeCLI:
    """Tests for --slice-mode and --masks-dir CLI options."""

    def test_synthesize_slice_mode_default(self):
        from eval.synthesize import parse_synthesize_args

        args = parse_synthesize_args([
            "--data-dir", "/data",
            "--output-dir", "/out",
        ])
        assert args.slice_mode == "max_tumor"

    def test_synthesize_slice_mode_custom(self):
        from eval.synthesize import parse_synthesize_args

        args = parse_synthesize_args([
            "--data-dir", "/data",
            "--output-dir", "/out",
            "--slice-mode", "all_tumor",
        ])
        assert args.slice_mode == "all_tumor"

    def test_synthesize_slice_mode_center(self):
        from eval.synthesize import parse_synthesize_args

        args = parse_synthesize_args([
            "--data-dir", "/data",
            "--output-dir", "/out",
            "--slice-mode", "center_tumor",
        ])
        assert args.slice_mode == "center_tumor"

    def test_synthesize_invalid_mode_fails(self):
        from eval.synthesize import parse_synthesize_args

        with pytest.raises(SystemExit):
            parse_synthesize_args([
                "--data-dir", "/data",
                "--output-dir", "/out",
                "--slice-mode", "invalid",
            ])

    def test_synthesize_masks_dir_explicit(self):
        from eval.synthesize import parse_synthesize_args

        args = parse_synthesize_args([
            "--data-dir", "/data",
            "--output-dir", "/out",
            "--masks-dir", "/masks",
        ])
        assert args.masks_dir == Path("/masks")

    def test_synthesize_masks_dir_auto_from_data_dir(self):
        from eval.synthesize import parse_synthesize_args

        with tempfile.TemporaryDirectory() as tmp:
            seg_dir = Path(tmp) / "segmentations"
            seg_dir.mkdir()
            args = parse_synthesize_args([
                "--data-dir", tmp,
                "--output-dir", "/out",
            ])
            assert args.masks_dir == seg_dir

    def test_synth_and_eval_slice_mode(self):
        from eval.synthesize import parse_synthesize_and_evaluate_args

        args = parse_synthesize_and_evaluate_args([
            "--predictions-dir", "/preds",
            "--ground-truth-path", "/gt",
            "--output-file", "m.json",
            "--slice-mode", "center_tumor",
        ])
        assert args.slice_mode == "center_tumor"

    def test_synth_and_eval_masks_dir_explicit(self):
        from eval.synthesize import parse_synthesize_and_evaluate_args

        args = parse_synthesize_and_evaluate_args([
            "--predictions-dir", "/preds",
            "--ground-truth-path", "/gt",
            "--output-file", "m.json",
            "--masks-dir", "/my/masks",
        ])
        assert args.masks_dir == Path("/my/masks")

    def test_synth_and_eval_masks_dir_falls_back_to_masks_path(self):
        from eval.synthesize import parse_synthesize_and_evaluate_args

        args = parse_synthesize_and_evaluate_args([
            "--predictions-dir", "/preds",
            "--ground-truth-path", "/gt",
            "--output-file", "m.json",
            "--masks-path", "/eval/masks",
        ])
        # No --masks-dir given → falls back to --masks-path
        assert args.masks_dir == Path("/eval/masks")

    def test_synth_and_eval_masks_dir_overrides_masks_path(self):
        from eval.synthesize import parse_synthesize_and_evaluate_args

        args = parse_synthesize_and_evaluate_args([
            "--predictions-dir", "/preds",
            "--ground-truth-path", "/gt",
            "--output-file", "m.json",
            "--masks-dir", "/synth/masks",
            "--masks-path", "/eval/masks",
        ])
        # Explicit --masks-dir takes precedence
        assert args.masks_dir == Path("/synth/masks")
        # --masks-path is still available for evaluation
        assert args.masks_path == Path("/eval/masks")

    def test_synth_and_eval_masks_dir_auto_from_data_dir(self):
        from eval.synthesize import parse_synthesize_and_evaluate_args

        with tempfile.TemporaryDirectory() as tmp:
            seg_dir = Path(tmp) / "segmentations"
            seg_dir.mkdir()
            args = parse_synthesize_and_evaluate_args([
                "--data-dir", tmp,
                "--output-dir", "/out",
                "--output-file", "m.json",
            ])
            # Both masks_path and masks_dir auto-resolve
            assert args.masks_path == seg_dir
            assert args.masks_dir == seg_dir


# ---------------------------------------------------------------------------
# SLICE_MODE_CHOICES constant
# ---------------------------------------------------------------------------


class TestSliceModeChoices:
    """Tests for SLICE_MODE_CHOICES constant."""

    def test_includes_expected_modes(self):
        from eval.synthesize import SLICE_MODE_CHOICES

        assert "max_tumor" in SLICE_MODE_CHOICES
        assert "center_tumor" in SLICE_MODE_CHOICES
        assert "all_tumor" in SLICE_MODE_CHOICES
        assert len(SLICE_MODE_CHOICES) == 3


# ---------------------------------------------------------------------------
# PNG output format verification
# ---------------------------------------------------------------------------


class TestPngOutputFormat:
    """Tests that synthesis produces PNG output (not NIfTI)."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        pytest.importorskip("SimpleITK")
        pytest.importorskip("PIL")

    def test_output_is_png_not_nifti(self):
        """synthesize_with_medigan should produce .png files, not .nii.gz."""
        from eval.synthesize import synthesize_with_medigan

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            input_dir = tmp / "input" / "P001"
            input_dir.mkdir(parents=True)
            output_dir = tmp / "output"

            _make_test_nifti(
                input_dir / "P001_0000.nii.gz", shape=(3, 8, 8)
            )

            mock_gen_instance = MagicMock()

            def fake_generate(**kwargs):
                in_dir = Path(kwargs["input_path"])
                out_dir = Path(kwargs["output_path"]) / "batch_0"
                out_dir.mkdir(parents=True, exist_ok=True)
                import shutil as _sh
                for png in sorted(in_dir.glob("*.png")):
                    _sh.copy(png, out_dir / png.name)

            mock_gen_instance.generate.side_effect = fake_generate
            mock_medigan = MagicMock()
            mock_medigan.Generators.return_value = mock_gen_instance

            original_cwd = os.getcwd()
            original_path = list(sys.path)
            try:
                os.chdir(tmp)
                with patch.dict("sys.modules", {"medigan": mock_medigan}):
                    with patch("eval.synthesize._ensure_medigan_importable"):
                        result = synthesize_with_medigan(
                            input_dir=tmp / "input",
                            output_dir=output_dir,
                            gpu_id="-1",
                            image_size=256,
                        )
            finally:
                os.chdir(original_cwd)
                sys.path = original_path

            # All output files should be PNGs
            for path in result:
                assert path.suffix == ".png", f"Expected .png, got {path.suffix}"
            # No NIfTI files in the output directory
            nifti_files = list(output_dir.glob("*.nii.gz"))
            assert len(nifti_files) == 0, f"Unexpected NIfTI files: {nifti_files}"

    def test_output_resized_to_native_resolution(self):
        """When medigan output differs from native slice size, resize to match."""
        from eval.synthesize import synthesize_with_medigan
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            input_dir = tmp / "input" / "P001"
            input_dir.mkdir(parents=True)
            output_dir = tmp / "output"

            # Native NIfTI has 48×48 slices (not 512)
            _make_test_nifti(
                input_dir / "P001_0000.nii.gz", shape=(3, 48, 48),
            )

            mock_gen_instance = MagicMock()

            def fake_generate(**kwargs):
                """Simulate medigan producing 512×512 output."""
                in_dir = Path(kwargs["input_path"])
                out_dir = Path(kwargs["output_path"]) / "batch_0"
                out_dir.mkdir(parents=True, exist_ok=True)
                for png in sorted(in_dir.glob("*.png")):
                    # Upscale input to 512×512 to simulate medigan behaviour
                    img = Image.open(png).resize((512, 512))
                    img.save(out_dir / png.name)

            mock_gen_instance.generate.side_effect = fake_generate
            mock_medigan = MagicMock()
            mock_medigan.Generators.return_value = mock_gen_instance

            original_cwd = os.getcwd()
            original_path = list(sys.path)
            try:
                os.chdir(tmp)
                with patch.dict("sys.modules", {"medigan": mock_medigan}):
                    with patch("eval.synthesize._ensure_medigan_importable"):
                        result = synthesize_with_medigan(
                            input_dir=tmp / "input",
                            output_dir=output_dir,
                            gpu_id="-1",
                            image_size=512,
                        )
            finally:
                os.chdir(original_cwd)
                sys.path = original_path

            assert len(result) == 1
            # Output must be resized back to native 48×48
            out_img = Image.open(result[0])
            assert out_img.size == (48, 48), (
                f"Expected (48, 48), got {out_img.size}"
            )


# ---------------------------------------------------------------------------
# _normalize_gpu_id
# ---------------------------------------------------------------------------


class TestNormalizeGpuId:
    """Tests for _normalize_gpu_id helper."""

    def test_bare_zero(self):
        from eval.synthesize import _normalize_gpu_id
        assert _normalize_gpu_id("0") == "cuda:0"

    def test_bare_one(self):
        from eval.synthesize import _normalize_gpu_id
        assert _normalize_gpu_id("1") == "cuda:1"

    def test_negative_one_is_cpu(self):
        from eval.synthesize import _normalize_gpu_id
        assert _normalize_gpu_id("-1") == "cpu"

    def test_cpu_string_passthrough(self):
        from eval.synthesize import _normalize_gpu_id
        assert _normalize_gpu_id("cpu") == "cpu"

    def test_cuda_string_passthrough(self):
        from eval.synthesize import _normalize_gpu_id
        assert _normalize_gpu_id("cuda") == "cuda"

    def test_cuda_colon_passthrough(self):
        from eval.synthesize import _normalize_gpu_id
        assert _normalize_gpu_id("cuda:0") == "cuda:0"

    def test_cuda_colon_1_passthrough(self):
        from eval.synthesize import _normalize_gpu_id
        assert _normalize_gpu_id("cuda:1") == "cuda:1"

    def test_whitespace_stripped(self):
        from eval.synthesize import _normalize_gpu_id
        assert _normalize_gpu_id("  0  ") == "cuda:0"


# ---------------------------------------------------------------------------
# --skip-synthesis GT slice extraction
# ---------------------------------------------------------------------------


class TestSkipSynthesisGTExtraction:
    """Tests that GT slices are correctly handled when --skip-synthesis is used.

    The combined pipeline must still produce matching 2D GT PNG slices
    for evaluation, even when synthesis itself is skipped.  Three cases:

    1. Fresh synthesis (not skipped) → always extract GT slices.
    2. Skip synthesis + previous ``.gt_slices/`` with PNGs → reuse.
    3. Skip synthesis + no previous extraction → extract now.
    """

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        pytest.importorskip("SimpleITK")
        pytest.importorskip("PIL")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _build_env(tmp: Path, *, with_gt_slices: bool = False,
                   with_mask_slices: bool = False):
        """Create a minimal filesystem layout for skip-synthesis tests.

        Returns (predictions_dir, gt_dir, masks_dir).
        """
        # predictions: some fake PNGs
        preds = tmp / "predictions"
        preds.mkdir()
        from PIL import Image as _PIL
        for pid in ("P001", "P002"):
            img = _PIL.new("L", (8, 8), color=128)
            img.save(preds / f"{pid}.png")

        # Ground-truth NIfTI volumes (nested MAMA-MIA layout)
        gt_dir = tmp / "images"
        for pid in ("P001", "P002"):
            d = gt_dir / pid
            d.mkdir(parents=True)
            _make_test_nifti(d / f"{pid}_0001.nii.gz", shape=(3, 8, 8))

        # Masks
        masks_dir = tmp / "segmentations"
        for pid in ("P001", "P002"):
            d = masks_dir / pid
            d.mkdir(parents=True)
            _make_test_nifti(d / f"{pid}.nii.gz", shape=(3, 8, 8))

        # Optionally pre-populate .gt_slices / .mask_slices inside preds
        if with_gt_slices:
            gs = preds / ".gt_slices"
            gs.mkdir()
            for pid in ("P001", "P002"):
                _PIL.new("L", (8, 8), color=64).save(gs / f"{pid}.png")
        if with_mask_slices:
            ms = preds / ".mask_slices"
            ms.mkdir()
            for pid in ("P001", "P002"):
                _PIL.new("L", (8, 8), color=255).save(ms / f"{pid}.png")

        return preds, gt_dir, masks_dir

    # -- Case 2: skip + previous extraction exists → reuse -----------------

    def test_reuses_existing_gt_slices(self):
        """When .gt_slices/ already has PNGs, they are reused (no re-extract)."""
        from eval.synthesize import synthesize_and_evaluate_main

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            preds, gt_dir, masks_dir = self._build_env(
                tmp, with_gt_slices=True, with_mask_slices=True,
            )
            out_json = tmp / "metrics.json"

            with patch("eval.synthesize.run_evaluation", return_value={"aggregates": {}}) as mock_eval, \
                 patch("eval.synthesize.extract_ground_truth_slices") as mock_extract:
                synthesize_and_evaluate_main([
                    "--predictions-dir", str(preds),
                    "--ground-truth-path", str(gt_dir),
                    "--masks-path", str(masks_dir),
                    "--output-file", str(out_json),
                ])

            # extract_ground_truth_slices should NOT have been called
            mock_extract.assert_not_called()

            # Evaluation should receive the pre-existing .gt_slices dir
            mock_eval.assert_called_once()
            call_kwargs = mock_eval.call_args
            assert call_kwargs[1]["ground_truth_dir"] == preds / ".gt_slices"
            assert call_kwargs[1]["masks_dir"] == preds / ".mask_slices"

    # -- Case 3: skip + no previous → extract now -------------------------

    def test_extracts_gt_slices_when_missing(self):
        """When .gt_slices/ does not exist, GT slices are extracted even with skip."""
        from eval.synthesize import synthesize_and_evaluate_main

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            preds, gt_dir, masks_dir = self._build_env(tmp)
            out_json = tmp / "metrics.json"

            with patch("eval.synthesize.run_evaluation", return_value={"aggregates": {}}) as mock_eval, \
                 patch("eval.synthesize.extract_ground_truth_slices") as mock_extract:
                synthesize_and_evaluate_main([
                    "--predictions-dir", str(preds),
                    "--ground-truth-path", str(gt_dir),
                    "--masks-dir", str(masks_dir),
                    "--output-file", str(out_json),
                ])

            # extract_ground_truth_slices SHOULD have been called
            mock_extract.assert_called_once()
            call_kwargs = mock_extract.call_args
            assert call_kwargs[1]["gt_dir"] == gt_dir
            assert call_kwargs[1]["output_dir"] == preds / ".gt_slices"

    # -- Case 1: fresh synthesis → always extract --------------------------

    def test_always_extracts_after_fresh_synthesis(self):
        """After fresh synthesis, GT slices are always (re-)extracted."""
        from eval.synthesize import synthesize_and_evaluate_main

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            preds, gt_dir, masks_dir = self._build_env(
                tmp, with_gt_slices=True,  # pre-existing should be overwritten
            )
            input_dir = gt_dir  # reuse as input for synthesis
            output_dir = preds
            out_json = tmp / "metrics.json"

            with patch("eval.synthesize.run_evaluation", return_value={"aggregates": {}}) as mock_eval, \
                 patch("eval.synthesize.extract_ground_truth_slices") as mock_extract, \
                 patch("eval.synthesize.synthesize_with_medigan"):
                synthesize_and_evaluate_main([
                    "--data-dir", str(tmp),
                    "--output-dir", str(output_dir),
                    "--output-file", str(out_json),
                ])

            # extract_ground_truth_slices SHOULD be called even though
            # .gt_slices/ already exists
            mock_extract.assert_called_once()

    # -- Edge case: skip + empty .gt_slices dir → extract ------------------

    def test_extracts_when_gt_slices_dir_empty(self):
        """An empty .gt_slices/ directory triggers fresh extraction."""
        from eval.synthesize import synthesize_and_evaluate_main

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            preds, gt_dir, masks_dir = self._build_env(tmp)
            # Create empty .gt_slices (no PNGs inside)
            (preds / ".gt_slices").mkdir()
            out_json = tmp / "metrics.json"

            with patch("eval.synthesize.run_evaluation", return_value={"aggregates": {}}), \
                 patch("eval.synthesize.extract_ground_truth_slices") as mock_extract:
                synthesize_and_evaluate_main([
                    "--predictions-dir", str(preds),
                    "--ground-truth-path", str(gt_dir),
                    "--masks-dir", str(masks_dir),
                    "--output-file", str(out_json),
                ])

            # Empty dir ≠ valid extraction → must re-extract
            mock_extract.assert_called_once()

    # -- Edge case: skip + reuse when only GT (no masks) -------------------

    def test_reuses_gt_slices_without_mask_slices(self):
        """Reuse works when .gt_slices/ exists but .mask_slices/ does not."""
        from eval.synthesize import synthesize_and_evaluate_main

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            preds, gt_dir, _ = self._build_env(
                tmp, with_gt_slices=True, with_mask_slices=False,
            )
            out_json = tmp / "metrics.json"

            with patch("eval.synthesize.run_evaluation", return_value={"aggregates": {}}) as mock_eval, \
                 patch("eval.synthesize.extract_ground_truth_slices") as mock_extract:
                synthesize_and_evaluate_main([
                    "--predictions-dir", str(preds),
                    "--ground-truth-path", str(gt_dir),
                    "--output-file", str(out_json),
                ])

            mock_extract.assert_not_called()
            call_kwargs = mock_eval.call_args
            assert call_kwargs[1]["ground_truth_dir"] == preds / ".gt_slices"
            # masks_dir should be None since .mask_slices/ does not exist
            assert call_kwargs[1]["masks_dir"] is None


# ---------------------------------------------------------------------------
# Per-task --clf-model-dir-* CLI options
# ---------------------------------------------------------------------------


class TestPerTaskClfModelDir:
    """Tests for per-task classifier model directory CLI arguments."""

    # -- CLI parsing -------------------------------------------------------

    def test_per_task_args_parsed(self):
        """All four per-task --clf-model-dir-* args are parsed correctly."""
        from eval.synthesize import parse_synthesize_and_evaluate_args

        args = parse_synthesize_and_evaluate_args([
            "--predictions-dir", "/preds",
            "--ground-truth-path", "/gt",
            "--output-file", "m.json",
            "--clf-model-dir-contrast", "/c",
            "--clf-model-dir-tumor-roi", "/t",
            "--clf-model-dir-luminal", "/l",
            "--clf-model-dir-tnbc", "/n",
        ])
        assert args.clf_model_dir_contrast == Path("/c")
        assert args.clf_model_dir_tumor_roi == Path("/t")
        assert args.clf_model_dir_luminal == Path("/l")
        assert args.clf_model_dir_tnbc == Path("/n")
        assert args.clf_model_dir is None  # legacy not set

    def test_legacy_fallback_still_works(self):
        """--clf-model-dir alone still sets fallback for all tasks."""
        from eval.synthesize import parse_synthesize_and_evaluate_args

        args = parse_synthesize_and_evaluate_args([
            "--predictions-dir", "/preds",
            "--ground-truth-path", "/gt",
            "--output-file", "m.json",
            "--clf-model-dir", "/all",
        ])
        assert args.clf_model_dir == Path("/all")
        assert args.clf_model_dir_contrast is None
        assert args.clf_model_dir_tumor_roi is None

    def test_per_task_overrides_legacy(self):
        """Per-task args take precedence over --clf-model-dir in MamaSynthEval."""
        from eval.evaluation import MamaSynthEval

        evaluator = MamaSynthEval(
            ground_truth_path="/gt",
            predictions_path="/pred",
            output_file="/out.json",
            clf_model_dir="/fallback",
            clf_model_dir_contrast="/contrast_only",
        )
        assert evaluator.clf_model_dir_contrast == Path("/contrast_only")
        assert evaluator.clf_model_dir_tumor_roi == Path("/fallback")
        assert evaluator.clf_model_dir_luminal == Path("/fallback")
        assert evaluator.clf_model_dir_tnbc == Path("/fallback")

    def test_legacy_propagates_to_all_tasks(self):
        """When only --clf-model-dir is given, all tasks inherit it."""
        from eval.evaluation import MamaSynthEval

        evaluator = MamaSynthEval(
            ground_truth_path="/gt",
            predictions_path="/pred",
            output_file="/out.json",
            clf_model_dir="/shared",
        )
        assert evaluator.clf_model_dir_contrast == Path("/shared")
        assert evaluator.clf_model_dir_tumor_roi == Path("/shared")
        assert evaluator.clf_model_dir_luminal == Path("/shared")
        assert evaluator.clf_model_dir_tnbc == Path("/shared")

    def test_none_when_nothing_provided(self):
        """Without any clf dirs, all tasks have None."""
        from eval.evaluation import MamaSynthEval

        evaluator = MamaSynthEval(
            ground_truth_path="/gt",
            predictions_path="/pred",
            output_file="/out.json",
        )
        assert evaluator.clf_model_dir is None
        assert evaluator.clf_model_dir_contrast is None
        assert evaluator.clf_model_dir_tumor_roi is None
        assert evaluator.clf_model_dir_luminal is None
        assert evaluator.clf_model_dir_tnbc is None

    # -- _clf_model_dir_for_task helper ------------------------------------

    def test_clf_model_dir_for_task_returns_correct_dir(self):
        """_clf_model_dir_for_task resolves the right per-task directory."""
        from eval.evaluation import MamaSynthEval

        evaluator = MamaSynthEval(
            ground_truth_path="/gt",
            predictions_path="/pred",
            output_file="/out.json",
            clf_model_dir="/fallback",
            clf_model_dir_contrast="/c",
            clf_model_dir_tumor_roi="/t",
        )
        assert evaluator._clf_model_dir_for_task("contrast") == Path("/c")
        assert evaluator._clf_model_dir_for_task("tumor_roi") == Path("/t")
        assert evaluator._clf_model_dir_for_task("luminal") == Path("/fallback")
        assert evaluator._clf_model_dir_for_task("tnbc") == Path("/fallback")

    def test_clf_model_dir_for_task_unknown_falls_back(self):
        """Unknown task names fall back to legacy clf_model_dir."""
        from eval.evaluation import MamaSynthEval

        evaluator = MamaSynthEval(
            ground_truth_path="/gt",
            predictions_path="/pred",
            output_file="/out.json",
            clf_model_dir="/legacy",
        )
        assert evaluator._clf_model_dir_for_task("unknown_task") == Path("/legacy")

    # -- run_evaluation bridge ---------------------------------------------

    def test_run_evaluation_passes_per_task_dirs(self):
        """run_evaluation forwards per-task dirs to MamaSynthEval."""
        from unittest.mock import patch, MagicMock

        mock_eval = MagicMock()
        mock_eval.evaluate.return_value = {"aggregates": {}, "results": []}

        with patch("eval.evaluation.MamaSynthEval", return_value=mock_eval) as mock_cls:
            from eval.synthesize import run_evaluation

            run_evaluation(
                predictions_dir=Path("/preds"),
                ground_truth_dir=Path("/gt"),
                output_file=Path("/out.json"),
                clf_model_dir=Path("/fallback"),
                clf_model_dir_contrast=Path("/c"),
                clf_model_dir_tumor_roi=Path("/t"),
                clf_model_dir_luminal=Path("/l"),
                clf_model_dir_tnbc=Path("/n"),
            )

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["clf_model_dir"] == "/fallback"
        assert call_kwargs["clf_model_dir_contrast"] == "/c"
        assert call_kwargs["clf_model_dir_tumor_roi"] == "/t"
        assert call_kwargs["clf_model_dir_luminal"] == "/l"
        assert call_kwargs["clf_model_dir_tnbc"] == "/n"

    # -- Integration: synthesize_and_evaluate_main -------------------------

    def test_main_passes_per_task_dirs_to_run_evaluation(self):
        """synthesize_and_evaluate_main forwards per-task dirs correctly."""
        from eval.synthesize import synthesize_and_evaluate_main

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(os.path.realpath(tmp))
            preds = tmp / "predictions"
            preds.mkdir()
            gt_dir = tmp / "gt"
            gt_dir.mkdir()
            # Need at least one matching pair for evaluation not to fail early
            from PIL import Image as _PIL
            _PIL.new("L", (8, 8), 128).save(preds / "P001.png")

            # Create .gt_slices with matching file so GT extraction is skipped
            gs = preds / ".gt_slices"
            gs.mkdir()
            _PIL.new("L", (8, 8), 64).save(gs / "P001.png")

            out_json = tmp / "metrics.json"

            with patch("eval.synthesize.run_evaluation", return_value={"aggregates": {}}) as mock_eval:
                synthesize_and_evaluate_main([
                    "--predictions-dir", str(preds),
                    "--ground-truth-path", str(gt_dir),
                    "--output-file", str(out_json),
                    "--clf-model-dir", str(tmp / "all"),
                    "--clf-model-dir-contrast", str(tmp / "c"),
                    "--clf-model-dir-tumor-roi", str(tmp / "t"),
                ])

            call_kwargs = mock_eval.call_args[1]
            assert call_kwargs["clf_model_dir"] == Path(tmp / "all")
            assert call_kwargs["clf_model_dir_contrast"] == Path(tmp / "c")
            assert call_kwargs["clf_model_dir_tumor_roi"] == Path(tmp / "t")
            assert call_kwargs["clf_model_dir_luminal"] is None
            assert call_kwargs["clf_model_dir_tnbc"] is None
