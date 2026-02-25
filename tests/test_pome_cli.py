import runpy
import sys
from pathlib import Path

import pytest


def _copy_pipeline_to_tmp(tmp_path: Path) -> Path:
    """
    Copy pome.py into an isolated temp directory so that PATH2DATA points there.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "pome.py"
    assert src.exists(), f"Cannot find pome.py at {src}"

    dst = tmp_path / "pome.py"
    dst.write_text(src.read_text(), encoding="utf-8")
    return dst


def _make_dummy_scripts(tmp_path: Path, script_names: list[str]) -> None:
    """
    Create minimal placeholder scripts so os.path.exists checks pass.
    They should not be executed in tests (we mock subprocess.run), but we create
    them to match the pipeline's existence checks.
    """
    for name in script_names:
        (tmp_path / name).write_text("# dummy\n", encoding="utf-8")


def _run_pome_as_main(pome_path: Path, argv: list[str], monkeypatch, run_spy):
    """
    Execute pome.py as if run from CLI, with subprocess.run mocked.
    """
    import subprocess

    monkeypatch.setattr(subprocess, "run", run_spy)
    monkeypatch.setattr(sys, "argv", argv)

    # run as __main__ so the script executes its main block
    runpy.run_path(str(pome_path), run_name="__main__")


@pytest.mark.parametrize(
    "mode, expected_scripts",
    [
        (
            "vortex",
            [
                "eris_calib_dark.py",
                "eris_calib_dark_flat.py",
                "eris_calib_flat.py",
                "eris_calib_dark_psf.py",
                "eris_elab_psf.py",
                "eris_elab_sky.py",
                "eris_elab_obj.py",
            ],
        ),
        (
            "coro_app",
            [
                "eris_calib_dark.py",
                "eris_calib_dark_flat.py",
                "eris_calib_flat.py",
                "vapp_eris_background.py",
                "vapp_eris_elab.py",
                "vapp_eris_dh_estraction.py",
            ],
        ),
        (
            "img_pupil",
            [
                "eris_calib_dark.py",
                "eris_calib_dark_flat.py",
                "eris_calib_flat.py",
                "eris_background_img_pupil.py",
                "eris_elab_img_pupil.py",
            ],
        ),
        (
            "img_field",
            [
                "eris_calib_dark.py",
                "eris_calib_dark_flat.py",
                "eris_calib_flat.py",
            ],
        ),
    ],
)
def test_runs_expected_scripts_in_order(tmp_path, monkeypatch, mode, expected_scripts):
    pome_path = _copy_pipeline_to_tmp(tmp_path)
    _make_dummy_scripts(tmp_path, expected_scripts)

    calls = []

    def run_spy(cmd, check):
        # cmd is like: ["python3", script, PATH2DATA, TARGETNAME]
        calls.append((cmd, check))
        return 0

    target = "TESTTARGET"
    argv = ["pome.py", target, mode]

    _run_pome_as_main(pome_path, argv, monkeypatch, run_spy)

    # Ensure image output dir is created
    image_dir = tmp_path / "reduced" / target / "image"
    assert image_dir.exists() and image_dir.is_dir()

    # Ensure subprocess.run called for each script in order
    assert len(calls) == len(expected_scripts)
    for (cmd, check), script in zip(calls, expected_scripts):
        assert check is True
        assert cmd[0] == "python3"
        assert cmd[1] == script
        # PATH2DATA should be tmp_path + "/"
        assert Path(cmd[2]).resolve() == tmp_path.resolve()
        assert cmd[2].endswith("/")  # pipeline appends trailing slash
        assert cmd[3] == target


def test_missing_script_exits_with_code_1(tmp_path, monkeypatch, capsys):
    pome_path = _copy_pipeline_to_tmp(tmp_path)

    # create all but one required script for vortex
    scripts = [
        "eris_calib_dark.py",
        "eris_calib_dark_flat.py",
        "eris_calib_flat.py",
        "eris_calib_dark_psf.py",
        "eris_elab_psf.py",
        "eris_elab_sky.py",
        "eris_elab_obj.py",
    ]
    missing = "eris_elab_obj.py"
    _make_dummy_scripts(tmp_path, [s for s in scripts if s != missing])

    def run_spy(cmd, check):
        pytest.fail("subprocess.run should not be called if a required script is missing")

    argv = ["pome.py", "TESTTARGET", "vortex"]

    with pytest.raises(SystemExit) as e:
        _run_pome_as_main(pome_path, argv, monkeypatch, run_spy)

    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "missing from the directory" in out
    assert missing in out


def test_help_or_wrong_args_exits_0(tmp_path, monkeypatch):
    pome_path = _copy_pipeline_to_tmp(tmp_path)

    def run_spy(cmd, check):
        pytest.fail("subprocess.run should not be called when printing help")

    # wrong number of args => help => exit 0
    argv = ["pome.py", "ONLY_TARGET"]
    with pytest.raises(SystemExit) as e:
        _run_pome_as_main(pome_path, argv, monkeypatch, run_spy)
    assert e.value.code == 0

    # explicit help => exit 0
    argv = ["pome.py", "--help", "vortex"]
    with pytest.raises(SystemExit) as e:
        _run_pome_as_main(pome_path, argv, monkeypatch, run_spy)
    assert e.value.code == 0


def test_unknown_mode_exits_0_via_help(tmp_path, monkeypatch):
    pome_path = _copy_pipeline_to_tmp(tmp_path)

    def run_spy(cmd, check):
        pytest.fail("subprocess.run should not be called for unknown mode")

    argv = ["pome.py", "TESTTARGET", "not_a_mode"]
    with pytest.raises(SystemExit) as e:
        _run_pome_as_main(pome_path, argv, monkeypatch, run_spy)

    # pome.py calls print_help() after printing "Unknown mode"
    assert e.value.code == 0
