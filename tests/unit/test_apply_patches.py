"""
Tests for scripts/apply_patches.py.
Uses real file fixtures to achieve actual code coverage.
"""

import os
from types import SimpleNamespace

import pytest

from scripts import apply_patches


@pytest.fixture
def mock_venv(tmp_path, monkeypatch):
    """Create a mock venv structure and chdir to tmp_path."""
    old_cwd = os.getcwd()
    os.chdir(tmp_path)

    # Create base venv structure
    venv_site = tmp_path / "venv" / "Lib" / "site-packages"
    venv_site.mkdir(parents=True)
    monkeypatch.setattr(apply_patches, "REPO_ROOT", tmp_path)
    # Keep tests deterministic by preventing discovery of the developer's active Poetry env.
    monkeypatch.setattr(apply_patches.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""))

    yield tmp_path, venv_site

    os.chdir(old_cwd)


# ---------------------------------------------------------
# patch_resemble_enhance Tests
# ---------------------------------------------------------


def test_patch_resemble_enhance_no_venv(tmp_path, capsys, monkeypatch):
    """Test when venv doesn't exist."""
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    monkeypatch.setattr(apply_patches, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(apply_patches.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""))
    try:
        with pytest.raises(FileNotFoundError, match="site-packages not found"):
            apply_patches.patch_resemble_enhance()
        captured = capsys.readouterr()
        assert "Checking Resemble-Enhance" in captured.out
    finally:
        os.chdir(old_cwd)


def test_patch_resemble_enhance_no_package(mock_venv, capsys):
    """Test when resemble_enhance package doesn't exist."""
    tmp_path, venv_site = mock_venv
    with pytest.raises(FileNotFoundError, match="resemble_enhance package not found"):
        apply_patches.patch_resemble_enhance()


def test_patch_resemble_enhance_already_patched_v2(mock_venv, capsys):
    """Test when files are already patched with V2."""
    tmp_path, venv_site = mock_venv

    resemble_dir = venv_site / "resemble_enhance"
    resemble_dir.mkdir()
    enhancer_dir = resemble_dir / "enhancer"
    enhancer_dir.mkdir()

    # File already patched with V2
    test_file = resemble_dir / "module.py"
    test_file.write_text("# PATCHED V2\nimport deepspeed\n", encoding="utf-8")

    # __main__.py with custom_load already present
    main_py = enhancer_dir / "__main__.py"
    main_py.write_text("import torchaudio\ncustom_load = lambda: None\n", encoding="utf-8")

    apply_patches.patch_resemble_enhance()
    captured = capsys.readouterr()
    # Should silently skip already patched files
    assert "already monkeypatched" in captured.out


def test_patch_resemble_enhance_deepspeed_patch(mock_venv, capsys):
    """Test actual DeepSpeed patching."""
    tmp_path, venv_site = mock_venv

    resemble_dir = venv_site / "resemble_enhance"
    resemble_dir.mkdir()
    enhancer_dir = resemble_dir / "enhancer"
    enhancer_dir.mkdir()

    # Create file needing patching
    test_file = resemble_dir / "model.py"
    test_file.write_text("import deepspeed\nfrom deepspeed import config\nclass Model: pass\n", encoding="utf-8")

    # Create __main__.py needing torchaudio patch
    main_py = enhancer_dir / "__main__.py"
    main_py.write_text("import torchaudio\ndef main(): pass\n", encoding="utf-8")

    apply_patches.patch_resemble_enhance()

    # Verify patching occurred
    patched = test_file.read_text(encoding="utf-8")
    assert "MockDeepSpeed" in patched
    assert "# PATCHED" in patched

    main_patched = main_py.read_text(encoding="utf-8")
    assert "custom_load" in main_patched

    captured = capsys.readouterr()
    assert "Patching" in captured.out


def test_patch_resemble_enhance_no_deepspeed_files(mock_venv, capsys):
    """Test when no files contain deepspeed imports."""
    tmp_path, venv_site = mock_venv

    resemble_dir = venv_site / "resemble_enhance"
    resemble_dir.mkdir()
    enhancer_dir = resemble_dir / "enhancer"
    enhancer_dir.mkdir()

    # File without deepspeed
    test_file = resemble_dir / "utils.py"
    test_file.write_text("import torch\ndef helper(): pass\n", encoding="utf-8")

    # __main__.py without torchaudio
    main_py = enhancer_dir / "__main__.py"
    main_py.write_text("def main(): pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"Could not find 'import torchaudio'"):
        apply_patches.patch_resemble_enhance()


def test_patch_resemble_enhance_torchaudio_not_found(mock_venv, capsys):
    """Test when __main__.py exists but has no torchaudio import."""
    tmp_path, venv_site = mock_venv

    resemble_dir = venv_site / "resemble_enhance"
    resemble_dir.mkdir()
    enhancer_dir = resemble_dir / "enhancer"
    enhancer_dir.mkdir()

    main_py = enhancer_dir / "__main__.py"
    main_py.write_text("import torch\ndef main(): pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Could not find 'import torchaudio'"):
        apply_patches.patch_resemble_enhance()


def test_patch_resemble_enhance_read_exception(mock_venv, capsys):
    """Test bad.py read failure is aggregated after continuing the DeepSpeed scan."""
    tmp_path, venv_site = mock_venv

    resemble_dir = venv_site / "resemble_enhance"
    resemble_dir.mkdir()
    enhancer_dir = resemble_dir / "enhancer"
    enhancer_dir.mkdir()

    good_file = resemble_dir / "model.py"
    good_file.write_text("import deepspeed\nclass Model: pass\n", encoding="utf-8")

    # Create a directory pretending to be a .py file (will fail read)
    bad_file = resemble_dir / "bad.py"
    bad_file.mkdir()  # Directory, not file

    with pytest.raises(RuntimeError, match=r"bad\.py"):
        apply_patches.patch_resemble_enhance()

    captured = capsys.readouterr()
    assert "Failed to patch" in captured.out
    assert "bad.py" in captured.out

    # Confirm loop continuation by checking another file was still patched.
    patched_good = good_file.read_text(encoding="utf-8")
    assert "MockDeepSpeed" in patched_good


# ---------------------------------------------------------
# patch_resemble_cli_args Tests
# ---------------------------------------------------------


def test_patch_resemble_cli_args_no_file(mock_venv, capsys):
    """Test when __main__.py doesn't exist."""
    tmp_path, venv_site = mock_venv

    with pytest.raises(FileNotFoundError, match=r"__main__\.py"):
        apply_patches.patch_resemble_cli_args()


def test_patch_resemble_cli_args_already_patched(mock_venv, capsys):
    """Test when chunk_seconds is present but chunks_overlap still needs upgrading."""
    tmp_path, venv_site = mock_venv

    enhancer_dir = venv_site / "resemble_enhance" / "enhancer"
    enhancer_dir.mkdir(parents=True)

    main_py = enhancer_dir / "__main__.py"
    main_py.write_text(
        """
def process():
    hwav, sr = enhance(
        input_file,
        lambd=args.lambd,
        tau=args.tau,
        chunk_seconds=25,
    )
""",
        encoding="utf-8",
    )

    apply_patches.patch_resemble_cli_args()
    patched = main_py.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert "chunks_overlap=2" in patched
    assert "Patch upgraded" in captured.out


def test_patch_resemble_cli_args_pattern_not_found(mock_venv, capsys):
    """Test when enhance() pattern is not found."""
    tmp_path, venv_site = mock_venv

    enhancer_dir = venv_site / "resemble_enhance" / "enhancer"
    enhancer_dir.mkdir(parents=True)

    main_py = enhancer_dir / "__main__.py"
    main_py.write_text("def process():\n    pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"Could not find 'enhance\(\.\.\.\)' call pattern to patch"):
        apply_patches.patch_resemble_cli_args()


def test_patch_resemble_cli_args_success(mock_venv, capsys):
    """Test successful CLI args patching."""
    tmp_path, venv_site = mock_venv

    enhancer_dir = venv_site / "resemble_enhance" / "enhancer"
    enhancer_dir.mkdir(parents=True)

    main_py = enhancer_dir / "__main__.py"
    content = """
def process():
    hwav, sr = enhance(
        input_file,
        lambd=args.lambd,
        tau=args.tau,
        run_dir=run_dir,
    )
"""
    main_py.write_text(content, encoding="utf-8")

    apply_patches.patch_resemble_cli_args()

    patched = main_py.read_text(encoding="utf-8")
    assert "chunk_seconds" in patched

    captured = capsys.readouterr()
    assert "Successfully patched" in captured.out


def test_patch_resemble_cli_args_already_in_regex(mock_venv, capsys):
    """Test when chunk_seconds is already in the enhance call but overlap still needs upgrading."""
    tmp_path, venv_site = mock_venv

    enhancer_dir = venv_site / "resemble_enhance" / "enhancer"
    enhancer_dir.mkdir(parents=True)

    main_py = enhancer_dir / "__main__.py"
    content = """
def process():
    hwav, sr = enhance(
        input_file,
        lambd=args.lambd,
        tau=args.tau,
        chunk_seconds=25,
    )
"""
    main_py.write_text(content, encoding="utf-8")

    apply_patches.patch_resemble_cli_args()
    patched = main_py.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert "chunks_overlap=2" in patched
    assert "Patch upgraded" in captured.out


def test_patch_resemble_cli_args_upgrades_whitespace_variant(mock_venv, capsys):
    """Test upgrade logic matches optional whitespace around values."""
    _, venv_site = mock_venv

    enhancer_dir = venv_site / "resemble_enhance" / "enhancer"
    enhancer_dir.mkdir(parents=True)

    main_py = enhancer_dir / "__main__.py"
    main_py.write_text(
        """
def process():
    hwav, sr = enhance(
        input_file,
        lambd=args.lambd,
        tau=args.tau,
        chunk_seconds = 10,
        chunks_overlap = 1,
    )
""",
        encoding="utf-8",
    )

    apply_patches.patch_resemble_cli_args()

    patched = main_py.read_text(encoding="utf-8")
    assert "chunk_seconds=25" in patched
    assert "chunks_overlap=2" in patched


def test_patch_resemble_cli_args_ignores_unrelated_chunk_settings(mock_venv, capsys):
    """Ensure unrelated chunk_seconds assignments do not short-circuit target enhance() patching."""
    _, venv_site = mock_venv

    enhancer_dir = venv_site / "resemble_enhance" / "enhancer"
    enhancer_dir.mkdir(parents=True)

    main_py = enhancer_dir / "__main__.py"
    main_py.write_text(
        """
chunk_seconds = 25

def process():
    hwav, sr = enhance(
        input_file,
        lambd=args.lambd,
        tau=args.tau,
        run_dir=run_dir,
    )
""",
        encoding="utf-8",
    )

    apply_patches.patch_resemble_cli_args()

    patched = main_py.read_text(encoding="utf-8")
    assert "chunk_seconds=25" in patched
    assert "chunks_overlap=2" in patched
    captured = capsys.readouterr()
    assert "Successfully patched" in captured.out


# ---------------------------------------------------------
# patch_common_separator_force_soundfile Tests
# ---------------------------------------------------------


def test_patch_separator_no_file(mock_venv, capsys):
    """Test when common_separator.py doesn't exist."""
    _, venv_site = mock_venv
    (venv_site / "audio_separator").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=r"common_separator\.py not found"):
        apply_patches.patch_common_separator_force_soundfile()


def test_patch_separator_needs_patching(mock_venv, capsys):
    """Test patching common_separator to force soundfile."""
    tmp_path, venv_site = mock_venv

    sep_dir = venv_site / "audio_separator" / "separator"
    sep_dir.mkdir(parents=True)

    sep_py = sep_dir / "common_separator.py"
    sep_py.write_text(
        """
class Separator:
    def __init__(self, config):
        self.use_soundfile = config.get("use_soundfile")
""",
        encoding="utf-8",
    )

    apply_patches.patch_common_separator_force_soundfile()

    patched = sep_py.read_text(encoding="utf-8")
    assert "True # Forced" in patched

    captured = capsys.readouterr()
    assert "Successfully forced" in captured.out


def test_patch_separator_already_forced(mock_venv, capsys):
    """Test when already forced."""
    tmp_path, venv_site = mock_venv

    sep_dir = venv_site / "audio_separator" / "separator"
    sep_dir.mkdir(parents=True)

    sep_py = sep_dir / "common_separator.py"
    sep_py.write_text("self.use_soundfile = True # Forced by apply_patches.py", encoding="utf-8")

    apply_patches.patch_common_separator_force_soundfile()
    captured = capsys.readouterr()
    assert "already forced" in captured.out


def test_patch_separator_target_not_found(mock_venv, capsys):
    """Test when target line not found."""
    tmp_path, venv_site = mock_venv

    sep_dir = venv_site / "audio_separator" / "separator"
    sep_dir.mkdir(parents=True)

    sep_py = sep_dir / "common_separator.py"
    sep_py.write_text("class Separator: pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Could not find target line to force soundfile usage"):
        apply_patches.patch_common_separator_force_soundfile()


def test_apply_runtime_patches_aggregates_failures(monkeypatch):
    monkeypatch.setattr(apply_patches, "patch_resemble_enhance", lambda: None)
    monkeypatch.setattr(apply_patches, "patch_resemble_cli_args", lambda: (_ for _ in ()).throw(RuntimeError("cli failed")))
    monkeypatch.setattr(
        apply_patches,
        "patch_common_separator_force_soundfile",
        lambda: (_ for _ in ()).throw(FileNotFoundError("separator missing")),
    )

    with pytest.raises(RuntimeError, match="Required runtime patches failed"):
        apply_patches.apply_runtime_patches()


# ---------------------------------------------------------
# Module Main Block Test
# ---------------------------------------------------------


def test_module_import():
    """Test that module imports cleanly."""
    import importlib

    importlib.reload(apply_patches)
    assert hasattr(apply_patches, "patch_resemble_enhance")
    assert hasattr(apply_patches, "patch_resemble_cli_args")
    assert hasattr(apply_patches, "patch_common_separator_force_soundfile")
    assert hasattr(apply_patches, "apply_runtime_patches")
