import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEEPSPEED_TRIGGERS = ("import deepspeed", "from deepspeed")


def _site_packages_for_root(root_dir):
    root = Path(root_dir)
    res = [root / "Lib/site-packages", root / "lib/site-packages"]
    lib_dir = root / "lib"
    if lib_dir.is_dir():
        for py_dir in lib_dir.glob("python3.*"):
            res.append(py_dir / "site-packages")
    return res


def _site_packages_candidates():
    candidates = []
    for prefix in [REPO_ROOT / ".venv", REPO_ROOT / "venv"]:
        candidates.extend(_site_packages_for_root(prefix))
    return candidates


def _prepend_poetry_site_packages(candidates):
    try:
        result = subprocess.run(["poetry", "env", "info", "-p"], cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return candidates

    if result.returncode != 0:
        return candidates
    poetry_env = result.stdout.strip()
    if poetry_env:
        return [*_site_packages_for_root(poetry_env), *candidates]
    return candidates


def _contains_target_package(candidate, target_package):
    return (candidate / target_package).exists() or (candidate / f"{target_package}.py").exists()


def _require_site_packages_dir(target_package):
    venv_site = _resolve_site_packages_dir(target_package)
    if venv_site is None:
        raise FileNotFoundError(f"venv site-packages not found for '{target_package}'")
    return venv_site


def _resolve_site_packages_dir(target_package):
    target_package = target_package.replace("-", "_")
    candidates = _prepend_poetry_site_packages(_site_packages_candidates())

    for candidate in candidates:
        if _contains_target_package(candidate, target_package):
            return candidate

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _patch_deepspeed_usage(resemble_dir):
    mock_code = """
# PATCHED V2: DeepSpeed removed for inference-only usage
import types
class MockDeepSpeed:
    def init_distributed(self, *args, **kwargs): pass
    class accelerator:
        def get_accelerator(self): return self
        def communication_backend_name(self): return "nccl"
    def zero_optim_partition(self, *args, **kwargs): return lambda x: x

deepspeed = MockDeepSpeed()
get_accelerator = lambda: MockDeepSpeed().accelerator()
DeepSpeedConfig = lambda x: None
class DeepSpeedEngine: pass
# End Patch
"""
    patched_count = 0
    failures = []
    for filepath in resemble_dir.rglob("*.py"):
        patched_count += _attempt_deepspeed_patch(filepath, mock_code, failures)

    if patched_count == 0:
        print(" -> DeepSpeed patches already applied or not needed.")

    if failures:
        raise RuntimeError("; ".join(f"{path}: {exc}" for path, exc in failures))


def _attempt_deepspeed_patch(filepath, mock_code, failures):
    try:
        return int(_patch_deepspeed_file(filepath, mock_code))
    except Exception as e:
        print(f" -> Failed to patch {filepath}: {e}")
        failures.append((filepath, e))
        return 0


def _is_deepspeed_trigger(line):
    return any(trigger in line for trigger in DEEPSPEED_TRIGGERS)


def _requires_deepspeed_patch(content):
    if not any(trigger in content for trigger in DEEPSPEED_TRIGGERS):
        return False
    return "# PATCHED V2" not in content


def _inject_deepspeed_mock(lines, mock_code):
    new_lines = []
    injected = False
    for line in lines:
        if not _is_deepspeed_trigger(line):
            new_lines.append(line)
            continue
        new_lines.append(f"# {line} # PATCHED")
        if not injected:
            new_lines.append(mock_code)
            injected = True
    return new_lines


def _patch_deepspeed_file(filepath, mock_code):
    content = filepath.read_text(encoding="utf-8")
    if not _requires_deepspeed_patch(content):
        return False

    print(f" -> Patching {filepath.name} for DeepSpeed...")
    new_lines = _inject_deepspeed_mock(content.splitlines(), mock_code)
    filepath.write_text("\n".join(new_lines), encoding="utf-8")
    return True


def _patch_torchaudio_loader(resemble_dir):
    monkeypatch_code = """
import soundfile
import torch
import torchaudio
def custom_load(filepath, **kwargs):
    w, sr = soundfile.read(filepath)
    # soundfile returns [frames, channels] or [frames]
    t = torch.from_numpy(w).float()
    if t.ndim == 1:
        t = t.unsqueeze(0)  # [channels=1, frames]
    elif t.ndim == 2:
        t = t.permute(1, 0)  # [channels, frames]

    return t, sr

def custom_save(filepath, src, sample_rate, **kwargs):
    # src is [channels, time] or [1, channels, time]
    if src.ndim == 3:
        src = src.squeeze(0)
    # torchaudio uses [channels, frames], soundfile expects [frames, channels]
    if src.ndim == 2:
        src = src.permute(1, 0)

    src = src.detach().cpu().numpy() # [frames, channels] or [frames]
    subtype = kwargs.get("subtype", "FLOAT")
    soundfile.write(filepath, src, sample_rate, subtype=subtype)

torchaudio.load = custom_load
torchaudio.save = custom_save
"""
    main_py = resemble_dir / "enhancer/__main__.py"
    if main_py.exists():
        content = main_py.read_text(encoding="utf-8")
        if "custom_load" in content:
            print(f" -> {main_py.name} already monkeypatched (Torchaudio).")
        else:
            print(f" -> Monkeypatching {main_py.name} for Torchaudio...")
            if "import torchaudio" in content:
                content = content.replace("import torchaudio", "import torchaudio\n" + monkeypatch_code)
                main_py.write_text(content, encoding="utf-8")
            else:
                raise RuntimeError(f"Could not find 'import torchaudio' in {main_py.name}")
    else:
        raise FileNotFoundError(f"{main_py} not found.")


def _patch_resemble_cfm(resemble_dir):
    cfm_py = resemble_dir / "enhancer/lcfm/cfm.py"
    if not cfm_py.exists():
        print(f" -> Warning: {cfm_py} not found; NumPy compatibility patch was not applied.")
        return
    content = cfm_py.read_text(encoding="utf-8")
    target = "a = float(scipy.optimize.fsolve(lambda a: h(1 / n, a) - 0.5, x0=0))"
    replacement = "a = float(scipy.optimize.fsolve(lambda a: h(1 / n, a) - 0.5, x0=0)[0])"
    if replacement in content:
        print(f" -> {cfm_py.name} already patched for NumPy 2.x.")
    elif target in content:
        print(f" -> Patching {cfm_py.name} for NumPy 2.x scalar conversion...")
        content = content.replace(target, replacement)
        cfm_py.write_text(content, encoding="utf-8")
    else:
        raise RuntimeError(f"Could not find target anchor in {cfm_py.name}")


def patch_resemble_enhance():
    print("[Patch] Checking Resemble-Enhance (DeepSpeed Removal)...")
    venv_site = _require_site_packages_dir("resemble_enhance")

    resemble_dir = venv_site / "resemble_enhance"
    if not resemble_dir.exists():
        raise FileNotFoundError("resemble_enhance package not found.")

    _patch_deepspeed_usage(resemble_dir)
    _patch_torchaudio_loader(resemble_dir)
    _patch_resemble_cfm(resemble_dir)


def patch_resemble_cli_args():
    print("[Patch] Fix Resemble-Enhance CLI Arguments...")
    venv_site = _require_site_packages_dir("resemble_enhance")

    resemble_main = venv_site / "resemble_enhance/enhancer/__main__.py"

    if not resemble_main.exists():
        raise FileNotFoundError(f"{resemble_main} not found.")

    try:
        content = resemble_main.read_text(encoding="utf-8")

        # Regex to match the enhance() call with arguments
        pattern = r"(hwav,\s*sr\s*=\s*enhance\()([\s\S]*?)" r"(lambd=args.lambd,)([\s\S]*?)(\))"

        match = re.search(pattern, content)
        if not match:
            raise RuntimeError("Could not find 'enhance(...)' call pattern to patch.")

        full_match = match.group(0)
        updated_content, status_message = _update_cli_patch_content(content, full_match)
        if updated_content is None:
            print(status_message)
            return

        resemble_main.write_text(updated_content, encoding="utf-8")
        print(status_message)

    except Exception as e:
        print(f" -> Failed to patch CLI args: {e}")
        raise


def _upgrade_cli_patch(full_match):
    chunk_seconds_pattern = r"(chunk_seconds\s*=\s*)\d+(?:\.\d+)?(?=\s*(?:,|\)))"
    chunks_overlap_pattern = r"(chunks_overlap\s*=\s*)\d+(?:\.\d+)?(?=\s*(?:,|\)))"

    upgraded_call = re.sub(chunk_seconds_pattern, "chunk_seconds=25", full_match, count=1)
    if re.search(chunks_overlap_pattern, upgraded_call):
        return re.sub(chunks_overlap_pattern, "chunks_overlap=2", upgraded_call, count=1)

    last_paren_idx = upgraded_call.rfind(")")
    call_content = upgraded_call[:last_paren_idx].rstrip()
    if not call_content.endswith(","):
        call_content += ","
    return f"{call_content}\n                chunks_overlap=2,\n            )"


def _build_cli_patch(full_match):
    last_paren_idx = full_match.rfind(")")
    call_content = full_match[:last_paren_idx].rstrip()
    if not call_content.endswith(","):
        call_content += ","
    new_args = "\n                chunk_seconds=25,\n                chunks_overlap=2,"
    return f"{call_content}{new_args}\n            )"


def _update_cli_patch_content(content, full_match):
    chunk_seconds_ok = bool(re.search(r"chunk_seconds\s*=\s*25(?=\s*(?:,|\)))", full_match))
    chunks_overlap_ok = bool(re.search(r"chunks_overlap\s*=\s*2(?=\s*(?:,|\)))", full_match))

    if chunk_seconds_ok and chunks_overlap_ok:
        return None, " -> CLI arguments already patched."

    if re.search(r"chunk_seconds\s*=\s*", full_match):
        print(" -> Upgrading patch to enforce 25s chunks and 2s overlap...")
        return content.replace(full_match, _upgrade_cli_patch(full_match)), " -> Patch upgraded."

    print(" -> Patching CLI arguments with improved logic...")
    return content.replace(full_match, _build_cli_patch(full_match)), " -> Successfully patched CLI arguments."


def patch_common_separator_force_soundfile():
    print("[Patch] Forcing Audio-Separator to use 'soundfile' (and ignoring pydub)...")
    venv_site = _require_site_packages_dir("audio_separator")

    sep_py = venv_site / "audio_separator/separator/common_separator.py"

    if not sep_py.exists():
        raise FileNotFoundError("common_separator.py not found.")

    try:
        content = sep_py.read_text(encoding="utf-8")
        target_line = 'self.use_soundfile = config.get("use_soundfile")'
        replacement = "self.use_soundfile = True # Forced by apply_patches.py"

        if replacement in content:
            print(" -> usage of soundfile already forced.")
        elif target_line in content:
            new_content = content.replace(target_line, replacement)
            sep_py.write_text(new_content, encoding="utf-8")
            print(" -> Successfully forced 'use_soundfile = True'.")
        else:
            raise RuntimeError("Could not find target line to force soundfile usage.")

    except Exception as e:
        print(f" -> Failed to patch common_separator: {e}")
        raise


def apply_runtime_patches():
    failures = []
    patchers = [
        patch_resemble_enhance,
        patch_resemble_cli_args,
        patch_common_separator_force_soundfile,
    ]

    for patcher in patchers:
        try:
            patcher()
        except Exception as exc:
            print(f"[Patch] FAILED {patcher.__name__}: {exc}")
            failures.append(f"{patcher.__name__}: {exc}")

    if failures:
        raise RuntimeError("Required runtime patches failed: " + "; ".join(failures))


if __name__ == "__main__":
    try:
        apply_runtime_patches()
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1) from exc
    print("Optimization patches applied.")
