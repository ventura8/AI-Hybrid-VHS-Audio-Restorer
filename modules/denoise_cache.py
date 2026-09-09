"""Where the full-mix neural denoiser writes, and what may already be there.

The UVR step takes any valid WAV already in its directory as a finished result: that is how
a rerun resumes after a failure. It also means the directory has to say what it holds, and
hold nothing that another stage wrote.
"""

import hashlib

from .utils import log_msg


def neural_denoise_dir(audio_dir, surgical_wav, model):
    """The UVR step's directory for this input and model, keyed so another pair cannot reuse it.

    The key is the input's name (which carries every upstream stage's fingerprint) and
    size, and the model. A rerun after a settings change lands in a different directory
    and denoises afresh.
    """
    size = surgical_wav.stat().st_size if surgical_wav.exists() else 0
    key = hashlib.sha256(f"{surgical_wav.name}|{size}|{model}".encode("utf-8")).hexdigest()[:12]
    return audio_dir / f"neural_denoised_{key}"


def without_stale_neural_output(denoise_sub_dir):
    """The UVR step's directory with no DeepFilterNet output left in it, or a fresh one.

    The optional stage writes to a directory of its own, but a DeepFilterNet output here --
    a partial one from a failed run, or one an earlier build left -- would be handed back
    as the fallback's own. A stale file that cannot be removed is left where it is and the
    fallback runs in a sibling directory it cannot reach.
    """
    denoise_sub_dir.mkdir(parents=True, exist_ok=True)
    for stale in denoise_sub_dir.glob("dfn_*.wav"):
        try:
            stale.unlink()
        except OSError as exc:
            log_msg(f"    [Warning] Could not remove stale {stale.name} ({exc}); denoising into a clean directory.", is_error=True)
            clean = denoise_sub_dir.with_name(f"{denoise_sub_dir.name}_clean")
            clean.mkdir(exist_ok=True)
            return clean
    return denoise_sub_dir
