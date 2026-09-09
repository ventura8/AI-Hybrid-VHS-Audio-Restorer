"""DeepFilterNet3 as the neural denoising stage, for the modes that opt in. Off by default.

On paired synthetic fixtures it looked like the better neural stage by a wide margin:
against the clean reference it improves log-spectral distance by 3.52 dB where UVR-DeNoise
improves it by 0.71, on the quiet combined fixture it moves SI-SDR from -6.75 to +16.64 dB
where UVR-DeNoise moves it nowhere, and it is seventeen times faster.

On real tape it loses, and badly. In the `auto_pure_linear` chain across 50 real captures
it triples programme deviation, 0.22 to 0.66 dB, with noise removal flat, and the count of
captures where the mode beats cathar on both halves of the trade falls from 29 to 7.
Capping its attenuation only walks a curve that never reaches the UVR-DeNoise chain. A
speech-enhancement model trained on clean speech treats band-limited VHS dialogue, and
everything that is not dialogue, as noise. It is the fifth constant on this branch that a
fixture set wrongly, and the reason the stage is opt-in rather than the default.

It is an optional dependency, and a demanding one: its Rust core builds from source under
MSVC, and it pins numpy below 2 and imports a torchaudio module that no longer exists. The
pins are ignored (`--no-deps`) and the removed module is shimmed here, because the code
only uses it as a type. When any of that is absent the stage reports itself unavailable and
the chain falls back to UVR-DeNoise, so nothing about this mode depends on the build.

Long captures are streamed from the file in overlapping chunks, each crossfaded onto the
last and written out as it is done. The model is frame-based with a short look-ahead, so a
half-second overlap hides the seams; read whole, a two-hour tape is 1.4 GB of float32 before
the model has allocated anything, so the file is never read whole.
"""

import dataclasses
import sys
import types
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

from .utils import log_msg

CHUNK_SECONDS = 60.0
OVERLAP_SECONDS = 0.5
# What a model that is installed can still fail with: a missing checkpoint or an unreadable
# file (OSError), a CUDA or inference fault (RuntimeError), a shape or rate it rejects
# (ValueError), a build whose API moved (AttributeError, TypeError). Each is a supported
# state of an optional stage, and each falls back to UVR-DeNoise.
STAGE_FAILURES = (OSError, RuntimeError, ValueError, AttributeError, TypeError)

_MODEL = {}


@dataclasses.dataclass
class _AudioMetaData:
    """Stand-in for torchaudio.backend.common.AudioMetaData, removed in torchaudio 2.x.

    DeepFilterNet imports it as a return type only. The fields are the ones it names.
    """

    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 32
    encoding: str = "PCM_F"


def _shim_torchaudio_backend():
    """Provides the module DeepFilterNet imports from, on torchaudio builds without it.

    Only the missing child is added: a `torchaudio.backend` that torchaudio has already
    loaded keeps its own state and gains a `common`.
    """
    if "torchaudio.backend.common" in sys.modules:
        return
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = _AudioMetaData
    backend = sys.modules.get("torchaudio.backend")
    if backend is None:
        backend = types.ModuleType("torchaudio.backend")
        sys.modules["torchaudio.backend"] = backend
    backend.common = common
    sys.modules["torchaudio.backend.common"] = common


def _load():
    """Imports DeepFilterNet and initialises the model once, or returns None."""
    if "model" in _MODEL:
        return _MODEL["model"]
    _shim_torchaudio_backend()
    # The import guard is its own block with `except ImportError`, which is the form pylint
    # recognises as an optional dependency; the model initialisation that follows can fail
    # for other reasons and is guarded separately.
    try:
        import torch
        import torchaudio
        from df.enhance import enhance, init_df
    except ImportError as exc:
        log_msg(f"    [DeepFilterNet] Not installed ({exc}); falling back.")
        _MODEL["model"] = None
        return None
    try:
        model, state, _ = init_df(log_level="ERROR")
        _MODEL["model"] = (model, state, enhance, torch, torchaudio)
    except STAGE_FAILURES as exc:
        log_msg(f"    [DeepFilterNet] Unavailable ({type(exc).__name__}: {str(exc)[:80]}); falling back.")
        _MODEL["model"] = None
    return _MODEL["model"]


def available():
    """Whether DeepFilterNet can run on this host."""
    return _load() is not None


def _join_two(head, chunk, overlap):
    """Appends one processed chunk to what came before, blending the overlap linearly."""
    fade = min(overlap, len(head), len(chunk))
    if fade <= 0:
        return np.concatenate([head, chunk])
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32).reshape(-1, 1)
    seam = head[-fade:] * (1.0 - ramp) + chunk[:fade] * ramp
    return np.concatenate([head[:-fade], seam, chunk[fade:]])


def _enhance_array(audio, rate, loaded):
    """Runs the model over one (samples, channels) array at the pipeline rate."""
    model, state, enhance, torch, torchaudio = loaded
    x = torch.from_numpy(np.ascontiguousarray(audio.T))
    x48 = torchaudio.functional.resample(x, rate, state.sr())
    with torch.no_grad():
        y48 = enhance(model, state, x48)
    y = torchaudio.functional.resample(y48, state.sr(), rate)
    return y.cpu().numpy().T.astype(np.float32)


def _stream_chunks(source, chunk, overlap):
    """Yields overlapping blocks of a capture without holding more than one block at a time."""
    total = source.frames
    start = 0
    while start < total:
        source.seek(start)
        block = source.read(min(chunk, total - start), dtype="float32", always_2d=True)
        if len(block) == 0:
            return
        yield block
        if start + len(block) >= total:
            return
        start = start + len(block) - overlap


def _stream_denoise(source, target, loaded, chunk, overlap):
    """Processes a capture block by block, crossfading each onto the last and writing as it goes.

    Only the previous block's overlap tail is held back between blocks: everything before it
    is final and goes to the file at once.
    """
    held = None
    written = 0
    for block in _stream_chunks(source, chunk, overlap):
        processed = _enhance_array(block, source.samplerate, loaded)
        joined = processed if held is None else _join_two(held, processed, overlap)
        final = len(joined) - min(overlap, len(joined))
        target.write(joined[:final])
        written += final
        held = joined[final:]
    if held is not None:
        target.write(held[: max(0, source.frames - written)])


def target_path(input_wav, output_dir):
    """Where the stage writes its result for a given input."""
    return Path(output_dir) / f"dfn_{Path(input_wav).stem}.wav"


def denoise(input_wav, output_dir):
    """Denoises a WAV with DeepFilterNet3, returning the output path or None if unavailable."""
    loaded = _load()
    if loaded is None or sf is None:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = target_path(input_wav, output_dir)
    with sf.SoundFile(str(input_wav)) as source:
        rate = source.samplerate
        with sf.SoundFile(str(target), "w", samplerate=rate, channels=source.channels, subtype="FLOAT") as out:
            _stream_denoise(source, out, loaded, int(CHUNK_SECONDS * rate), int(OVERLAP_SECONDS * rate))
    log_msg("    [DeepFilterNet] Denoised the full mix.")
    return target


def denoise_or(input_wav, output_dir, wanted, fallback):
    """Runs DeepFilterNet when asked for and working, else whatever the caller falls back to.

    A failure inside the stage -- an unreadable file, a resampling or inference error, a
    write that cannot complete -- is the same as the stage being absent: the chain goes on
    with the fallback rather than ending the restoration.
    """
    if wanted:
        try:
            produced = denoise(input_wav, output_dir)
        except STAGE_FAILURES as exc:
            log_msg(f"    [DeepFilterNet] Failed ({type(exc).__name__}: {str(exc)[:80]}); falling back.")
            # The stage streams into its output as it goes, so a failure part-way leaves a
            # file that reads as valid audio; it must not survive to be picked up later. A
            # cleanup that fails is logged, since the fallback matters more than the tidy-up.
            try:
                target_path(input_wav, output_dir).unlink(missing_ok=True)
            except OSError as cleanup:
                log_msg(f"    [DeepFilterNet] Could not remove partial output ({cleanup}).")
            produced = None
        if produced is not None:
            return produced
    return fallback()
