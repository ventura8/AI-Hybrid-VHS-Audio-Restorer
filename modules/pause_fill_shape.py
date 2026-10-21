"""The pause floor's hiss shelf: a shaped fill that cuts the source's hiss and keeps its programme.

The pause floor's quiet frames are not only pauses: on dense or compressed programme 57-99 % of
the frames sit within 10 dB of the 15th percentile. Where the neural stage emptied soft programme
there, the shipped flat fill puts the programme back whole and the shelf puts back its low band
alone. The IA confirmation of the `auto_pure_linear` v9 plateau (2026-10-21, 202 hunt clips, ear
v4's constraints against the shipped settings) caught it on a clean capture (capture-a-6398:
`dsp.hiss_prog_kept_hf_db` 0.1 to -8.5, the presence -0.5 to -3.5) and a royal wedding broadcast
(kept HF -17.2 to -32.7).

'hiss_shelf' cuts the shelf only from the cells the source holds as hiss: the MMSE-LSA gain read
on the source's SNR against its whole-file quantile print (`spectral_suppress.reference_noise`),
floored per bin by the shelf's own response, so a hiss cell takes exactly the shelf's cut and a
cell where the programme stands over the print passes; it can only keep more than the shelf.
Replayed on the same clips it cleared capture-a-6398 (no guard tripped) and took the wedding to
-19.3, with the pause hiss as low as the shelf's. A partial held through the whole file is the
print's 5th percentile itself and is cut as hiss: a children's tape whose music never stops
(thomas-friends, `dsp.hiss_prog_kept_soft_db` -18.5 shipped, -33.3 shelved, -31.3 hiss-shelved).
"""

from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from . import spectral_suppress
from .blend_weights import FRAME, HOP
from .hygiene import atomic_target
from .spectral_suppress import _spectra, _stream
from .stage_names import stage_output
from .utils import is_valid_audio, log_msg

# The estimator's own settings (apl_suppress_* defaults; reference_suppress.CATHAR_SETTINGS).
NOISE_BIAS = 1.0
DD_ALPHA = 0.96
STAGE = "pause_floor"


class HissShelf(spectral_suppress.ReferenceSuppressor):
    """The estimator's gain read on the source's own SNR and floored per bin by the shelf's response.

    A cell the source holds as hiss gets the shelf's cut; a cell where the programme stands over
    the print keeps up to its whole level; under the corner the floor is 1 and the source passes.
    """

    def __init__(self, noise, floor):
        super().__init__(noise, NOISE_BIAS, 0.0, DD_ALPHA)
        self.floor = floor


def shelf_response(shaper, rate):
    """The shaped fill's amplitude response per bin of the suppressor's grid: |H|^2 of the zero-phase low band plus
    `shaper.dry` of the rest, the gain `pause_floor.shaped` applies to a steady tone."""
    _, response = scipy.signal.sosfreqz(shaper.sos, worN=np.fft.rfftfreq(FRAME, 1.0 / rate), fs=rate)
    low = np.abs(response) ** 2
    return low + shaper.dry * (1.0 - low)


def source_print(source_wav):
    """The source's whole-file quantile print on the suppressor's grid, or None under its minimum of live windows."""
    noise, _depth = spectral_suppress.reference_noise(source_wav)
    return noise


def write_hiss_shelved(source_wav, target_wav, noise, floor):
    """Writes the source with its hiss shelved (`HissShelf`) a block at a time; returns the target."""
    with (
        atomic_target(target_wav) as partial,
        sf.SoundFile(str(source_wav)) as source,
        sf.SoundFile(str(partial), "w", samplerate=source.samplerate, channels=source.channels, subtype="FLOAT") as out,
    ):
        shelf = HissShelf(noise, floor)

        def gained(first, last):
            """The source frames [first, last) gained by their own SNR."""
            spectrum = _spectra(_read_span(source, first * HOP - FRAME // 2, (last - 1) * HOP + FRAME // 2))
            return shelf.apply_pair(spectrum, spectrum)

        _stream(out, source.frames, source.channels, gained)
    return Path(target_wav)


def hiss_shelved(source_wav, audio_dir, shaper, noise, settings):
    """The source with its hiss shelved by `shaper`, reused when already written; `settings` is (corner Hz, cut dB),
    which the name carries."""
    corner_hz, cut_db = settings
    target = stage_output(Path(audio_dir) / STAGE, f"hiss_shelved_{corner_hz:g}c{cut_db:.2f}", source_wav)
    if is_valid_audio(target):
        log_msg(f"    [Pause Floor] Reusing {target.name}.")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    log_msg("    [Pause Floor] Shelving the source's hiss, not its programme, for the fill...")
    return write_hiss_shelved(source_wav, target, noise, shelf_response(shaper, sf.info(str(source_wav)).samplerate))


def _read_span(handle, start, stop):
    """Samples [start, stop) of an open file as float32, zero outside it."""
    span = np.zeros((stop - start, handle.channels), dtype=np.float32)
    low, high = max(0, start), min(handle.frames, stop)
    if high > low:
        handle.seek(low)
        into = slice(low - start, high - start)
        span[into] = handle.read(high - low, dtype="float32", always_2d=True)
    return span
