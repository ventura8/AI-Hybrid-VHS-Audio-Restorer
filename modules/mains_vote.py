"""The scan's 'confirmed' mains vote (`scanner_mains_vote: confirmed`): the head's notch, kept only where the hum is.

The scan's own vote (`filters._head_mains_vote`) reads the first 32768 samples and asks one bin
per candidate to stand 1.8 times over the bins two either side. On the IA hunt (2026-10-12,
cathar 0.8.0 on CPU) it notched 123 of the 203 clips R0 reads as hum-free, and 29 of the 200
hum-free realistic-v2 speech and chord fixtures. Here that notch is kept only where the whole
analysed signal's eight harmonics of its family stand `MIN_EXCESS_DB` over their own
neighbourhood (`spectral_denoise.harmonic_excess_db`, the reading `detect_mains_hz` votes on):
the vote never adds a notch, so the chord beds a whole-signal vote reads as hum (13.4-15.2 dB of
excess on targets that carry none) cannot gain one.
"""

from . import spectral_denoise
from .filters import MAINS_FUNDAMENTAL_BY_HARMONIC
from .utils import log_msg

# measure_hum's "carries hum" line and the gate detect_mains_hz holds auto_pure_linear's canceller to.
# At the head's family, 122 of the 123 hunt false fires read under it, while the 35 right-family
# notches on the 75 hum tapes of the hunt and catalog_hum48 read 6.07 dB or more.
MIN_EXCESS_DB = 6.0


def confirmed_notch(signal_data, sample_rate, notch):
    """The head's `notch` where the whole signal's harmonics of its family carry hum, else 0.0 (logged).

    100 Hz reads at 50 and 120 at 60, so the line-rate constraint the head applied still holds.
    A signal shorter than the reading's 16384-sample frame reads 0 dB and keeps no notch.
    """
    if notch <= 0:
        return notch
    family = MAINS_FUNDAMENTAL_BY_HARMONIC.get(notch, notch)
    excess_db = spectral_denoise.harmonic_excess_db(signal_data, sample_rate, family)
    if excess_db >= MIN_EXCESS_DB:
        return notch
    log_msg(
        f"    [Auto-Scan] Mains vote: the head's {notch:g} Hz not confirmed "
        f"({excess_db:.2f} dB of harmonic excess at {family:g} Hz, under {MIN_EXCESS_DB:g}); no notch."
    )
    return 0.0
