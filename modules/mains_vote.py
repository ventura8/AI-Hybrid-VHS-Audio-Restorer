"""The scan's 'confirmed' mains vote (`scanner_mains_vote: confirmed`): the head's notch, kept only where the hum is.

The scan's own vote (`filters._head_mains_vote`) reads the first 32768 samples and asks one bin
per candidate to stand 1.8 times over the bins two either side. On the IA hunt (2026-10-12,
cathar 0.8.0 on CPU) it notched 123 of the 203 clips R0 reads as hum-free, and 29 of the 200
hum-free realistic-v2 speech and chord fixtures. Here that notch is kept only where the whole
analysed signal's eight harmonics of its family stand `MIN_EXCESS_DB` over their own
neighbourhood (`spectral_denoise.harmonic_excess_db`, the reading `detect_mains_hz` votes on):
the vote never adds a notch, so the chord beds a whole-signal vote reads as hum (13.4-15.2 dB of
excess on targets that carry none) cannot gain one.

R0 names the IA clips' hum with `measure_hum.hum_excess_db`, the same computation at the same
6 dB (0.002 dB apart at R0's family, median of the 284 hunt and catalog_hum48 sources), so on the
IA sets this gate agrees with R0 by construction and those counts do not test it. The fixture
labels do: it clears 17 of the 29 false fires on hum-free realistic-v2 fixtures (the 12 it keeps
read 6.1-10.7 dB, 8 of them music) and drops 5 of the 71 right-family notches on hum fixtures,
whose injected 50 Hz hum reads 3.9-5.9 dB over the whole 15 s.
"""

from . import spectral_denoise
from .filters import MAINS_FUNDAMENTAL_BY_HARMONIC
from .utils import log_msg

# measure_hum's "carries hum" line and the gate detect_mains_hz holds auto_pure_linear's canceller to.
# R0 names the IA hum tapes by the same reading at this line, so the IA sets cannot place it (163 of
# the 166 head notches there fall on R0's side of it by construction; the three others are where R0's
# family is not the head's). The labelled fixtures put 12 of the 29 false fires over it and 5 of the
# 71 right-family hum notches under it.
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
