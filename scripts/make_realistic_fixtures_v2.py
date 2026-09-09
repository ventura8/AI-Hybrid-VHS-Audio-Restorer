#!/usr/bin/env python3
"""Paired fixtures whose programme and degradation match the real corpus, measured property by
property, because the previous sets did not and it cost five wrong conclusions.

Every fixture set on this branch has agreed with real tape about which stage to look at and
disagreed about what it is worth: the subtraction factor, the margin gate, the tonal cleanup,
the mains dehum and the DeepFilterNet stage all ranked one way on fixtures and the other way
on 174 real captures. Measured on the properties that decided those inversions, the blend's
training set is band-limited to 3 kHz where real tape reaches 5.1, carries 8 dB more
low-frequency energy relative to speech than real tape (injected hum dominating), is more
tonal than real tape rather than less, and is 2 dB peakier.

This generator sets each of those from the corpus rather than from a guess:

- **Programme.** Piper speech is the only clean source, so it is conditioned toward what a
  tape carries: a room around it, mild compression to a broadcast-like crest factor, a
  low-level broadband ambience bed that is part of the reference (real programme has room
  tone, and a stage that strips it is disturbing programme), and on a share of fixtures a
  sustained harmonic bed -- the music that broke the speech-tuned subtraction factor on
  real tape. The room is what DeepFilterNet's verdict turned on: dry synthesised speech is
  the material a speech enhancer was trained to keep, and on it the stage deviated *less*
  than UVR-DeNoise where 36 of 50 real captures had it deviating more. With a 0.4 s room
  it deviates 0.79 dB against 0.19, which is the real figure (0.66 against 0.22).
- **Degradation.** Real tape noise from the corpus, at margins drawn from the corpus
  distribution rather than three fixed values. Hum at a level that puts the low band where
  real tape has it, 8 dB under the speech band, not on top of it.
- **What the pauses hold, and how loud the rest is.** With pauses that held nothing but
  noise, room tone and a reverb tail, the chain emptied them: 42 dB of removal against the
  8.75 dB it manages on real tape. The frame-level profile of 50 real captures says why.
  Above the scanner's floor, real tape is a ramp -- the 20th percentile frame sits 2.3 dB
  up, the median 8.1, the 90th 14.5 -- where the fixtures were a step: 40% of frames within
  a decibel of the floor, then speech 15-20 dB above it. A tape's quiet frames are not
  empty and its loud ones are not far above them, because a camcorder's gain control and
  a broadcast chain flatten the level, pumping the noise up in every pause. So each
  speech-led fixture gets another voice from another language far across the room, the
  pauses are shorter, and the programme goes through a slow gain control that the
  reference goes through with it. The tape hiss is added after it, because that is where
  a tape adds it: the pauses then carry pumped room tone with a spectrum of its own, the
  speech carries hiss, and a noise profile learned in the quietest window describes one
  and not the other -- which is why real tape at a 10 dB margin gives up 8 dB of noise
  where a fixture with one stationary noise throughout gave up 22.
- **Reference and target are different files.** The reference is everything the tape
  carried: the voice in its room, the voice across it, the room tone. It is what the
  realism check scores against, because a stage that strips it is disturbing what was
  recorded. The target is the programme somebody restoring the tape wants back: the voice
  in its room and the music, without the room tone and the voice across it. It is what
  the per-bin blend is fitted against. Fitted against the reference instead, the blend
  learned to put the pauses back and gave up 6 dB of noise removal on real tape for 0.02 dB
  of deviation -- real tape's quiet frames are exactly the content the reference says to
  keep and the listener says to remove.
- **Every fault in the defect catalog, and the ones the literature adds.**
  `docs/vhs_audio_defects_research.md` lists fourteen; each has a class here, on the
  speech-led base above, drawn rather than stepped:
  crackle, dropouts, clipping, azimuth skew, channel imbalance, a line whistle, rumble, wow
  and flutter, a DC bias, a quiet capture, Hi-Fi head-switching buzz, video crosstalk, the
  SP roll-off, an EP tape, the camcorder's enclosure resonance, plosives and handling
  noise, a band-limited tape, and a worn tape carrying most of the transport faults at
  once. Each is written to trip the detector the chain gates its stage on where a stage
  exists; the realism check's coverage matrix says which ones the scanner sees, and the
  classes with no stage -- plosives, handling noise, crosstalk -- show what that costs.
  Checked against the preservation literature (IASA TC-04, the Library of Congress
  sticky-shed study, the BAVC glossary, VHS Hi-Fi service notes) ten more were missing from
  the catalog and are here: print-through, sticky-shed squeal, modulation noise, Hi-Fi
  compander breathing, Hi-Fi-to-linear track switching, undecoded Dolby B, edge damage, a
  clogging head, the archive's lossy codec, and the ghost of an earlier recording; the AV
  Artifact Atlas and the VCR service literature added scrape flutter and an EMI buzz whose
  harmonic series runs far past the eight a dehum takes out.
- **Music-led programme.** A share of real tapes is music with speech under it, or no
  speech at all, and that is where the chain and the DeepFilterNet stage do their damage:
  on the speech-led classes above DeepFilterNet deviates *less* than UVR-DeNoise, the
  opposite of real tape, and it takes dense non-stationary music -- chord changes, a bass
  line, percussion -- for the fixtures to show what 50 real captures showed. A sustained
  chord under speech, the first attempt, was not enough. These classes are drawn on their
  own random stream so adding them left the speech-led fixtures bit-identical.

A fixture set earns its keep by predicting real tape. The acceptance test is rank agreement:
run the comparisons that inverted before, and require that they now rank the way real tape
did. That check lives in scripts/validate_fixture_realism.py.
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import scipy.signal
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.realistic_defects import (
    NOISE_WOBBLE_DB,
    assemble_stereo,
    build_noise_bank,
    inject_breathing,
    inject_clipping,
    inject_codec,
    inject_crackle,
    inject_dc_offset,
    inject_dropouts,
    inject_edge_damage,
    inject_emi_buzz,
    inject_enclosure_resonance,
    inject_ghost_recording,
    inject_handling_noise,
    inject_head_clog,
    inject_head_switching,
    inject_line_whistle,
    inject_mains_hum,
    inject_modulation_noise,
    inject_plosives,
    inject_print_through,
    inject_rumble,
    inject_scrape_flutter,
    inject_squeal,
    inject_tape_noise,
    inject_track_switch,
    inject_undecoded_dolby_b,
    inject_video_crosstalk,
    inject_wow_and_flutter,
)

# The rate every fixture is written at, as scripts/audio_matrix/cli.py defines it; that module
# imports librosa at load, which the static CI job does not carry, so the value is restated.
FIXTURE_SAMPLE_RATE = 44100
# Corpus medians on 59 real captures, the targets every knob below is set against.
CORPUS_CREST_DB = 14.4
CORPUS_LF_VS_SPEECH_DB = -7.7
CORPUS_FLATNESS = 0.05
# Programme-above-noise margins as measured across the corpus: median 13.9 dB with most of
# the mass between 8 and 22. The scanner reads a speech fixture within a dB of the margin
# the injector set up to about 19 dB, then compresses toward a 22.7 dB ceiling set by the
# room tone and reverberation in the quietest frames; the sustained-bed class reads 10-14
# whatever the noise, because the bed fills the pauses. These values put the speech
# readings at 10, 13, 15, 19 and 21 and the set's median at the corpus figure.
MARGIN_DB = (9.0, 12.0, 15.0, 18.0, 24.0)
# Hum level that lands the low band at the corpus figure against Piper speech. The old 0.02
# put it at +0.15 dB; loud variants at 0.12 dominated the reference outright.
HUM_LEVEL = 0.045
AMBIENCE_DB = -30.0
# Room tone and the room's tail stop where a room's do. Opened up to 6 and 9 kHz for the
# 99% bandwidth figure, they put 3-5 dB more of 600-4800 Hz into the quiet frames, relative
# to the loud ones, than the quiet frames of 102 real clips carry; the bandwidth the corpus
# reads comes from the speech, and is 4.2-4.9 kHz on the wider corpus.
AMBIENCE_LOWPASS_HZ = 3000.0
MUSIC_DB = -14.0
# Soft saturation drive. An RMS-tracked compressor was tried first and raised the crest
# factor, because its 20 ms envelope lags the instantaneous peaks it was meant to catch;
# saturation clips them, which is also what a broadcast or VHS chain does. Drive 1.0
# lands Piper's 16.8 dB crest at 14.3 on its own; with the gaps below lowering the RMS,
# 1.4 is what lands the finished programme there.
# Re-read on 455 real clips at 15 s -- 172 of the original corpus and 283 of the wider one --
# the crest is 16.2-16.4 dB; at 1.6 the fixtures sat at 14.4, so the drive comes down.
SATURATION_DRIVE = 1.1
# Real programme is not wall-to-wall speech. Synthesised speech is, and that alone put the
# fixtures' spectral flatness at 0.02 against the corpus's 0.05: the median frame on a real
# tape is often ambience or a pause. A median frame flatness of 0.05 means more than half
# of real frames are not clean speech, so more than half of the programme here is not either.
GAP_SHARE = 0.0
# The voice's tilt. Relative to its 300-600 Hz band, a real tape's loud frames carry
# 600-1200 Hz at -1.5 dB, 1200-2400 at -5.5, 2400-4800 at -11 and 20-120 at -10, on 102
# clips of two corpora; Piper's voices carry -7, -10, -17 and -23. A camcorder microphone,
# a broadcast chain and a chest close to a microphone all put presence and weight on a
# voice that a synthesiser does not, so a high shelf and a low shelf put them on here.
# Breath. A speaker inhales before a phrase and a synthesiser does not; the inhale is a
# broadband noise burst, 400-4000 Hz, a few hundred milliseconds, 20-odd dB under the
# voice, and it sits in exactly the frames a noise profile is learned from and judged on.
# Real tapes' quiet frames vary 11-14 dB frame to frame per band; without breath the
# fixtures' varied 5-6 in the mids.
# Pauses. A synthesiser leaves over a second of dead silence between sentences; on a tape
# something else fills that -- a reply, a music bed, the room -- and a real tape's quietest
# fifth is quiet speech, not silence: on real clips the chain takes 2-8 dB out of the
# 300-2400 Hz band of the quiet frames, where on fixtures with the silences left in it
# took 7-20, because those frames held nothing a profile would spare. Pauses are cut to
# what a breath needs.
PAUSE_MAX_S = 0.3
# One longer pause survives per take -- a turn, a scene change -- because a tape's quietest
# window, the one the noise profile is learned from, usually holds one; with every pause
# cut to a breath the profile was learned on speech and the subtraction deviated 0.95 dB
# against real tape's 0.3-0.5.
PAUSE_LONGEST_S = 1.5
BREATH_DB = -22.0
BREATH_BAND_HZ = (400.0, 4000.0)
BREATH_MAX_S = 0.35
BREATH_PAUSE_S = 0.15
PRESENCE_SHELF_HZ = 700.0
PRESENCE_SHELF_DB = 5.5
WEIGHT_SHELF_HZ = 120.0
WEIGHT_SHELF_DB = 7.0
GAP_SECONDS = 0.3
# Recorder gain control: a camcorder or broadcast limiter riding the level. Slow enough to
# pump rather than distort, with a range that flattens speech the way the corpus profile
# shows, and it is applied to programme and noise together, which is where it sits.
AGC_TARGET = 0.1
AGC_MAX_GAIN_DB = 12.0
AGC_ATTACK_S = 0.05
AGC_RELEASE_S = 0.8
AGC_STEP_S = 0.001
# Room reverberation on the speech-led classes: a domestic or studio room, RT60 and the
# direct-to-reverberant ratio drawn per fixture. A camcorder across a living room sits
# well below 0 dB; a presenter on a close microphone sits above it. The tail darkens with
# a gentle first-order roll-off, since a steeper one pulled the set's bandwidth to 2.9 kHz
# against the corpus's 5.1.
ROOM_RT60_S = (0.3, 0.6)
ROOM_DIRECT_TO_REVERB_DB = (-10.0, 3.0)
ROOM_EARLY_REFLECTIONS = ((7, 0.5), (13, 0.4), (23, 0.3), (31, 0.25))
ROOM_TAIL_LOWPASS_HZ = 5000.0
# A second voice across the room, in the pauses as much as under the speech.
BACKGROUND_DB = -18.0
BACKGROUND_DIRECT_TO_REVERB_DB = -10.0
# The defect classes, each on the speech-led base at one margin. "worn" is the tape that
# has most of it at once, at the noisiest margin the set carries.
DEFECT_MARGIN_DB = 15.0
DEFECT_CLASSES = {
    "crackle": ("crackle",),
    "dropout": ("dropout",),
    "clip": ("clip",),
    "azimuth": ("azimuth",),
    "imbalance": ("imbalance",),
    "whistle": ("whistle",),
    "rumble": ("rumble",),
    # The transport classes carry a whine recorded on the tape as well as the warp: the drift
    # detector reads the wander of a stable tone, and speech alone gives it none to read.
    "flutter": ("recordedwhine", "flutter"),
    "dc": ("dc",),
    "low_level": ("low_level",),
    "bandlimited": ("bandlimited",),
    "hifibuzz": ("hifibuzz",),
    "crosstalk": ("crosstalk",),
    "sprolloff": ("sprolloff",),
    "ep": ("eprolloff", "recordedwhine", "flutter", "dropout"),
    "resonance": ("resonance",),
    "plosive": ("plosive",),
    "handling": ("handling",),
    "worn": ("crackle", "dropout", "recordedwhine", "flutter", "dc", "hum"),
    "printthrough": ("printthrough",),
    "squeal": ("squeal",),
    "modnoise": ("modnoise",),
    "breathing": ("breathing",),
    "trackswitch": ("trackswitch",),
    "dolbyb": ("dolbyb",),
    "edgedamage": ("edgedamage",),
    "headclog": ("headclog",),
    "codec": ("codec",),
    "ghost": ("ghost",),
    "crowd": ("crowd",),
    "scrapeflutter": ("scrapeflutter",),
    "buzz": ("buzz",),
}
# A crowd under the voice: the sport and comedy genres, a quarter of the wider corpus.
CROWD_DB = -8.0
CROWD_BAND_HZ = (200.0, 4000.0)
CROWD_SWELLS = (2, 4)
CLASS_MARGIN_DB = {"worn": 9.0, "ep": 12.0}
LOW_LEVEL_DB = -24.0
# Where a clipped capture's peaks would have landed before the digitiser flattened them;
# the scanner's clipping detector looks for flat tops at 0.985.
CLIP_SCALE = 1.97
# Tape response: a worn or narrow tape at 3 kHz, the linear track's SP cliff at 8 kHz,
# EP's at 4.5 kHz. (cutoff, order)
BAND_LIMITS = {"bandlimited": (3000.0, 4), "sprolloff": (8000.0, 2), "eprolloff": (4500.0, 4)}
# Faults that move audio in time. The blend is fitted bin against bin, so these are left
# out of its dataset; the realism check reads them through the reference-free metric.
TIMING_DEFECTS = ("flutter", "scrapeflutter")
# Music-led classes: music at full level with speech 12 dB under it, and music alone.
MUSIC_LED_SPEECH_DB = -12.0
MUSIC_NOTE_SECONDS = 1.2
MUSIC_HIT_SECONDS = 0.5
MUSIC_BANDWIDTH_HZ = 8000.0
MUSIC_ROOTS_HZ = (110.0, 130.8, 146.8, 164.8, 196.0, 220.0)


def _shelf(mono, rate, frequency, gain_db, high):
    """A second-order shelving filter (RBJ), boosting above `frequency` when high, below it otherwise."""
    amplitude = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * frequency / rate
    alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)
    cos_w0, root = np.cos(w0), 2.0 * np.sqrt(amplitude) * alpha
    sign = -1.0 if high else 1.0
    b = [
        amplitude * ((amplitude + 1) - sign * (amplitude - 1) * cos_w0 + root),
        sign * 2.0 * amplitude * ((amplitude - 1) - sign * (amplitude + 1) * cos_w0),
        amplitude * ((amplitude + 1) - sign * (amplitude - 1) * cos_w0 - root),
    ]
    a = [
        (amplitude + 1) + sign * (amplitude - 1) * cos_w0 + root,
        sign * -2.0 * ((amplitude - 1) + sign * (amplitude + 1) * cos_w0),
        (amplitude + 1) + sign * (amplitude - 1) * cos_w0 - root,
    ]
    return scipy.signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], mono).astype(np.float32)


def _voiced_for_tape(mono, rate):
    """Presence and weight on the synthesised voice, to the tilt real tapes carry."""
    return _shelf(_shelf(mono, rate, PRESENCE_SHELF_HZ, PRESENCE_SHELF_DB, True), rate, WEIGHT_SHELF_HZ, WEIGHT_SHELF_DB, False)


def _pauses(mono, rate, minimum_s):
    """(start, stop) sample bounds of the pauses in a dry voice, 40 dB under its peak frame."""
    frame = max(rate // 100, 1)
    count = len(mono) // frame
    level = 20.0 * np.log10(np.sqrt(np.mean(mono[: count * frame].reshape(count, frame) ** 2, axis=1)) + 1e-9)
    quiet = level < level.max() - 40.0
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8)))
    bounds = np.concatenate(([0], edges + 1, [count]))
    return [(a * frame, b * frame) for a, b in zip(bounds[:-1], bounds[1:]) if quiet[a] and (b - a) * frame >= minimum_s * rate]


def _trim_pauses(mono, rate):
    """Cuts every pause down to PAUSE_MAX_S but the longest, which keeps up to PAUSE_LONGEST_S."""
    keep = np.ones(len(mono), dtype=bool)
    pauses = _pauses(mono, rate, PAUSE_MAX_S)
    longest = max(pauses, key=lambda span: span[1] - span[0], default=None)
    for start, stop in pauses:
        allowed = PAUSE_LONGEST_S if (start, stop) == longest else PAUSE_MAX_S
        half = int(allowed * rate / 2)
        keep[start + half : stop - half] = False
    return mono[keep]


def _with_breaths(mono, rate, rng):
    """An inhale in every pause long enough to hold one, rising into the phrase that follows."""
    out = mono.astype(np.float32).copy()
    b, a = scipy.signal.butter(2, [BREATH_BAND_HZ[0] / (rate / 2), BREATH_BAND_HZ[1] / (rate / 2)], "band")
    level = float(np.sqrt(np.mean(mono**2))) * (10.0 ** (BREATH_DB / 20.0))
    for start, stop in _pauses(mono, rate, BREATH_PAUSE_S):
        length = min(stop - start, int(BREATH_MAX_S * rate))
        breath = scipy.signal.lfilter(b, a, rng.normal(0.0, 1.0, length))
        envelope = np.sin(np.pi * np.arange(length) / length) ** 0.7
        breath *= envelope * level / (float(np.sqrt(np.mean(breath**2))) + 1e-12)
        out[stop - length : stop] += breath.astype(np.float32)
    return out


def _saturate(mono):
    """Soft peak limiting toward a broadcast crest factor."""
    peak = float(np.max(np.abs(mono))) + 1e-9
    return (np.tanh(mono / peak * SATURATION_DRIVE) / np.tanh(SATURATION_DRIVE) * peak).astype(np.float32)


def _with_gaps(mono, rate, rng):
    """Interleaves the speech with silent gaps that the ambience bed will fill."""
    if GAP_SHARE <= 0.0:
        return mono
    gap = int(GAP_SECONDS * rate)
    phrase = int(gap * (1.0 - GAP_SHARE) / GAP_SHARE)
    pieces, start = [], 0
    while start < len(mono):
        pieces.append(mono[start : start + phrase])
        pieces.append(np.zeros(gap, dtype=np.float32))
        start += phrase
    out = np.concatenate(pieces)
    return out[: len(mono)] if len(out) >= len(mono) else np.pad(out, (0, len(mono) - len(out)))


def _room_response(rate, rng, direct_to_reverb_db=None):
    """A synthetic room: the direct sound, a few early reflections, then a diffuse tail."""
    rt60 = float(rng.uniform(*ROOM_RT60_S))
    length = int(rt60 * 1.5 * rate)
    time = np.arange(length) / rate
    tail = rng.normal(0.0, 1.0, length) * np.exp(-6.908 * time / rt60)
    tail[: int(0.005 * rate)] = 0.0
    b, a = scipy.signal.butter(1, ROOM_TAIL_LOWPASS_HZ / (rate / 2), "low")
    tail = scipy.signal.lfilter(b, a, tail)
    direct = np.zeros(length)
    direct[0] = 1.0
    for delay_ms, gain in ROOM_EARLY_REFLECTIONS:
        direct[int(delay_ms * rate / 1000)] += gain * rng.choice([-1.0, 1.0])
    if direct_to_reverb_db is None:
        direct_to_reverb_db = float(rng.uniform(*ROOM_DIRECT_TO_REVERB_DB))
    ratio = 10.0 ** (direct_to_reverb_db / 20.0)
    tail *= np.sqrt(np.sum(direct**2)) / (ratio * np.sqrt(np.sum(tail**2)) + 1e-12)
    response = direct + tail
    return response / np.sqrt(np.sum(response**2))


def _in_a_room(mono, rate, rng, direct_to_reverb_db=None):
    """Puts the speech in a room, tails decaying into the gaps rather than cut at them."""
    response = _room_response(rate, rng, direct_to_reverb_db)
    return scipy.signal.fftconvolve(mono, response)[: len(mono)].astype(np.float32)


def _background_voice(other, length, rate, rng):
    """Another speaker, far across the room, phrasing out of step with the main voice."""
    rolled = np.roll(other, int(rng.integers(len(other))))[:length]
    if len(rolled) < length:
        rolled = np.pad(rolled, (0, length - len(rolled)))
    return _in_a_room(_with_gaps(rolled, rate, rng), rate, rng, BACKGROUND_DIRECT_TO_REVERB_DB)


def _ambience(length, rate, rng):
    """A room-tone bed: pink-ish noise, low-passed, the way a live room sits under dialogue."""
    white = rng.normal(0.0, 1.0, length)
    b, a = scipy.signal.butter(2, AMBIENCE_LOWPASS_HZ / (rate / 2), "low")
    pink = scipy.signal.lfilter([0.049922, -0.095993, 0.050612, -0.004408], [1, -2.494956, 2.017265, -0.522189], white)
    return scipy.signal.lfilter(b, a, pink).astype(np.float32)


def _music_bed(length, rate, rng):
    """Sustained harmonic content with slow vibrato: the material a speech-tuned factor subtracts."""
    time = np.arange(length) / rate
    chord = np.zeros(length, dtype=np.float64)
    for root in rng.choice([110.0, 146.8, 196.0, 220.0], size=3, replace=False):
        vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * 5.5 * time + rng.uniform(0, 6.28))
        for harmonic in range(1, 6):
            chord += (0.5 / harmonic) * np.sin(2 * np.pi * root * harmonic * np.cumsum(vibrato) / rate)
    return (chord / np.max(np.abs(chord))).astype(np.float32)


def _music_notes(length, rate, rng):
    """Chords that change, each note with its own envelope, over a bass line."""
    time = np.arange(length) / rate
    out = np.zeros(length, dtype=np.float64)
    note = int(MUSIC_NOTE_SECONDS * rate)
    for start in range(0, length, note):
        stop = min(start + note, length)
        local = time[start:stop] - time[start]
        envelope = np.minimum(local / 0.03, 1.0) * np.exp(-local / 0.9)
        for root in rng.choice(MUSIC_ROOTS_HZ, size=3, replace=False):
            vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * 5.5 * local + rng.uniform(0, 6.28))
            for harmonic in range(1, 7):
                out[start:stop] += envelope * (0.6 / harmonic) * np.sin(2 * np.pi * root * harmonic * np.cumsum(vibrato) / rate)
        out[start:stop] += 0.8 * envelope * np.sin(2 * np.pi * (rng.choice(MUSIC_ROOTS_HZ) / 2.0) * local)
    return out


def _music_hits(length, rate, rng):
    """Percussive bursts: band-passed noise with a fast decay, twice a second, jittered."""
    out = np.zeros(length, dtype=np.float64)
    b, a = scipy.signal.butter(2, [200.0 / (rate / 2), 6000.0 / (rate / 2)], "band")
    for start in range(0, length, int(MUSIC_HIT_SECONDS * rate)):
        hit_start = max(0, min(length - 1, start + int(rng.uniform(-0.05, 0.05) * rate)))
        hit_stop = min(length, hit_start + int(rng.uniform(0.02, 0.06) * rate))
        burst = scipy.signal.lfilter(b, a, rng.normal(0.0, 1.0, hit_stop - hit_start))
        out[hit_start:hit_stop] += 1.5 * burst * np.exp(-np.arange(hit_stop - hit_start) / (0.012 * rate))
    return out


def _music_programme(length, rate, rng):
    """Dense, non-stationary music, band-limited the way a linear track carries it."""
    b, a = scipy.signal.butter(4, MUSIC_BANDWIDTH_HZ / (rate / 2), "low")
    out = scipy.signal.lfilter(b, a, _music_notes(length, rate, rng) + _music_hits(length, rate, rng))
    return (out / max(float(np.max(np.abs(out))), 1e-9)).astype(np.float32)


def _scale_to(bed, reference, target_db):
    """Scales a bed so its RMS sits target_db below the reference's."""
    ref = float(np.sqrt(np.mean(reference**2))) + 1e-12
    bed_rms = float(np.sqrt(np.mean(bed**2))) + 1e-12
    return bed * (ref * 10.0 ** (target_db / 20.0) / bed_rms)


def _normalised(reference, target):
    """Scales both to the reference's 0.7 peak, so the target sits at the reference's level."""
    scale = 0.7 / max(float(np.max(np.abs(reference))), 1e-9)
    return (reference * scale).astype(np.float32), (target * scale).astype(np.float32)


def condition_programme(mono, rate, rng, music, other=None):
    """Turns clean synthesised speech into something with a tape programme's properties.

    Returns (reference, target): the reference is everything the tape carried, the target
    is the programme wanted back -- the voice in its room and the music, without the room
    tone and the voice across the room.
    """
    voice = _saturate(_in_a_room(_with_breaths(_with_gaps(_voiced_for_tape(mono, rate), rate, rng), rate, rng), rate, rng))
    out = voice
    if other is not None:
        out = out + _scale_to(_background_voice(other, len(out), rate, rng), out, BACKGROUND_DB)
    out = out + _scale_to(_ambience(len(out), rate, rng), out, AMBIENCE_DB)
    target = voice
    if music:
        bed = _scale_to(_music_bed(len(out), rate, rng), out, MUSIC_DB)
        out = out + bed
        target = target + bed
    return _normalised(out, target)


def condition_music_led(mono, rate, rng, with_speech):
    """Music-led programme: music at full level, with speech well under it or absent."""
    music = _music_programme(len(mono), rate, rng) * 0.5
    voice = _with_gaps(_saturate(_voiced_for_tape(mono, rate)), rate, rng)
    target = music + (_scale_to(voice, music, MUSIC_LED_SPEECH_DB) if with_speech else 0.0)
    out = target + _scale_to(_ambience(len(target), rate, rng), target, AMBIENCE_DB)
    return _normalised(out, target)


def _gain_control(mix, rate):
    """The gain trajectory a slow automatic gain control would apply to this mix.

    The level follower runs on a millisecond grid of block levels, the way the defect
    injectors' followers do, and the gain is interpolated back to samples: at attack and
    release times of 50 and 800 ms a one-millisecond step is far below the follower's own
    resolution, and it makes the loop a thousandth of the length it was per sample. Against
    the per-sample follower it replaced, the gain is identical away from onsets and within
    0.5 dB during the millisecond after one.
    """
    magnitude = np.abs(mix, dtype=np.float64)
    block = max(int(AGC_STEP_S * rate), 1)
    count = -(-len(magnitude) // block)
    levels = np.mean(np.pad(magnitude, (0, count * block - len(magnitude))).reshape(count, block), axis=1)
    attack = np.exp(-block / (AGC_ATTACK_S * rate))
    release = np.exp(-block / (AGC_RELEASE_S * rate))
    followed = np.zeros(count)
    level = 0.0
    for index, block_level in enumerate(levels):
        coefficient = attack if block_level > level else release
        level = coefficient * level + (1.0 - coefficient) * block_level
        followed[index] = level
    envelope = np.interp(np.arange(len(mix)), (np.arange(count) + 0.5) * block, followed)
    gain = AGC_TARGET / (envelope + 1e-6)
    return np.clip(gain, 1.0, 10.0 ** (AGC_MAX_GAIN_DB / 20.0)).astype(np.float32)


def degrade(programme, rate, noise, margin_db, hum, rng):
    """The recorder rides the programme and its hum; the tape adds hiss at a corpus-drawn margin.

    Takes (reference, target) and returns (reference, target, degraded): the gain control is
    part of the recording, so both carry the same gain trajectory as the recorded signal.
    """
    return degrade_with(programme, rate, noise, margin_db, ("hum",) if hum else (), rng)


def _write_pair(out_dir, name, signals, defects, music, margin):
    """Writes one fixture's reference, target and degraded files and returns its manifest record."""
    clean, target, degraded = signals
    sf.write(str(out_dir / f"{name}_clean.wav"), clean, FIXTURE_SAMPLE_RATE, subtype="FLOAT")
    sf.write(str(out_dir / f"{name}_target.wav"), target, FIXTURE_SAMPLE_RATE, subtype="FLOAT")
    sf.write(str(out_dir / f"{name}_vhs.wav"), degraded, FIXTURE_SAMPLE_RATE, subtype="FLOAT")
    return {
        "name": name,
        "defects": defects,
        "music": music,
        "margin_db": margin,
        "sample_rate": FIXTURE_SAMPLE_RATE,
        "frames": int(len(clean)),
        "clean": f"{name}_clean.wav",
        "target": f"{name}_target.wav",
        "degraded": f"{name}_vhs.wav",
    }


def _discover_languages(fixtures_dir):
    """Returns {language: [clean reference paths]} from a generated fixture tree."""
    languages = {}
    for language_dir in sorted(p for p in fixtures_dir.iterdir() if p.is_dir()):
        cleans = sorted(language_dir.glob("*_clean.wav"))
        if cleans:
            languages[language_dir.name] = cleans
    return languages


def _segments(samples, segment_frames, max_segments):
    """Splits a reference into independent segments to widen a thin fixture set."""
    if segment_frames <= 0 or len(samples) <= segment_frames:
        return [samples]
    starts = list(range(0, len(samples) - segment_frames + 1, segment_frames))[:max_segments]
    segments = []
    for start in starts:
        stop = start + segment_frames
        segments.append(samples[start:stop])
    return segments


def _load_clean(path):
    """Reads a clean reference at the pipeline rate. Resampling needs librosa, which the static CI job lacks, so it loads here."""
    from scripts.make_reference_fixtures import _load_clean as load_clean

    return load_clean(path)


def _clean_segments(clean_paths, segment_frames, max_segments):
    """Yields (stem, segment index, mono segment) for every clean source."""
    for path in clean_paths:
        # The clean sources are named `<take>_clean.wav`; the fixture family is read from the
        # first underscore on, so the suffix has to go or every fixture reads as "clean...".
        stem = path.stem[: -len("_clean")] if path.stem.endswith("_clean") else path.stem
        samples = _load_clean(path)
        mono = samples.mean(axis=1) if samples.ndim > 1 else samples
        # Cut long, trim the synthesiser's pauses, then cut to length, so every fixture is the
        # same length and none of it is dead silence.
        for index, segment in enumerate(_segments(mono, int(segment_frames * 1.25), max_segments)):
            trimmed = _trim_pauses(segment, FIXTURE_SAMPLE_RATE)[:segment_frames]
            yield stem, index, np.pad(trimmed, (0, max(0, segment_frames - len(trimmed))))


def _build_speech_led(clean_paths, out_dir, segment_frames, max_segments, noise_bank, others, rng):
    """The speech-led classes: speech, and speech over a sustained bed, at every margin."""
    records = []
    for stem, index, segment in _clean_segments(clean_paths, segment_frames, max_segments):
        for music in (False, True):
            for margin in MARGIN_DB:
                for hum in (False, True):
                    other = others[int(rng.integers(len(others)))] if others else None
                    programme = condition_programme(segment, FIXTURE_SAMPLE_RATE, rng, music, other)
                    noise = noise_bank[int(rng.integers(len(noise_bank)))]
                    signals = degrade(programme, FIXTURE_SAMPLE_RATE, noise, margin, hum, rng)
                    name = f"{stem}{index:02d}_{'music' if music else 'speech'}_m{int(margin):02d}{'_hum' if hum else ''}"
                    defects = ["tape_noise"] + (["hum"] if hum else [])
                    records.append(_write_pair(out_dir, name, signals, defects, music, margin))
    return records


def _build_music_led(clean_paths, out_dir, segment_frames, max_segments, noise_bank, rng):
    """The music-led classes: music with speech under it, and music alone, at every margin."""
    records = []
    for stem, index, segment in _clean_segments(clean_paths, segment_frames, max_segments):
        for with_speech in (True, False):
            for margin in MARGIN_DB:
                programme = condition_music_led(segment, FIXTURE_SAMPLE_RATE, rng, with_speech)
                noise = noise_bank[int(rng.integers(len(noise_bank)))]
                signals = degrade(programme, FIXTURE_SAMPLE_RATE, noise, margin, False, rng)
                name = f"{stem}{index:02d}_{'musicled' if with_speech else 'musiconly'}_m{int(margin):02d}"
                records.append(_write_pair(out_dir, name, signals, ["tape_noise"], True, margin))
    return records


def _band_limit(signals, rate, defects):
    """The tape's response: what it never carried the reference does not carry either."""
    for name, (cutoff, order) in BAND_LIMITS.items():
        if name in defects:
            b, a = scipy.signal.butter(order, cutoff / (rate / 2), "low")
            signals = tuple(scipy.signal.lfilter(b, a, x).astype(np.float32) for x in signals)
    return signals


def _camcorder_faults(programme, rate, defects, rng, other=None):
    """What the camcorder adds before the tape: on the reference, since it was recorded, not on the target."""
    reference, target = programme
    if "resonance" in defects:
        reference = inject_enclosure_resonance(reference, rate, rng)
    if "plosive" in defects:
        reference = inject_plosives(reference, rate, rng)
    if "handling" in defects:
        reference = inject_handling_noise(reference, rate, rng)
    if "ghost" in defects and other is not None:
        reference = inject_ghost_recording(reference, rate, other)
    return reference, target


# The tape's and the capture's faults after the noise, in the order they happen to a tape:
# a whine recorded on the tape (the transport classes carry one, see DEFECT_CLASSES) goes
# on before the wear that loses it (a dropout loses everything the tape carried; put on
# after, it filled every dropout with a tone 10 dB above the detector's silence and none
# was counted), the transport warps everything recorded, and the capture monitor, the deck
# and the archive come last. Each entry is (defect, injector taking (degraded, rate, noise,
# rng, work_dir)).
TAPE_FAULTS = (
    ("recordedwhine", lambda x, rate, _n, rng, _w: inject_line_whistle(x, rate, rng)),
    ("crackle", lambda x, rate, _n, rng, _w: inject_crackle(x, rate, rng)),
    ("dropout", lambda x, rate, _n, rng, _w: inject_dropouts(x, rate, rng)),
    ("flutter", lambda x, rate, _n, _r, _w: inject_wow_and_flutter(x, rate)),
    ("scrapeflutter", lambda x, rate, _n, rng, _w: inject_scrape_flutter(x, rate, rng)),
    ("whistle", lambda x, rate, _n, rng, _w: inject_line_whistle(x, rate, rng)),
    ("rumble", lambda x, rate, _n, rng, _w: inject_rumble(x, rate, rng)),
    ("dc", lambda x, _rate, _n, _r, _w: inject_dc_offset(x)),
    ("hifibuzz", lambda x, rate, _n, rng, _w: inject_head_switching(x, rate, rng)),
    ("crosstalk", lambda x, rate, _n, rng, _w: inject_video_crosstalk(x, rate, rng)),
    ("buzz", lambda x, rate, _n, rng, _w: inject_emi_buzz(x, rate, rng)),
    ("printthrough", lambda x, rate, _n, _r, _w: inject_print_through(x, rate)),
    ("squeal", lambda x, rate, _n, rng, _w: inject_squeal(x, rate, rng)),
    ("modnoise", lambda x, rate, _n, rng, _w: inject_modulation_noise(x, rate, rng)),
    ("breathing", lambda x, rate, noise, rng, _w: inject_breathing(x, rate, noise, rng)),
    ("trackswitch", lambda x, rate, noise, rng, _w: inject_track_switch(x, rate, noise, rng)),
    ("dolbyb", lambda x, rate, _n, _r, _w: inject_undecoded_dolby_b(x, rate)),
    ("edgedamage", lambda x, rate, _n, rng, _w: inject_edge_damage(x, rate, rng)),
    ("headclog", lambda x, rate, _n, rng, _w: inject_head_clog(x, rate, rng)),
    ("codec", lambda x, rate, _n, _r, work: inject_codec(x, rate, work)),
)


def _tape_faults(degraded, rate, defects, noise, rng, work_dir):
    """Applies every fault the class carries, in the order a tape acquires them."""
    for defect, inject in TAPE_FAULTS:
        if defect in defects:
            degraded = inject(degraded, rate, noise, rng, work_dir)
    return degraded


def _record(programme, rate, hum, rng):
    """The recorder: hum from its supply, and its gain control riding the level.

    Levelled back to the programme's 0.7 peak afterwards, since the gain control lifted a
    dense music-led fixture past full scale and the clipping detector read it as clipped.
    """
    clean, target = programme
    recorded = inject_mains_hum(clean, rate, level=HUM_LEVEL, rng=rng) if hum else clean
    gain = _gain_control(recorded, rate)
    gain = gain * (0.7 / max(float(np.max(np.abs(recorded * gain))), 1e-9))
    return (clean * gain).astype(np.float32), (target * gain).astype(np.float32), (recorded * gain).astype(np.float32)


def _clipped_capture(reference, target, noisy):
    """A capture whose digitiser was driven too hard: flat tops at the ceiling, noise and all.

    The reference is scaled by the same gain and left unclipped, which is what the declip
    stage is asked to give back.
    """
    scale = CLIP_SCALE / max(float(np.max(np.abs(noisy))), 1e-9)
    return reference * scale, target * scale, inject_clipping(noisy)


def _crowd_bed(length, rate, rng):
    """A crowd: band-passed noise breathing slowly, with a few swells, the way a stadium or an audience sits under commentary."""
    b, a = scipy.signal.butter(2, [CROWD_BAND_HZ[0] / (rate / 2), CROWD_BAND_HZ[1] / (rate / 2)], "band")
    bed = scipy.signal.lfilter(b, a, rng.normal(0.0, 1.0, length))
    time = np.arange(length) / rate
    envelope = 1.0 + 0.3 * np.sin(2.0 * np.pi * float(rng.uniform(0.2, 0.5)) * time + rng.uniform(0, 6.28))
    for _ in range(int(rng.integers(*CROWD_SWELLS))):
        centre = float(rng.uniform(0.5, length / rate - 0.5))
        envelope += 1.0 * np.exp(-((time - centre) ** 2) / (2 * 0.5**2))
    return (bed * envelope).astype(np.float32)


def _with_crowd(programme, rate, rng):
    """Puts a crowd under both the reference and the target: it is programme, not noise."""
    reference, target = programme
    bed = _scale_to(_crowd_bed(len(reference), rate, rng), reference, CROWD_DB)
    return _normalised(reference + bed, target + bed)


def degrade_with(programme, rate, noise, margin_db, defects, rng, other=None, work_dir=None):
    """The full recording chain with a chosen set of faults; returns (reference, target, degraded)."""
    if "crowd" in defects:
        programme = _with_crowd(programme, rate, rng)
    programme = _camcorder_faults(_band_limit(programme, rate, defects), rate, defects, rng, other)
    reference, target, recorded = _record(programme, rate, "hum" in defects, rng)
    noisy = inject_tape_noise(recorded, noise, margin_db, rng=rng, wobble_db=NOISE_WOBBLE_DB)
    if "clip" in defects:
        reference, target, noisy = _clipped_capture(reference, target, noisy)
    degraded = _tape_faults(noisy, rate, defects, noise, rng, work_dir)
    if "low_level" in defects:
        scale = 10.0 ** (LOW_LEVEL_DB / 20.0)
        reference, target, degraded = reference * scale, target * scale, degraded * scale
    if "azimuth" in defects or "imbalance" in defects:
        degraded = assemble_stereo(degraded, rate, rng, "azimuth" in defects, "imbalance" in defects)
        reference, target = np.column_stack((reference, reference)), np.column_stack((target, target))
    return reference.astype(np.float32), target.astype(np.float32), degraded.astype(np.float32)


def _build_defect_classes(clean_paths, out_dir, segment_frames, noise_bank, others, rng):
    """One fixture per voice and defect class, on the speech-led base."""
    records = []
    for stem, index, segment in _clean_segments(clean_paths, segment_frames, 1):
        for name, defects in DEFECT_CLASSES.items():
            other = others[int(rng.integers(len(others)))] if others else None
            programme = condition_programme(segment, FIXTURE_SAMPLE_RATE, rng, False, other)
            noise = noise_bank[int(rng.integers(len(noise_bank)))]
            margin = CLASS_MARGIN_DB.get(name, DEFECT_MARGIN_DB)
            # The ghost is another voice under this one; with a single language there is no
            # other voice, so the same segment, rolled by a third, stands in.
            ghost = None
            if "ghost" in defects:
                ghost = others[int(rng.integers(len(others)))] if others else np.roll(segment, len(segment) // 3)
            signals = degrade_with(programme, FIXTURE_SAMPLE_RATE, noise, margin, defects, rng, ghost, out_dir / "_codec")
            fixture = f"{stem}{index:02d}_{name}_m{int(margin):02d}"
            records.append(_write_pair(out_dir, fixture, signals, ["tape_noise", *defects], False, margin))
    return records


def _build(clean_paths, out_dir, segment_frames, max_segments, noise_bank, others, seed, voice=0):
    """Writes every variant for one language and returns the manifest records.

    Every class draws on a stream seeded by the voice as well as the run, so five voices do
    not share one noise window, one room, one chord sequence, one resonance centre, one
    dropout pattern, one field rate: a median over five voices is then a median over five
    draws of everything, not five voices under one draw.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _build_speech_led(
        clean_paths, out_dir, segment_frames, max_segments, noise_bank, others, np.random.default_rng([seed, voice])
    )
    records += _build_music_led(clean_paths, out_dir, segment_frames, max_segments, noise_bank, np.random.default_rng([seed + 1, voice]))
    records += _build_defect_classes(clean_paths, out_dir, segment_frames, noise_bank, others, np.random.default_rng([seed + 2, voice]))
    return records


def _first_segment(path, segment_frames):
    """The opening segment of a clean source, mono: what a fixture in another language hears behind it."""
    samples = _load_clean(path)
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    return _segments(mono, segment_frames, 1)[0]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=Path("artifacts/multi-clean"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/realistic-v2"))
    # The corpus clips are 15 s, and the metrics read a clip in proportion: the quietest
    # fifth of an 8 s fixture with one 1.2 s pause is that pause and nothing else, so the
    # chain took 15 dB out of it where a 15 s tape gives up 9. Fixtures are the clips' length.
    parser.add_argument("--segment-seconds", type=float, default=15.0)
    parser.add_argument("--max-segments", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--corpus-dir", type=Path, default=Path("experiments/ia_corpus_1000"))
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--noise-windows", type=int, default=24, help="Captures to sample tape noise from")
    return parser.parse_args()


def main():
    """Builds the calibrated fixture set for every language found."""
    args = _parse_args()
    catalog = json.loads((args.catalog or args.corpus_dir / "catalog_1000.json").read_text(encoding="utf-8"))
    noise_bank = build_noise_bank(args.corpus_dir, catalog, args.output_dir / "_noise_probe", count=args.noise_windows)
    if not noise_bank:
        raise SystemExit(f"No usable tape noise found under {args.corpus_dir}")
    print(f"  sampled real tape noise from {len(noise_bank)} captures")
    manifest = {"sample_rate": FIXTURE_SAMPLE_RATE, "languages": {}}
    segment_frames = int(args.segment_seconds * FIXTURE_SAMPLE_RATE)
    languages = _discover_languages(args.fixtures_dir)
    voices = {language: _first_segment(paths[0], segment_frames) for language, paths in languages.items()}
    for index, (language, clean_paths) in enumerate(languages.items()):
        others = [voice for name, voice in voices.items() if name != language]
        records = _build(clean_paths, args.output_dir / language, segment_frames, args.max_segments, noise_bank, others, args.seed, index)
        manifest["languages"][language] = records
        print(f"  {language}: {len(records)} paired fixtures")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total = sum(len(r) for r in manifest["languages"].values())
    print(f"\nWrote {total} paired fixtures and {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
