#!/usr/bin/env python3
"""Acceptance probe for ear v4's hiss readings on the realistic-v2 fixtures (`restoration_quality/hiss_*.py`).

    python -m scripts.check_hiss_fixtures [--fixtures artifacts/realistic-v2] [--languages de,en,es,fr,it]
        [--programme linear|full] [--shape corpus|white|SHAPE.npy] [--shape-from WAV --shape-out artifacts/hiss-v1/shape.npy]
        [--out experiments/hiss_v4/fixtures.jsonl]

Run it as a module from the repository root. Per language, every m12 tape-noise fixture becomes
a gapped programme (two 1.2 s gaps spliced in, `degradations_hiss.with_gaps`, so R4 finds true
pauses and the anchor validates): `--programme linear` low-passes the fixture's target at
4.5 kHz first (the Tata tapes' band), `full` keeps it. The hiss is the fixture's own recorded
noise (`--shape corpus`, `vhs - clean`, repeated on through the gaps), stationary tape hiss at
its 3-10 kHz level (`hiss`, `degradations_hiss.tape_hiss`, the calibration's linear capture), a
flat noise at that level (`white`), or noise shaped by a pause spectrum (a `SHAPE.npy` written by
`--shape-from WAV`: rows freqs, pause spectrum, loud spectrum, in `hiss_floor.stft_power`'s
scaling), placed against the programme's loud frames as the tape's hiss stands against its own.
A shape derived from a family tape stays under the gitignored `artifacts/`; never commit it.

Each source (programme + hiss) is read with its own capture profile and hiss anchor (the
runner's: `hiss_stationary.source_anchor`, the pause anchor or, when that one is not validated,
the continuous-programme anchor the programme reading may validate on), and these
outputs are scored through the two readings: the identity; the oracle, the hiss 6 / 12 / 20 /
30 dB down; the programme alone; a fresh realisation of the hiss; the source delayed 0.37
samples, +-0.1 dB, through a soxr round trip and requantised to 16 bits; `hf_left` (the hiss
above 4 kHz untouched, the rest 20 dB down); `lp4k_all` (everything low-passed at 4 kHz);
`gated_m0` (the hiss kept where the source's VAD calls speech, 30 dB down elsewhere) and
`islands_15` (the hiss broken into islands, 15 % of its cells kept). One JSON row per fixture
goes to `--out`, and the summary rows of the acceptance table (docs/ear_v4_hiss.md) are printed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf

from scripts import degradations_hiss, quality_degradations, same_percept
from scripts.cli_paths import confined_path
from scripts.restoration_quality import hiss_floor, hiss_pause, hiss_programme, hiss_stationary, pause_metrics, source_profile

DEFAULT_FIXTURES = Path("artifacts/realistic-v2")
DEFAULT_OUT = Path("experiments/hiss_v4/fixtures.jsonl")
LANGUAGES = ("de", "en", "es", "fr", "it")
MARGIN = "hiss_prog_margin_db"
READINGS = (
    MARGIN,
    "hiss_prog_excess_db",
    "hiss_prog_margin_all_db",
    "hiss_prog_excess_all_db",
    "hiss_pause_margin_db",
    "hiss_pause_excess_db",
    "hiss_prog_removed_hf_db",
    "hiss_pump_db",
    "hiss_island_kurt",
    "hiss_prog_kept_hf_db",
    "hiss_prog_kept_soft_db",
    "hiss_prog_validated",
)
ORACLE_DB = (6.0, 12.0, 20.0, 30.0)
LADDER = ("identity", "oracle_m6", "oracle_m12", "oracle_m20", "oracle_m30", "prog_only")
HF_LEFT_HZ = 4000.0
LP_ALL_HZ = 4000.0
GATED_LEFT_DB = 30.0
ISLAND_SHARE = 0.15
ISLAND_FRAME = 256
ISLAND_SEED = 5
SHAPE_SEEDS = (3, 4)
FRESH_SEED = 7
CORE_HZ = (500.0, 3000.0)
WHITE_HZ = (3000.0, 10000.0)
LADDER_SLACK_DB = 0.05
SPEECH = "speech"


def read_mono(path):
    """`(mono float64, rate)` of a WAV."""
    audio, rate = sf.read(str(path), dtype="float64", always_2d=True)
    return audio.mean(axis=1), int(rate)


def tape_noise_records(manifest, language):
    """The language's m12 fixtures that carry tape noise alone."""
    return [rec for rec in manifest["languages"].get(language, []) if rec["defects"] == ["tape_noise"] and rec["name"].endswith("m12")]


def mean_spectrum(power, frames):
    """The per-bin mean of `power` over `frames` (all frames when none is selected)."""
    return power[frames].mean(axis=0) if frames.any() else power.mean(axis=0)


def loud_spectrum(mono, rate):
    """`(freqs, per-bin mean power of the loud frames, >= p70)` in `hiss_floor.stft_power`'s scaling."""
    freqs, power = hiss_floor.stft_power(mono, rate)
    level = power.sum(axis=1)
    return freqs, mean_spectrum(power, level >= np.percentile(level, 70.0))


def shape_from(wav):
    """A shape file's rows from a capture: freqs, the mean power of its true pauses, the mean power of its loud frames."""
    mono, rate = read_mono(wav)
    freqs, power = hiss_floor.stft_power(mono, rate)
    paused = hiss_floor.frame_map(rate, len(power), hiss_floor.chunked_pause_mask(mono, rate))
    return np.vstack([freqs, mean_spectrum(power, paused), loud_spectrum(mono, rate)[1]])


def white_spectrum(noise, rate):
    """`(freqs, flat power)` at the noise's own mean per-bin power over 3-10 kHz."""
    freqs, power = hiss_floor.stft_power(noise, rate)
    band = (freqs >= WHITE_HZ[0]) & (freqs < WHITE_HZ[1])
    return freqs, np.full(len(freqs), power[:, band].mean())


def placed_spectrum(rows, programme, rate):
    """A shape file's pause spectrum moved so it stands against the programme's loud core as against the tape's own."""
    freqs, loud = loud_spectrum(programme, rate)
    core, tape_core = ((grid >= CORE_HZ[0]) & (grid < CORE_HZ[1]) for grid in (freqs, rows[0]))
    return rows[0], rows[1] * (loud[core].mean() / rows[2][tape_core].mean())


def shaped_noise(count, rate, spectrum, seed):
    """Seeded noise whose `hiss_floor.stft_power` reads `spectrum` (`(freqs, per-bin power)`)."""
    white = np.fft.rfft(np.random.default_rng(seed).standard_normal(count))
    power = np.interp(np.fft.rfftfreq(count, 1.0 / rate), *spectrum) * hiss_floor.FRAME / 2.0
    return np.fft.irfft(white * np.sqrt(power), count)


def corpus_noise(_shape, corpus, programme, rate):
    """The fixture's recorded noise repeated on through the gaps, and a fresh realisation of it."""
    noise = np.resize(corpus, len(programme))
    return noise, same_percept.fresh_like(noise, rate, FRESH_SEED)


def tape_hiss_noise(_shape, corpus, programme, rate):
    """Two realisations of stationary tape hiss at the recorded noise's 3-10 kHz level (the calibration's linear capture)."""
    return tuple(degradations_hiss.tape_hiss(corpus, rate, len(programme), seed) for seed in SHAPE_SEEDS)


def _shaped(spectrum, programme, rate):
    return tuple(shaped_noise(len(programme), rate, spectrum, seed) for seed in SHAPE_SEEDS)


def white_noise(_shape, corpus, programme, rate):
    """Two realisations of a flat noise at the recorded noise's 3-10 kHz level."""
    return _shaped(white_spectrum(corpus, rate), programme, rate)


def file_noise(shape, _corpus, programme, rate):
    """Two realisations of noise shaped by a shape file's pause spectrum, placed against the programme."""
    return _shaped(placed_spectrum(np.load(shape), programme, rate), programme, rate)


NOISE_MAKERS = {"corpus": corpus_noise, "hiss": tape_hiss_noise, "white": white_noise}
NAMED_SHAPES = tuple(NOISE_MAKERS)


def make_noise(shape, corpus, programme, rate):
    """`(hiss, a fresh realisation of it)` for one programme: the corpus noise, tape hiss, a flat noise or a shape file's."""
    return NOISE_MAKERS.get(str(shape), file_noise)(shape, corpus, programme, rate)


def programme_of(target, rate, kind):
    """The gapped programme: the target low-passed at 4.5 kHz for `linear`, as it is for `full`."""
    base = degradations_hiss.lowpass(target, rate, degradations_hiss.LINEAR_LOWPASS_HZ) if kind == "linear" else target
    return degradations_hiss.with_gaps(base, rate)


def load_fixture(folder, rec, options):
    """`(programme, hiss, fresh hiss, rate)` of one fixture."""
    vhs, rate = read_mono(folder / rec["degraded"])
    clean, _rate = read_mono(folder / rec["clean"])
    programme = programme_of(read_mono(folder / rec["target"])[0], rate, options.programme)
    noise, fresh = make_noise(options.shape, vhs - clean, programme, rate)
    return programme, noise, fresh, rate


def speech_gate(source, rate):
    """1 per sample where the source's DSP VAD calls speech."""
    frame = max(1, int(pause_metrics.FRAME_S * rate))
    speech = np.repeat(pause_metrics.dsp_vad(source, rate, frame), frame)
    return np.pad(speech, (0, max(0, len(source) - len(speech))))[: len(source)].astype(np.float64)


def islands(noise, rate):
    """The noise with 15 % of its 256-point STFT cells kept (rescaled): spectral-subtraction islands."""
    _freqs, _times, spectrum = scipy.signal.stft(noise, rate, nperseg=ISLAND_FRAME)
    cells = np.random.default_rng(ISLAND_SEED).random(spectrum.shape) < ISLAND_SHARE
    _times, out = scipy.signal.istft(np.where(cells, spectrum / np.sqrt(ISLAND_SHARE), 0.0), rate, nperseg=ISLAND_FRAME)
    return out[: len(noise)]


def cases_for(programme, noise, fresh, rate):
    """Every output the probe scores, by name (module docstring)."""
    source = programme + noise
    low = degradations_hiss.lowpass(noise, rate, HF_LEFT_HZ)
    oracle = {f"oracle_m{int(gain)}": programme + noise * 10.0 ** (-gain / 20.0) for gain in ORACLE_DB}
    return {
        "identity": source,
        **oracle,
        "prog_only": programme,
        "fresh": programme + fresh,
        "delay037": same_percept.fractional_delay(source, same_percept.FRACTIONAL_DELAY_SAMPLES),
        "gain_up": same_percept.gain_db(source, same_percept.GAIN_STEP_DB),
        "gain_down": same_percept.gain_db(source, -same_percept.GAIN_STEP_DB),
        "soxr": same_percept.soxr_round_trip(source, rate, same_percept.round_trip_rate(rate)),
        "requantise": quality_degradations.benign_requantise(source, np.random.default_rng(FRESH_SEED)).astype(np.float64),
        "hf_left": programme + (noise - low) + 0.1 * low,
        "lp4k_all": degradations_hiss.lowpass(source, rate, LP_ALL_HZ),
        "gated_m0": programme + noise * speech_gate(source, rate) + noise * 10.0 ** (-GATED_LEFT_DB / 20.0),
        "islands_15": programme + islands(noise, rate),
    }


def _rounded(value):
    return None if value is None else round(float(value), 3)


def case_readings(source, output, rate, setting):
    """The output side of `READINGS` for one case; `setting` is `(anchor, top)`."""
    anchor, top = setting
    context = hiss_floor.HissContext(anchor, top, hiss_floor.spl_offset_db(source), hiss_floor.loud_db(source), hiss_floor.loud_db(output))
    found = {**hiss_programme.programme_readings(source, output, rate, context), **hiss_pause.pause_readings(source, output, rate, context)}
    return {name: _rounded(found[name][1]) for name in READINGS}


def anchor_error(anchor, noise, rate, top):
    """`[min, median, max]` of the anchor over the noise's true band means, dB, from 1 kHz up; None without an anchor."""
    if anchor.level is None:
        return None
    freqs, power = hiss_floor.stft_power(noise, rate)
    index, centres = hiss_floor.band_layout(freqs, rate, top)
    error = 10.0 * np.log10(anchor.level / hiss_floor.band_means(power, index, len(centres)))[centres >= 1000.0]
    return [round(float(stat(error[np.isfinite(error)])), 2) for stat in (np.min, np.median, np.max)]


def score_fixture(programme, noise, fresh, rate):
    """One fixture's row: its capture profile, its anchor (with the runner's second path) and every case's readings.

    `validated` is the programme reading's anchor (`hiss_floor.Anchor.for_programme`), `programme_how` its path.
    """
    source = programme + noise
    profile = source_profile.capture_profile(source.astype(np.float32), rate)
    top = profile.get("brickwall_hz")
    anchor = hiss_stationary.source_anchor(source, rate, profile.get("prog_bandwidth_hz"), top)
    cases = {name: case_readings(source, output, rate, (anchor, top)) for name, output in cases_for(programme, noise, fresh, rate).items()}
    row = {
        "prog_bw": profile.get("prog_bandwidth_hz"),
        "how": anchor.how,
        "pause_s": round(anchor.pause_s, 2),
        "validated": anchor.for_programme().validated,
        "programme_how": anchor.programme_how(),
    }
    return {**row, "anchor_err_1k_up": anchor_error(anchor, noise, rate, top), "cases": cases}


# ----------------------------------------------------------------------------- the summary rows


def _value(row, case, reading=MARGIN):
    return row["cases"][case][reading]


def deviations(rows, case, reading):
    """|case - identity| of one reading over the rows where both read."""
    pairs = [(_value(row, case, reading), _value(row, "identity", reading)) for row in rows]
    return [abs(out - base) for out, base in pairs if out is not None and base is not None]


def p95_max(values):
    """`(p95, max)` of the values, `(None, None)` with none."""
    return (round(float(np.percentile(values, 95.0)), 3), round(float(np.max(values)), 3)) if values else (None, None)


def jitter_line(rows, cases, label):
    """The p95 / max deviation of the hiss margins and excesses over `cases`, as one summary line."""
    parts = []
    for reading in (
        "hiss_prog_margin_db",
        "hiss_prog_excess_db",
        "hiss_prog_excess_all_db",
        "hiss_pause_margin_db",
        "hiss_pause_excess_db",
    ):
        found = [deviation for case in cases for deviation in deviations(rows, case, reading)]
        parts.append(f"{reading.removeprefix('hiss_')} p95/max {p95_max(found)}")
    return f"{label}: " + "; ".join(parts)


def monotone(row):
    """Whether the margin never rises by more than 0.05 dB through the oracle ladder."""
    margins = [_value(row, case) for case in LADDER]
    return None not in margins and all(later <= earlier + LADDER_SLACK_DB for earlier, later in zip(margins, margins[1:]))


def share(flags):
    """`"n/m"` of the true flags."""
    return f"{sum(bool(flag) for flag in flags)}/{len(flags)}"


def _stats(values):
    return {"min": round(min(values), 3), "median": round(float(np.median(values)), 3), "max": round(max(values), 3)} if values else {}


def _readings_of(rows, case, reading=MARGIN):
    return [_value(row, case, reading) for row in rows if _value(row, case, reading) is not None]


def removal_line(rows):
    """The median of removed_hf - g at the oracle's 6 and 12 dB."""
    medians = [
        round(float(np.median([value - gain for value in _readings_of(rows, f"oracle_m{int(gain)}", "hiss_prog_removed_hf_db")])), 3)
        for gain in (6.0, 12.0)
    ]
    return f"removed_hf - g, median at 6 / 12 dB: {medians}"


def guard_lines(rows):
    """The gated, islands and low-pass rows of the acceptance table."""
    within = [_value(row, "gated_m0") is not None and _value(row, "gated_m0") <= _value(row, "identity") + 0.5 for row in rows]
    pumps = _readings_of(rows, "gated_m0", "hiss_pump_db")
    kept = _readings_of(rows, "lp4k_all", "hiss_prog_kept_hf_db")
    return [
        f"gated_m0: margin within +0.5 dB of identity {share(within)}, pump >= 15 {share([p >= 15.0 for p in pumps])} {_stats(pumps)}",
        f"islands_15: island kurtosis {_stats(_readings_of(rows, 'islands_15', 'hiss_island_kurt'))}",
        f"lp4k_all: kept HF <= -1 dB {share([k <= -1.0 for k in kept])} {_stats(kept)}",
        f"prog_only: kept soft (the hiss's share) {_stats(_readings_of(rows, 'prog_only', 'hiss_prog_kept_soft_db'))}",
    ]


JITTER_ROWS = (
    (("gain_up", "gain_down"), "gain +-0.1 dB"),
    (("delay037",), "delay 0.37"),
    (("soxr",), "soxr"),
    (("requantise",), "requantise"),
    (("fresh",), "fresh"),
)


def ladder_line(rows):
    """The oracle ladder's monotone share and the perfect output's margins."""
    perfect = _readings_of(rows, "prog_only")
    inaudible = share([value <= 0.0 for value in perfect])
    return f"oracle ladder monotone {share([monotone(row) for row in rows])}; perfect <= 0 {inaudible} {_stats(perfect)}"


def hf_left_line(rows):
    """|median delta| of the margin when only the hiss under 4 kHz was taken."""
    found = deviations(rows, "hf_left", MARGIN)
    return f"hf_left |median delta| {round(float(np.median(found)), 3) if found else None}"


def summary_lines(rows):
    """The acceptance table's fixture rows, over the validated fixtures."""
    valid = [row for row in rows if row["validated"]]
    lines = [f"validated {share([row['validated'] for row in rows])}"]
    lines += [jitter_line(valid, cases, label) for cases, label in JITTER_ROWS]
    return lines + [ladder_line(valid), removal_line(valid), hf_left_line(valid), *guard_lines(valid)]


# ----------------------------------------------------------------------------- command line


def _csv(value):
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def parse_args(argv=None):
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--languages", type=_csv, default=LANGUAGES)
    parser.add_argument("--programme", choices=("linear", "full"), default="linear")
    parser.add_argument("--shape", default="corpus", help="corpus, hiss, white or a shape .npy (--shape-from writes one)")
    parser.add_argument("--shape-from", type=Path, default=None, help="a capture (WAV) to read a pause spectrum from")
    parser.add_argument("--shape-out", type=Path, default=Path("artifacts/hiss-v1/shape.npy"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def _confined(args):
    """`args` with every path checked where it is used (`scripts/cli_paths.py`)."""
    args.fixtures = confined_path(args.fixtures, "--fixtures", must_exist=True)
    args.out = confined_path(args.out, "--out")
    if args.shape not in NAMED_SHAPES:
        args.shape = confined_path(args.shape, "--shape", must_exist=True)
    return args


def write_shape(args):
    """`--shape-from`: the capture's shape rows at `--shape-out`, which then becomes `--shape`."""
    target = confined_path(args.shape_out, "--shape-out")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, shape_from(confined_path(args.shape_from, "--shape-from", must_exist=True)))
    args.shape = str(target)
    print(f"wrote {target}")


def run(args):
    """Every fixture's row, written to `--out` one JSON line each, in language and manifest order."""
    manifest = json.loads((args.fixtures / "manifest.json").read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.out, "w", encoding="utf-8") as stream:
        for language in args.languages:
            for rec in tape_noise_records(manifest, language):
                row = {"lang": language, "name": rec["name"], "programme": args.programme, "shape": str(args.shape)}
                row.update(score_fixture(*load_fixture(args.fixtures / language, rec, args)))
                stream.write(json.dumps(row) + "\n")
                rows.append(row)
    return rows


def main(argv=None):
    """Writes the fixture rows and prints the acceptance table's summary rows."""
    args = parse_args(argv)
    if args.shape_from:
        write_shape(args)
    rows = run(_confined(args))
    print("\n".join(summary_lines(rows)))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
