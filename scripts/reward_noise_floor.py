#!/usr/bin/env python3
"""Test-retest noise level of every harness reading: score an output, then near-copies of it, against the same source.

    python -m scripts.reward_noise_floor SOURCE=OUTPUT [SOURCE[=OUTPUT] ...] --out experiments/reward/noise_floor.json
        [--transforms shift_1,requantise_16] [--repeats 1] [--families dsp]
        [--windows 15 --hop 7.5] [--max-seconds 0] [--percentile 95]
        [--cache-dir experiments/quality_cache] [--language ro] [--device cuda]

Run it as a module from the repository root (`python -m scripts.reward_noise_floor`, or
`poetry run python -m ...`): it imports the `scripts` package and does not patch sys.path.

The operating point. The reward ranks (source, restored candidate) pairs, so the noise it
must tie away is the scatter of the readings when the CANDIDATE changes inaudibly while the
source stays fixed. Every `SOURCE=OUTPUT` pair (WAVs or videos; audio is extracted once
into the cache) is scored as the `reference` (source, output) and then as (source,
near-copy of the output) for every transform, the source side untouched. Run it on the
shipped APL and cathar outputs of the tuning tapes, the point the reward works at. A bare
`SOURCE` falls back to the identity point (the source is its own output). That point
under-reads: every delta reading sits at 0, and 16-bit dither (about -96 dBFS) vanishes
under raw tape hiss (about -50 dBFS) where it would not under a restored output's pause
floor near -70 dBFS. Measured on a synthetic 6 s pair (source hiss ~-46 dBFS, output hiss
~-70 dBFS, DSP family), requantisation moved `dsp.residual_noise_db.delta.median` by 4e-5
at identity and 8.1e-3 at the output point, `dsp.lkr` by 3e-4 against 1.5e-2 and
`dsp.whistle_db` by 3.4e-3 against 5.4e-2: identity floors read 16-200x too narrow.
The report names the point of every pair (`sources[].operating_point`)
and lists the points it pooled (`operating_points`): pool one point per report. A path
holding an '=' is read whole when it exists; a pair splits at the first '=' that leaves an
existing file on both sides.

The near-copies no listener can tell apart from the output:

- `shift_1`: one sample of head delay; the alignment must absorb it (on the synthetic 6 s
  pair above no DSP reading moved by more than 1.2e-7), and the audibility check (plan
  1.2) must call it inaudible;
- `requantise_16`: 16-bit with TPDF dither (`quality_degradations.benign_requantise`), a
  change to every sample at about -96 dBFS;
- `resample_roundtrip` (opt-in): a polyphase round trip through 48 kHz, or through 44.1 kHz
  when the output already runs at 48 kHz, so it never passes through the output's own rate
  as a no-op. Its anti-alias filters soften the band edge near 20 kHz, which the readings
  above 16 kHz may see.

Each scoring is flattened the way a sweep ranks it (`runner.aggregates_for_tuning`,
`{metric.side.stat: value}` plus the `gates.*` counts). Per reading, the deviation of every
scoring after the reference is |x - x_reference|; the reading's `floor` is the
`--percentile` (default 95) of those deviations over every pair, transform and repeat,
beside the `max`, the count `n` and the largest deviation per transform.

`--repeats N` scores the whole set N times. The DSP family is deterministic (repeats read
0); the learned families on the GPU are not, and their jitter is part of the floor. The
scorer caches what is dear per file: Whisper transcripts per file key, lag and language
(`<cache-dir>/asr/`), separated stems (`<cache-dir>/stems/`), resampled audio. A repeat
that hit those caches would record 0 jitter for the CER/WER and stem readings, so every
repeat after the first scores through its own cache directory,
`<cache-dir>/noise_floor/repeat<r>/` (listed in the report as `repeat_caches`). That costs
the disk of one more set of those caches per repeat, and a later run reuses them.

Why (ear v3 plan, 3.0): the group reward (`restoration_quality/reward.py`) calls two
candidates a tie when their difference lies within the floors of the readings they are
compared on, and the learned judges veto only beyond 3x their benign floor. Without a
floor the v3 sibilance loop accepted moves that differed from the v2 plateau by -70..-100
dBFS. `calibrate_quality_metrics.noise_floor` measures a benign floor on synthetic fixtures
(p95 |delta - centre| over identity, requantisation, resampling and shifts); this one
measures it on the material the reward will rank (real tapes and their restored outputs,
125 s excerpts), as the retest difference of the same flat keys the reward reads.

Cost: a DSP-only scoring of a 6 s clip takes ~0.8 s here; all families on the four tuning
tapes (~22 min of audio) took ~7.3 min per candidate in the v4 round, so a full-family run
costs that per pair, transform and repeat. `--max-seconds` scores only the head of the
source and the output. Near-copies are written once into `<cache-dir>/noise_floor/` under
names keyed by the output (path, size, mtime) and reused by later runs, so the scorer's own
caches keep hitting; nothing is written next to a source or an output.
"""

import argparse
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from scripts import quality_degradations as deg
from scripts.cli_paths import confined_path, existing_path_arg, language_arg, path_arg
from scripts.restoration_quality import audio_io, runner

SCHEMA = 2
KIND = "reward_noise_floor"
REFERENCE = "reference"
IDENTITY = "identity"
OUTPUT_POINT = "output"
WORK_DIR = "noise_floor"
SEED = 20261009
SHIFT_SAMPLES = 1
ROUND_TRIP_RATES = (48000, 44100)
FLOOR_PERCENTILE = 95.0
SUMMARY_TOP = 10
DEFAULT_CACHE = Path("experiments/quality_cache")


def _per_channel(audio, function):
    return np.stack([function(np.ascontiguousarray(audio[:, channel])) for channel in range(audio.shape[1])], axis=1)


def shift_one_sample(audio, _rate, _rng):
    """One sample of silence ahead of the programme (`(frames, channels)` in, one frame longer out)."""
    pad = np.zeros((SHIFT_SAMPLES, audio.shape[1]), dtype=np.float32)
    return np.concatenate([pad, np.asarray(audio, dtype=np.float32)])


def requantise_16(audio, _rate, rng):
    """16-bit with TPDF dither, channel by channel."""
    return _per_channel(audio, lambda mono: deg.benign_requantise(mono, rng))


def round_trip_rate(rate):
    """The rate a resample round trip passes through: 48 kHz, or 44.1 kHz for audio already at 48 kHz."""
    return next(via for via in ROUND_TRIP_RATES if via != int(rate))


def _round_trip(mono, rate):
    rate = int(rate)
    via = round_trip_rate(rate)
    up, down = via // gcd(rate, via), rate // gcd(rate, via)
    return resample_poly(resample_poly(mono, up, down), down, up)[: len(mono)].astype(np.float32)


def resample_roundtrip(audio, rate, _rng):
    """A polyphase round trip through another rate (`round_trip_rate`), channel by channel, at the original length."""
    return _per_channel(audio, lambda mono: _round_trip(mono, rate))


TRANSFORMS = {"shift_1": shift_one_sample, "requantise_16": requantise_16, "resample_roundtrip": resample_roundtrip}
DEFAULT_TRANSFORMS = ("shift_1", "requantise_16")


@dataclass(frozen=True)
class Plan:
    """What is scored per pair: the near-copies, how many times each, how much of the audio, and the floor percentile."""

    transforms: tuple = DEFAULT_TRANSFORMS
    repeats: int = 1
    max_seconds: float = 0.0
    percentile: float = FLOOR_PERCENTILE


@dataclass(frozen=True)
class Material:
    """One source and the output whose near-copies are scored against it (`output` None: the source itself)."""

    source: Path
    output: Path | None
    source_wav: Path
    output_wav: Path

    @property
    def operating_point(self):
        """`identity` when the source is scored against itself, `output` when a restored output is."""
        return IDENTITY if self.output is None else OUTPUT_POINT


def _write_atomic(target, audio, rate):
    """Writes float WAV `target` through a partial file, so an interrupted run never leaves a half file to reuse."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".partial.wav")
    sf.write(str(partial), np.asarray(audio, dtype=np.float32), rate, subtype="FLOAT")
    partial.replace(target)
    return target


def prepared_source(source, cache_dir, max_seconds=0.0):
    """The file as a WAV (extracted into the cache when it is a video), cut to its first `max_seconds` when given."""
    wav = audio_io.extract_wav(source, cache_dir)
    if not max_seconds:
        return Path(wav)
    target = Path(cache_dir) / WORK_DIR / f"{audio_io.file_key(wav)}_first{max_seconds:g}s.wav"
    if not target.exists():
        audio, rate = sf.read(str(wav), dtype="float32", always_2d=True)
        _write_atomic(target, audio[: int(round(max_seconds * rate))], rate)
    return target


def _pair(entry):
    """`(source, output or None)` from a bare source or a `(source, output)` pair."""
    return (Path(entry[0]), entry[1]) if isinstance(entry, tuple) else (Path(entry), None)


def prepared(entry, cache_dir, max_seconds=0.0):
    """The `Material` of one entry (a source, or a `(source, output)` pair), both sides as WAVs cut alike."""
    source, output = _pair(entry)
    source_wav = prepared_source(source, cache_dir, max_seconds)
    output_wav = source_wav if output is None else prepared_source(output, cache_dir, max_seconds)
    return Material(source, None if output is None else Path(output), source_wav, output_wav)


def variant_wav(output_wav, name, cache_dir):
    """The near-copy `name` of `output_wav`, written once into `<cache_dir>/noise_floor/` and reused after."""
    target = Path(cache_dir) / WORK_DIR / f"{audio_io.file_key(output_wav)}_{name}.wav"
    if not target.exists():
        audio, rate = sf.read(str(output_wav), dtype="float32", always_2d=True)
        _write_atomic(target, TRANSFORMS[name](audio, rate, np.random.default_rng(SEED)), rate)
    return target


def flat_readings(card, path):
    """The flat `{metric.side.stat: value}` a sweep ranks one output on (`runner.aggregates_for_tuning`)."""
    return runner.aggregates_for_tuning({"variants": {"output": runner.card_to_dict(card, path)}}, "output")


def repeat_options(options, repeat):
    """The scorer options of repeat `repeat` (0-based): every later one scores through its own cache directory."""
    if not repeat:
        return options
    return {**options, "cache_dir": Path(options["cache_dir"]) / WORK_DIR / f"repeat{repeat + 1}"}


def _outputs(material, transforms, cache_dir):
    """`[(name, output WAV)]`: the output itself as the reference, then its near-copies."""
    return [(REFERENCE, material.output_wav)] + [(name, variant_wav(material.output_wav, name, cache_dir)) for name in transforms]


def score_instances(material, plan, scorer, options):
    """`[(name, flat readings, family status)]` for one pair, the reference (source, output) first.

    Then (source, near-copy of the output) per transform, the source side untouched; the
    whole set `plan.repeats` times, each repeat after the first in its own scorer cache.
    """
    outputs = _outputs(material, plan.transforms, options["cache_dir"])
    instances = []
    for repeat in range(plan.repeats):
        scoring = repeat_options(options, repeat)
        for name, output in outputs:
            card, _view = scorer(material.source_wav, output, **scoring)
            instances.append((name, flat_readings(card, output), dict(card.families)))
    return instances


def _finite(value):
    return isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool) and bool(np.isfinite(value))


def _add(out, key, name, value, base):
    if _finite(value) and _finite(base):
        out.setdefault(key, []).append((name, abs(float(value) - float(base))))


def deviations(instances):
    """`{reading: [(transform, |x - reference|)]}` over every scoring after the reference (the first one)."""
    reference = instances[0][1]
    out = {}
    for name, flat, _families in instances[1:]:
        for key, value in flat.items():
            _add(out, key, name, value, reference.get(key))
    return out


def _merge(into, more):
    for key, items in more.items():
        into.setdefault(key, []).extend(items)


def _summary(items, percentile):
    values = np.asarray([deviation for _name, deviation in items], dtype=np.float64)
    by_transform = {}
    for name, deviation in items:
        by_transform[name] = max(by_transform.get(name, 0.0), deviation)
    return {
        "floor": float(np.percentile(values, percentile)),
        "max": float(values.max()),
        "n": int(len(values)),
        "by_transform": by_transform,
    }


def summarise(per_reading, percentile=FLOOR_PERCENTILE):
    """`{reading: {floor, max, n, by_transform}}`, readings sorted by name."""
    return {key: _summary(items, percentile) for key, items in sorted(per_reading.items())}


def _optional(path):
    return None if path is None else str(path)


def _described(material, instances):
    return {
        "path": str(material.source),
        "output": _optional(material.output),
        "operating_point": material.operating_point,
        "wav": str(material.source_wav),
        "output_wav": str(material.output_wav),
        "scorings": len(instances),
        "families": instances[0][2],
    }


def _report(plan, options, described, per_reading):
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "operating_points": sorted({entry["operating_point"] for entry in described}),
        "percentile": plan.percentile,
        "transforms": [REFERENCE, *plan.transforms],
        "repeats": plan.repeats,
        "repeat_caches": [str(repeat_options(options, repeat)["cache_dir"]) for repeat in range(plan.repeats)],
        "max_seconds": plan.max_seconds,
        "score_options": {key: value for key, value in options.items() if key != "registry"},
        "sources": described,
        "readings": summarise(per_reading, plan.percentile),
    }


def measure(sources, plan=None, *, cache_dir, scorer=None, **score_options):
    """The noise-floor report for `sources` (bare sources or `(source, output)` pairs) under `plan`.

    `score_options` (windows, hop, families, registry, language) reach the scorer.
    """
    plan, scorer = plan or Plan(), scorer or runner.score_pair
    options = {**score_options, "cache_dir": Path(cache_dir)}
    per_reading, described = {}, []
    for entry in sources:
        material = prepared(entry, cache_dir, plan.max_seconds)
        instances = score_instances(material, plan, scorer, options)
        _merge(per_reading, deviations(instances))
        described.append(_described(material, instances))
    return _report(plan, options, described, per_reading)


def render_summary(report, top=SUMMARY_TOP):
    """A few lines for the console: how much was scored, at which operating point, and the widest floors."""
    readings = report["readings"]
    points = ", ".join(report["operating_points"])
    lines = [f"{len(readings)} readings from {len(report['sources'])} pair(s) at {points}, transforms {', '.join(report['transforms'])}"]
    widest = sorted(readings.items(), key=lambda item: -item[1]["floor"])[:top]
    lines += [f"  {key}: floor {entry['floor']:.4g} (max {entry['max']:.4g}, n {entry['n']})" for key, entry in widest]
    return "\n".join(lines)


# --------------------------------------------------------------------------- command line


def _csv(value):
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _known_names(value, known, what):
    names = _csv(value)
    unknown = [name for name in names if name not in known]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown {what}: {', '.join(unknown)} (known: {', '.join(known)})")
    return names


def transforms_arg(value):
    """An argparse type: comma-separated near-copy names (the reference is always scored; an empty list scores only it)."""
    return _known_names(value, tuple(TRANSFORMS), "transform")


def families_arg(value):
    """An argparse type: `all` or comma-separated harness families."""
    names = runner.ALL_FAMILIES if str(value).strip() == "all" else _known_names(value, runner.ALL_FAMILIES, "family")
    if not names:
        raise argparse.ArgumentTypeError("name at least one family")
    return names


def _halves(parts, cut):
    return "=".join(parts[:cut]), "=".join(parts[cut:])


def _both_files(halves):
    return all(Path(half).is_file() for half in halves)


def _split_pair(text):
    """`(source, output)` split at the first '=' that leaves an existing file on both sides, or None."""
    parts = text.split("=")
    return next((halves for halves in (_halves(parts, cut) for cut in range(1, len(parts))) if _both_files(halves)), None)


def pair_arg(value):
    """An argparse type: `SOURCE` (the identity point) or `SOURCE=OUTPUT` (the output's near-copies against the source)."""
    text = str(value)
    if "=" not in text or Path(text).is_file():
        return existing_path_arg(text), None
    split = _split_pair(text)
    if split is None:
        raise argparse.ArgumentTypeError(f"neither an existing file nor SOURCE=OUTPUT of two existing files: {text}")
    return existing_path_arg(split[0]), existing_path_arg(split[1])


def _number(value, kind):
    try:
        number = kind(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from error
    if not np.isfinite(number):
        raise argparse.ArgumentTypeError(f"not a finite number: {value!r}")
    return number


def _at_least(value, kind, minimum, inclusive):
    number = _number(value, kind)
    if number < minimum or (number == minimum and not inclusive):
        raise argparse.ArgumentTypeError(f"must be {'>=' if inclusive else '>'} {minimum}: {value!r}")
    return number


def positive_float_arg(value):
    """An argparse type: a number > 0 (window and hop lengths)."""
    return _at_least(value, float, 0.0, False)


def seconds_arg(value):
    """An argparse type: a number >= 0 (0 = the whole source)."""
    return _at_least(value, float, 0.0, True)


def repeats_arg(value):
    """An argparse type: an integer >= 1."""
    return _at_least(value, int, 1, True)


def percentile_arg(value):
    """An argparse type: a percentile in (0, 100]."""
    number = _at_least(value, float, 0.0, False)
    if number > 100.0:
        raise argparse.ArgumentTypeError(f"a percentile is at most 100: {value!r}")
    return number


def parse_args(argv=None):
    """The command line, every value checked by its argparse type (paths are confined later, where they are used)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "sources", nargs="+", type=pair_arg, help="SOURCE=OUTPUT pairs (WAVs or videos), or a bare SOURCE for the identity point"
    )
    parser.add_argument("--out", type=path_arg, required=True, help="the noise-floor JSON to write")
    parser.add_argument("--transforms", type=transforms_arg, default=DEFAULT_TRANSFORMS)
    parser.add_argument("--repeats", type=repeats_arg, default=1)
    parser.add_argument("--families", type=families_arg, default=("dsp",))
    parser.add_argument("--windows", type=positive_float_arg, default=15.0)
    parser.add_argument("--hop", type=positive_float_arg, default=7.5)
    parser.add_argument("--max-seconds", type=seconds_arg, default=0.0)
    parser.add_argument("--percentile", type=percentile_arg, default=FLOOR_PERCENTILE)
    parser.add_argument("--cache-dir", type=path_arg, default=DEFAULT_CACHE)
    parser.add_argument("--language", type=language_arg, default="ro")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args(argv)


def _confined_pair(pair):
    source, output = pair
    return confined_path(source, "source", must_exist=True), None if output is None else confined_path(output, "output", must_exist=True)


def _confined(args):
    """`args` with every path checked where it is used (scripts/cli_paths.py): inside an allowed root, inputs existing."""
    args.sources = [_confined_pair(pair) for pair in args.sources]
    args.out = confined_path(args.out, "--out")
    args.cache_dir = confined_path(args.cache_dir, "--cache-dir")
    return args


def _utf8_console():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main(argv=None):
    """Scores every pair's output and its near-copies against the source and writes the noise-floor JSON to --out."""
    _utf8_console()
    args = _confined(parse_args(argv))
    report = measure(
        args.sources,
        Plan(args.transforms, args.repeats, args.max_seconds, args.percentile),
        cache_dir=args.cache_dir,
        windows=args.windows,
        hop=args.hop,
        families=args.families,
        registry=runner.ModelRegistry(device=args.device),
        language=args.language,
    )
    runner.write_result(report, args.out)
    print(render_summary(report))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
