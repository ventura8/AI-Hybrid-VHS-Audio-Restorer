# Validation

## Canonical Command

Run full local validation with:

```bash
./run_pipeline_locally.sh
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_pipeline_locally.ps1
```

## What Validation Covers

- PowerShell script linting.
- Python linting for modules and tests (Black, isort, Ruff, and Flake8).
- TOML formatting check via `taplo`.
- Static analysis gate (Pylint).
- Security checks via Bandit and pip-audit.
- Markdown formatting verified via `mdformat --check`.
- Markdown linting via `pymarkdown --config .pymarkdown.json scan`.
- Test execution with total coverage threshold enforcement.
- Radon complexity, maintainability, raw, and Halstead checks for modules and
  the test suite.
- Strict per-file coverage enforcement from `coverage.json`.
- Coverage badge regeneration.

## Expected Outcome

- Lint passes without suppressions.
- Tests pass.
- Total coverage remains >= 90%.
- Every measured source file remains >= 90% coverage.
- assets/coverage.svg is updated.

## Fast Spot Checks

```powershell
.\.venv\Scripts\python.exe -m poetry run ruff check modules tests `
    restore_audio_hybrid.py scripts/apply_patches.py
.\.venv\Scripts\python.exe -m poetry run black --check modules tests `
    restore_audio_hybrid.py scripts/apply_patches.py
.\.venv\Scripts\python.exe -m poetry run isort --check-only --diff `
    modules tests `
    restore_audio_hybrid.py scripts/apply_patches.py
.\.venv\Scripts\python.exe -m poetry run flake8 modules tests `
    restore_audio_hybrid.py scripts/apply_patches.py
$tomlFiles = @(git ls-files "*.toml")
if ($tomlFiles.Count -gt 0) {
    .\.venv\Scripts\python.exe -m poetry run taplo fmt --check $tomlFiles
}
.\.venv\Scripts\python.exe -m poetry run bandit -ll -ii -r modules `
    restore_audio_hybrid.py scripts/apply_patches.py
.\.venv\Scripts\python.exe -m poetry run pip-audit
.\.venv\Scripts\python.exe -m poetry run radon cc `
    modules `
    tests\conftest.py tests\unit tests\integration restore_audio_hybrid.py `
    scripts/apply_patches.py -s
.\.venv\Scripts\python.exe -m poetry run pytest -o addopts= `
    --cov=restore_audio_hybrid --cov=modules --cov-branch `
    --cov-report=xml --cov-report=json --cov-report=term `
    --cov-fail-under=90 tests/
.\.venv\Scripts\python.exe -m poetry run python `
    tests/tooling/quality_gate.py coverage.json `
    --threshold 90.0
```

## Hardware Validation Matrix

Hardware validation is opt-in and supplements the canonical quality gate.
The host audit and core dry run are required for applicable audio, DSP, sync,
and hardware changes. Physical fixture execution remains opt-in.

The Piper catalog covers 50 language-native voices with checksum-pinned downloads.
Piper is installed in `tools/piper-tts/.venv`, separate from the application's
CUDA/TensorRT environment, so its CPU ONNX Runtime cannot change GPU provider
selection.

```powershell
.\.venv\Scripts\python.exe -m poetry run python scripts/audit_hardware.py
.\.venv\Scripts\python.exe -m poetry run python `
    scripts/generate_audio_matrix.py core --language all
$env:AI_RESTORE_HARDWARE_TESTS = "1"
.\.venv\Scripts\python.exe -m poetry run pytest tests/hardware -v
```

Hardware this machine does not have is validated on a machine that does,
over SSH:

```bash
scripts/remote_validate.sh <user>@<host> --stage execute
```

See [remote_validation.md](remote_validation.md) for the stages, the host
requirements, and what the reports contain.

Use `scripts/run_hardware_validation.py --execute` only on a prepared machine;
it drives selected modes through temporary video fixtures and writes timing and
peak-VRAM data beneath `artifacts/`.

## Fixtures That Predict Real Tape

Every synthetic fixture set used to tune `auto_pure_linear` before v1.2.1 agreed
with real tape about *which* stage mattered and disagreed about what it was
worth: the subtraction factor, the margin gate, the tonal cleanup, the mains
dehum and the DeepFilterNet stage each ranked one way on fixtures and the other
way on the corpus. A fixture set earns its keep only by predicting real tape, so
there is now a calibrated set and a check that says whether it does.

```powershell
.\.venv\Scripts\python.exe -m poetry run python `
    scripts/expand_ia_corpus.py
.\.venv\Scripts\python.exe -m poetry run python `
    scripts/make_realistic_fixtures_v2.py `
    --catalog experiments/ia_corpus_1000/catalog_all.json --noise-windows 48
.\.venv\Scripts\python.exe -m poetry run python `
    scripts/validate_fixture_realism.py
```

### The corpus behind it

`expand_ia_corpus.py` widened the Internet Archive corpus the set is calibrated
against from 192 clips of 192 tapes to 1,982 clips of 847 tapes: nineteen
searches with genre labels that mean what they say (children's television,
sport, documentary, comedy, music, adverts, home video, and tapes in other
languages), PAL and NTSC, paged, up to three 15 s slices of each tape at
different offsets -- only the offsets a tape reaches, since the extractor's
fallback to the opening had labelled 167 openings as later slices before they
were found by hash and dropped. The tape-noise bank samples 48 captures from
the whole merged catalog,
and a window qualifies as noise on steadiness, crest factor and level together
-- sampled by level alone, eight windows in twelve were fades, dropouts or
speech, and one was a test tone -- with the capture's line whine and mains
series notched out, since each is a class of its own.

### The set

`make_realistic_fixtures_v2.py` writes 460 paired fixtures under
`artifacts/realistic-v2`, each 15 s like the corpus clips: five Piper voices,
speech-led and music-led programme at five margins with and without mains hum,
and 32 defect classes on the speech-led base -- every fault in
`docs/vhs_audio_defects_research.md`, the ten the preservation literature
added to it, and a crowd under commentary.
Each fixture carries three files: `_vhs` (the tape), `_clean` (everything the
tape carried, the room and the voice across it included) and `_target` (the
programme wanted back), and each property was set against a measurement of the
corpus, recorded beside the knob in the generator. The ones that decided the
inversions:

| Property | Real tape | Fixtures |
| :--- | ---: | ---: |
| Frame level above the floor, p20 | 2.3 dB | 2.2 dB |
| Frame level above the floor, median | 8.1 dB | 6.8 dB |
| Frame level above the floor, p90 | 14.5 dB | 11.4 dB |
| Noise removed, speech reading a 10-11 dB margin | 8.1-8.4 dB | 9.5-11.5 dB |
| Programme deviation, speech classes | 0.33 dB | 0.37 dB |
| Loud frames, 2400-4800 Hz against 300-600 | -11 dB | -11 dB |

The level profile took three changes: no inserted pauses, since a tape's
quietest fifth is quiet programme rather than silence; a slow gain control that
rides programme and reference alike; and tape hiss added after it, where a tape
adds it. The DeepFilterNet verdict took a room, because dry synthesised speech
is exactly what a speech enhancer was trained to keep. The voice itself carries
the tilt a tape's does -- relative to its 300-600 Hz band a real tape's loud
frames hold 600-1200 Hz at -1.5 dB and 2400-4800 at -11, where Piper's hold
-7 and -17, so a presence shelf and a low shelf put on what a microphone and a
broadcast chain put on -- and it breathes before its phrases, since real quiet
frames vary 11-14 dB per band frame to frame and a synthesiser's varied 5.

### Does it predict real tape

`validate_fixture_realism.py` reruns the four comparisons whose real-tape
verdict is known, on the metric those verdicts were reached with, and reports
whether each ranks the same way. On the v1.2.1 set every one does:

| Comparison | Real tape | Fixtures |
| :--- | :--- | :--- |
| Factor 3.0 against 1.8 | +1.27 dB for +0.09 | +1.29 dB for +0.24 |
| Probe 2.5 s against 0.75 s | +3.39 dB for +0.12 | +3.80 dB for +0.24 |
| UVR-DeNoise against DeepFilterNet | 0.22 against 0.66 | 0.58 against 0.92 |
| Tonal gate on tonal material | 0.49 to 0.33 | 1.24 to 0.92 |

The first two rows read noise removed for programme deviation, both in dB; the
last two read programme deviation in dB, the winner first.

Two things the check makes explicit. The full-reference scores against the
clean fixture, SI-SDR and log-spectral distance on the finished output, are
reported but cannot rank subtraction strength at these margins: fed a noise-free
fixture the finished chain comes back at 16.9 dB SI-SDR, because the polish,
expander and loudness stages colour the waveform, and at a 14 dB margin that
colour is a larger error than the noise was. They catch destruction --
DeepFilterNet on music-led programme lands at -9 dB -- and nothing finer.

The last gap was the metric reading the fixture's length. On an 8 s fixture
holding one 1.2 s pause, the quietest fifth the metric reads is that pause and
nothing else, and the chain took 15 dB out of it; on a 15 s tape the quietest
fifth runs on into quiet speech and the chain takes 9. Margins are compared as
the scanner reads them, since a fixture built at a nominal 15 dB reads 11 and
a real tape's margin is only ever a reading: at the clips' own length, with the
synthesiser's pauses trimmed to what a breath needs but one, the speech class
reading 10 dB gives up 9.5 dB and the class reading 11 gives up 11.5, against
8.1-8.4 on real tapes reading up to 13, and they deviate 0.24 against real
tape's 0.15-0.47. Each class is five voices under five draws of noise window
and room, and the five spread from 6.0 to 14.9 dB of removal at the one margin
and 7.1 to 14.0 at the other. That is the width a median of five draws
carries, and the reason the check judges rank agreement rather than a
magnitude.

### What the scanner sees

The check runs the scanner and the dropout detector over every defect class and
reports which detector each trips against what the class was built to trip.
Every class trips its own detector on at least three voices in five, with two
exceptions, both recorded in the check as known and excused from its exit
status: the enclosure resonance, a 16 dB ring at a Q of 5, which the scanner
does not read at all, because its width test walks off the ring at the first
dip between the voice's harmonics; and the worn tape's mains hum, which under
that class's 9 dB margin of noise reads on two voices in five where the same
hum at a 15 dB margin reads on seven captures in ten. Any other class going
blind to its own detector, any comparison ranking against real tape, or a
repair stage hurting an undamaged class fails the check. The matrix is also
the record of what the scanner, which is shared with `cathar` and therefore
left alone in this release, sees on material that does not have the fault:

| Detector | Fires on material without the fault |
| :--- | :--- |
| Clicks | every class, plain speech included, with corpus noise |
| CRT whistle | 40-100% of every class; noise alone clears its prominence |
| Drift | flutter (right), music-led programme, and noise with no whine |
| Mains hum | 20-80% of hum-free classes, the codec and the buzz most |
| DC bias | rumble and clipping |
| Rumble | mains hum, and music-led programme |

A crackle fixture the scanner cannot see exercises nothing; a detector that
fires on everything runs its stage on everything -- pop removal and `decrackle`
run on nearly every tape, which is why both are held to being free there. Both
are findings the set now measures rather than the chain assumes.

### Repair on undamaged material

The repair stages are run on and off over every class and held to the real-tape
verdict that gated repair is close to free: across 174 captures it moved removal
from 9.66 to 9.74 dB and deviation from 0.48 to 0.50. On the 28 classes carrying
no physical damage the two configurations land within 0.5 dB of removal and
0.1 dB of deviation of each other on every one. On the damaged classes the
finished-output metrics show the azimuth stage working (+4.2 dB SI-SDR, -3.8 dB
log-spectral distance) and the crackle, dropout, clipping and head-switching
stages changing almost nothing -- the defect-region metric of
`scripts/score_defect_repair.py` reads decrackle and inpaint as inert on
crackle drawn at eight pops a second with bright tails and on dropouts drawn at
5-50 ms, where the stepped fixtures they were adopted on read them at +10 dB.

Every knob was then tried. `decrackle` at sensitivity 5 through 10 and
`declick` at thresholds 8 down to 1 recover 0.3-2.3 dB inside the damaged
samples whatever the crackle -- dense and 20-30 dB down at a hundred a second,
or sparse and 6 dB down at three a second, with tails of four samples or
sixty-four -- and `declick` at 1 costs 7 dB of collateral on undamaged speech.
FFmpeg's `adeclick`, which the pre-conditioning already runs, takes minutes per
fixture at the settings that would do more. `inpaint` is a different case: on
the exact dropout spans it fills every hole, with programme-like audio 13.5 dB
under the level of what was lost, which the defect-region metric reads as +156
dB of spectral repair and +0.6 dB of sample repair -- the hole is no longer a
hole, and the waveform is not the one that was lost, which is what
autoregressive interpolation is. The impulse tools the chain had did not
remove pops the size a tape carries in the noise a tape carries them in, so
`modules/impulse_repair.py` now does: a short autoregressive model per block,
a pop where the prediction residual stands seven robust scales out, a
least-squares autoregressive refill of each isolated span. At the true pop
positions of the crackle class it repairs +5.5 dB (98% of pops found); on
undamaged speech it changes 40 dB under the programme; on the music-led
classes, once spans crowded by other outliers were left alone, 34 dB under on
music with speech and nothing on music alone; across 50 real captures noise
removal and deviation move by 0.00 dB. The finished-output metrics still do not
show it, for the reason they show none of the impulse stages: the chain's own
colour is a larger error inside a pop's few milliseconds than the pop was.

The set is also what the per-bin blend is fitted on, through
`scripts/build_blend_dataset.py --reference target` pointed at
`artifacts/realistic-v2`, using the shipping chain's own subtraction so the
features the model learns from are the features it is asked about. Three
retrains on it captured 63-77% of the fixture headroom and moved nothing on
real tape; the shipped weights stay.

## CI Parity

CI workflow mirrors local validation ordering and tooling to avoid environment
drift.
