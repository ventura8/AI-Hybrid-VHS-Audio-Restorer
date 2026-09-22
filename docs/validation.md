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
whether each ranks the same way. On this release's build every one does:

| Comparison | Real tape | Fixtures |
| :--- | :--- | :--- |
| Factor 3.0 vs 1.8, speech-led | +3.35 dB for +0.07 | +2.30 dB for +0.31 |
| Probe (now 4 s) against 0.75 s | +3.39 dB for +0.12 | +3.02 dB for +0.13 |
| UVR-DeNoise against DeepFilterNet | 0.22 against 0.66 | 0.71 against 1.05 |
| Tonal gate on tonal material | 0.49 to 0.33 | 1.26 to 0.89 |

The first two rows read noise removed for programme deviation, both in dB; the
last two read programme deviation in dB, the winner first.

The factor row is judged on the speech-led classes -- speech, and speech over
a music bed, thirty of the forty fixtures and what the corpus is made of --
and its real-tape verdict was re-measured with this release's chain: +3.35 dB
of removal for +0.07 of deviation on 50 mixed clips, +2.52 for +0.23 on the
most tonal third, +2.77 for +0.00 on the most tonal tenth. The two music-only
classes read the factor the other way, -1.34 dB of removal for +2.36 of
deviation over their ten fixtures, which no real subset does: a fixture with
no quiet frames that are noise reads a stronger factor as programme lost. The
check prints that reading under the table as a known blind spot and excuses
it from the exit status, the way it does the two classes the scanner cannot
see. Judged over all forty the row had agreed by 5.6:1 at the 4 s probe and
read 4.2:1 on the gated hum canceller, against the 5:1 rule: the ungated
canceller had been cancelling the music fixtures' chord partials, which the
metric's broadband gain match read as deviation avoided, and against the
clean reference the gated chain is the better one on both classes.

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

The matrix exists for two reasons. A class its own detector could not see would
exercise nothing, which is what the exit gate above catches (the crackle class
is read on every voice); and a detector that fires on everything runs its
stage on everything -- pop removal and `decrackle` run on nearly every tape,
which is why both are held to being free there. Both are findings the set now
measures rather than the chain assumes.

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

### The mode's own stages

v1.3.0 gives `auto_pure_linear` stages of its own for the rows where `cathar`'s
cascade still led on paper, and each was adopted or held back on the same
terms as the repair stages: measured on the calibrated classes, then on real
tape, and free on undamaged material.

**Hum.** The mode's hum canceller (`modules/hum_cancel.py`) tracks each mains
harmonic that stands out as a line of its own in the quietest frames -- at the
frequency it actually sits at, since on the tapes measured the harmonics sit
one to eight hertz off the exact series -- and subtracts it per channel ahead
of the noise probe. Two gates decide it. `scripts/measure_hum.py --mains auto`
reads the harmonic excess at whichever of 50 and 60 Hz the recording's
harmonics support, on the 48 corpus tapes that carry hum; run alone on their
source audio the stage removes a median 3.01 dB of excess (upper quartile
5.98) at 0.155 dB of low-band movement, where `cathar`'s `dehum` at the right
frequency removes 2.07 at 0.25. In the chain, through the subtraction and
the neural stage, hum removed goes from a median 0.19 dB to 2.49, better on
34 of 48 tapes and worse by more than a decibel on none, with the low band
moving 1.41 to 1.48 dB and the broadband trade 12.91/0.33 to 12.90/0.35 --
inside the free band. Four tapes whose lines wander more than five hertz
between frames are not helped; they are recorded, not forced. On the
calibrated set the realism check now runs the shipped configuration against
`no_hum_cancel` over every class, held to the repair stages' free band.

Two readings were added to the hum instrument after the stage was adopted.
The excess reading is blind to a chain that lowers the floor around a line it
left behind -- the line then stands out more although it is no louder -- so
`hum_line_drop_db` reads the summed power at the harmonics themselves on
gain-matched audio, and the instrument now refuses the saturated or
constant-level sources the trade metric refuses (15 of the 48), on which the
gain-matched low band moved by tens of dB identically for both modes. On the
33 readable tapes the final chain removes a median 1.19 dB of excess against
`cathar`'s -0.74 and takes the lines themselves down 3.04 dB against 0.75,
ahead of `cathar` on 24 and 26 of the 33. Two switches for the notches that
precede the canceller were measured there, before the gates, and left at
their defaults: the mode's own third-to-fifth harmonic notches earn their
place (without them hum removal fell 2.43 to 1.71 dB, worse by more than a
decibel on 8 tapes),
and leaving the two pre-conditioned harmonics out of the canceller's plan
changes nothing.

The check also caught the canceller firing on the music-only class -- a chord's
partials sit near multiples of 50 Hz, a G major at 98, 147 and 196 Hz reading as
the second to fourth harmonics of 49 Hz -- and both instruments above are blind
to programme removed at mains multiples by construction. A series whose lines
place the fundamental outside the canceller's half-hertz window is now refused
as a chord, and a second gate follows the physics of a mains line: it sits at an
exact multiple of the fundamental, wobbling with the transport by a fraction of
a hertz, so a gated line further off its multiple than half the band the
canceller tracks it in (0.75 Hz at the fundamental, a quarter more per harmonic,
to 2.5 Hz) is a partial and is left alone -- the fixtures' chord partials at 98,
147 and 196 Hz sit 1.1 to 2.0 Hz off. A tighter tolerance was measured and lost
the strongest real hum tapes: their lines wobble past it with the transport, and
a 40-harmonic series was refused whole. A third gate, a line's level in the loud
frames against the quiet ones, was tried and dropped: programme energy shares a
line's bins when the programme is loud, and on the strongest real hum tapes
every harmonic read 15 to 65 dB louder there and the whole series was refused. A
third gate stands: a series with nothing at the fundamental is not hum -- the
`ep` class put the canceller on a synthesised voice whose pitch sits at twice a
mains-like fundamental, every harmonic on the series and steady, and the lines
it took were in the clean reference -- so it is refused unless the shared
scanner reported the hum and the pre-conditioning notched the fundamental ahead
of the stage. With the three gates the canceller stands aside on every
music-only and `ep` fixture, and both classes read the same with it as without
(13.67/1.69 and 9.71/0.58). On two real tonal tapes the low band moved 0.91 and
1.50 dB before the refusal and 0.11 and 0.01 after: what read as hum removal
there was programme. On the 33 readable hum tapes the chain then removes a
median 1.19 dB of excess and takes the lines themselves down 3.04 dB, against
`cathar`'s -0.74 and 0.75, ahead of it on 24 and 26 of the 33, with the low band
moved 0.48 dB against `cathar`'s 0.55 -- less hum removed than the 2.43 dB the
ungated canceller read, and less programme taken with it.

**Plosives.** The plosive tamer (`modules/plosive_tamer.py`) finds a blast as
a run of hops where the low band stands 12 dB over its own running level and
leads the mid band's rise by 6 dB, peaks within three hops, lasts 10-120 ms
and stands alone, and takes each down to the level the band held just before
it -- a downward expander on the low band, bit-identical elsewhere. The
calibrated `plosive` class injects its bursts into the reference (they were
recorded, not added by the tape) and its target differs from that reference
everywhere, so `scripts/score_defect_repair.py --reference target` reads the
damage from the low-band difference between the two files and measures the
error against the target inside those spans. There the stage recovers 0.89
dB at 0.03 dB of collateral, against `cathar deplosive`'s 1.55 at 0.34; on the
undamaged classes it costs 0.00-0.01 dB where `deplosive` costs 0.5-0.9 and
reads -6.7 dB on music-led programme. On 50 real captures it is free:
10.02/0.23 to 10.02/0.23, 37 captures untouched to the hundredth.

**Held back.** A per-bin MMSE log-spectral suppressor in the subtraction slot
(`modules/spectral_suppress.py`) is the branch's second DeepFilterNet: on the
calibrated classes it lands at 5.9-6.2 dB of log-spectral distance where the
subtraction lands at 10-12.9, and on 50 real captures it removes 5.83 dB at
0.19 against 10.02 at 0.22 (3.06 against 7.96 on the tonal 45). It keeps the
low-level programme the quiet frames hold, and the trade metric reads that
as noise left behind. The Mel-Roformer denoiser measured 9.39/0.22 against
UVR-DeNoise's 10.02/0.23, and Resemble-Enhance's denoiser 10.78/0.58 -- more
removal, the programme moved two and a half times as far, and 12 captures won
outright against `cathar` where UVR-DeNoise wins 28. A canceller for
persistent lines
(`modules/tone_cancel.py`) at its first setting read sustained notes and
missed mains lines as persistent lines in the speech range and cost 0.06 dB
of deviation on 50 captures; restricted to lines above 4 kHz it finds lines
on 19 of the 50 and moves the medians not at all, 10.02/0.23 to 10.02/0.23,
while one capture loses 10.45 dB of noise removal to it. A line that holds
still is already in the noise profile, and the subtraction removes it
outright where the tracker's smoothed envelope leaves a residual; a line
that wanders defeats both. It stays off.

### A perceptual cross-check

The trade metric is blind to what it does not measure, and every ranking on
this branch rests on it. `scripts/score_perceptual.py` reads DNSMOS P.835
(Microsoft's non-intrusive estimator of P.835 listening scores: speech
quality SIG, background BAK, overall OVRL, plus a P.808 overall MOS; fetched
by `scripts/download_dnsmos.py`, CC BY 4.0) on the source and on each mode's
restoration of every clip. It is speech-trained and reads at 16 kHz, so on a
corpus that carries music and archive material it is a cross-check and not a
gate: a change the trade metric and the defect gates approve and that DNSMOS
reads clearly worse is a change to listen to before it ships. On the 174
restored clips of the v1.2.1 corpus run it reads:

| Configuration | SIG | BAK | OVRL | P808 | OVRL vs source (paired median) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| source | 1.67 | 1.37 | 1.32 | 2.45 | |
| `cathar` | 2.16 | 2.22 | 1.62 | 2.45 | +0.07 |
| `auto_pure_linear` | 2.38 | 2.95 | 1.82 | 2.50 | +0.30 |

`auto_pure_linear` reads better than `cathar` on 111 of 174 clips for speech
quality, 149 for background and 132 for overall, and leaves fewer clips
reading worse than their own source (31 against 65 on OVRL). The scale is
compressed -- the source reads 1.3 on a 1-5 scale, where a clean studio
recording reads above 4 -- which is what a tape corpus looks like to a model
trained on suppressor outputs, and the reason the figure is read paired,
clip by clip, rather than as an absolute.

## Output validation harness

The trade metric reads noise removed and programme deviation in 300-3400 Hz,
and nothing else. This week two failures a listener heard at once scored well
on it: the multiband de-esser bug (a 4 kHz brick wall, speech "under water")
and a single 4 s noise probe on dialogue (the print learned sibilance, 8-12 kHz
down 14 dB). `scripts/validate_restoration.py` scores an output against its
own source the way a listener would, in four families, and vetoes with hard
gates:

- `dsp` (native rate, no model): presence and air band on the loud frames
  (muffling), a log-kurtosis ratio in the quiet frames (musical noise), click
  density (an impulse must stand 12x over its 50 ms floor and clear -60 dBFS,
  so dither-scale residue in a pause a denoiser gated to -85 dBFS is not a
  click), holes in the programme, the 15.6 kHz line, mains-harmonic excess,
  LUFS / loudness range, the trade metric itself, and two readings on the
  pauses (the source's quiet 20 ms frames): pumping, the spread of the floor's
  level (a denoiser's mask opening and closing between words; a source pause
  moves 4 dB, every denoiser measured 11-23 dB), and tilt, the low band's
  residual minus the high band's (rumble kept and air taken reads positive,
  hiss left reads negative, a uniform reduction reads 0). Both sides are
  read with their DC offset removed: three Internet Archive music clips sat
  at +0.49 with the programme 23-27 dB below, and against such a source the
  app's 2 Hz blocker read as destruction of the music. A window with more
  than 90 % of its power below 80 Hz (the same captures: 5 Hz harmonics under
  the offset) is routed as silence, not as the held tones the persistence
  reading would take it for.
- `stems`: the app's own BS-RoFormer splits source and output; on the
  non-vocal stem SI-SDR, log-spectral distance, the worst octave (from
  125 Hz: both engines run an 80 Hz rumble high-pass, so the 63-125 Hz octave
  of a bass-heavy source always reads as lost) and the envelope correlation
  say whether music and ambience survived. SI-SDR is shown, never ranked: a
  filter's phase shift turns it negative while the ear hears nothing. Read only
  where the source's stem carries a background (above -45 dBFS and within
  20 dB of the mix); an interview without music skips it.
- `speech` (16 kHz): Whisper large-v3-turbo transcribes source and output
  with the language forced (`--language ro`) and the character error rate
  between the two reads words changed, with the confidence delta beside it;
  WavLM-base-plus-SV cosine reads timbre against the tape's own intra-speaker
  floor; UTMOS reads naturalness.
- `mos`: SIGMOS (ITU-T P.804: coloration, discontinuity, noise, reverb,
  loudness, signal, overall) at 48 kHz, DNSMOS P.835 / P.808, and Audiobox
  Aesthetics (production quality, production complexity, enjoyment,
  usefulness) on music as well as speech. Both sides are brought to -23 LUFS
  with one gain first.

Every reading is a paired delta, output minus source, over 15 s windows every
7.5 s, aggregated as a median and a tail (p10 or p90, always the bad end).
Windows are routed by the scanner's band ratios and by tonal persistence
(`modules/tonal_persistence.py`, the share of prominent spectral peaks held
for eight 93 ms frames; the band ratios alone read every archive music clip as
dialogue): speech metrics run on speech and mixed windows, stems and Audiobox
on every window that is not silent. The same reading, as a whole-file median,
is what the scanner reports as `tonal_persistence` and what cathar's music
profile keys on. Learned predictors are guardrails, never objectives: in the URGENT
2024 challenge the systems that topped DNSMOS and NISQA ranked at the bottom
with listeners, and single-scalar MOS models cannot tell an over-suppressed
voice from a noisy one. Only SIGMOS's coloration and discontinuity axes make
that split, which is why they carry gates.

The gates live as data in `scripts/restoration_quality/gates.py` and a
`gates.json` from the calibration below overrides them. Weights are fetched by
`scripts/download_quality_models.py` into `models/<name>/` (SIGMOS, Whisper,
WavLM, UTMOS: MIT; Audiobox: CC-BY-4.0), pinned to upstream revisions with
the sha256 of every file recorded on first fetch and refused on mismatch.

```powershell
.\.venv\Scripts\python.exe scripts\download_quality_models.py
.\.venv\Scripts\python.exe scripts\validate_restoration.py source.mov `
    cathar=source_Cathar_Cleaned.mov apl=source_PureLinear_Cleaned.mov `
    --report out.json --markdown out.md --listen-dir listen
```

`--listen-dir` renders the windows each metric found worst, cut from the
source and from every output, with an `index.md`: the user is the judge, the
harness only points at where to listen. `--metrics dsp` needs no model.

### Calibrating the output-quality metrics

None of the models saw VHS degradation or Romanian, so before the harness
ranks anything `scripts/calibrate_quality_metrics.py` proves each metric
moves the right way. It applies the single-factor failures in
`scripts/quality_degradations.py` at three levels to realistic-v2 fixtures
(hiss from the fixture's own recorded noise, mains hum, an 8th-order lowpass
for the underwater voice, over-subtraction for musical noise, Griffin-Lim
resynthesis for the robotic voice, muted and spliced words, a music bed
attenuated, crackle, dropouts, the line whistle, gain and compression) and
asks, per metric and failure: does the delta point the expected way at the
severest level, is it monotonic across the levels, and does the mildest level
clear three times the benign floor (identity, 16-bit requantisation, a
resample round trip, 5-60 ms shifts)? A second set is the user's own tapes
with this week's outputs judged by ear: the de-esser bug must be flagged
muffled, the single 4 s probe must lose to the stitched one on coloration and
highs while reading at least as quiet, and APL must not read as altered.
Thresholds are then derived per gate: the midpoint between the worst
known-good and the best known-bad reading when that gap clears the floor,
else three floors past the benign centre. The report and `gates.json` land
in `experiments/quality_calibration/`.

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_quality_metrics.py `
    --fixtures artifacts\realistic-v2 --languages en `
    --known-ordering experiments\quality_calibration\known_ordering.json
```

Opt-in model tests: `AI_RESTORE_QUALITY_MODELS=1 pytest tests/quality`.

What the first calibration (2026-09-20, English fixtures, Tele7abc / SOTI /
Vaccin) found:

- Every DSP guardrail passes direction, monotonicity and effect on its
  failure: residual noise on hiss, hum excess, both high bands on the lowpass,
  the kurtosis ratio on over-subtraction, clicks, holes on muted words and on
  dropouts, the 15.6 kHz line, LUFS on gain. The benign floor of the DSP
  readings is essentially zero.
- The learned predictors are far less sensitive on 15 s Piper fixtures than
  on real tape: SIGMOS coloration reads the 4 kHz lowpass (-0.64) but not
  6 or 8 kHz, and reads Griffin-Lim resynthesis as *better*; UTMOS moves by
  hundredths; Whisper's CER barely moves for 0.3 s mutes (it infers the
  words) and not at all for a 0.5-2 s splice. Audiobox reads the lowpass
  (-1.0 at 4 kHz) and a stripped music bed (PC -0.9 / -2.6 / -3.7 at -6 /
  -12 / -24 dB) clearly. SI-SDR on the stem is blind to a uniform
  attenuation by construction; the loudness range of a 15 s fixture is too
  small for the compressor to move. Those pairs are reported as blind.
- On the real tapes the same predictors separate what the ear separated:
  SIGMOS coloration +0.01 on the single 4 s probe against +0.48 for the
  stitched print and APL, discontinuity -0.60 against -0.16..+0.37, UTMOS
  -0.59 against -0.14..-0.29, Audiobox PQ -1.29 against about 0, CER median
  0.10 against 0.02-0.04; speaker cosine 0.97-0.99 everywhere. With the
  derived gates every ordering rule holds on all three tapes: the de-esser
  bug is flagged muffled and ranks in the bottom two, the 4 s probe is
  duller than the stitched one, APL is not altered, and the outputs the
  user accepted rank on top.
- Derived gates: the high bands come from the good/bad gap (-5.3 dB at
  4-8 kHz, -7.4 dB at 8-16 kHz); most other thresholds are bounded by the
  worst accepted output plus three floors, because the known-bad outputs
  fail on other readings (every denoiser raises the kurtosis ratio and
  Audiobox's production complexity, so the hand-set 0.3 and -0.5 would have
  vetoed the accepted outputs). Whisper's worst-decile CER is 0.3-0.7 on
  every restoration of hissy tape, so that gate is soft.

### Fine-tuning on real tapes

`scripts/tune_restoration.py` runs both engines over a grid of settings on
excerpts of the user's tapes and ranks the variants with the harness. Excerpts
are 125 s: cathar's stitched noise probe engages only when the material is at
least twenty times `cathar_noiseprint_duration_s` (6 s, so 120 s), and a cut
of 125 s probes at 125.0 s. Each variant runs in a fresh interpreter with its
own `config.yaml` in the launch directory, which is how the app resolves
configuration; the driver validates every override against the app's own
typed settings and re-reads the resolved configuration from a child before
spending a run. Outputs are scored against their source excerpt, hard gates
veto a variant that fails on more excerpts than the grid allows and than its
own engine's baseline does (the gates were calibrated on three tapes; a long
tape's lead-in fails them for every variant, defaults included; per tape the
tolerance is zero), every variant is ranked by the mean rank over the grid's
`ranking` metrics and the recommendation takes the best survivor; the trade
metric is shown, never ranked. The
scoreboard names the best variant per engine and the best engine per tape,
and prints the confirmation commands for the full tapes: an excerpt's noise
probe is not the full tape's. An engine in the grid may carry `env`, extra
environment for the app's child processes; `tata_v1.yaml` runs the cathar
chain a second time as `cathar075` with `AI_RESTORE_CATHAR_BIN` pointing at
the 0.7.5 build, so a cathar upgrade is scored beside the validated binary
before anything is replaced.

```powershell
.\.venv\Scripts\python.exe scripts\tune_restoration.py all --name tata_v1 `
    --tapes-dir "D:\Tata\New folder" --grid scripts\tune_grids\tata_v1.yaml
```

Corpus clips are taken whole instead of cut, with
`scripts/tune_grids/ia_v1.yaml`, whose variants leave the stitched
probe alone (at 15 s cathar keeps its single 0.75 s window) and tune the
subtraction factor and floor, Wiener, the de-esser, enhance, the coherent
mask, repair and deplosive, and for APL the factor, probe, blend, model, air
shelf and expander:

```powershell
.\.venv\Scripts\python.exe scripts\tune_restoration.py all --name ia_v1 `
    --tapes-dir experiments\ia_corpus_1000 --whole --limit 40 --language en `
    --catalog experiments\ia_corpus_1000\catalog_1000.json `
    --grid scripts\tune_grids\ia_v1.yaml
```

Never run it beside another restoration. A cathar default that changes as a
result alters the five reference clips' outputs (`experiments/cathar_ab_head.json`)
unless it is gated by material length as the noise probe is, and the
`auto_cathar_*` routing was calibrated with cathar on its shipped settings;
both are the user's call.

## CI Parity

CI workflow mirrors local validation ordering and tooling to avoid environment
drift.
