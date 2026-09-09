# Head-to-Head Evaluation: Cathar vs. Auto Pure Linear

## Summary

Measured on the v1.3.0 build with `cathar-cli` 0.7.3 across 136 Internet Archive
VHS clips spanning European PAL (50 Hz mains, 15,625 Hz CRT line whistle) and
American NTSC (60 Hz mains, 15,734 Hz CRT line whistle). 174 were run; 38 are
excluded as degenerate sources, explained below. `cathar` is bit-identical to
the v1.2.1 build, and its column reproduces that release's figures.

**`auto_pure_linear` leads `cathar` on broadband noise without trading
programme fidelity for it, and it now removes the mains hum that neither mode
touched in v1.2.1.** The stages this release adds are gated on their own
defect and measured free where the defect is absent; the 1.35 dB of extra
removal over v1.2.1 comes mostly from a noise probe lengthened to 4 s on the
evidence of this corpus, at 0.31 dB of median deviation against 0.33.

| Metric (median, 136 clips) | `cathar` | `auto_pure_linear` |
| :--- | ---: | ---: |
| Noise removed (dB, higher better) | 6.02 | **10.10** |
| Programme deviation (dB, lower better) | 0.44 | **0.31** |

`auto_pure_linear` removes more noise on 114 of 136 clips and disturbs
the programme less on 84 of 136. On 70 clips it wins both at once, against 8
where `cathar` wins both.

| Mains hum (median, 33 readable hum tapes) | `cathar` | `auto_pure_linear` |
| :--- | ---: | ---: |
| Hum removed (dB of harmonic excess, higher better) | -0.74 | **1.19** |
| Hum lines taken down (dB, gain-matched) | 0.75 | **3.04** |
| Low band moved (dB, 50-400 Hz in loud frames) | 0.55 | **0.48** |

______________________________________________________________________

## How to read these numbers

**The metrics changed between v1.2.0 and v1.2.1, and the old ones should not be
compared against these.** The previous editions ranked the modes on attenuation
ratios and a `noise_reduction_db` figure. Those were withdrawn because they
cannot distinguish the two things a restoration does:

> An attenuation ratio rises whether a mode removed the defect or removed the
> programme along with it.

That is not hypothetical. Measured against clean references, the deterministic
dehum stage those metrics rewarded cuts 6.7-10.7 dB *below* the truth in the hum
harmonic bins, because speech shares the 50-400 Hz range with mains hum. The
metric scored that as an improvement.

The two headline figures are measured separately and never combined:

- **Noise removed** is the level drop in the frames the *source* says are quiet.
  Those frames are mostly noise, so a drop there is noise going away.
- **Programme deviation** is how far the restored spectrum moves in the frames
  the source says are loud, inside the 300-3400 Hz speech band. Those frames are
  dominated by content, so movement there is the restoration altering it.

Both are computed after gain-matching on the loud frames, so the final loudness
normalisation can neither flatter nor penalise either number. Which of the two
matters more is a judgement; collapsing them into one score is precisely the
mistake this replaces.

**Hum has a metric of its own** (`scripts/measure_hum.py --mains auto`),
because the broadband pair cannot see it: a mains series is a few narrow lines
that a quiet-frame level drop barely registers and the speech band excludes.
Hum removed is the drop in *harmonic excess* -- the energy at the fundamental
and its harmonics against their own spectral neighbourhood, at whichever of 50
and 60 Hz the recording's harmonics support -- and the low band moved is the
spectral distance in 50-400 Hz on loud frames with the harmonic bins masked
out, so a stage that takes the speech between the lines is seen and a stage
that takes only the lines is not. The excess reading is blind to a chain that
lowers the floor around a line it left behind, so the instrument also reads
the summed power at the harmonics themselves on gain-matched audio (the line
drop), and it refuses the same degenerate sources the trade metric refuses,
15 of the 48 tapes that carry hum.

**Medians, not means.** Every headline figure here is a median.

**38 sources are excluded, and the reason is the metric's own premise.** It
reads noise from the frames the source says are quiet and programme from the
frames it says are loud. On 38 of the 174 clips those are the same frames: the
20th and 70th percentile frame levels sit within 3 dB of each other, and on 20
of them the crest factor is under 1.3 dB -- a saturated or constant-level track,
not audio. On those the metric reports deviations of 20-50 dB *identically for
both modes* and a "noise removed" as low as -69 dB. That is the metric failing,
not a restoration, so the metric refuses a source with under 3 dB of
quiet-to-loud spread, and every figure below is over the 136 it can read.

______________________________________________________________________

## 1. Regional results (median)

### Europe, PAL (69 clips)

| Metric | `cathar` | `auto_pure_linear` |
| :--- | ---: | ---: |
| Noise removed (dB) | 7.76 | **12.18** |
| Programme deviation (dB) | 0.32 | **0.27** |

`auto_pure_linear` removes more noise on 61 of 69 clips and deviates less on
41; it wins both on 36 against 3.

### America, NTSC (67 clips)

| Metric | `cathar` | `auto_pure_linear` |
| :--- | ---: | ---: |
| Noise removed (dB) | 4.68 | **8.34** |
| Programme deviation (dB) | 0.57 | **0.45** |

NTSC was the mode's worst region in v1.2.0, with a median noise reduction of
-0.05 dB. It removes more noise on 53 of 67 clips there and deviates less on
43, winning both on 34 against 5. The 4 s probe spent part of the mode's NTSC
fidelity lead (0.33 dB at 2.5 s) for 1.4 dB of removal; it still leads
`cathar`'s 0.57 there.

______________________________________________________________________

## 2. Noise floor regressions

Both modes are normalised to the same loudness target at the final mux, so a
mode that raises the programme level without removing noise raises the measured
floor with it. This was `auto_pure_linear`'s dominant defect in v1.2.0.

| | Floor got worse | NTSC only |
| :--- | ---: | ---: |
| `cathar` | 18/136 (13%) | 12/67 |
| `auto_pure_linear` | **4/136 (3%)** | **3/67** |

For comparison, the v1.2.0 figures for `auto_pure_linear` were 82/174 (47%) and
40/77 (52%), over the unfiltered corpus.

The distribution shows the same thing without relying on a threshold:

| Noise removed (dB) | p10 | p25 | median | p75 | p90 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `cathar` | -1.15 | 2.45 | 6.02 | 9.62 | 20.08 |
| `auto_pure_linear` | **2.55** | **6.34** | **10.10** | **18.14** | **37.50** |

______________________________________________________________________

## 3. What changed in v1.3.0

The broadband trade was already won in v1.2.1; this release closes the rows of
the defect-coverage table where `cathar`'s cascade still led on paper, with
stages of the mode's own, each gated on its defect. Every candidate was
measured on the calibrated fixture set and then on real tape, and adopted or
held on the tape.

- **A hum canceller** (`modules/hum_cancel.py`), ahead of the noise probe. Each
  mains harmonic that stands out as a line of its own in the quietest frames is
  tracked by complex demodulation at the frequency it actually sits at -- on the
  tapes measured the harmonics sit one to eight hertz off the exact series --
  and subtracted per channel, with an envelope bandwidth that widens with the
  harmonic number, Wiener shrinkage against the line's own neighbourhood floor
  and a 6 dB cap over its running level. Run alone on the 48 hum tapes it
  removes a median 3.01 dB of harmonic excess at 0.155 dB of low-band movement,
  where `cathar dehum` at the right frequency removes 2.07 at 0.25; in the
  chain, hum removed goes from 0.19 dB to 2.49 on all 48 at the probe the stage
  was adopted at, better on 34 of 48 and worse by more than a decibel on none;
  section 4 has the final build's table on the readable tapes. Three gates keep
  programme out of the series: a series whose lines place the fundamental
  outside the half-hertz window is refused; a gated line further off an exact
  multiple than half the band the canceller tracks it in (0.75 Hz at the
  fundamental, a quarter more per harmonic) is a partial; and a series with
  nothing at the fundamental is a voice or a note pitched at a multiple of the
  mains frequency, refused unless the scanner reported the hum and the
  pre-conditioning notched the fundamental upstream. The calibrated music-only
  class caught the canceller taking partials at 98, 147 and 196 Hz for hum, and
  on two tonal tapes the low band moved 0.91 and 1.50 dB before the gates and
  0.11 and 0.01 after.
- **Event-gated plosive control** (`modules/plosive_tamer.py`). A blast under
  150 Hz that stands 12 dB over the low band's running level and leads the mid
  band's rise is taken down to the level the band held just before it; nothing
  else is touched. On the calibrated class it recovers 0.89 dB at 0.03 dB of
  collateral against `cathar deplosive`'s 1.55 at 0.34, and where `deplosive`
  costs 0.5-0.9 dB on undamaged classes this costs 0.00-0.01. On real tape it is
  free.
- **A 4 s noise probe**, up from 2.5. v1.2.1 chose 2.5 s on 25 clips where the
  curve looked flat above it; measured on the full corpus, with the hum
  canceller now ahead of the probe, it is not: 4 s removes a median 9.99 dB
  against 8.73 at the same 0.32 dB of deviation, wins both halves against
  `cathar` on the same 72 clips and loses both on 10 rather than 11. The cost is
  in the deviation tail -- the upper quartile moves 0.76 to 0.91 dB, 17 clips
  move up by more than 0.2 dB and 8 move down -- and on NTSC, whose median
  deviation goes 0.33 to 0.45 against `cathar`'s 0.57. 6 s removes 10.64 at 0.35
  and loses both on 13; 8 s read 12.64/0.27 on 50 clips and was not taken
  further. On a long tape the quietest 4 s is far likelier to be pure noise than
  it is inside a 15 s corpus clip, so the corpus reads this setting's risk high
  rather than low. An adaptive length -- extend the quietest window while its
  spectrum, level or spread stays close to the 2.5 s window's -- was calibrated
  against that per-clip cost and predicts none of it (correlations of 0.01 to
  0.06), so the length is fixed.
- **One chain runner** (`modules/apl_chain.py`) for the stages between
  pre-conditioning and the neural denoiser, each behind a switch, logging which
  of them changed the audio.
- **Candidates measured and held back**, with their numbers in the configuration
  comments so the next round does not repeat them:
  - A per-bin MMSE log-spectral suppressor in the subtraction slot
    (`modules/spectral_suppress.py`), built for the tonal third of the corpus.
    On 50 real captures it removes 5.83 dB at 0.19 of deviation against the
    subtraction's 10.02 at 0.22 (3.06/0.22 against 7.96/0.28 on the tonal 45). A
    per-bin estimator keeps the low-level programme the quiet frames hold, which
    the trade metric reads as noise left behind -- the DeepFilterNet lesson a
    second time.
  - The Mel-Roformer denoiser as the neural stage: 9.39/0.22 against
    UVR-DeNoise's 10.02/0.23.
  - Resemble-Enhance's denoiser as the neural stage (`--denoise_only`, a masking
    model; the package's enhancer is generative and was never a candidate):
    10.78/0.58 against UVR-DeNoise's 10.02/0.23 on 50 captures. More removal,
    the programme moved two and a half times as far, and 12 captures won
    outright against `cathar` where UVR-DeNoise wins 28..
  - A noise probe of its own on tonal material, and the neural stage left out
    there: on the most tonal 45 clips a 2.5 s probe reads 8.09/0.28 against
    9.05/0.30 and loses both halves to `cathar` on 6 clips against 4 (1 s:
    5.77/0.28, 10 losses); without the neural stage 8.84/0.31, the deviation's
    upper quartile 0.96 to 0.84, 27 clips won against 24 and 5 lost against 4.
    The tonal losses are neither the probe's nor the neural stage's.
  - A canceller for persistent non-mains lines above 4 kHz
    (`modules/tone_cancel.py`): finds lines on 19 of 50 captures, mostly the
    field-rate sidebands the surgical notch leaves either side of the CRT line,
    moves the medians not at all and costs one capture 10.45 dB of noise
    removal. A line that holds still is already in the noise profile.

`cathar` is unchanged and is verified so: its decoded output is bit-identical
to the v1.2.1 build on every clip tested, after every change to the chain, and
its per-clip figures here match the v1.2.1 run to the hundredth on 136 of
136\. An earlier run of this release's corpus in a checkout without a venv of
its own had resolved FFmpeg 8.0.1 from the system path where the venv carries
9.0.1, and read `cathar` at 5.87/0.44: the two builds' outputs differ by
55-105 dB below the programme, the deep floor of near-silent passages, which
a quiet-frame level reading magnifies. Binaries now resolve beside the
running interpreter, and this run is the venv's build.

______________________________________________________________________

## 4. Mains hum on the tapes that carry it

Hum is measured on the 48 corpus tapes whose source reads more than 6 dB of
harmonic excess, at whichever of 50 and 60 Hz the tape's own harmonics
support -- on 10 of the 48 that is not the region's nominal frequency, which
is where `cathar`'s negative figure comes from: its `dehum` runs at the
frequency the shared scanner reports, and the scanner reports 0 Hz whenever
it misses. Fifteen of the 48 are the saturated or constant-level captures the
trade metric refuses, and the instrument now refuses them too; every figure
here is over the 33 it can read (median source excess 8.9 dB, upper quartile
13.9).

| Median over the 33 | Excess removed | p75 | Lines down | Low band |
| :--- | ---: | ---: | ---: | ---: |
| `cathar` (unchanged since v1.2.1) | -0.74 dB | 1.27 dB | 0.75 dB | 0.55 dB |
| `auto_pure_linear` | **1.19 dB** | **3.71 dB** | **3.04 dB** | **0.48 dB** |
| `cathar dehum` alone, right frequency, all 48 | 2.07 dB | | | 0.25 dB |
| this canceller alone on the source, all 48 | 3.01 dB | 5.70 dB | | 0.155 dB |

`auto_pure_linear` removes more excess than `cathar` on 24 of the 33 tapes
and takes the lines themselves further down on 26; on 13 it ends past the
2.07 dB `cathar`'s own `dehum` manages run alone at the right frequency. The
two readings differ on purpose: excess is a line against its neighbourhood,
and a chain that lowers the floor around a line it left behind reads as
having removed less hum than it did, which is why the lines' own level is
read beside it (on 5 of the 33 it rises after `auto_pure_linear`, on 13
after `cathar`). The low-band figure is the whole chain's movement in 50-400
Hz with the harmonic bins masked; the canceller alone moves the band 0.155
dB. Four tapes whose lines wander more than five hertz between frames are
not helped by any narrow tracker; they are recorded here, not forced.

**Rumble**, the other low-band defect, is read by the same instrument
(`--band rumble`: quiet-frame energy below 100 Hz, loud-frame deviation in
50-400 Hz). `auto_pure_linear` has no stage for it beyond the shared
pre-conditioning highpass, where `cathar` runs `dewind`, and on the corpus that
reads a median 20.49 dB drop in the
quiet-frame energy below 100 Hz for `auto_pure_linear` against 12.91 for
`cathar`, at 0.59 against 0.69 dB of low-band movement on the loud frames,
over the 135 clips the instrument reads (the degenerate sources refused);
`auto_pure_linear` removes more on 113 of them. That is the shared
highpass and the subtraction seen from below 100 Hz, and it is why no rumble
stage of the mode's own was built this round: the reading had to come first,
and it does not show a gap.

______________________________________________________________________

## 5. A perceptual cross-check

Every ranking above rests on the trade metric and the hum gate, and both are
blind to what they do not measure. DNSMOS P.835 (`scripts/score_perceptual.py`;
Microsoft's non-intrusive estimator of P.835 listening scores, speech-trained,
16 kHz) is read on the source and on each mode's restoration of every clip as a
cross-check and not a gate: a change the metrics approve and DNSMOS reads
clearly worse is a change to listen to before it ships.

| Configuration (174 clips) | SIG | BAK | OVRL | OVRL vs source (paired median) |
| :--- | ---: | ---: | ---: | ---: |
| source | 1.67 | 1.37 | 1.32 | |
| `cathar` | 2.16 | 2.22 | 1.62 | +0.07 |
| `auto_pure_linear` v1.2.1 | 2.38 | 2.95 | 1.82 | +0.30 |
| `auto_pure_linear` v1.3.0 | 2.41 | 2.97 | 1.81 | +0.32 |

`auto_pure_linear` reads better than `cathar` on 140 of 174 clips
for overall quality. The scale is compressed -- the source reads 1.3 on a 1-5
scale, where a clean studio recording reads above 4 -- which is what a tape
corpus looks like to a model trained on suppressor outputs, and the reason the
figure is read paired, clip by clip, rather than as an absolute.

______________________________________________________________________

## 6. Defect coverage

The headline metrics measure broadband noise against programme fidelity. They
do not see impulsive damage at all -- a click is a handful of samples, and
log-spectral distance averages it away -- so physical repair is measured
separately, against paired fixtures that carry the damage, with the repair and
its collateral reported apart.

| Defect | `cathar` | `auto_pure_linear` |
| :--- | :--- | :--- |
| Broadband tape hiss | noiseprint subtraction | subtraction, blend, neural |
| Surface crackle, pops | `decrackle` | pop removal, then `decrackle`, gated |
| Gap dropouts | `inpaint` | `inpaint`, gated on detection |
| Saturation / clipping | `declip` | `declip`, gated on detection |
| Azimuth phase skew | `azimuth` | `azimuth`, gated on detection |
| CRT line whistle | surgical notch | surgical notch |
| Rumble / wind | `dewind` | shared highpass |
| Mains hum | `dehum`, measured -0.74 dB | per-harmonic canceller, 1.19 dB |
| Plosives | `deplosive`, harmful on controls | event-gated, free on controls |
| Spectral spikes | `repair`, harmful on controls | none -- see below |
| Persistent whines | -- | `tone_cancel`, off: measured no gain |

Each `auto_pure_linear` stage is gated on its own defect being detected, which is
a correctness requirement rather than an optimisation: applied to everything,
`decrackle` scores -11.09 dB on material whose only defect is azimuth skew, and
`cathar`'s `repair` and `deplosive` score -15.89 and -5.60 dB on fixtures that
carry no physical damage at all. The v1.2.1 repair stages and the pop stage of
the mode's own are unchanged; see `docs/validation.md`, "Fixtures That Predict
Real Tape" and "The mode's own stages".

### Where `cathar` still leads

- **Spectral spikes.** `repair` was measured harmful on the controls and is not
  in this mode; a gated version was not built this round, and the whine
  canceller that was built measured no gain because a line that holds still is
  already in the mode's noise profile.
- **The clips it wins outright.** Eight clips, against eleven in v1.2.1:
  seven of that release's eleven, of which four moved to a split, and one
  newcomer. Five of the eight are NTSC and four sit in the most tonal third
  of the corpus (flatness under 0.036), and on every one the pattern is the
  same: `cathar` removes a little more and, on the music, deviates a lot
  less. The figures are noise removed / programme deviation in dB.

| Clip | Standard | `cathar` | `auto_pure_linear` |
| :--- | :--- | ---: | ---: |
| JiminyCricketsChristmasD207472PalVHS | PAL | 10.92/0.17 | 8.97/0.53 |
| arthur-arthurs-lost-library-book-1997 | NTSC | 3.22/2.34 | 1.65/2.62 |
| opening-to-private-parts-1997-vhs-true-hq... | NTSC | 5.13/0.70 | 4.29/0.91 |
| rhino-home-video-black-history-in-music... | NTSC | 39.88/0.29 | 27.09/2.53 |
| universal-pictures-ad-family-collection-2... | PAL | 10.11/0.54 | 8.27/0.81 |
| vhspreviewsfromairforceonetouchstone... | NTSC | 13.82/0.00 | 12.30/0.90 |
| vid-20230624-142023 | PAL | 44.39/0.07 | 39.36/0.10 |
| videoplayback_20251013 | NTSC | -2.91/0.00 | -3.95/0.06 |

What the mode still loses on is sustained, tonal programme under a global
subtraction factor, and this round's candidate for it -- the per-bin
suppressor -- kept that programme and lost 4 dB of removal everywhere else.
The gentler factor on tonal material (2.0 under a flatness of 0.035) is what
holds the count at ten; a factor that follows the programme bin by bin
without giving up the removal is the open problem, and it is recorded rather
than forced.

- **Deterministic mathematics throughout**, so instruments, brass and applause
  are not warped. `auto_pure_linear`'s one non-linear stage is its neural
  denoiser, and this release held the line at masking models: the generative
  candidates were out of scope, and the two masking candidates measured lost to
  UVR-DeNoise.

______________________________________________________________________

## 7. Which mode to use

- **General restoration, and any capture with tape hiss, mains hum, rumble
  or plosive-heavy dialogue** -> `auto_pure_linear`, the default from v1.3.0.
  It leads on every row measured on real tape and repairs the same physical
  damage `cathar` does, gated on the defect being there.
- **Crackle, dropouts, clipping or azimuth skew** -> either. Both repair
  these; `auto_pure_linear` runs the stages only where the defect is
  detected, where `cathar` applies its cascade throughout.
- **Sustained tonal programme -- music albums, a held score under
  dialogue** -> `cathar` still deviates less on the most tonal tenth of the
  corpus, where four of its eight outright wins sit, at the cost of removing
  less noise everywhere else. Listen before choosing.
- **Spectral spikes** -> `cathar repair`, with the caveat above: it is the
  only stage for them, and it was measured harmful on undamaged material.
- **A tape whose hum wanders more than five hertz** -> neither; four of the
  48 hum tapes do, and no narrow tracker follows them.
- **CRT line whistle** -> either; both remove it completely.

`auto_pure_linear` is the default from this release. Changing a released
default is a decision for a release, not a benchmark, and this one was taken
on the table above: every row measured on real tape, with `cathar` unchanged
and one line of `config.yaml` away.

______________________________________________________________________

## 8. Reproducing this

- **Corpus provisioning** builds the clip set:

  ```bash
  poetry run python scripts/curate_massive_ia_corpus.py \
    --target-count 1000 \
    --output-dir experiments/ia_corpus_1000 \
    --workers 8
  ```

- **Benchmark** measures both modes on the trade metric and keeps the restored
  outputs for the readings that follow:

  ```bash
  poetry run python scripts/measure_tradeoff.py \
    --corpus-dir experiments/ia_corpus_1000 \
    --modes cathar auto_pure_linear \
    --limit 192 \
    --work-dir experiments/v130_full_work \
    --report experiments/v130_full.json
  ```

  192 is the catalog's clip count, so the limit takes every clip: 174 of them
  restored and were scored, and of those the metric refuses the 38 whose
  quiet-to-loud spread is under 3 dB, leaving the 136 every figure above is
  over.

- **Hum and rumble** read the kept outputs on the 48 hum tapes and the whole
  corpus:

  ```bash
  poetry run python scripts/measure_hum.py --mains auto \
    --work-dirs experiments/v130_full_work \
    --catalog experiments/catalog_hum48.json \
    --report experiments/v130_hum48.json
  poetry run python scripts/measure_hum.py --mains auto --band rumble \
    --work-dirs experiments/v130_full_work \
    --report experiments/v130_rumble.json
  ```

- **The perceptual cross-check** needs the DNSMOS models
  (`scripts/download_dnsmos.py`, pinned checksums) and reads the same outputs:

  ```bash
  poetry run python scripts/score_perceptual.py \
    --work-dirs experiments/v130_full_work \
    --report experiments/v130_perceptual.json
  ```

- **Settings sweep** re-runs the variant comparison behind every adopted and
  held-back setting. Each variant patches the file that actually resolves the
  setting and the merged configuration is re-read before a run is spent on it:

  ```bash
  poetry run python scripts/sweep_denoise_settings.py --limit 50 \
    --catalog experiments/catalog_dfn50.json
  ```
