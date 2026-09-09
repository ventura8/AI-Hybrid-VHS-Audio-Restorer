# Head-to-Head Evaluation: Cathar vs. Auto Pure Linear

## Summary

Measured on the v1.2.1 build with `cathar-cli` 0.7.3 across 136 Internet Archive
VHS clips spanning European PAL (50 Hz mains, 15,625 Hz CRT line whistle) and
American NTSC (60 Hz mains, 15,734 Hz CRT line whistle). 174 were run; 38 are
excluded as degenerate sources, explained below.

**`auto_pure_linear` is now the stronger mode on broadband noise, and it leads
without trading programme fidelity to get there.** This reverses the v1.2.0
finding in every particular, including on NTSC, which was previously the mode's
worst region by a wide margin.

| Metric (median, 136 clips) | `cathar` | `auto_pure_linear` |
| :--- | ---: | ---: |
| Noise removed (dB, higher better) | 6.02 | **8.75** |
| Programme deviation (dB, lower better) | 0.44 | **0.33** |

`auto_pure_linear` removes more noise on 113 of 136 clips and disturbs the
programme less on 86 of 136. On 74 clips it wins both at once, against 11 where
`cathar` wins both.

This did not come from a new algorithm. The mode learned its noise profile from
a 0.75 s probe, which is too little audio to estimate a noise floor from.
Taking 2.5 s instead accounts for essentially the whole change.

______________________________________________________________________

## How to read these numbers

**The metrics changed between v1.2.0 and v1.2.1, and the old ones should not be
compared against these.** The previous edition of this document ranked the modes
on attenuation ratios and a `noise_reduction_db` figure. Those were withdrawn
because they cannot distinguish the two things a restoration does:

> An attenuation ratio rises whether a mode removed the defect or removed the
> programme along with it.

That is not hypothetical. Measured against clean references, the deterministic
dehum stage those metrics rewarded cuts 6.7-10.7 dB *below* the truth in the hum
harmonic bins, because speech shares the 50-400 Hz range with mains hum. The
metric scored that as an improvement. It is now off by default in this mode.

The two figures above are measured separately and never combined:

- **Noise removed** is the level drop in the frames the *source* says are quiet.
  Those frames are mostly noise, so a drop there is noise going away.
- **Programme deviation** is how far the restored spectrum moves in the frames
  the source says are loud, inside the 300-3400 Hz speech band. Those frames are
  dominated by content, so movement there is the restoration altering it.

Both are computed after gain-matching on the loud frames, so the final loudness
normalisation can neither flatter nor penalise either number. Which of the two
matters more is a judgement; collapsing them into one score is precisely the
mistake this replaces.

**Medians, not means.** Every headline figure here is a median.

**38 sources are excluded, and the reason is the metric's own premise.** It
reads noise from the frames the source says are quiet and programme from the
frames it says are loud. On 38 of the 174 clips those are the same frames: the
20th and 70th percentile frame levels sit within 3 dB of each other, and on 20
of them the crest factor is under 1.3 dB -- a saturated or constant-level track,
not audio. On those the metric reports deviations of 20-50 dB *identically for
both modes* and a "noise removed" as low as -69 dB. That is the metric failing,
not a restoration, and an earlier edition of this document reported those
figures as a fidelity tail. The metric now refuses a source with under 3 dB of
quiet-to-loud spread, and every figure below is over the 136 it can read.

______________________________________________________________________

## 1. Regional results (median)

### Europe, PAL (69 clips)

| Metric | `cathar` | `auto_pure_linear` |
| :--- | ---: | ---: |
| Noise removed (dB) | 7.76 | **10.66** |
| Programme deviation (dB) | 0.32 | 0.32 |

`auto_pure_linear` removes more noise on 59 of 69 clips and deviates less on
46\. On PAL the two modes are level on the median deviation; the lead is in
noise removal, and in the count of clips it disturbs less.

### America, NTSC (67 clips)

| Metric | `cathar` | `auto_pure_linear` |
| :--- | ---: | ---: |
| Noise removed (dB) | 4.68 | **7.17** |
| Programme deviation (dB) | 0.57 | **0.41** |

NTSC was the mode's worst region in v1.2.0, with a median noise reduction of
-0.05 dB: on half the American clips it left the floor no better than it found
it. It now leads by 2.48 dB there, removing more noise on 54 of 67 clips and
deviating less on 40 -- and NTSC is where its fidelity lead lives.

______________________________________________________________________

## 2. Noise floor regressions

Both modes are normalised to the same loudness target at the final mux, so a
mode that raises the programme level without removing noise raises the measured
floor with it. This was `auto_pure_linear`'s dominant defect in v1.2.0.

| | Floor got worse | NTSC only |
| :--- | ---: | ---: |
| `cathar` | 18/136 (13%) | 12/67 (18%) |
| `auto_pure_linear` | **4/136 (3%)** | **3/67 (4%)** |

For comparison, the v1.2.0 figures for `auto_pure_linear` were 82/174 (47%) and
40/77 (52%), over the unfiltered corpus.

The distribution shows the same thing without relying on a threshold. At the
tenth percentile -- the worst tenth of the corpus -- `auto_pure_linear` still
removes 2.32 dB where `cathar` adds 1.21 dB:

| Noise removed (dB) | p10 | p25 | median | p75 | p90 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `cathar` | -1.21 | 1.70 | 6.02 | 9.45 | 17.14 |
| `auto_pure_linear` | **2.32** | **5.34** | **8.75** | **14.56** | **33.30** |

______________________________________________________________________

## 3. What changed in v1.2.1

- **A 2.5 s noise probe, for this mode only.** The shared 0.75 s value is what
  held the mode back, and it hid behind a settings sweep that could not move it:
  the value is pinned in `config.yaml`, which overrides the defaults the sweep
  was patching, so every earlier attempt reported no effect. On 25 real captures
  2.5 s removes 3.39 dB more noise for 0.12 dB more deviation, reproduced
  exactly on a repeat run. `cathar`'s own probe stays at 0.75 s.

- **A learned per-bin blend** between the subtracted signal and the original,
  which repairs over-subtraction rather than trading against it. Fitted across
  five voices and held out by voice.

- **Context features for that blend.** Capacity was not the limit -- a model
  four times wider bought 1% of the available headroom -- so the blend now also
  sees bin stationarity and its spectral and temporal neighbourhood. Worth
  +0.069 dB held out, winning all five folds.

- **Deterministic tonal cleanup off by default**, on the ground-truth evidence
  described above.

- **Physical damage repair**, gated per defect. Crackle, dropouts, clipping and
  azimuth skew are now repaired where they are detected; three further `cathar`
  stages were measured and rejected for making undamaged material worse. Across
  the corpus it is close to free: noise removal moves from 9.66 to 9.74 dB and
  deviation from 0.48 to 0.50. It is switchable with
  `apl_enable_physical_repair`.

  A 25-clip sample had put the cost at 0.43 dB of noise removal, and the full
  corpus does not bear that out. The smaller measurement was paired and correct
  for its own clips; it simply did not generalise, which is worth stating because
  it is the second time on this branch that a figure from a small sample has
  pointed the wrong way.

`cathar` is unchanged and is verified so: its decoded output is bit-identical to
the v1.2.0 build on every clip tested.

______________________________________________________________________

## 4. Defect coverage

The headline metrics measure broadband noise against programme fidelity. They do
not see impulsive damage at all -- a click is a handful of samples, and
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
| Mains hum | 8-harmonic, measured -0.58 dB | off by default, measured 0.00 dB |
| Spectral spikes | `repair` | not adopted -- see below |
| Plosives | `deplosive` | not adopted -- see below |

Each `auto_pure_linear` stage is gated on its own defect being detected, which is
a correctness requirement rather than an optimisation: applied to everything,
`decrackle` scores -11.09 dB on material whose only defect is azimuth skew.

The stages were adopted on stepped synthetic damage, and the calibrated fixture
set that arrived later in the release reads them very differently: on crackle at
eight pops a second, dropouts at 5-50 ms in real tape noise, and clipping and
azimuth skew drawn from ranges, `declip` recovers +4.86 dB and `azimuth` +10.99,
where `decrackle` recovers +1.19 (+10.00 on the stepped clicks) and `inpaint`
+0.43 (+115.71 dB spectral on the clean 50 ms hole). Inside the finished chain
neither of the last two moves the metrics on its class. They cost nothing on
undamaged material, so they stay; the case for them is now the tape's to make.
The gap on pops is closed by a stage of the mode's own -- residual-outlier
detection and autoregressive refill -- which repairs +5.5 dB at the pop
positions of the same class and is free on real tape. See
`docs/validation.md`, "Fixtures That Predict Real Tape".

**Three of `cathar`'s stages were measured and deliberately not adopted.**
`repair` and `deplosive` fail on the controls rather than on their targets: on
fixtures carrying no physical damage at all, `repair` scores -15.89 dB and lifts
injected error to -4.68 dB against the programme, and `deplosive` -5.60 and
-15.61. Both make clean material worse. `declick` is dominated by `decrackle` on
the same defect, +1.02 dB against +10.00, and degrades as its threshold is
lowered. They remain available in `cathar`, which applies them as a fixed
cascade.

### Where `cathar` still leads

- **Mains hum: neither mode, and the earlier claim is withdrawn.** v1.2.0
  reported `cathar` attenuating hum 17.98x against 2.74x. Measured on this
  build across the 48 tapes that carry hum, with harmonic excess rather than a
  saturating ratio, `cathar` removes **-0.58 dB** and `auto_pure_linear`
  **0.00 dB**. `cathar`'s dehum runs at whatever frequency the shared scanner
  reports, and the scanner reports 0 Hz whenever it misses, which it does on
  nearly half of hum-carrying tapes. The stage itself works when run at the
  right frequency (a median 2.07 dB on those tapes), but in
  `auto_pure_linear`'s chain that collapses to +0.66 dB on a coin flip, so it
  ships available and off (`apl_enable_dehum`). What `cathar` retains is the
  capability; on this corpus it does not deliver it.
- **Spike repair and plosive control**, on the terms above.
- **Deterministic mathematics throughout**, so instruments, brass and applause
  are not warped.

______________________________________________________________________

## 5. Which mode to use

- **General restoration, and any capture with audible tape hiss** ->
  `auto_pure_linear`. It removes more noise than `cathar` in both regions while
  disturbing the programme less, and it no longer degrades quiet captures.
- **Crackle, dropouts, clipping or azimuth skew** -> either. Both repair these
  now; `auto_pure_linear` runs the stages only where the defect is detected,
  where `cathar` applies its cascade throughout.
- **Strong mains hum** -> neither removes it by default on this corpus. `cathar`'s
  stage runs at the wrong frequency whenever the scanner misses; `auto_pure_linear`
  can switch dehum on (`apl_enable_dehum`) for a marginal, measured gain.
- **Spectral spikes, or plosive-heavy dialogue** -> `cathar`. Those two stages
  measured harmful on undamaged material and are not in the other mode.
- **Muffled or muddy speech** -> `auto_pure_linear`, for the linear air shelf.
- **CRT line whistle** -> either; both remove it completely.

Physical repair can be switched off with `apl_enable_physical_repair: false`.
Across the corpus it costs nothing measurable to leave on -- noise removal 9.66
to 9.74 dB, deviation 0.48 to 0.50 -- so the switch is there for control rather
than because the default has a price worth paying attention to.

`cathar` remains the default mode. It is the safer choice on unknown material
because its defect coverage is broader, and changing a released default is a
decision for a release, not a benchmark.

______________________________________________________________________

## 6. Reproducing this

- **Corpus provisioning** builds the clip set:

  ```bash
  poetry run python scripts/curate_massive_ia_corpus.py \
    --target-count 1000 \
    --output-dir experiments/ia_corpus_1000 \
    --workers 8
  ```

- **Benchmark** measures both modes on the trade metric:

  ```bash
  poetry run python scripts/measure_tradeoff.py \
    --corpus-dir experiments/ia_corpus_1000 \
    --modes cathar auto_pure_linear \
    --limit 192 \
    --report experiments/v13_full.json
  ```

  192 is the catalog's clip count, so the limit takes every clip: 174 of them
  restored and were scored, and of those the metric refuses the 38 whose
  quiet-to-loud spread is under 3 dB, leaving the 136 every figure above is
  over.

- **Settings sweep** re-runs the variant comparison behind the tuned constants.
  Each variant patches the file that actually resolves the setting and the
  merged configuration is re-read before a run is spent on it:

  ```bash
  poetry run python scripts/sweep_denoise_settings.py --limit 25
  ```
