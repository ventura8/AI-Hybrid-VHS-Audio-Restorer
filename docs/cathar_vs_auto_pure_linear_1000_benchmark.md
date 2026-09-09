# Head-to-Head Evaluation: Cathar vs. Auto Pure Linear

## Summary

Measured on the shipping build with `cathar-cli` 0.7.3 across 174 Internet Archive
VHS clips spanning European PAL (50 Hz mains, 15,625 Hz CRT line whistle) and
American NTSC (60 Hz mains, 15,734 Hz CRT line whistle).

**`cathar` is the stronger general-purpose mode.** It leads on every metric in
both regions. The two modes are at parity on CRT line whistle: both remove it
completely on the clips that carry one. The gap is in broadband noise and mains
hum, and it is widest on NTSC, where `auto_pure_linear` leaves the noise floor
essentially unchanged.

| Metric (median, all 174 clips) | `cathar` | `auto_pure_linear` |
| :--- | ---: | ---: |
| Noise reduction (dB) | **6.85** | 1.04 |
| SNR gain (dB) | **9.45** | 0.70 |
| CRT attenuation (x) | **2.58** | 0.87 |
| Mains attenuation (x) | **7.19** | 1.83 |
| Rumble reduction (pp) | **2.40** | 0.65 |

______________________________________________________________________

## How to read these numbers

**Medians, not means.** CRT and mains attenuation are ratios of the form
`original / max(restored, 0.05)`. When a mode removes a tone completely the
denominator hits that 0.05 clamp and the ratio explodes, so a handful of clips
dominate any average. The mean CRT attenuation for this same run is 742.02 for
`cathar` and 849.69 for `auto_pure_linear` -- which would suggest
`auto_pure_linear` is the better de-whistler, and it is not. Every headline
figure in this document is a median.

**Most clips do not carry the defect being measured.** Of the 174 clips:

- 59 carry a genuine CRT whistle (original whistle-to-background ratio >= 3)
- 45 carry genuine mains hum (original hum-to-background ratio >= 3)

A corpus-wide attenuation average therefore mixes clips with a defect to remove
and clips with nothing to remove, where the ratio is measuring noise against
noise. The conditioned figures below are the ones that describe restoration
quality.

______________________________________________________________________

## 1. Regional results (median)

### Europe, PAL (97 clips)

| Metric | `cathar` | `auto_pure_linear` |
| :--- | ---: | ---: |
| Noise reduction (dB) | **10.55** | 3.11 |
| SNR gain (dB) | **11.44** | 1.46 |
| CRT attenuation (x) | **1.40** | 0.51 |
| Mains attenuation (x) | **8.57** | 2.54 |
| Rumble reduction (pp) | **2.26** | 0.65 |

### America, NTSC (77 clips)

| Metric | `cathar` | `auto_pure_linear` |
| :--- | ---: | ---: |
| Noise reduction (dB) | **4.32** | -0.05 |
| SNR gain (dB) | **6.66** | 0.42 |
| CRT attenuation (x) | **2.66** | 1.02 |
| Mains attenuation (x) | **5.89** | 1.36 |
| Rumble reduction (pp) | **3.00** | 0.65 |

NTSC is where the modes diverge most. `auto_pure_linear`'s median noise
reduction is -0.05 dB: on half of the American clips it leaves the measured
noise floor no better than it found it.

______________________________________________________________________

## 2. Conditioned on the defect actually being present

### CRT line whistle, 59 clips that carry one

| Region | Clips | `cathar` | `auto_pure_linear` |
| :--- | ---: | ---: | ---: |
| All | 59 | 719.80 | 719.80 |
| Europe (PAL) | 35 | 973.40 | **1118.60** |
| America (NTSC) | 24 | 189.50 | **228.10** |

**The two modes are equivalent here.** Both drive the whistle below the
measurement floor, so the ratio saturates against the 0.05 clamp and the
remaining spread reflects the original tone strength rather than any difference
in removal. The corpus-wide CRT figures in the summary table are dominated by
the 115 clips with no whistle to remove and should not be read as a quality gap.

### Mains hum, 45 clips that carry it

| Metric | `cathar` | `auto_pure_linear` |
| :--- | ---: | ---: |
| Mains attenuation (x) | **17.98** | 2.74 |

This gap is real. `cathar` cancels mains hum up to 8 harmonics with adaptive
tracking; `auto_pure_linear` applies fixed narrow notches at the fundamental and
a small set of harmonics.

### Noise floor regressions

Both modes are normalised to the same loudness target at the final mux, so a
mode that raises the programme level without removing noise raises the measured
floor with it.

| | Clips where the floor got worse | NTSC only |
| :--- | ---: | ---: |
| `cathar` | 45/174 (26%) | 23/77 (30%) |
| `auto_pure_linear` | 82/174 (47%) | 40/77 (52%) |

On quiet captures `UVR-DeNoise` removes very little, and the subsequent
normalisation amplifies the untouched hiss along with the programme. Measured on
the worst affected clips, pre-gaining the input to the model's nominal operating
range before inference changes this by less than 0.3 dB, so it is not a
gain-staging problem: the model finds little to remove in this material at any
input level. `cathar`'s noiseprint-based spectral denoise suppresses the same
floor by roughly 29 dB on the same clip.

______________________________________________________________________

## 3. Architecture

### `cathar` (Python-orchestrated Rust CLI DSP pipeline)

The Python pipeline invokes the separately installed Rust `cathar` CLI through
subprocess calls. The installers provision `cathar-cli` 0.7.3 from the verified
upstream release; the runtime resolves `CATHAR_BIN` from its executable search
paths, including the Cargo bin directory.

#### Cathar strengths

- Broad physical defect coverage: pops (`declick`), surface crackle
  (`decrackle`), gap dropouts (`inpaint`), saturation (`declip`), mains hum to
  8 harmonics (`dehum`), spectral spikes (`repair`), tape azimuth phase skew.
- Deterministic mathematics throughout: no neural vocoder hallucination, so
  instruments, brass and applause are not warped.
- Transient integrity: fast attacks and drums stay punchy.
- Spectral Band Replication exciter restores lost tape harmonics.

#### Cathar weaknesses

- Does not separate vocal formants from complex background music.
- Computationally heavy across its cascaded DSP stages.

### `auto_pure_linear` (full-mix neural denoising engine)

#### Auto Pure Linear strengths

- Removes continuous broadband tape hiss via `UVR-DeNoise` where the material
  is at a normal recording level.
- Surgical pre-denoise notching and post-denoise residual cleanup keep tonal
  content away from the neural stage.
- Linear air high-shelf restores presence on muffled dialogue.
- Fewer DSP stages than `cathar`.

#### Auto Pure Linear weaknesses

- Little click, crackle or dropout repair; impulsive scratches survive.
- Mains hum cancellation is far weaker than `cathar`'s (2.74x vs 17.98x on
  clips that carry hum).
- On quiet captures it leaves the noise floor largely intact, which after
  loudness normalisation reads as a floor regression on 47% of the corpus.

______________________________________________________________________

## 4. Which mode to use

- **Default, and anything with mains hum, clicks, dropouts or a damaged tape**
  -> `cathar`. It leads on noise reduction, SNR, mains and rumble in both
  regions, and is the only mode with dropout inpainting and spike repair.
- **Quiet or low-level captures** -> `cathar`. This is where
  `auto_pure_linear`'s neural stage contributes least.
- **CRT line whistle** -> either. Both remove it completely; pick on other
  grounds.
- **Dialogue at a healthy level with constant broadband hiss, where a quiet
  background matters more than transient fidelity** -> `auto_pure_linear`.
- **Muffled or muddy speech** -> `auto_pure_linear`, for the linear air shelf.

______________________________________________________________________

## 5. Reproducing this

- **Corpus provisioning** builds the clip set:

  ```bash
  poetry run python scripts/curate_massive_ia_corpus.py \
    --target-count 1000 \
    --output-dir experiments/ia_corpus_1000 \
    --workers 8
  ```

- **Benchmark** measures both modes:

  ```bash
  poetry run python scripts/benchmark_ia_corpus_batch.py \
    --catalog experiments/ia_corpus_1000/catalog_1000.json \
    --corpus-dir experiments/ia_corpus_1000 \
    --modes cathar auto_pure_linear \
    --output-dir experiments/benchmark_results \
    --gpu "NVIDIA GeForce RTX 5090"
  ```

  Run into a clean `--output-dir`. The script checkpoints and resumes, so an
  existing directory will return earlier results rather than re-measuring.

- **Detector attribution** explains why a mode did or did not act on a clip:

  ```bash
  poetry run python scripts/diagnose_apl_gap.py \
    --corpus-dir experiments/ia_corpus_1000
  ```

- **Provenance.** Metrics are computed by `scripts/ia_benchmark_common.py`.
  Results are only comparable across runs that share both that file and the
  `cathar-cli` version, since either changes the numbers. The figures here were
  produced with `cathar-cli` 0.7.3.

- **Not re-measured in this run.** Per-clip processing latency is not recorded
  by the benchmark report, so no throughput comparison is claimed here. Use
  `scripts/run_hardware_validation.py --execute`, which reports elapsed time and
  a real-time factor per mode.

- Corpus clips and result JSON are excluded from version control due to
  Internet Archive stream licensing and size, so a clean checkout cannot verify
  these numbers without re-running the commands above.
