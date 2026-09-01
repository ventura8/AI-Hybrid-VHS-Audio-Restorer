# Comprehensive Head-to-Head Evaluation: Cathar vs. Auto Pure Linear

## Executive Summary & The Verdict

Based on empirical benchmarking across an Internet Archive VHS corpus
spanning European PAL (50 Hz mains, 15,625 Hz CRT line whistle) and American
NTSC (60 Hz mains, 15,734 Hz CRT line whistle), here is the head-to-head
evaluation and verdict.

### Reproducibility & Benchmark Environment

- **Corpus Manifest**: `experiments/ia_corpus_1000/catalog_1000.json`
  curated via `scripts/curate_massive_ia_corpus.py`.

- **Benchmark Command**:

  ```bash
  poetry run python scripts/benchmark_ia_corpus_batch.py \
    --catalog experiments/ia_corpus_1000/catalog_1000.json \
    --corpus-dir experiments/ia_corpus_1000 \
    --modes cathar auto_pure_linear \
    --output-dir experiments/benchmark_1000_results \
    --gpu
  ```

- **Mode Configuration**:

  - `cathar`: High-order harmonic de-hum (8 harmonics, adaptive tracking),
    spectral denoise with quiet-window noiseprint, de-click/de-crackle, and
    phase azimuth correction.
  - `auto_pure_linear`: Dual-resolution acoustic scan, analog pre-conditioning
    (CRT notch, mains notch, rumble highpass), single-pass UVR-DeNoise neural
    inference, and linear air presence shelf.

- **Generated Result Artifacts**:

  - Checkpoint: `experiments/benchmark_1000_results/checkpoint.json`
  - Full Report: `experiments/benchmark_1000_results/report_1000.json`

  These generated artifacts are intentionally excluded from version control due
  to corpus licensing and size. A release claiming these metrics must attach
  both JSON files as release assets alongside this document.

> **The Overall Winner**: **`cathar`** is the superior all-round VHS audio
> restoration engine, achieving higher SNR improvement (**+16.96 dB vs +13.72
> dB**), stronger broadband noise reduction (**+11.22 dB vs +8.48 dB**),
> drastically deeper CRT line whistle rejection (**96.17x vs 26.09x**), and
> deeper mains hum cancellation (**4.66x vs 1.85x**), while maintaining pure
> deterministic DSP integrity without neural vocoder hallucinations.
>
> **When `auto_pure_linear` Wins**: `auto_pure_linear` excels specifically on
> **clean dialogue tracks suffering from high stationary tape hiss** where neural
> spectral mask inference (`UVR-DeNoise`) removes broadband hiss without
> transient smearing, and where high-frequency linear tape presence ("air shelf")
> is needed to restore muffled vocal tracks.

______________________________________________________________________

## 1. Large-Scale Empirical Benchmark Metrics

Out of the 1,000 candidate items cataloged by `curate_massive_ia_corpus.py`,
174 representative VHS clips (spanning European PAL and American NTSC
captures) completed full dual-mode restoration and evaluation in
`benchmark_ia_corpus_batch.py`. Clips were excluded if upstream IA streams
were unavailable, truncated, missing audio streams, or if clip extraction or
audio validation failed during batch processing:

| Metric | `cathar` | `auto_pure_linear` | Superior |
| :--- | :---: | :---: | :---: |
| **Broadband Noise Red.** | **+11.22 dB** | +8.48 dB | **Cathar (+2.7 dB)** |
| **SNR Improvement** | **+16.96 dB** | +13.72 dB | **Cathar (+3.2 dB)** |
| **CRT Notch Attenuation** | **96.17x** | 26.09x | **Cathar (+3.7x)** |
| **Mains Atten. (50/60Hz)** | **4.66x** | 1.85x | **Cathar (+2.5x)** |
| **Motor Rumble Reduction** | **+5.93 pp** | +2.86 pp | **Cathar (+3.07 pp)** |
| **Processing Latency** | 5.31s | **3.42s** | **APL (1.5x faster)** |

______________________________________________________________________

## 2. Regional Breakdown: Europe (PAL) vs. America (NTSC)

### Europe PAL (50 Hz Mains / 15,625 Hz CRT Whistle)

- **`cathar`**:
  - Noise Reduction: **+15.83 dB**
  - SNR Gain: **+21.25 dB**
  - CRT Attenuation: **126.18x**
  - Mains Attenuation: **5.23x**
- **`auto_pure_linear`**:
  - Noise Reduction: +14.82 dB
  - SNR Gain: +19.78 dB
  - CRT Attenuation: 31.15x
  - Mains Attenuation: 1.83x

*Analysis*: On European tapes, Cathar achieves near-total erasure of the
infamous 15,625 Hz flyback tone (126.18x attenuation) and 50 Hz buzz.

### America NTSC (60 Hz Mains / 15,734 Hz CRT Whistle)

- **`cathar`**:
  - Noise Reduction: **+5.33 dB**
  - SNR Gain: **+11.48 dB**
  - CRT Attenuation: **57.87x**
  - Mains Attenuation: **3.94x**
- **`auto_pure_linear`**:
  - Noise Reduction: +0.40 dB
  - SNR Gain: +5.98 dB
  - CRT Attenuation: 19.64x
  - Mains Attenuation: 1.87x

*Analysis*: On NTSC captures, `cathar` maintains robust positive noise reduction
(+5.33 dB) and SNR gain (+11.48 dB), whereas `auto_pure_linear` struggles with
quieter NTSC tracks (+0.40 dB noise reduction) due to conservative UVR-DeNoise
masking.

______________________________________________________________________

## 3. Deep Architectural Comparison

### `cathar` (Pure-Rust Native DSP Pipeline)

- **Strengths**:
  - **Comprehensive Physical Defect Coverage**: Cleans pops (`declick`),
    surface crackle (`decrackle`), gap dropouts (`inpaint`), saturation
    (`declip`), mains hum up to 8 harmonics (`dehum`), spectral spikes
    (`repair`), and tape azimuth phase skew.
  - **Zero AI Hallucination**: Pure deterministic mathematics. Musical
    instruments, brass sections, and applause are never warped or "watery."
  - **Transient Integrity**: Fast attacks and drums remain punchy.
  - **High-Frequency Reconstruction**: Incorporates Spectral Band Replication
    (SBR) harmonic exciter to restore lost tape harmonics naturally.
- **Weaknesses**:
  - Does not separate vocal formants from complex background music.
  - High computational intensity across its 10+ cascaded DSP stages.

### `auto_pure_linear` (AI Full-Mix Denoising Engine)

- **Strengths**:
  - **Deep Broadband Hiss Eradication**: Leverages `UVR-DeNoise` neural network
    to identify and subtract continuous tape hiss across the entire spectrum.
  - **Surgical Pre/Post Filtering**: Integrates pre-denoise tonal notching and
    post-denoise residual cleanup to prevent neural network hallucination.
  - **Adaptive Model Selection**: Dynamically switches to `UVR-DeNoise-Lite` on
    quiet recordings ($\<-50\\text{ dB}$) to preserve transients.
  - **Fast Execution**: Streamlined pipeline processes ~1.5x faster than
    Cathar.
- **Weaknesses**:
  - Limited click/crackle/dropout repair (leaves impulsive scratches
    partially intact).
  - Can slightly attenuate delicate background ambience or reverberation tails
    on music tracks.

______________________________________________________________________

## 4. Decision Matrix: Which Mode Should You Choose?

- **Damaged or Degraded Tape** (Pops, clicks, dropouts, mains buzz):
  - *Recommended Mode*: **`cathar`**
  - *Rationale*: Only Cathar features AR dropout inpainting and transient
    spectral spike repair.
- **CRT Line Whistle Present** (15.625 kHz PAL / 15.734 kHz NTSC):
  - *Recommended Mode*: **`cathar`**
  - *Rationale*: Cathar delivers 96x–126x whistle attenuation vs APL's
    26x–31x.
- **Hi-Fi Stereo Music & Concerts**:
  - *Recommended Mode*: **`cathar`**
  - *Rationale*: Preserves complex musical harmony and stereo phase without AI
    gating artifacts.
- **Dialogue / Spoken Word with Constant Hiss**:
  - *Recommended Mode*: **`auto_pure_linear`**
  - *Rationale*: UVR-DeNoise achieves pristine, quiet backgrounds behind clear
    voices.
- **Muffled / Muddy Speech**:
  - *Recommended Mode*: **`auto_pure_linear`**
  - *Rationale*: Linear air high-shelf filter brightens muffled voices
    naturally.
- **Fastest Turnaround Time**:
  - *Recommended Mode*: **`auto_pure_linear`**
  - *Rationale*: 1.55x faster execution with fewer DSP stages.
