# Architecture

## Overview

The project is a high-performance cross-platform audio restoration engine for
VHS-era material supporting Linux, macOS (Apple Silicon MPS / Intel), and
Windows.

Core goals:

- Preserve non-vocal ambience and retro analog characteristics.
- Improve vocal clarity and speech intelligibility.
- Keep the pipeline resilient and resumable across operating systems.
- Run consistently in local and CI quality gates.

## Module Map

- `restore_audio_hybrid.py`: entry point and batch orchestration.
- `modules/config.py`: configuration loading, process-mode normalization, and
  mix-volume sanitizing.
- `modules/hardware.py`: CPU/GPU detection, CUDA/MPS profile selection, and
  dynamic linker preparation.
- `modules/auto_scanner.py`: acoustic profiling, tape speed drift detection, and
  restoration strategy selection.
- `modules/filters.py`: defect detectors and every FFmpeg DSP filter graph.
- `modules/processing.py`: stage orchestration, stem separation, denoising,
  mastering, and container muxing.
- `modules/sync.py`: cross-correlation shift and DTW alignment.
- `modules/ui.py`: interactive input handling and startup banner.
- `modules/utils.py`: logging, progress rendering, subprocess handling, media
  validation, and the package-anchored model store.
- `scripts/batch_restore.py`: unattended multi-folder batch runner.

## Restoration Modes

| Mode | Alias | Output suffix |
|---|---|---|
| `auto_pure` (default) | `pure` | `*_Pure_Cleaned` |
| `auto` | — | `*_Auto_Cleaned` |
| `multipass_auto` | `multipass` | `*_MultiPass_Cleaned` |
| `hybrid` | — | `*_Hybrid_Cleaned` |
| `denoise_only` | — | `*_Denoised_Cleaned` |
| `auto_ffmpeg_native` | `auto_vhs_native` | `*_AutoFFmpeg_Cleaned` |
| `vhs_native` | `ffmpeg_native` | `*_FFmpeg_Cleaned` |
| `arnndn_speech` | — | `*_Speech_Cleaned` |

## Mode Dispatch

Every run extracts audio first, then branches on `process_mode`. `auto`,
`auto_pure`, and `multipass_auto` scan the audio and select a restoration
strategy; only `auto` may dispatch that strategy to a different process mode.

```mermaid
flowchart TD
    classDef io fill:#DAE2F9,stroke:#3F5F91,color:#001B3E,rx:30,ry:30;
    classDef step fill:#DCE5DD,stroke:#526350,color:#101E10,rx:10,ry:10;
    classDef ai fill:#F8DAC2,stroke:#825500,color:#2D1600,rx:10,ry:10;
    classDef gate fill:#F9DEDC,stroke:#8C1D18,color:#410E0B,rx:6,ry:6;

    IN(["Input video"]):::io --> EX["Extract audio<br/>pcm_f32le 44.1 kHz stereo"]:::step
    EX --> SKIP{"Valid output<br/>already exists?"}:::gate
    SKIP -->|yes| DONE(["Skip, keep existing"]):::io
    SKIP -->|no| MODE{"process_mode"}:::gate

    MODE -->|auto_pure / pure| M1["4-pass pure restoration"]:::ai
    MODE -->|auto| M2["Scan, then dispatch"]:::ai
    MODE -->|multipass_auto| M3["4-pass cascaded restoration"]:::ai
    MODE -->|hybrid| M4["2-stem separation + enhancement"]:::ai
    MODE -->|denoise_only| M5["Full-track neural denoise"]:::ai
    MODE -->|auto_ffmpeg_native| M6["Auto-tuned FFmpeg DSP"]:::step
    MODE -->|vhs_native| M7["Fixed FFmpeg DSP"]:::step
    MODE -->|arnndn_speech| M8["RNNoise speech denoise"]:::step

    M1 --> OUT(["Restored container<br/>video stream copied"]):::io
    M2 --> OUT
    M3 --> OUT
    M4 --> OUT
    M5 --> OUT
    M6 --> OUT
    M7 --> OUT
    M8 --> OUT
```

## Shared Stage: Acoustic Scan and Strategy

`scan_and_decide_restoration_strategy` builds one profile that every downstream
stage reads. Model choices, enhancement parameters, sync method, and mix gains
all come from this single decision.

```mermaid
flowchart TD
    classDef step fill:#DCE5DD,stroke:#526350,color:#101E10,rx:10,ry:10;
    classDef ai fill:#F8DAC2,stroke:#825500,color:#2D1600,rx:10,ry:10;
    classDef gate fill:#F9DEDC,stroke:#8C1D18,color:#410E0B,rx:6,ry:6;

    WAV(["Extracted WAV"]) --> MACRO
    MACRO["Macro scan<br/>speech / music / ambient<br/>noise floor"]:::step
    WAV --> MICRO["Micro scan<br/>sliding 5 s windows, capped at 60"]:::step
    WAV --> DEF["Defect detectors"]:::step

    DEF --> D1["Mains hum, constrained by line rate"]:::step
    DEF --> D2["CRT flyback, band search 15450-15900 Hz"]:::step
    DEF --> D3["Rumble, clicks, clipping, DC offset"]:::step
    DEF --> D4["Stereo balance and azimuth"]:::step
    DEF --> D5["Tape speed drift<br/>tracks the line whine, not pitch"]:::step
    DEF --> D6["Onset periodicity<br/>a beat, not a spectrum"]:::step

    MACRO --> SEL{"Select mode<br/>and parameters"}:::gate
    MICRO --> SEL
    D1 --> SEL
    D2 --> SEL
    D3 --> SEL
    D4 --> SEL
    D5 --> SEL
    D6 --> SEL

    SEL --> STRAT
    STRAT["Strategy<br/>models, NFE/tau, sync,<br/>gains, filters"]:::ai
```

The strategy carries `mode`, `vocals_model`, `denoise_model`, `arnndn_model`,
`enhance_nfe`, `enhance_tau`, `sync_method`, the two mix gains, and the
`precondition_filters` block. The noise floor is sampled across the whole
recording rather than its opening seconds, which on real tapes swung the
estimate by a median of 9.8 dB.

Tape speed drift is measured against the PAL/NTSC horizontal line whine recorded
on the tape, a fixed physical frequency. Musical pitch moves the dominant
spectral peak but never moves that reference, so program content cannot be
mistaken for drift. A recording carrying no usable reference is reported stable,
which keeps the fast, artifact-free `shift` alignment as the default.

## Shared Stage: Analog Pre-Conditioning

Applied by `auto_pure` and `multipass_auto` before any AI stage. Every filter is
conditional: the chain collapses to `anull` when no defect is detected.

```mermaid
flowchart TD
    classDef step fill:#DCE5DD,stroke:#526350,color:#101E10,rx:10,ry:10;
    classDef gate fill:#F9DEDC,stroke:#8C1D18,color:#410E0B,rx:6,ry:6;

    A(["Extracted WAV"]) --> B{"DC offset?"}:::gate
    B -->|yes| B1["highpass=f=2"]:::step
    B --> C{"Channel balance"}:::gate
    C -->|"gap > 25 dB"| C1["Mirror live channel<br/>pan to dual mono"]:::step
    C -->|"0.5 to 12 dB"| C2["Attenuate the louder side"]:::step
    C -->|"12 to 25 dB"| C3["Leave untouched<br/>too uncertain to act"]:::step
    C1 --> D{"Clipping?"}:::gate
    C2 --> D
    C3 --> D
    B1 --> C
    D -->|yes| D1["adeclip"]:::step
    D --> E{"Azimuth skew<br/>and channels correlated?"}:::gate
    E -->|yes| E1["adelay"]:::step
    E --> F{"Motor rumble?"}:::gate
    F -->|yes| F1["highpass=f=45/60/75"]:::step
    F --> G{"Clicks?"}:::gate
    G -->|yes| G1["adeclick"]:::step
    G --> H["Notches"]:::step
    H --> H1["Mains fundamental + 2nd harmonic"]:::step
    H --> H2["CRT flyback 15625/15734 Hz"]:::step
    H --> H3["Enclosure resonance<br/>only if a real bump, not a formant"]:::step
    H1 --> Z(["Pre-conditioned WAV"])
    H2 --> Z
    H3 --> Z
```

A channel gap wider than 25 dB means a dead channel rather than a level
mismatch, so the live side is mirrored to both rather than attenuating the only
channel carrying audio. The azimuth delay is applied only when the two channels
actually correlate, because a lag measured against a silent channel is noise.

## Shared Stage: Mastering and Mux

```mermaid
flowchart TD
    classDef step fill:#DCE5DD,stroke:#526350,color:#101E10,rx:10,ry:10;
    classDef io fill:#DAE2F9,stroke:#3F5F91,color:#001B3E,rx:30,ry:30;
    classDef gate fill:#F9DEDC,stroke:#8C1D18,color:#410E0B,rx:6,ry:6;

    S1(["Aligned vocal stem"]) --> MIX["amix with per-stem gains"]:::step
    S2(["Aligned background stem"]) --> MIX
    MIX --> LN{"Loudness enabled?"}:::gate
    LN -->|yes| P1["Pass 1: measure programme loudness"]:::step
    P1 --> P2["Pass 2: loudnorm with measured values<br/>linear, targets -16 LUFS"]:::step
    P2 --> RS["aresample to 44.1 kHz<br/>loudnorm leaves the graph at 96 kHz"]:::step
    RS --> LIM["True-peak limiter, -1.0 dBTP"]:::step
    LN -->|no| ENC
    LIM --> ENC["Container-dependent encode<br/>AAC, MP2, or PCM"]:::step
    ENC --> MUX["Mux with -c:v copy"]:::step
    MUX --> VAL{"Output valid?"}:::gate
    VAL -->|yes| OUT(["Atomic rename to final name"]):::io
    VAL -->|no| ERR(["Remove temp, raise"]):::io
```

If the measurement pass fails or returns an incomplete block, the pipeline logs
a warning and falls back to single-pass normalization rather than aborting.

Single-track modes share this chain. They carry one processed stream instead of
two stems, so the graph starts at `[1:a]` rather than an `amix`, but the
measurement pass, resample, and limiter are the same code. The mastered stream is
routed through a `filter_complex` label rather than `-af`, because the optional
archival track is a second audio output that must not be normalized.

## Mode: `auto_pure` (default)

Pure speech and ambient restoration with no generative vocoder synthesis. The
speech stem is denoised and de-essed, never resynthesized.

```mermaid
flowchart TD
    classDef step fill:#DCE5DD,stroke:#526350,color:#101E10,rx:10,ry:10;
    classDef ai fill:#F8DAC2,stroke:#825500,color:#2D1600,rx:10,ry:10;
    classDef io fill:#DAE2F9,stroke:#3F5F91,color:#001B3E,rx:30,ry:30;

    A(["Extracted WAV"]):::io --> P1["Pass 1: acoustic scan -> strategy"]:::ai
    P1 --> P2["Pass 2: analog pre-conditioning"]:::step
    P2 --> SEP["Pass 3: stem separation<br/>strategy vocals_model"]:::ai
    SEP --> V["Speech stem"]:::io
    SEP --> B["Background stem"]:::io
    V --> VD["UVR-DeNoise<br/>strategy denoise_model"]:::ai
    VD --> DE["Dynamic de-esser<br/>written to polished_vocals/"]:::step
    B --> BD["UVR-DeNoise<br/>strategy denoise_model"]:::ai
    BD --> BE["Downward expander below -45 dB"]:::step
    DE --> A1["Pass 4: align to source timing"]:::step
    BE --> A2["Pass 4: align to source timing"]:::step
    A1 --> MIX["Mastering and mux"]:::step
    A2 --> MIX
    MIX --> OUT(["*_Pure_Cleaned"]):::io
```

De-essed output is written to a dedicated `polished_vocals/` directory so a
resumed run cannot mistake an already de-essed file for a fresh stem.

## Mode: `auto`

Profiles the audio, then runs whichever pipeline the strategy selects. When
`enable_multipass` is set and the strategy chooses `hybrid`, the multipass
pipeline runs instead.

```mermaid
flowchart TD
    classDef step fill:#DCE5DD,stroke:#526350,color:#101E10,rx:10,ry:10;
    classDef ai fill:#F8DAC2,stroke:#825500,color:#2D1600,rx:10,ry:10;
    classDef gate fill:#F9DEDC,stroke:#8C1D18,color:#410E0B,rx:6,ry:6;
    classDef io fill:#DAE2F9,stroke:#3F5F91,color:#001B3E,rx:30,ry:30;

    A(["Extracted WAV"]):::io --> SC["Acoustic scan"]:::ai
    SC --> ST{"Strategy mode"}:::gate
    ST -->|"dialogue present"| H{"enable_multipass?"}:::gate
    H -->|yes| MP["multipass_auto pipeline"]:::ai
    H -->|no| HY["hybrid pipeline"]:::ai
    ST -->|"sustained beat"| DO["denoise_only pipeline"]:::ai
    ST -->|"music or ambience only"| DO
    ST -->|"tape noise, no dialogue"| FN["auto_ffmpeg_native pipeline"]:::step
    MP --> OUT(["*_Auto_Cleaned"]):::io
    HY --> OUT
    DO --> OUT
    FN --> OUT
```

Rhythm is tested before the dialogue gate. Sung vocals occupy the same
300–3400 Hz band as speech, so a music video trips every speech test and would
otherwise always reach stem separation. Measured on a labelled corpus of 21
speech tapes and 27 music-video slices, the shipped tonal-peak ratio scored at
chance; onset periodicity keeps all 21 speech tapes on the separation path while
routing 19 of 27 music slices to `denoise_only`.

Note that `auto` is the only mode that acts on this decision. `auto_pure`,
`hybrid`, and the rest name one specific pipeline and run it regardless.

## Mode: `multipass_auto`

Same four passes as `auto_pure`, but the speech stem goes through
Resemble-Enhance generative reconstruction instead of pure denoising.

```mermaid
flowchart TD
    classDef step fill:#DCE5DD,stroke:#526350,color:#101E10,rx:10,ry:10;
    classDef ai fill:#F8DAC2,stroke:#825500,color:#2D1600,rx:10,ry:10;
    classDef io fill:#DAE2F9,stroke:#3F5F91,color:#001B3E,rx:30,ry:30;

    A(["Extracted WAV"]):::io --> P1["Pass 1: acoustic scan -> strategy"]:::ai
    P1 --> P2["Pass 2: analog pre-conditioning"]:::step
    P2 --> SEP["Pass 3: stem separation"]:::ai
    SEP --> V["Speech stem"]:::io
    SEP --> B["Background stem"]:::io
    V --> VE["Resemble-Enhance<br/>strategy NFE and tau, GPU retry"]:::ai
    VE --> DE["Dynamic de-esser"]:::step
    B --> BD["UVR-DeNoise"]:::ai
    BD --> BE["Downward expander"]:::step
    DE --> A1["Pass 4: align"]:::step
    BE --> A2["Pass 4: align"]:::step
    A1 --> MIX["Mastering and mux"]:::step
    A2 --> MIX
    MIX --> OUT(["*_MultiPass_Cleaned"]):::io
```

## Mode: `hybrid`

The same separation and enhancement chain, run directly on the extracted audio
with no pre-conditioning pass.

```mermaid
flowchart TD
    classDef step fill:#DCE5DD,stroke:#526350,color:#101E10,rx:10,ry:10;
    classDef ai fill:#F8DAC2,stroke:#825500,color:#2D1600,rx:10,ry:10;
    classDef io fill:#DAE2F9,stroke:#3F5F91,color:#001B3E,rx:30,ry:30;

    A(["Extracted WAV"]):::io --> SEP["BS-Roformer separation"]:::ai
    SEP --> V["Vocals stem"]:::io
    SEP --> B["Background stem<br/>subtractive, keeps all ambience"]:::io
    V --> VE["Resemble-Enhance"]:::ai
    VE --> DE["Dynamic de-esser"]:::step
    B --> BD["UVR-DeNoise-Lite"]:::ai
    BD --> BE["Downward expander"]:::step
    DE --> A1["Align vocals"]:::step
    BE --> A2["Align background"]:::step
    A1 --> MIX["Mastering and mux"]:::step
    A2 --> MIX
    MIX --> OUT(["*_Hybrid_Cleaned"]):::io
```

The background stem is the companion output of vocal separation rather than a
dedicated music model, which guarantees non-vocal audio such as birds and room
tone is retained in full.

## Mode: `denoise_only`

```mermaid
flowchart TD
    classDef step fill:#DCE5DD,stroke:#526350,color:#101E10,rx:10,ry:10;
    classDef ai fill:#F8DAC2,stroke:#825500,color:#2D1600,rx:10,ry:10;
    classDef io fill:#DAE2F9,stroke:#3F5F91,color:#001B3E,rx:30,ry:30;

    A(["Extracted WAV"]):::io --> D
    D["UVR-DeNoise-Lite on the full track<br/>no separation, no enhancement"]:::ai
    D --> S["Align to source timing"]:::step
    S --> MUX["Mastering + single-track mux<br/>-c:v copy"]:::step
    MUX --> OUT(["*_Denoised_Cleaned"]):::io
```

## Mode: `auto_ffmpeg_native`

```mermaid
flowchart TD
    classDef step fill:#DCE5DD,stroke:#526350,color:#101E10,rx:10,ry:10;
    classDef io fill:#DAE2F9,stroke:#3F5F91,color:#001B3E,rx:30,ry:30;

    A(["Extracted WAV"]):::io --> AN
    AN["Measure noise floor, hum,<br/>rumble, and click density"]:::step
    AN --> F["Auto-tuned filter graph<br/>highpass + adeclick + afftdn + notches"]:::step
    F --> S["Align to source timing"]:::step
    S --> MUX["Mastering + single-track mux<br/>-c:v copy"]:::step
    MUX --> OUT(["*_AutoFFmpeg_Cleaned"]):::io
```

## Mode: `vhs_native`

```mermaid
flowchart TD
    classDef step fill:#DCE5DD,stroke:#526350,color:#101E10,rx:10,ry:10;
    classDef io fill:#DAE2F9,stroke:#3F5F91,color:#001B3E,rx:30,ry:30;

    A(["Extracted WAV"]):::io --> F
    F["Fixed filter graph from config<br/>highpass, adeclick, afftdn, bandreject"]:::step
    F --> S["Align to source timing"]:::step
    S --> MUX["Mastering + single-track mux<br/>-c:v copy"]:::step
    MUX --> OUT(["*_FFmpeg_Cleaned"]):::io
```

## Mode: `arnndn_speech`

```mermaid
flowchart TD
    classDef step fill:#DCE5DD,stroke:#526350,color:#101E10,rx:10,ry:10;
    classDef ai fill:#F8DAC2,stroke:#825500,color:#2D1600,rx:10,ry:10;
    classDef io fill:#DAE2F9,stroke:#3F5F91,color:#001B3E,rx:30,ry:30;

    A(["Extracted WAV"]):::io --> HP["highpass + adeclick"]:::step
    HP --> RN["RNNoise arnndn<br/>strategy-selected .rnnn model"]:::ai
    RN --> S["Align to source timing"]:::step
    S --> MUX["Mastering + single-track mux<br/>-c:v copy"]:::step
    MUX --> OUT(["*_Speech_Cleaned"]):::io
```

## Synchronization

Both alignment methods restore the processed track to the source timing so the
result stays in sync with the untouched video stream.

- `shift`: global cross-correlation lag correction. Fast and artifact-free.
- `dtw`: Dynamic Time Warping for variable drift, GPU `torch.cdist` distance
  with CPU pathfinding and a `fastdtw` fallback.

The scanner selects `dtw` only when the line-whine reference shows measurable
speed instability; otherwise `shift` is used.

## Runtime Composition

- Python 3.12.x.
- Poetry-managed dependencies with platform-specific wheel markers.
- NVIDIA CUDA (Linux / Windows) and Apple Silicon MPS (macOS) acceleration.
- FFmpeg and FFprobe auto-discovery across `.venv/bin`, `.venv/Scripts`, and
  system PATH.
- Separator models resolve to a single package-anchored store, so running the
  CLI from any working directory reuses the same downloads.

## Reliability Design

- Every expensive stage checks for a valid existing output before running.
- Resume checks use validity helpers, never bare existence.
- Media is written to a temporary name and atomically renamed once verified.
- Downloaded models (ARNNDN `.rnnn` and separator checkpoints) undergo
  integrity validation; corrupted or truncated files are automatically deleted
  and re-downloaded cleanly.
- Intermediate audio stays 32-bit float end to end; stems that a separator
  emits as fixed-point are converted back.
- Stem detection accepts every naming convention `audio-separator` emits,
  case-insensitively.
- Temporary work directories are isolated per input and preserved on failure.

## Quality Gates

Local canonical gate:

```bash
./run_pipeline_locally.sh
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_pipeline_locally.ps1
```

Gate stages: PowerShell lint, Ruff, Black, isort, Taplo, Flake8, Pylint, Bandit,
pip-audit, Radon CC/MI pass gates, Markdown format check and lint, pytest with
the shared coverage threshold, strict per-file coverage from `coverage.json`,
and coverage badge regeneration.
