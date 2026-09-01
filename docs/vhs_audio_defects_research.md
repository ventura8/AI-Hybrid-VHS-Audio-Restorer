# VHS Tape Audio Defects & Noise — Technical Research

> Comprehensive technical reference on all known VHS audio
> degradation mechanisms, noise signatures, and restoration
> strategies. Compiled from professional archival literature,
> broadcast engineering standards, and empirical tape analysis.

______________________________________________________________________

## 1. VHS Audio Recording Systems

VHS tapes use two fundamentally different audio recording
systems, each with distinct noise characteristics:

### 1.1 Linear Audio (Edge Tracks)

| Parameter | SP Mode | EP/SLP Mode |
| :--- | :--- | :--- |
| **Tape Speed** | 1.31 ips (33.35 mm/s) | 0.44 ips (11.12 mm/s) |
| **Track Width** | ~1.0 mm (mono) / ~0.35 mm (stereo) | Same physical width |

| **Frequency Response** | 100 Hz – 10 kHz | 100 Hz – 4–7 kHz |
| **SNR** | ~40–42 dB | ~35–38 dB |
| **Dynamic Range** | ~40–50 dB | ~35–42 dB |

- Recorded by a **stationary head** along the tape edge.
- Inherently low-fidelity: narrow track, slow speed, high
  hiss floor.
- **Stereo linear** splits the already-narrow track into two
  thinner channels, degrading SNR further.

### 1.2 Hi-Fi FM Audio (Helical Scan)

| Parameter | Value |
| :--- | :--- |
| **Frequency Response** | 20 Hz – 20 kHz |
| **SNR** | ~70–80 dB |
| **Dynamic Range** | ~80+ dB |
| **Recording Method** | FM modulated, helical scan with video heads |

- Recorded **beneath** the video signal by the rotating drum.
- Near-CD quality when tracking is optimal.
- Introduces unique artifacts: head-switching buzz, tracking
  noise, intermodulation with video signal.

______________________________________________________________________

## 2. Comprehensive Defect Catalog

### 2.1 Tape Hiss (Broadband Noise)

- **Source**: Magnetic grain of the tape oxide and the inherent
  noise floor of the analog head-to-tape interface.
- **Frequency Range**: Broadband, but concentrated 2–16 kHz.
- **Severity**: Constant and pervasive. Worse on:
  - Linear tracks (vs Hi-Fi)
  - EP/SLP recordings (3x worse than SP)
  - Stereo linear (halved track width = ~6 dB worse SNR)
  - Aged tapes (oxide degradation weakens signal)
- **Measurement**: 10th-percentile windowed RMS during quiet
  passages.
- **Correction**:
  - Spectral subtraction with learned noise profile
  - Adaptive oversubtraction (alpha > 1) with spectral
    flooring (beta) to suppress musical noise artifacts
  - UVR-DeNoise neural broadband denoiser
  - Downward dynamic expander (gentle 1:2–1:3 ratio)

### 2.2 Mains Hum & Harmonics

- **Source**: Power supply ripple, ground loops between VCR
  and TV/capture device, electromagnetic interference from
  transformer windings.
- **Fundamental Frequencies**:
  - **PAL regions (Europe, UK, Australia)**: 50 Hz
  - **NTSC regions (North America, Japan)**: 60 Hz
- **Harmonic Series**:
  - 50 Hz system: 50, 100, 150, 200, 250, 300 Hz...
  - 60 Hz system: 60, 120, 180, 240, 300, 360 Hz...
  - Typically 5–8 harmonics are audible.
- **Odd harmonics** (150, 250, 350 Hz for 50 Hz) indicate
  asymmetric saturation in the power supply. **Even harmonics**
  (100, 200, 300 Hz) indicate full-wave rectifier ripple.
- **Correction**:
  - Adaptive I/Q heterodyne tracking comb filter (follows
    tape speed wander and per-harmonic amplitude variation)
  - Narrowband notch filters at fundamental + N harmonics
  - Q factor: 30–50 for surgical notching without affecting
    adjacent programme content

### 2.3 CRT Flyback Transformer Whistle

- **Source**: Magnetostriction in the horizontal output
  transformer (flyback transformer). Physical vibration of
  transformer core at the horizontal scanning frequency.
- **Frequencies**:
  - **PAL**: 15,625 Hz (625 lines x 25 fps)
  - **NTSC**: 15,734.264 Hz (525 lines x 29.97 fps)
- **Characteristics**:
  - Very narrowband tonal peak (~1–5 Hz wide)
  - Often captured acoustically by camcorder microphone when
    recording near a CRT television
  - Can be extremely strong (>40 dB above noise floor) on
    home recordings made in front of a TV
  - Intensity varies with TV electrical load
- **Correction**:
  - Very narrow notch (Q = 50–100) at detected frequency
  - Band search in 15,500–15,800 Hz range to find exact peak
  - Must avoid false positives from programme harmonics

### 2.4 Motor Rumble & Mechanical Vibration

- **Source**: Capstan motor, head drum motor, pinch roller
  vibration, belt-driven mechanism resonance.
- **Frequency Range**: Sub-100 Hz, typically 20–80 Hz.
- **Characteristics**:
  - Low-frequency energy that masks bass content
  - More pronounced on worn VCR mechanisms
  - Out-of-phase rumble between L/R on stereo recordings
    (caused by head drum asymmetry)
- **Correction**:
  - Butterworth highpass filter (4th order, 40–75 Hz cutoff)
  - Sub-bass stereo mono collapse (mono below 100 Hz)
    eliminates out-of-phase mechanical rumble while preserving
    mono bass energy
  - DC offset blocking (2 Hz highpass)

### 2.5 Wow & Flutter (Speed Drift)

- **Source**: Capstan motor speed instability, pinch roller
  deformation, belt slippage, back-tension brake issues, worn
  bearings, dried electrolytic capacitors in servo circuit.
- **Measurement**:
  - **Wow**: Slow fluctuations (0.5–6 Hz rate), causes
    audible pitch wobble on sustained notes
  - **Flutter**: Fast fluctuations (6–100 Hz rate), causes
    roughness or "gargling" quality
  - VHS spec: \<=0.3% weighted wow & flutter (SP mode)
- **Characteristics**:
  - Worse on EP/SLP (slower tape = more sensitive to speed
    variation)
  - Worsens with age (belt stretch, lubricant degradation)
  - Measurable by tracking a reference tone's instantaneous
    frequency deviation
- **Correction**:
  - Instantaneous frequency estimation and time-domain
    resampling correction (cathar dewow)
  - Control track servo reference extraction
  - Cross-correlation with video line rate as timing reference

### 2.6 Clicks, Pops & Impulsive Transients

- **Source**: Tape oxide dropouts (brief signal loss), crease
  damage, splice points, head-switching transients, static
  discharge, tape oxide particles on heads.
- **Characteristics**:
  - Impulsive: \<5 ms duration, full-bandwidth energy burst
  - Can be isolated (single click) or dense (surface crackle
    from degraded oxide binder)
  - Head-switching clicks repeat at field rate (50/60 Hz)
- **Correction**:
  - Autoregressive (AR Janssen) interpolation for isolated
    clicks (cathar declick)
  - Statistical outlier detection + interpolation
  - FFmpeg adeclick for moderate density
  - Surface decrackle for high-density noise
  - AR dropout inpainting for gaps up to 50 ms

### 2.7 Tape Dropout & Oxide Shedding

- **Source**: Physical loss of magnetic oxide coating due to:
  - Sticky-shed syndrome (binder hydrolysis from moisture)
  - Mechanical abrasion (head contact wear)
  - Crease/fold damage
  - Mold growth
- **Audio Effect**: Brief silence, loud pop, or distorted
  burst lasting 1–50 ms.
- **Visual Analogue**: White horizontal streak on video.
- **Correction**:
  - AR dropout inpainting (cathar inpaint, up to 50 ms gap
    reconstruction via linear prediction)
  - Spectral interpolation across the gap
  - Cannot recover truly lost signal data, only estimate

### 2.8 Azimuth Misalignment (Stereo Phase Error)

- **Source**: Angular offset between playback head gap and
  tape track orientation. Occurs when:
  - Playback VCR differs from recording VCR
  - Head mount has physically drifted over time
  - Tape has stretched unevenly
- **Effects**:
  - High-frequency loss (short wavelengths cancel)
  - Inter-channel time delay (phase shift)
  - Comb filtering when summed to mono
  - "Dull", "muddy", or "hollow" sound
- **Measurement**: GCC-PHAT cross-correlation peak offset
  between L and R channels.
- **Typical Magnitude**: 0.5–50 microseconds (0.02–2.2
  samples at 44.1 kHz).
- **Correction**:
  - GCC-PHAT sub-sample stereo alignment
  - Correlation-gated: only correct when correlation > 0.5
    (avoid correcting uncorrelated content like true stereo)

### 2.9 Tape Saturation & Clipping

- **Source**: Recording levels exceeding tape's magnetic
  remanence. Common on:
  - Home camcorder recordings (automatic gain control
    overshoot)
  - Recordings from line-level sources without attenuation
- **Characteristics**:
  - Flat-topped waveform peaks
  - Harsh harmonic distortion
  - Spectral spread into upper harmonics
- **Correction**:
  - SPADE sparse peak reconstruction (cathar declip)
  - FFmpeg adeclip threshold detection
  - Cannot fully reverse, only soften the harshness

### 2.10 Hi-Fi Head Switching Buzz

- **Source**: Transition between rotating head pairs during
  helical scan playback. Brief signal discontinuity at
  vertical blanking interval.
- **Frequency**: Field rate (50 Hz PAL / 60 Hz NTSC) and
  harmonics, but manifests as broadband impulsive buzz.
- **Characteristics**:
  - Periodic buzz synchronised to video field rate
  - Worsens with tracking misalignment
  - Can modulate with video content brightness
- **Correction**:
  - Treated as periodic click removal
  - Transient spectral repair (cathar repair)
  - Not applicable to linear-only audio recordings

### 2.11 Intermodulation & Video Crosstalk

- **Source**: VHS is a "color-under" system; the Hi-Fi FM
  audio signal is recorded beneath the video signal by the
  same rotating heads. Signal separation is imperfect.
- **Effects**:
  - High-frequency buzzing that varies with video brightness
  - Intermodulation products between chroma subcarrier and
    audio carrier
  - More severe on worn heads or misaligned tracking
- **Correction**:
  - Notch filtering at known intermodulation frequencies
  - Spectral gating to suppress video-correlated noise
  - Use linear audio track as fallback if Hi-Fi is unusable

### 2.12 High-Frequency Roll-Off (Tape Speed Loss)

- **Source**: Inherent limitation of linear tape recording.
  The slow tape speed (1.31 ips SP) limits the shortest
  recordable wavelength, causing progressive HF loss.
- **Frequency Response Cliff**:
  - SP: begins rolling off around 8–10 kHz
  - EP/SLP: rolls off around 4–6 kHz
- **Correction**:
  - Studio high-shelf EQ ("air" presence: treble g=+2 dB
    at 7.5 kHz)
  - Spectral Band Replication (SBR) to synthesize harmonics
    above roll-off point (cathar enhance --method replicate)
  - Must apply after denoising to avoid amplifying hiss

### 2.13 EP/SLP Extended Play Degradation

- **Source**: Tape speed reduced to 1/3 of SP mode.
- **Combined Effects**:
  - Track width effectively unchanged but signal density
    tripled
  - Increased crosstalk between adjacent tracks
  - Higher wow and flutter sensitivity
  - Dramatically reduced HF response
  - Increased susceptibility to dropout and oxide shedding
  - Tracking becomes extremely sensitive to VCR alignment
- **Audio Symptoms**: Muffled, hissy, prone to buzzing and
  tracking glitches, frequent dropouts.

### 2.14 Acoustic Room Noise (Camcorder Recordings)

- **Source**: Built-in microphone on consumer camcorders picks
  up room reflections, HVAC noise, and enclosure resonance.
- **Specific Issues**:
  - **Plastic housing resonance**: 1.5–3.5 kHz peak from
    camcorder body vibration
  - **Plosive pops**: Low-frequency air blasts from P/B
    consonants when speaking close to mic
  - **Handling noise**: Mechanical vibration transmitted
    through the camcorder body
  - **Room reverberation**: Boomy indoor sound from hard
    wall reflections
- **Correction**:
  - Enclosure resonance notching (1.5–3.5 kHz, gated to
    avoid notching speech formants)
  - Deplosive filter for sub-250 Hz microphone blasts
  - WPE dereverberation for room reflections
  - High-shelf compensation for budget microphone roll-off

______________________________________________________________________

## 3. Spectral Subtraction: Musical Noise Problem

Professional spectral denoising often creates "musical noise",
isolated tonal artifacts caused by random fluctuations in
the noise estimate.

### Mitigation Techniques

1. **Oversubtraction factor (alpha > 1)**: Remove more than
   the estimated noise to eliminate broadband peaks. Typical
   values: alpha = 1.5–4.0 depending on local SNR.
1. **Spectral flooring (beta > 0)**: Prevent spectral bins
   from reaching zero. Fills "valleys" with low-level masking
   noise. Typical values: beta = 0.002–0.05.
1. **Adaptive alpha**: Higher oversubtraction in low-SNR
   frames, lower in high-SNR frames. Preserves speech detail
   while aggressively suppressing hiss in silence.
1. **Temporal smoothing**: Smooth the gain function across
   time frames to prevent abrupt on/off switching.
1. **Spectral smoothing**: Average across adjacent frequency
   bins to reduce isolated "twinkle" peaks.

______________________________________________________________________

## 4. Professional Restoration Workflow (Best Practices)

Based on iZotope RX, SpectraLayers, and archival standards:

### Order of Operations ("Mud Flows Downstream")

1. **DC offset removal**: 2 Hz highpass filter
1. **Stereo balance/azimuth correction**: Channel leveling and delay
1. **De-hum**: Mains fundamental and harmonic rejection
1. **De-click/de-crackle**: Impulsive noise suppression
1. **Dropout inpainting**: Autoregressive gap reconstruction
1. **De-clip**: Peak saturation reconstruction
1. **Broadband denoise**: Spectral subtraction or neural denoising
1. **De-plosive**: Low-frequency microphone blast attenuation
1. **De-esser**: Multiband sibilance control
1. **De-reverb**: Room reflection suppression
1. **HF restoration**: Linear air shelf or SBR after denoising
1. **Dynamic expansion**: Noise gating below programme level
1. **Loudness normalization**: Two-pass EBU R128 mastering
1. **True-peak limiting**: Brickwall limiter at -1 dBTP

> **Critical Rule**: Always denoise before applying HF
> enhancement. Boosting treble before removing hiss amplifies
> the very noise you want to eliminate.

### Conservative vs Aggressive Processing

| Approach | Noise Removal | Artifacts | Best For |
| :--- | :--- | :--- | :--- |
| Conservative (6-10 dB) | Partial | Minimal | Music, archival |
| Moderate (10-20 dB) | Good | Some thinning | Dialogue, home video |
| Aggressive (>20 dB) | Maximum | "Underwater" | Last resort, speech |

______________________________________________________________________

## 5. VHS Defect-to-Filter Mapping Matrix

| Defect | Freq Range | DSP Correction | Priority |
| :--- | :---: | :--- | :---: |
| DC offset | 0-2 Hz | Highpass 2 Hz | 1 |
| Motor rumble | 20-80 Hz | Highpass 45-75 Hz + mono-below | 2 |
| Mains hum | 50/60 Hz + harmonics | Adaptive comb / notch x 5-8 | 3 |
| Handling pops | \<250 Hz | Deplosive filter | 4 |
| Clicks/pops | Broadband impulsive | AR interpolation / adeclick | 5 |
| Tape dropout | Broadband gaps | AR inpainting (\<=50 ms) | 6 |
| Clipping | Broadband harmonic | SPADE declip | 7 |
| Tape hiss | 2-16 kHz broadband | Spectral subtraction / neural | 8 |
| CRT whistle | 15,625/15,734 Hz | Narrowband notch Q=50-100 | 9 |
| Azimuth skew | HF phase error | GCC-PHAT alignment | 10 |
| HF roll-off | >8 kHz | Air shelf / SBR synthesis | 11 |
| Sibilance | 4-9 kHz | Multiband de-esser | 12 |
| Room reverb | Broadband late | WPE dereverberation | 13 |
| Wow/flutter | Pitch modulation | Instantaneous freq correction | 14 |

______________________________________________________________________

## 6. Key Insights for Our Pipeline

### 6.1 Auto Pure Linear Weaknesses (from 39-tape benchmark)

The reproducible report is `experiments/benchmark_ia_corpus_report.md`, with
per-clip source metadata in `experiments/ia_corpus_catalog.json`. Metrics use
the shared definitions in `scripts/ia_benchmark_common.py`; record the Git
commit and benchmark run identifier beside each generated report.

1. **NTSC negative noise reduction (-5.11 dB avg)**:
   UVR-DeNoise model occasionally over-processes NTSC content,
   potentially because the model was trained predominantly on
   music and speech at higher sample rates. The lower dynamic
   range of some NTSC captures may trigger the neural network
   into treating programme content as noise.

1. **Music content degradation (-2.17 dB noise reduction)**:
   The neural denoiser sometimes strips musical transients and
   harmonics, particularly on already-clean music content.
   This results in a "thinner" or "darker" sound.

1. **Rumble insensitivity on home videos (-8.74% vs Cathar's
   -15.45%)**: The UVR-DeNoise model does not specifically
   target sub-100 Hz mechanical rumble. A dedicated
   pre-denoising rumble suppression step would help.

### 6.2 Cathar Strengths (from 39-tape benchmark)

1. **Consistent across all content types**: Wins 77% of clips.
1. **Superior CRT whistle kill**: 11,665x attenuation (7.3x
   better than APL).
1. **Superior mains hum suppression**: 13.25x (2.2x better).
1. **Superior rumble removal**: 3.99% residual (vs 8.36%).
1. **Pure DSP, no AI hallucination risk**.

### 6.3 Improvement Opportunities

- **APL**: Add pre-denoising sub-bass mono collapse and CRT
  notch filter to match Cathar's surgical precision before
  the neural stage.
- **APL**: Implement adaptive denoising aggressiveness based
  on measured SNR, lighter processing for already-clean
  content.
- **APL**: Add a post-denoise residual hum/CRT cleanup pass.
- **Cathar**: Extend harmonic hum series to 8 harmonics for
  complete 50/60 Hz family suppression.
- **Both**: Ensure order-of-operations follows the
  professional "mud flows downstream" principle.

______________________________________________________________________

*Last updated: 2026-09-05*
*Sources: iZotope RX Documentation, IASA Technical Guidelines,
VideoHelp Forums, Digital FAQ, gotape.eu, richardhess.com,
IIT Bombay spectral subtraction research, and empirical
analysis of 39 Internet Archive VHS captures.*
