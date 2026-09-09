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
| **Frequency Response** | 100 Hz–10 kHz NTSC, 80 Hz–8 kHz PAL | peak 4 kHz |
| **SNR** | ~41–42 dB | ~35–38 dB |
| **Dynamic Range** | ~40–50 dB | ~35–42 dB |

- Recorded by a **stationary head** along the tape edge. The response
  and SNR rows are the published format figures
  ([Wikipedia: VHS](https://en.wikipedia.org/wiki/VHS)); against a compact
  cassette's 1.875 ips, a linear track runs at roughly two thirds of that
  speed at SP and under a quarter at EP/SLP.
- Inherently low-fidelity: narrow track, slow speed, high
  hiss floor.
- **Stereo linear** splits the already-narrow track into two
  thinner channels, degrading SNR further.

### 1.2 Hi-Fi FM Audio (Helical Scan)

| Parameter | Value |
| :--- | :--- |
| **Frequency Response** | 20 Hz – 20 kHz |
| **SNR** | ~70 dB quoted; 45–50 dB off the heads before companding |
| **Dynamic Range** | ~90 dB quoted |
| **Recording Method** | AFM, 1.3 / 1.7 MHz carriers, under the video |

- Recorded **beneath** the video signal by the rotating drum: the
  carriers go on first and deeper, the video is re-recorded over them
  ([Wikipedia: VHS](https://en.wikipedia.org/wiki/VHS)).
- A dbx-like compander that cannot be switched off lifts the 45–50 dB
  the FM signal has off the heads to the quoted figure; it is also the
  source of the breathing in 2.18
  ([Tapeheads](https://www.tapeheads.net/threads/why-needed-vhs-hifi-nr.63936/)).
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
  - EP/SLP recordings (reduced linear-track bandwidth and signal margin versus
    SP; measure the capture's SNR rather than applying a universal multiplier)
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
  - Consumer linear tracks are specified in the 0.1–0.3% weighted
    range; a Hi-Fi track, riding the video heads, measures around
    0.005%. The linear track's figure is "much higher" than Hi-Fi's
    ([Wikipedia](https://en.wikipedia.org/wiki/Wow_and_flutter_measurement)),
    and a clean measured capture on this branch's corpus sits at
    0.0006% against the line whine.
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

### 2.15 Print-Through (Pre-Echo)

- **Source**: The magnetic pattern of one tape layer imprints
  the layer wound against it. Governed by wavelength, coating
  thickness and the coercivity spread of the particles
  ([IASA TC-04 5.4](https://www.iasa-web.org/tc04/removal-storage-related-signal-artefacts)).
  The linear track of a video cassette "may still have some
  print effects" ([Wikipedia](https://en.wikipedia.org/wiki/Print-through)).
- **Characteristics**: A faint copy of the programme one wrap
  ahead (pre-echo) and one wrap behind; one wrap is 3.5 s near
  the hub of a VHS pack at SP and about 11 s at the rim.
  Typically 40-60 dB below the programme, loudest before a loud
  passage that follows silence.
- **Correction**: None in the chain. Print-through is
  indistinguishable from programme to a denoiser, and the
  archival guidance is to note it, not to remove it.

### 2.16 Sticky-Shed Squeal

- **Source**: Binder hydrolysis leaves a tacky surface that
  sticks and slips at the heads and guides; the Library of
  Congress characterises the syndrome by the deposit it leaves
  and "an audible squeal"
  ([LoC](https://www.loc.gov/preservation/scientists/projects/sticky_shed.html)).
  The BAVC glossary lists squeal separately: debris on a guide
  or head, or lost lubrication
  ([BAVC](https://cool.culturalheritage.org/byorg/bavc/bavcterm.html)).
- **Characteristics**: A loud tone, typically 1.5-4 kHz, that
  wanders with the stick-slip cycle and modulates the programme
  at the slip rate; the chatter also adds flutter. Playing a
  squealing tape damages it further.
- **Correction**: Baking the tape before transfer; a wandering
  notch after the fact only where the tone is narrow enough to
  follow.

### 2.17 Modulation Noise

- **Source**: Noise made by the recording process itself --
  tape vibration, amplitude modulation of the bias, head
  magnetisation -- that "increases as record level increases
  and disappears when no signal is present"
  ([National Audio Company glossary](https://www.nationalaudiocompany.com/cassette-glossary/)).
- **Characteristics**: A noise skirt around the programme,
  absent in pauses. A noise profile learned in a pause does not
  describe it, which is why spectral subtraction leaves a
  "halo" around loud passages on some tapes.
- **Correction**: Only the neural stage sees it; subtraction
  cannot.

### 2.18 Hi-Fi Compander Breathing

- **Source**: VHS Hi-Fi runs a dbx-like companding noise
  reduction that cannot be switched off; the FM signal off the
  heads carries 45-50 dB of signal-to-noise and the compander
  lifts that to the quoted 70 dB (1.1). When the carrier off the tape
  is weak -- mistracking between the recording and playback
  decks -- the expander mistracks the level
  ([Tapeheads](https://www.tapeheads.net/threads/why-needed-vhs-hifi-nr.63936/)).
- **Characteristics**: "Pumping" or "breathing": the noise
  floor swells after loud passages and settles with the
  compander's release time, tens to hundreds of milliseconds.
- **Correction**: A noise profile is the wrong tool -- the
  noise is not stationary. A neural denoiser, or a transfer
  from a better-tracking deck.

### 2.19 Hi-Fi Carrier Loss and Track Switching

- **Source**: Most consumer Hi-Fi decks "could not track Hi-Fi
  without dropouts and buzz" ([Wikipedia](https://en.wikipedia.org/wiki/VHS)),
  and a deck losing the FM carrier falls back to the linear
  track until it locks again.
- **Characteristics**: The audio jumps from wideband and quiet
  to 4-8 kHz and hissy for a stretch of seconds and back, often
  with a click at each switch. On a stereo Hi-Fi tape the
  fallback is mono.
- **Correction**: None that restores the missing band;
  matching the level and tilt across the switch is the most a
  chain can do.

### 2.20 Undecoded Dolby B on the Linear Track

- **Source**: Stereo linear tracks halved the track width and
  "manufacturers applied Dolby B noise reduction" to counter the
  hiss ([Wikipedia](https://en.wikipedia.org/wiki/VHS)). A
  capture through a deck without the decoder, or with it off,
  leaves the encoding in; IASA notes that a fluctuating
  background hiss level is the sign of a wrong playback setting
  ([IASA TC-04 5.4](https://www.iasa-web.org/book/export/html/480)).
- **Characteristics**: Quiet passages come back bright and
  hissy -- the encoder boosted their top end and nothing took it
  back out -- while loud passages sound right.
- **Correction**: A level-dependent high shelf, the mirror of
  the encoder; not in the chain.

### 2.21 Edge Damage and Level Fluctuation

- **Source**: The linear track lives on the tape's edge, and a
  creased or curled edge -- "physical distortion of the top or
  bottom edge of the magnetic tape, usually caused by pack
  problems" -- affects the audio track and sometimes stops
  playback ([BAVC](https://cool.culturalheritage.org/byorg/bavc/bavcterm.html)).
  Tape memory from poor storage reduces head contact the same
  way ([IASA TC-04 5.4](https://www.iasa-web.org/book/export/html/480)).
- **Characteristics**: A slow wobble in level, 0.3-2 Hz, with
  the top of the band going with it as contact comes and goes.
- **Correction**: Slow gain riding; the roll-off is not
  recoverable.

### 2.22 Head Clogging

- **Source**: Debris and shed oxide on the audio head gap
  ([BAVC](https://cool.culturalheritage.org/byorg/bavc/bavcterm.html)).
- **Characteristics**: The top of the band comes and goes over
  seconds as debris passes; in the limit the track mutes.
- **Correction**: Cleaning the deck and transferring again.

### 2.23 Lossy Capture Codec

- **Source**: Not the tape: the archive. Every capture on the
  Internet Archive that this branch measured is an MP4 with AAC
  audio, most at 96-128 kbit/s, and every restoration runs on
  the decoded file.
- **Characteristics**: A hard cut-off around 16 kHz, pre-echo
  before transients, "birdies" in the hiss where the codec
  gates quiet bands on and off.
- **Correction**: None; the fixtures carry it so the chain is
  measured on the material it actually receives.

### 2.24 Incomplete Erasure

- **Source**: A tape recorded over: the erase head leaves a
  remnant of the earlier recording, loudest where the new
  programme is quiet.
- **Characteristics**: A faint, band-limited second programme
  underneath, continuous, unrelated to the picture.
- **Correction**: None; it is programme to every stage.

### 2.25 Constant Speed Error

- **Source**: A deck running off speed, or an SP recording
  played at the wrong speed setting; IASA treats speed
  inaccuracy as a documented replay fault distinct from wow and
  flutter ([IASA TC-04 5.4](https://www.iasa-web.org/book/export/html/480)).
- **Characteristics**: A fixed pitch and tempo offset, one to a
  few percent.
- **Correction**: Resampling by the measured ratio, with the
  line whine as the reference. Not in the chain, and not in the
  fixtures: neither metric can read a constant time-base offset.

### 2.26 Scrape Flutter

- **Source**: The tape vibrating against a fixed head or guide as it
  is dragged past -- stick-slip in the tape itself rather than in the
  transport. The wow-and-flutter measurement standards (IEC 386, DIN
  45507, AES6-2008) put it above 100 Hz
  ([Wikipedia](https://en.wikipedia.org/wiki/Wow_and_flutter_measurement));
  the AV Artifact Atlas lists it as an artefact of its own
  ([AVAA](https://www.avartifactatlas.com/tags.html)).
- **Characteristics**: Too fast to hear as pitch: a roughness or noise
  skirt around sustained tones, present only with signal, much like
  modulation noise. Worse on a dirty or worn tape path and on
  hydrolysed tape.
- **Correction**: None after the fact; a clean, lubricated tape path
  at transfer.

### 2.27 EMI Buzz

- **Source**: A switching power supply, a fluorescent fitting, a dimmer
  or a ground loop with a sharp waveform, picked up by the camcorder,
  the deck or the capture card. The AV Artifact Atlas separates "Hum
  and Buzz" and "Electromagnetic Interference" from the smooth mains
  hum of a transformer
  ([AVAA](https://www.avartifactatlas.com/tags.html)); the VCR service
  literature reports hum or buzz on one channel from the deck's own
  electronics
  ([Sci.Electronics.Repair FAQ](https://www.repairfaq.org/REPAIR/F_vcrfaq6.html)).
- **Characteristics**: A mains-rate series whose harmonics do not roll
  off -- tens of them at nearly even strength, reaching several kHz --
  so it reads as a rasp rather than a low hum. It drifts with tape
  speed like the hum does when it was recorded on the tape.
- **Correction**: An eight-harmonic dehum takes the bottom of it and
  leaves the rasp; a comb filter tracking the fundamental over the
  whole series, or spectral gating, is what it needs. Not in the
  chain.

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

Defects the catalog added after the literature check, with no priority
assigned in the chain:

| Defect | Freq Range | DSP Correction |
| :--- | :---: | :--- |
| Video crosstalk | 2-8 kHz buzz | Notch / gating |
| Enclosure resonance | 1.5-3.5 kHz | Gated bandreject |
| Channel imbalance | Level | Balance |
| Quiet capture | Level | Gain staging |
| Print-through | Pre-echo | none |
| Sticky-shed squeal | 1.5-4 kHz tone | Tape baking |
| Modulation noise | Around programme | Neural only |
| Compander breathing | Noise floor swell | Neural only |
| Track switching | Band and hiss jump | none |
| Undecoded Dolby B | HF in quiet passages | Level-dependent shelf |
| Edge damage | Slow level wobble | Gain riding |
| Head clog | HF comes and goes | Re-transfer |
| Lossy codec | >16 kHz cut, pre-echo | none |
| Incomplete erasure | Faint second programme | none |
| Scrape flutter | >100 Hz speed modulation | none |
| EMI buzz | Mains series to several kHz | Tracking comb |
| Constant speed error | Fixed pitch offset | Resample |

### Fixture coverage

| Defect | Fixture class |
| :--- | :--- |
| DC offset | `dc`, `worn` |
| Motor rumble | `rumble` |
| Mains hum | `*_hum`, `worn` |
| Handling pops | `plosive`, `handling` |
| Clicks/pops | `crackle`, `hifibuzz`, `worn` |
| Tape dropout | `dropout`, `ep`, `worn` |
| Clipping | `clip` |
| Tape hiss | every class |
| CRT whistle | `whistle`, `flutter` |
| Azimuth skew | `azimuth` |
| HF roll-off | `sprolloff`, `ep`, `bandlimited` |
| Sibilance | -- |
| Room reverb | every speech class |
| Wow/flutter | `flutter`, `ep`, `worn` |
| Video crosstalk | `crosstalk` |
| Enclosure resonance | `resonance` |
| Channel imbalance | `imbalance` |
| Quiet capture | `low_level` |
| Print-through | `printthrough` |
| Sticky-shed squeal | `squeal` |
| Modulation noise | `modnoise` |
| Compander breathing | `breathing` |
| Track switching | `trackswitch` |
| Undecoded Dolby B | `dolbyb` |
| Edge damage | `edgedamage` |
| Head clog | `headclog` |
| Lossy codec | `codec` |
| Incomplete erasure | `ghost` |
| Scrape flutter | `scrapeflutter` |
| EMI buzz | `buzz` |
| Crowd under commentary | `crowd` |
| Constant speed error | -- |

Every class is generated by `scripts/make_realistic_fixtures_v2.py` on a
speech programme conditioned to the corpus, and `scripts/validate_fixture_realism.py`
reports which detector each class trips. See `docs/validation.md`.

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

- **APL**: Tune the existing pre-denoise mono, CRT-notch, and surgical filtering
  stages against representative captures.
- **APL**: Implement adaptive denoising aggressiveness based
  on measured SNR, lighter processing for already-clean
  content.
- **APL**: Tune the existing post-denoise residual hum/CRT cleanup pass.
- **Cathar**: Verify the existing eight-harmonic dehum configuration on each
  capture's detected 50/60 Hz family.
- **Both**: Ensure order-of-operations follows the
  professional "mud flows downstream" principle.

______________________________________________________________________

*Last updated: 2026-09-11*
*Sources: iZotope RX Documentation, IASA TC-04 (5.4 Reproduction of
Analogue Magnetic Tapes; 5.4.13 Removal of Storage Related Signal
Artefacts), Library of Congress Preservation Science sticky-shed
research, the BAVC Video Preservation Glossary, VideoHelp Forums,
Tapeheads.net Hi-Fi service threads, the Sci.Electronics.Repair VCR
FAQ, the AV Artifact Atlas, the vhs-decode audio notes, Digital FAQ, gotape.eu,
richardhess.com, IIT Bombay spectral subtraction research, and
empirical analysis of Internet Archive VHS captures.*
