import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

VALID_PROCESS_MODES = {
    "auto",
    "multipass_auto",
    "multipass",
    "auto_pure",
    "auto_pure_linear",
    "pure",
    "hybrid",
    "denoise_only",
    "ffmpeg_native",
    "auto_ffmpeg_native",
    "vhs_native",
    "auto_vhs_native",
    "arnndn_speech",
    "cathar",
    "cathar_vhs",
}
DEFAULT_PROCESS_MODE = "auto"
DEFAULT_EXTENSIONS = [".mp4", ".mkv", ".avi", ".mov", ".mpg", ".mpeg", ".ts", ".m2ts"]

# Single source of truth for mode-specific output naming. Both the processing
# pipeline (suffix selection) and the UI file scanner (output exclusion) key off
# this map.
OUTPUT_SUFFIX_BY_MODE = {
    "auto": "_Auto_Cleaned",
    "multipass_auto": "_MultiPass_Cleaned",
    "multipass": "_MultiPass_Cleaned",
    "auto_pure": "_Pure_Cleaned",
    "auto_pure_linear": "_PureLinear_Cleaned",
    "pure": "_Pure_Cleaned",
    "hybrid": "_Hybrid_Cleaned",
    "denoise_only": "_Denoised_Cleaned",
    "ffmpeg_native": "_FFmpeg_Cleaned",
    "auto_ffmpeg_native": "_AutoFFmpeg_Cleaned",
    "vhs_native": "_FFmpeg_Cleaned",
    "auto_vhs_native": "_AutoFFmpeg_Cleaned",
    "arnndn_speech": "_Speech_Cleaned",
    "cathar": "_Cathar_Cleaned",
    "cathar_vhs": "_Cathar_Cleaned",
}

# Legacy VHS suffixes from earlier releases, retained so the scanner keeps
# excluding those outputs even though no current mode emits them.
_LEGACY_CLEANED_OUTPUT_SUFFIXES = ("_VHS_Cleaned", "_AutoVHS_Cleaned")
CLEANED_OUTPUT_SUFFIXES = tuple(dict.fromkeys(OUTPUT_SUFFIX_BY_MODE.values())) + _LEGACY_CLEANED_OUTPUT_SUFFIXES


def _config_paths():
    """Returns launch-directory and bundled configuration paths in priority order."""
    launch_config = Path.cwd() / "config.yaml"
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    bundled_config = bundle_root / "config.yaml"
    return (launch_config, bundled_config)


def _normalize_process_mode(raw_value):
    if not isinstance(raw_value, str):
        print(f"[Warning] Invalid process_mode={raw_value!r}; falling back to '{DEFAULT_PROCESS_MODE}'.")
        return DEFAULT_PROCESS_MODE

    normalized = raw_value.strip().lower()
    if normalized in VALID_PROCESS_MODES:
        return normalized

    print(f"[Warning] Invalid process_mode={raw_value!r} (normalized={normalized!r}); falling back to '{DEFAULT_PROCESS_MODE}'.")
    return DEFAULT_PROCESS_MODE


DEFAULT_DENOISE_MODEL = "UVR-DeNoise-Lite.pth"
DEFAULT_VOCALS_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_BACKGROUND_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"
MAX_ENHANCE_NFE = 128


def _parse_mix_float(raw_value):
    try:
        val = float(raw_value)
        if math.isfinite(val) and 0.0 <= val <= 10.0:
            return round(val, 4)
    except (TypeError, ValueError):
        pass
    return None


def _normalize_mix_volume(raw_value, param_name="mix_volume", default=1.0):
    val = _parse_mix_float(raw_value)
    if val is not None:
        return val
    print(f"[Warning] Invalid {param_name}={raw_value!r}; falling back to {default}.")
    return default


_NUMERIC_CONFIG_FIELDS = (
    ("enhance_nfe", int, MAX_ENHANCE_NFE, 1, MAX_ENHANCE_NFE),
    ("enhance_tau", float, 0.3, 0.0),
    ("dtw_resolution", int, 40, 1),
    # Files restored at once in a batch. cathar.exe is single-threaded on every stage
    # (CPU time equals wall time; RAYON_NUM_THREADS changes nothing) and a file's
    # stages run in sequence, so the only way onto the other cores is more files at a
    # time, each in its own interpreter with its own work directory: every file's
    # output is the same bytes as when it runs alone. Neural modes hold their models on
    # the GPU per job, so raise this with the GPU memory in mind.
    ("batch_jobs", int, 1, 1, 16),
    ("afftdn_nr", float, 10.0, 0.0),
    ("afftdn_nf", float, -55.0, None),
    ("highpass_freq", int, 80, 0),
    ("notch_freq", float, 50.0, 0.0),
    ("arnndn_highpass_freq", int, 80, 0),
    ("cathar_alpha", float, 2.0, 0.0),
    # cathar's music profile. The speech settings shave music: a print learned from music
    # is programme, and subtracting it at the speech factor takes 8-16 kHz down 14 dB and
    # removes no noise. The self-driving loop on 11 Internet Archive music clips and the
    # first minute of Gaudeamus (docs/validation.md) settled, over five rounds, on
    # subtraction at 0.5 with no learned print, the coherent path off and the deplosive off:
    # 22 hard-gate failures against 28 at the speech settings, ahead on 6-10 of 12 clips in
    # every accepted round; the 0.7.6 build, on by env in the loop, was not a knob. The profile
    # applies when the scanner's tonal persistence (median share of held spectral peaks
    # over 15 s windows, `modules/tonal_persistence.py`) reads at least
    # cathar_music_persistence_min: the music clips read 0.064-0.225, speech over a bed
    # 0.007-0.033, dry dialogue under 0.003; 0.05 sits in the gap.
    ("cathar_music_persistence_min", float, 0.05, 0.0, 1.0),
    ("cathar_music_alpha", float, 0.5, 0.0),
    ("cathar_beta", float, 0.01, 0.0),
    ("cathar_dewind_cutoff", int, 80, 0),
    ("cathar_declick_threshold", float, 8.0, 0.0),
    ("cathar_decrackle_sensitivity", int, 6, 0),
    ("cathar_declip_threshold", float, 0.95, 0.0, 1.0),
    ("cathar_azimuth_max_ms", float, 5.0, 0.0),
    ("cathar_repair_strength", int, 4, 0),
    ("cathar_inpaint_max_gap_ms", int, 50, 0),
    ("cathar_inpaint_iterations", int, 3, 1),
    # Total length of quiet tape cathar learns its noise print from on a tape of 120 s or
    # more: stitched from that many 0.75 s pauses spread over the quietest fifth of the
    # windows (see modules/cathar.py). A single 4 s window on a dialogue tape carries speech
    # and the print learns sibilance: 11.4 dB removed but 6 dB more off speech at 8-12 kHz
    # than the 0.75 s print; eight stitched pauses removed 6.7 dB at the 0.75 s print's
    # speech highs and 5-6 dB less hiss left in the pauses. Shorter material keeps the
    # single 0.75 s window cathar shipped with, so corpus clips and 60 s excerpts are
    # unchanged.
    ("cathar_noiseprint_duration_s", float, 6.0, 0.0),
    ("cathar_dehum_harmonics", int, 8, 1),
    ("cathar_mono_below_hz", int, 100, 0),
    ("cathar_deplosive_strength", int, 4, 0),
    ("cathar_deesser_bands", int, 3, 1),
    ("cathar_deesser_freq", int, 4000, 1),
    ("cathar_deesser_threshold", float, 6.0, None),
    ("cathar_dereverb_strength", float, 2.0, 0.0),
    ("linear_air_gain_db", float, 2.0, None),
    ("adaptive_denoise_threshold_db", float, -50.0, None),
    # The longest track the UVR denoiser is given in one pass; longer ones are cut into
    # equal overlapping chunks no longer than this. The separator needs about 30 GB of
    # host memory per hour of audio, so 0 (the default) sizes the chunk from the machine:
    # half its memory at that rate, between five minutes and two hours.
    ("neural_chunk_seconds", float, 0.0, 0.0),
    # Spectral subtraction factor, set from real tapes rather than synthetic fixtures.
    #
    # 1.8 was chosen on paired fixtures and is too gentle for real material. Measured across
    # 25 captures with noise removal and programme damage separated, the trade improves
    # steadily to 3.0 and then stops: removal reaches 7.66 dB and plateaus (7.67 at both 3.4
    # and 4.0) while programme deviation keeps creeping, so past the knee there is only cost.
    #
    #   alpha 1.8  6.39 dB removed  0.21 dB deviation
    #   alpha 2.6  7.48            0.29
    #   alpha 3.0  7.66            0.30   <- knee
    #   alpha 4.0  7.67            0.32
    #
    # Each dB of deviation buys 13.7 dB of removal here, against 4.7 for cathar, which is
    # the whole difference between removing noise and removing the programme with it.
    ("apl_spectral_alpha", float, 3.0, 0.0),
    # The factor for tonal material, and the spectral flatness below which a recording is
    # treated as tonal. 3.0 is right in aggregate and wrong on music: on the most tonal
    # third of the corpus (median flatness in 100-5000 Hz under 0.035) the mode deviated
    # 0.49 dB against cathar's 0.32, because sustained tones sit near the noise profile and
    # a speech-tuned factor subtracts them. At 2.0 on those clips deviation is 0.33 -- level
    # with cathar -- with removal still ahead, 7.97 against 7.00. The blend was the first
    # suspect and is cleared: without it the tonal deviation is worse, 0.56.
    ("apl_spectral_alpha_tonal", float, 2.0, 0.0),
    ("apl_tonal_flatness_max", float, 0.035, 0.0),
    # Where `auto` prefers cathar's fidelity: sustained tonal programme with no silence for
    # auto_pure_linear's 4 s noise probe to learn from. The tape reads as tonal under this
    # flatness, and the quietest 4 s carries the programme -- its speech-band spectrum
    # correlates with the loud frames' above auto_cathar_probe_similarity -- with no
    # sustained beat. Measured on 136 corpus clips and 81 excerpts of 21 local tapes, that
    # is the one condition under which cathar deviates less on most clips (10 of 14 and 9
    # of 15; 0.21 dB against 0.54 and 0.15 against 0.19), always for 2-3 dB less noise
    # removed; every other reading leaves cathar behind on both halves. A fidelity
    # preference, not a win: auto_cathar_tonal false keeps auto_pure_linear everywhere.
    ("auto_cathar_flatness_max", float, 0.04, 0.0, 1.0),
    ("auto_cathar_probe_similarity", float, 0.9, 0.0, 1.0),
    # Seconds of the quietest stretch used to learn the noise profile on tonal material.
    # Music has no true silence: its quietest stretch carries sustained partials, and a
    # profile learned over more of it is more programme. Measured against the general probe
    # on the most tonal 45 corpus clips, a shorter probe there buys nothing: 2.5 s reads
    # 8.09/0.28 against 9.05/0.30 and loses both halves of the trade to cathar on 6 clips
    # against 4, 1 s reads 5.77/0.28 and loses on 10. The tonal losses are not the probe's,
    # so it stays equal to the general one.
    ("apl_noiseprint_tonal_s", float, 4.0, 0.0),
    # Programme-above-noise-floor margin above which subtraction is skipped.
    #
    # Effectively off for real tape, and deliberately so. 20 dB came from paired synthetic
    # fixtures, where subtraction appeared to cost fidelity on healthy material. Measured on
    # real captures at the factor this build ships, it does not: raising the gate leaves the
    # median untouched but improves 6 of 25 tapes by 5.61 dB of noise removal for 0.00 dB of
    # programme deviation, and full ground truth on healthy fixtures agrees -- log-spectral
    # distance is unchanged on two and 0.68 dB better on the third.
    #
    # 60 dB still guards the one case worth guarding: audio that is already pristine, well
    # above anything a tape produces. The loudest margin measured across the corpus is
    # 51.7 dB, so every real capture is treated.
    ("apl_spectral_margin_db", float, 60.0, None),
    # Seconds of the quietest stretch used to learn the noise profile, for this mode only.
    #
    # This constant was the single largest gain measured on the branch, and it was hiding
    # behind a broken sweep: the shared 0.75 s value is pinned in config.yaml, so every
    # earlier attempt to move it was overwritten before the pipeline saw it and reported no
    # effect. Measured properly on 25 real captures, 2.5 s removes 3.39 dB more noise for
    # 0.12 dB more programme deviation, and a repeat run reproduced +3.39 / +0.12 exactly.
    #
    # v1.2.1 shipped 2.5 s because on those 25 captures the curve looked flat above it
    # (2.5, 4 and 6 s gave +3.39, +3.32 and +3.97). Measured on the full corpus of 136 in
    # v1.3.0, with the hum canceller now running ahead of the probe, it is not flat: 4 s
    # removes a median 9.99 dB against 8.73 at the same 0.32 dB of median deviation, wins
    # both halves of the trade against cathar on the same 72 clips and loses both on 10
    # rather than 11, and leaves the floor worse on the same 4. The cost is in the deviation
    # tail (upper quartile 0.76 to 0.91 dB; 17 clips move up by more than 0.2 dB, 8 move
    # down) and on NTSC, whose median deviation goes 0.33 to 0.45 against cathar's 0.57. 6 s
    # removes 10.64 at 0.35 but loses both to cathar on 13; 8 s was measured on 50 captures
    # at 12.64/0.27 and not taken further. On a two-hour tape the quietest 4 s is far
    # likelier to be pure noise than it is inside a 15 s corpus clip, so the corpus reads
    # this setting's risk high, not low.
    #
    # An adaptive probe -- extend the quietest window while its spectrum, level or spread
    # stays close to the 2.5 s window's -- was calibrated against the 4 s probe's per-clip
    # cost on the corpus and predicts none of it (correlations of 0.01 to 0.06 with the
    # deviation change; the 17 clips that pay are not the ones any such statistic marks),
    # so the length is fixed rather than adaptive.
    #
    # Raising alpha to 4.0 instead removes 11.85 on 25 captures but costs 0.41 dB of
    # deviation, past cathar, which is the one property worth keeping -- so alpha stays at
    # 3.0 and the probe carries the removal.
    #
    # cathar's own cathar_noiseprint_duration_s only applies from twenty times its length of material (see
    # modules/cathar.py), so this mode passes its own value explicitly instead.
    ("apl_noiseprint_duration_s", float, 4.0, 0.0),
    # Depth and length that mark a span as a dropout rather than a pause, for the physical
    # repair stage. A dropout is loss of head contact, so the audio falls away entirely for
    # a few tens of milliseconds; a speech pause is longer and never that deep.
    ("apl_mute_silence_db", float, -45.0, None),
    ("apl_mute_min_ms", float, 15.0, 0.0),
    ("apl_mute_max_ms", float, 200.0, 0.0),
    # Harmonic excess above which a recording is treated as carrying mains hum. Hum energy at
    # the fundamental and seven harmonics against their own spectral neighbourhood; clean
    # audio sits near 0 dB. 6 dB selects 48 of 174 corpus tapes, and on those a dehum at the
    # frequency the harmonics support removes a median 2.07 dB of hum for 0.25 dB of movement
    # in the speech that shares the band.
    ("apl_hum_min_excess_db", float, 6.0, 0.0),
    # Pop removal in the same mode, before decrackle: prediction-residual outliers this many
    # robust scales out are refilled by autoregressive interpolation. 0 switches it off. Set
    # on the calibrated crackle class, where cathar's decrackle recovers +1.7 dB inside the
    # pops at any sensitivity and this recovers +5.5 at 7, touching 61 spans of undamaged
    # speech per 15 s at 40 dB under the programme; 5 touches 312, 8 starts missing pops.
    ("apl_depop_threshold", float, 7.0, 0.0),
    # The mode's own harmonic hum canceller: how many mains harmonics it evaluates (each is
    # cancelled only where it stands above its own neighbourhood, so a high count costs
    # compute and nothing else -- an EMI buzz runs to eighty harmonics, mains hum to eight)
    # and the envelope bandwidth at the fundamental in Hz (it widens with harmonic number,
    # because recorded hum carries the transport's wow and the excursion scales with it).
    # Set by the measurement in `apl_enable_hum_cancel`.
    ("apl_hum_max_harmonics", int, 40, 1, 128),
    ("apl_hum_bandwidth_hz", float, 1.5, 0.1),
    # The mode's own noise suppressor, in the slot cathar's subtraction holds: the noise
    # estimate's bias (the over-subtraction analogue; 1.0 is the estimate as measured), the
    # gain floor in dB (cathar's beta of 0.01 is a -20 dB floor), and the decision-directed
    # smoothing of the a priori SNR at this mode's 1024-sample hop (0.98 is the textbook
    # value at ~100 frames per second; at 43 it would remember 1.2 s and smear onsets).
    # Set by the measurement in `apl_use_native_suppress`.
    ("apl_suppress_noise_bias", float, 1.0, 0.0),
    ("apl_suppress_gain_floor_db", float, -20.0, None),
    ("apl_suppress_dd_alpha", float, 0.96, 0.0, 1.0),
    # Plosive control: how far the low band must rise over its own running level, in dB,
    # to be read as a plosive burst. 0 switches detection off, like `apl_depop_threshold`.
    # Set by the measurement in `apl_enable_plosive_tamer`.
    ("apl_plosive_excess_db", float, 12.0, 0.0),
    # Deterministic tonal cleanup (dehum, dewind) for the same mode. Off by default: it
    # measures as harmful. Against a clean reference on fixtures whose hum genuinely
    # dominates, dehum drives the harmonic bins 6.7-10.7 dB *below* the truth -- speech
    # fundamentals share 50-400 Hz, so it removes programme along with the hum -- and
    # applying it alone scores worse than doing nothing at all (10.23 dB log-spectral
    # distance against an untouched 9.78). End to end the mode scores 6.71 dB with this
    # off against 8.26 with it on. Spectral subtraction already removes hum correctly,
    # landing 1.5 dB from the reference instead of 9.9 dB past it.
    #
    # It reads as an improvement only through the corpus mains-attenuation ratio, which
    # measures a peak against its local background and never penalises removing the
    # programme with it. Kept behind a switch rather than deleted, for material where the
    # hum is genuinely isolated from content.
)
_BOOL_CONFIG_FIELDS = (
    ("afftdn_tn", True),
    ("enable_adeclick", True),
    ("arnndn_enable_adeclick", True),
    ("enable_multipass", True),
    ("enable_deesser", True),
    ("enable_loudnorm", True),
    ("enable_dynamic_expander", True),
    ("enable_linear_air", True),
    ("apl_enable_spectral_denoise", True),
    ("apl_enable_tonal_cleanup", False),
    # Mains hum removal for this mode, separate from the rumble stage it used to share a
    # switch with. Available, and off: measured end to end it does not earn a default.
    #
    # The stage alone is a clear gain. Across the 48 corpus tapes that carry hum, cathar's
    # dehum run at the frequency the harmonics support removes a median 2.07 dB of harmonic
    # excess at 0.25 dB of movement in the speech that shares the band. In the chain that
    # collapses: +0.66 dB ahead of subtraction, on 20 of 48 tapes, with the speech band
    # moving 1.52 to 2.15 dB; +0.74 dB after subtraction, on 14 of 48, at a cost of 0.76 dB
    # of broadband noise removal. The 2.5 s noise profile already captures hum on the tapes
    # where it is stationary -- the chain without this stage shows a 2.94 dB upper quartile
    # of hum removed -- so the stage mostly moves that removal earlier and pays for it.
    #
    # It is the same finding as the CRT cascade: an isolated gain that does not survive the
    # chain does not ship. cathar's own dehum removes -0.58 dB on these tapes, because it
    # runs at whatever frequency the shared scanner reports and the scanner reports 0 Hz
    # whenever it misses; the detector here is kept correct so that anyone switching this on
    # gets the frequency the recording actually carries.
    ("apl_enable_dehum", False),
    # The mode's own hum canceller: each mains harmonic found as a line of its own in the
    # quiet frames is tracked by complex demodulation at the frequency it actually sits at
    # and subtracted, per channel, ahead of the noise probe. Measured on the 48 corpus tapes
    # that carry hum, in the chain, hum removed goes from a median 0.19 dB to 2.49 (upper
    # quartile 2.64 to 5.87), 26 tapes past the 2.07 dB cathar's dehum manages run alone at
    # the right frequency, better on 34 of 48 and worse by more than a dB on none; the low
    # band moves 1.41 to 1.48 dB across the chain, where cathar's stage alone costs 0.25;
    # the broadband trade on those tapes moves 12.91/0.33 to 12.90/0.35, inside the free
    # band. Stage alone on the same tapes: 3.01 dB at 0.155. Four tapes whose lines wander
    # more than five hertz are not helped; they are documented, not forced.
    #
    # Three gates keep a chord's partials out of the series, after the calibrated music-only
    # class caught the canceller taking a G major at 98, 147 and 196 Hz for the second to
    # fourth harmonics of 49 Hz (2.42 dB of deviation against 1.69 without it), and the
    # speech class whose voice sits at twice a mains-like fundamental had its harmonics read
    # as that fundamental's even lines: a series whose lines place the fundamental more than
    # half a hertz off the mains window is refused as a chord; a gated line further off an
    # exact multiple of the refined fundamental than half its tracking band is a partial and
    # is left alone; and a series with no line at the fundamental is not hum, unless the
    # shared pre-conditioning has already notched that line away. (A gate on a line's level
    # in the loud frames against the quiet ones was tried and dropped: programme energy
    # shares a line's bins, and the strongest real hum tapes lost their whole series to
    # it.) On real tape the chord reading was hiding as hum removal -- on two tonal tapes
    # the low band moved 0.91 and 1.50 dB before the gates and 0.11 and 0.01 after. Gated,
    # on the 33 readable hum tapes (15 of the 48 with under 3 dB of quiet-to-loud spread are
    # refused by the instrument), the chain removes a median 1.19 dB of excess (2.43 before
    # the gates; upper quartile 3.71) and takes the lines themselves down 3.04 dB (6.71),
    # against cathar's -0.74 and 0.75, ahead of it on 24 and 26 of the 33, 13 tapes past
    # 2.07 to cathar's 6, the low band moved 0.48 dB to cathar's 0.55; on the most tonal 45
    # clips deviation reads 0.30 against 0.32 ungated.
    ("apl_enable_hum_cancel", True),
    ("auto_cathar_tonal", True),
    # Skip the neural denoiser on tonal material. UVR-DeNoise earns its place in aggregate
    # (without it the mode removes 1.05 dB less noise on real tape); on the most tonal
    # third, where the mode loses to cathar on fidelity, its share of that loss was never
    # read on its own. Measured on the most tonal 45 corpus clips: without it 8.84/0.31
    # against 9.05/0.30, the deviation's upper quartile 0.96 to 0.84, both halves won
    # against cathar on 27 clips against 24 and lost on 5 against 4; with a 2.5 s probe as
    # well 7.78/0.28, upper quartile 0.71, lost on 7. A trade, not a gain: off.
    ("apl_tonal_skip_neural", False),
    # The mode's own bandrejects at the third to fifth mains harmonics, applied ahead of the
    # chain whenever the shared scanner reports mains hum. They predate the canceller, which
    # tracks each of those lines at the frequency it actually sits at; a Q-30 notch at the
    # exact multiple takes the centre of a line that sits on it and misses one that sits a
    # few hertz off, and either way hands the canceller a line it can no longer read whole.
    # Measured on the 33 readable hum tapes with the canceller on, leaving the harmonics to
    # the canceller loses hum removal, 2.43 to 1.71 dB of excess at the median and worse by
    # more than a decibel on 8 tapes against better on 1, while the broadband trade reads
    # 0.04 dB less deviation for the same removal (a gain-match effect of the energy left at
    # 150-250 Hz). The notch and the canceller do better together than the canceller alone.
    ("apl_surgical_mains_notch", True),
    # When the scanner reported mains hum, the shared pre-conditioning has already notched
    # the fundamental and its second harmonic (Q 15) before the canceller sees the audio, so
    # on those tapes the two lowest lines it finds are the notch's skirts, not lines, and
    # their peaks pull the refined fundamental to the edge of its range. With this on those
    # two harmonics are left to the notch and the refinement reads the harmonics above them.
    # Measured on the 33 readable hum tapes it changes nothing: the same 2.43 dB median, no
    # tape moved by a decibel, the broadband trade to the hundredth. The notch has already
    # taken those two lines, and cancelling what it leaves neither helps nor harms.
    ("apl_hum_skip_notched", False),
    # The mode's own noise suppressor in the subtraction slot: per-bin MMSE log-spectral
    # gain with a tracked noise level, in place of cathar's global factor. Off, on real-tape
    # evidence, the same way DeepFilterNet is: on the calibrated fixtures it lands at 5.9-6.2
    # dB of log-spectral distance where cathar's subtraction lands at 10-12.9, and on 50
    # real captures in the chain it removes 5.83 dB at 0.19 of deviation against the
    # subtraction's 10.02 at 0.22 (bias 1.5: 6.88/0.20; floor -30 dB: 5.80/0.20; without the
    # blend: 5.66/0.19), and on the tonal 45, where the mode's losses live, 3.06/0.22 against
    # 7.96/0.28. A per-bin estimator keeps the low-level programme the quiet frames hold,
    # which the trade metric reads as noise left behind; the metric is the judge this branch
    # chose, and by it the subtraction stays. Kept as an engine, selectable, for the day a
    # metric that can see fidelity in quiet frames is on the branch.
    ("apl_use_native_suppress", False),
    # Event-gated plosive control: a burst under 150 Hz that stands over the low band's
    # running level and leads the mid band's own rise is taken down to the level the band
    # held just before it, and nothing else is touched. On the calibrated plosive class, at
    # the low-band spans where the reference differs from the target, it recovers 0.89 dB at
    # 0.03 dB of collateral against cathar deplosive's 1.55 at 0.34, and on the undamaged
    # classes it costs 0.00-0.01 dB where deplosive costs 0.5-0.9 and reads -6.7 dB on
    # music-led programme. On 50 real captures in the chain it is free: 10.02/0.23 to
    # 10.02/0.23, 37 captures untouched to the hundredth; a 9 dB threshold costs 0.02 dB of
    # deviation for nothing and is not the default.
    ("apl_enable_plosive_tamer", True),
    # A canceller for persistent non-mains lines. Off: on 50 real captures at its first
    # setting it read persistent lines on 41 and cost 0.06 dB of deviation, with one capture
    # moved 4.2 dB, because sustained notes of the programme and mains lines the detector had
    # missed read as persistent lines in the speech range. Restricted to lines above 4 kHz,
    # where recorded whines live and programme lines do not, it finds lines on 19 of the 50
    # -- mostly the field-rate sidebands the surgical notch leaves either side of the CRT
    # line -- and moves the medians not at all, 10.02/0.23 to 10.02/0.23, while one capture
    # loses 10.45 dB of noise removal to it: a line that holds still is already in the
    # noise profile and the subtraction removes it outright, where the tracker's smoothed
    # envelope leaves a residual. What the profile cannot capture, a line that wanders, the
    # tracker cannot follow either. Selectable for a whine the ear finds and the probe missed.
    ("apl_enable_tone_cancel", False),
    # Resemble-Enhance's denoiser (a masking model; its enhancer is generative and not a
    # candidate) in place of UVR-DeNoise, where the package is installed. Off: on 50 real
    # captures it removes 10.78 dB against UVR-DeNoise's 10.02 and moves the programme 0.58
    # dB against 0.23, the upper quartile of deviation going 0.49 to 1.77; it wins both
    # halves of the trade against cathar on 12 captures where the shipped stage wins on 28.
    # A 44.1 kHz masking UNet trained on studio speech takes tape programme for noise the
    # way DeepFilterNet did. Absence or failure falls back to UVR.
    ("apl_use_resemble_denoise", False),
    # Physical tape damage repair for this mode: crackle, dropouts, saturation and azimuth
    # skew, each gated on its own defect being detected. Four stages earned a place against
    # paired fixtures and three were rejected -- `repair` and `deplosive` make undamaged
    # material measurably worse, and `declick` is dominated by `decrackle`. Gating is not an
    # optimisation: applied blanket-fashion `decrackle` scores -11.09 dB on material whose
    # only defect is azimuth skew.
    ("apl_enable_physical_repair", True),
    # Whether DeepFilterNet3 replaces UVR-DeNoise as the neural stage in this mode, where it
    # is installed. Off, on real-tape evidence. On paired synthetic fixtures it looked like a
    # clear win -- log-spectral distance improved 3.52 dB against UVR-DeNoise's 0.71, and it
    # is seventeen times faster -- and on 50 real captures in the same chain it triples
    # programme deviation, 0.22 to 0.66 dB, with noise removal flat. Capping its attenuation
    # only walks a curve (20 dB: 7.42/0.31, 12 dB: 5.22/0.27) that never reaches the
    # UVR-DeNoise chain's 9.91/0.22. A speech-enhancement model trained on clean speech
    # treats band-limited VHS dialogue, and everything that is not dialogue, as noise.
    #
    # Fifth constant on this branch that a fixture set wrongly. Kept as an opt-in stage: it
    # is a from-source optional dependency (Rust core under MSVC, numpy and torchaudio pins
    # ignored) and absence falls back to UVR-DeNoise.
    ("apl_use_deepfilternet", False),
    # Blend the subtracted signal back toward its input, per frequency bin, using weights
    # fitted against clean references. It repairs what subtraction over-cuts, so its value
    # depends on how hard subtraction is pushed -- the two settings interact, and measuring
    # them independently was misleading: at alpha 1.8 the blend preserved programme better
    # on only 21 of 40 tapes, a coin flip, and looked not worth keeping. At the alpha this
    # build actually ships it wins on 18 of 25 and cuts programme deviation by a third,
    # 0.30 dB to 0.20, for 0.60 dB of removal.
    #
    # The pair strictly dominates the previous setting: more noise removed (7.05 dB against
    # 6.17) and less programme disturbed (0.20 against 0.23).
    #
    # Re-checked at the 2.5 s probe, because settings on this branch have interacted before
    # and this one was last judged at 0.75 s. It still pays, and what it pays for is
    # fidelity rather than removal: switching it off removes 0.91 dB *more* noise but lifts
    # programme deviation from 0.32 to 0.41. cathar sits at 0.36 on the same captures, so
    # the blend is precisely what keeps this mode on the right side of it. Without it the
    # mode still wins on noise and loses on programme, which is the half worth having.
    ("apl_enable_learned_blend", True),
    ("preserve_original_audio_track", False),
    ("debug_logging", False),
    ("cathar_enable_coherent", True),
    ("cathar_enable_dewind", True),
    ("cathar_enable_azimuth", True),
    ("cathar_enable_declick", True),
    ("cathar_enable_decrackle", True),
    ("cathar_enable_inpaint", True),
    ("cathar_enable_declip", True),
    ("cathar_enable_dehum", True),
    ("cathar_dehum_adaptive", True),
    ("cathar_enable_repair", True),
    ("cathar_enable_dewow", False),
    ("cathar_enable_enhance", True),
    ("cathar_enable_noiseprint", True),
    ("cathar_enable_mono_below", True),
    ("cathar_enable_deplosive", True),
    # The music profile (see cathar_music_alpha): switched as a whole, then its three stage
    # switches. Off, every tape gets the speech settings.
    ("cathar_music_profile", True),
    ("cathar_music_enable_noiseprint", False),
    ("cathar_music_enable_coherent", False),
    ("cathar_music_enable_deplosive", False),
    ("cathar_enable_deesser", True),
    ("cathar_enable_dereverb", False),
    ("cathar_dereverb_wpe", True),
)


def _typed_config_defaults():
    """Returns the canonical numeric and Boolean defaults for configuration loading."""
    numeric = {field[0]: field[2] for field in _NUMERIC_CONFIG_FIELDS}
    boolean = {name: default for name, default in _BOOL_CONFIG_FIELDS}
    return {**numeric, **boolean}


_BOOL_STRINGS = {
    "1": True,
    "true": True,
    "yes": True,
    "on": True,
    "y": True,
    "t": True,
    "0": False,
    "false": False,
    "no": False,
    "off": False,
    "n": False,
    "f": False,
    "": False,
}
VALID_CATHAR_DENOISE_METHODS = {"spectral", "wiener"}
VALID_CATHAR_AZIMUTH_METHODS = {"correlation", "gcc-phat"}
VALID_CATHAR_ENHANCE_METHODS = {"replicate", "interpolate"}


def _reject_config_value(param_name, raw_value, default):
    """Logs a normalisation warning and returns the field's default."""
    print(f"[Warning] Invalid {param_name}={raw_value!r}; falling back to {default}.")
    return default


def _normalize_choice(raw_value, allowlist, field_name, fallback):
    """Validates string choice against an allowlist, warning and returning fallback on mismatch."""
    if isinstance(raw_value, str) and raw_value.strip().lower() in allowlist:
        return raw_value.strip().lower()
    return _reject_config_value(field_name, raw_value, fallback)


def _normalize_cathar_denoise_method(raw_value):
    return _normalize_choice(raw_value, VALID_CATHAR_DENOISE_METHODS, "cathar_denoise_method", "spectral")


def _normalize_cathar_azimuth_method(raw_value):
    return _normalize_choice(raw_value, VALID_CATHAR_AZIMUTH_METHODS, "cathar_azimuth_method", "gcc-phat")


def _normalize_cathar_enhance_method(raw_value):
    return _normalize_choice(raw_value, VALID_CATHAR_ENHANCE_METHODS, "cathar_enhance_method", "replicate")


def _is_bad_number(val):
    """True for a float that came back as NaN or infinity."""
    return isinstance(val, float) and not math.isfinite(val)


def _is_invalid_number(val, min_val, max_val):
    """True for non-finite floats or values below the minimum bound."""
    if _is_bad_number(val):
        return True
    return (min_val is not None and val < min_val) or (max_val is not None and val > max_val)


def _coerce_number(raw_value, caster, param_name, default, min_val=None, max_val=None):
    """Casts a config value to int/float, rejecting bools, None, non-finite, and bounds violations."""
    if raw_value is None or isinstance(raw_value, bool):
        return _reject_config_value(param_name, raw_value, default)
    try:
        val = caster(raw_value)
    except (TypeError, ValueError):
        return _reject_config_value(param_name, raw_value, default)
    if _is_invalid_number(val, min_val, max_val):
        return _reject_config_value(param_name, raw_value, default)
    return val


def _coerce_bool(raw_value, param_name, default):
    """Parses a config Boolean by content, so quoted 'false'/'0' resolve to False."""
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    if isinstance(raw_value, str):
        token = raw_value.strip().lower()
        if token in _BOOL_STRINGS:
            return _BOOL_STRINGS[token]
    return _reject_config_value(param_name, raw_value, default)


def _normalize_mute_span(defaults):
    """Keeps the dropout span's minimum under its maximum; a crossed pair falls back to both defaults.

    The repair stage clamps the maximum above the minimum otherwise, which silently turns a
    configured maximum into a different number.
    """
    numeric = {field[0]: field[2] for field in _NUMERIC_CONFIG_FIELDS}
    if defaults["apl_mute_min_ms"] > defaults["apl_mute_max_ms"]:
        print(
            f"[Warning] apl_mute_min_ms={defaults['apl_mute_min_ms']!r} exceeds apl_mute_max_ms={defaults['apl_mute_max_ms']!r}; "
            "falling back to both defaults."
        )
        defaults["apl_mute_min_ms"] = numeric["apl_mute_min_ms"]
        defaults["apl_mute_max_ms"] = numeric["apl_mute_max_ms"]


def _normalize_typed_config_fields(defaults):
    """Normalizes every typed numeric/Boolean field before module-level conversion."""
    for field in _NUMERIC_CONFIG_FIELDS:
        name, caster, default, min_val, *max_value = field
        max_val = max_value[0] if max_value else None
        defaults[name] = _coerce_number(defaults.get(name, default), caster, name, default, min_val=min_val, max_val=max_val)
    for name, default in _BOOL_CONFIG_FIELDS:
        defaults[name] = _coerce_bool(defaults.get(name, default), name, default)
    _normalize_mute_span(defaults)


def _find_config_path():
    """Returns the first configuration file that exists, or None when there is none."""
    for path in _config_paths():
        if path.exists():
            return path
    return None


def _read_user_config(config_path):
    """Parses a configuration file, returning None when it cannot be read."""
    try:
        with open(config_path, "r") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        print(f"[Warning] Failed to load config.yaml: {exc}")
        return None


def _apply_user_config(defaults, user_config):
    """Merges parsed configuration over the defaults, sanitising every value."""
    if not isinstance(user_config, Mapping):
        return False
    defaults.update(user_config)
    extensions = defaults.get("extensions")
    if isinstance(extensions, (str, bytes)) or not isinstance(extensions, (list, tuple, set)):
        print(f"[Warning] Invalid extensions={extensions!r}; falling back to default extensions.")
        defaults["extensions"] = list(DEFAULT_EXTENSIONS)
    else:
        defaults["extensions"] = list(extensions)
    defaults["process_mode"] = _normalize_process_mode(defaults.get("process_mode"))
    defaults["cathar_denoise_method"] = _normalize_cathar_denoise_method(defaults.get("cathar_denoise_method"))
    defaults["cathar_azimuth_method"] = _normalize_cathar_azimuth_method(defaults.get("cathar_azimuth_method"))
    defaults["cathar_enhance_method"] = _normalize_cathar_enhance_method(defaults.get("cathar_enhance_method"))
    defaults["vocal_mix_volume"] = _normalize_mix_volume(defaults.get("vocal_mix_volume"), "vocal_mix_volume")
    defaults["background_mix_volume"] = _normalize_mix_volume(defaults.get("background_mix_volume"), "background_mix_volume")
    _normalize_typed_config_fields(defaults)
    return True


def load_config():
    """Load bundled defaults and apply the optional user configuration."""
    defaults = {
        "vocal_mix_volume": 1.0,
        "background_mix_volume": 1.0,
        "extensions": list(DEFAULT_EXTENSIONS),
        "vocals_model": DEFAULT_VOCALS_MODEL,
        "background_model": DEFAULT_BACKGROUND_MODEL,
        "denoise_model": DEFAULT_DENOISE_MODEL,
        "sync_method": "shift",  # 'shift' or 'dtw'
        "process_mode": DEFAULT_PROCESS_MODE,  # includes aliases: 'multipass', 'pure', 'ffmpeg_native', 'auto_vhs_native'
        "arnndn_model": "cb.rnnn",
        "cathar_denoise_method": "spectral",
        "cathar_azimuth_method": "gcc-phat",
        "cathar_enhance_method": "replicate",
        # The neural model auto_pure_linear runs after subtraction, by file name, when set;
        # empty follows the chain's own choice (the light UVR model, upgraded to the deep
        # one once subtraction has run). A setting, so a candidate can be swept.
        "apl_neural_model": "",
    }
    defaults.update(_typed_config_defaults())
    config_path = _find_config_path()
    if config_path is None:
        return defaults, "Defaults"

    if yaml is None:
        print("[Warning] config.yaml exists but PyYAML is not installed; using defaults.")
        return defaults, "Defaults (PyYAML missing)"

    user_config = _read_user_config(config_path)
    if not user_config:
        return defaults, "Defaults"

    if not _apply_user_config(defaults, user_config):
        return defaults, "Defaults (invalid config.yaml)"
    return defaults, "config.yaml"


CONFIG, CONFIG_SOURCE = load_config()

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
LOG_FILE = Path("session_log.txt")

EXTS = set(CONFIG["extensions"])
KEEP_INPUT_FILES = os.environ.get("AI_RESTORE_TEST_MODE") == "1"
BATCH_JOBS = int(CONFIG.get("batch_jobs", 1))

# Audio mix levels
VOCAL_MIX_VOL = float(CONFIG["vocal_mix_volume"])
BACKGROUND_MIX_VOL = float(CONFIG["background_mix_volume"])


_MODEL_NAME_FORBIDDEN = ("/", "\\", ":", "\0")


def _is_bare_filename(name):
    """True when a model name cannot leave the model store: no separator, drive or dot-prefix."""
    return bool(name) and not name.startswith(".") and not any(ch in name for ch in _MODEL_NAME_FORBIDDEN)


def _model_filename(setting, value, default=""):
    """A UVR model setting reduced to a bare filename inside the model store.

    Model names are joined onto MODELS_DIR (and unlinked from it on a failed load), so a
    path from a planted config.yaml must not be able to escape the store. Only a string
    is a candidate: anything else, or a string with a separator, a parent reference, or a
    leading dot, or a Windows drive prefix (`D:model.ckpt`), falls back to the default.
    Unset and empty fall back silently.
    """
    if value is None or value == "":
        return default
    name = value.strip() if isinstance(value, str) else ""
    if _is_bare_filename(name):
        return name
    print(
        f"[Config] Ignoring '{setting}' value {value!r}: model names must be a bare filename.",
        file=sys.stderr,
    )
    return default


# AI Configs
VOCALS_MODEL = _model_filename("vocals_model", CONFIG["vocals_model"], DEFAULT_VOCALS_MODEL)
BACKGROUND_MODEL = _model_filename("background_model", CONFIG["background_model"], DEFAULT_BACKGROUND_MODEL)
DENOISE_MODEL = _model_filename("denoise_model", CONFIG["denoise_model"], DEFAULT_DENOISE_MODEL)
ADAPTIVE_DENOISE_THRESHOLD_DB = float(CONFIG.get("adaptive_denoise_threshold_db", -50.0))
NEURAL_CHUNK_SECONDS = float(CONFIG.get("neural_chunk_seconds", 0.0))
ENHANCE_NFE = str(CONFIG["enhance_nfe"])
ENHANCE_TAU = str(CONFIG["enhance_tau"])
SYNC_METHOD = CONFIG["sync_method"]
DTW_RESOLUTION = int(CONFIG["dtw_resolution"])
PROCESS_MODE = CONFIG["process_mode"]
ENABLE_MULTIPASS = bool(CONFIG.get("enable_multipass", True))
DEBUG_LOGGING = CONFIG.get("debug_logging", False)

# Native VHS filter configs
AFFTDN_NR = float(CONFIG.get("afftdn_nr", 10.0))
AFFTDN_NF = float(CONFIG.get("afftdn_nf", -55.0))
AFFTDN_TN = bool(CONFIG.get("afftdn_tn", True))
HIGHPASS_FREQ = int(CONFIG.get("highpass_freq", 80))
ENABLE_ADECLICK = bool(CONFIG.get("enable_adeclick", True))
NOTCH_FREQ = float(CONFIG.get("notch_freq", 50.0))

# ARNNDN Speech configs
ARNNDN_MODEL = str(CONFIG.get("arnndn_model", "cb.rnnn"))
ARNNDN_HIGHPASS_FREQ = int(CONFIG.get("arnndn_highpass_freq", 80))
ARNNDN_ENABLE_ADECLICK = bool(CONFIG.get("arnndn_enable_adeclick", True))

# Cathar Restoration Settings
CATHAR_DENOISE_METHOD = str(CONFIG.get("cathar_denoise_method", "spectral"))
CATHAR_ALPHA = float(CONFIG.get("cathar_alpha", 2.5))
CATHAR_BETA = float(CONFIG.get("cathar_beta", 0.01))
CATHAR_ENABLE_COHERENT = bool(CONFIG.get("cathar_enable_coherent", True))
CATHAR_ENABLE_DEWIND = bool(CONFIG.get("cathar_enable_dewind", True))
CATHAR_DEWIND_CUTOFF = int(CONFIG.get("cathar_dewind_cutoff", 80))
CATHAR_ENABLE_AZIMUTH = bool(CONFIG.get("cathar_enable_azimuth", True))
CATHAR_AZIMUTH_METHOD = str(CONFIG.get("cathar_azimuth_method", "gcc-phat"))
CATHAR_AZIMUTH_MAX_MS = float(CONFIG.get("cathar_azimuth_max_ms", 5.0))
CATHAR_ENABLE_DECLICK = bool(CONFIG.get("cathar_enable_declick", True))
CATHAR_DECLICK_THRESHOLD = float(CONFIG.get("cathar_declick_threshold", 8.0))
CATHAR_ENABLE_DECRACKLE = bool(CONFIG.get("cathar_enable_decrackle", True))
CATHAR_DECRACKLE_SENSITIVITY = int(CONFIG.get("cathar_decrackle_sensitivity", 6))
CATHAR_ENABLE_INPAINT = bool(CONFIG.get("cathar_enable_inpaint", True))
CATHAR_INPAINT_MAX_GAP_MS = int(CONFIG.get("cathar_inpaint_max_gap_ms", 50))
CATHAR_INPAINT_ITERATIONS = int(CONFIG.get("cathar_inpaint_iterations", 3))
CATHAR_ENABLE_DECLIP = bool(CONFIG.get("cathar_enable_declip", True))
CATHAR_DECLIP_THRESHOLD = float(CONFIG.get("cathar_declip_threshold", 0.95))
CATHAR_ENABLE_DEHUM = bool(CONFIG.get("cathar_enable_dehum", True))
CATHAR_DEHUM_ADAPTIVE = bool(CONFIG.get("cathar_dehum_adaptive", True))
CATHAR_DEHUM_HARMONICS = int(CONFIG.get("cathar_dehum_harmonics", 8))
CATHAR_ENABLE_REPAIR = bool(CONFIG.get("cathar_enable_repair", True))
CATHAR_REPAIR_STRENGTH = int(CONFIG.get("cathar_repair_strength", 4))
CATHAR_ENABLE_DEWOW = bool(CONFIG.get("cathar_enable_dewow", False))
CATHAR_ENABLE_ENHANCE = bool(CONFIG.get("cathar_enable_enhance", True))
CATHAR_ENHANCE_METHOD = str(CONFIG.get("cathar_enhance_method", "replicate"))
CATHAR_ENABLE_NOISEPRINT = bool(CONFIG.get("cathar_enable_noiseprint", True))
CATHAR_NOISEPRINT_DURATION_S = float(CONFIG.get("cathar_noiseprint_duration_s", 6.0))
CATHAR_ENABLE_MONO_BELOW = bool(CONFIG.get("cathar_enable_mono_below", True))
CATHAR_MONO_BELOW_HZ = int(CONFIG.get("cathar_mono_below_hz", 100))
CATHAR_ENABLE_DEPLOSIVE = bool(CONFIG.get("cathar_enable_deplosive", True))
CATHAR_DEPLOSIVE_STRENGTH = int(CONFIG.get("cathar_deplosive_strength", 4))
CATHAR_ENABLE_DEESSER = bool(CONFIG.get("cathar_enable_deesser", True))
CATHAR_DEESSER_BANDS = int(CONFIG.get("cathar_deesser_bands", 3))
CATHAR_DEESSER_FREQ = int(CONFIG.get("cathar_deesser_freq", 4000))
CATHAR_DEESSER_THRESHOLD = float(CONFIG.get("cathar_deesser_threshold", 6.0))
CATHAR_ENABLE_DEREVERB = bool(CONFIG.get("cathar_enable_dereverb", False))
CATHAR_DEREVERB_WPE = bool(CONFIG.get("cathar_dereverb_wpe", True))
CATHAR_DEREVERB_STRENGTH = float(CONFIG.get("cathar_dereverb_strength", 2.0))
CATHAR_MUSIC_PROFILE = bool(CONFIG.get("cathar_music_profile", True))
CATHAR_MUSIC_PERSISTENCE_MIN = float(CONFIG.get("cathar_music_persistence_min", 0.05))
CATHAR_MUSIC_ALPHA = float(CONFIG.get("cathar_music_alpha", 0.5))
CATHAR_MUSIC_ENABLE_NOISEPRINT = bool(CONFIG.get("cathar_music_enable_noiseprint", False))
CATHAR_MUSIC_ENABLE_COHERENT = bool(CONFIG.get("cathar_music_enable_coherent", False))
CATHAR_MUSIC_ENABLE_DEPLOSIVE = bool(CONFIG.get("cathar_music_enable_deplosive", False))

# Advanced Audio Polish & Archival Configs
ENABLE_DEESSER = bool(CONFIG.get("enable_deesser", True))
ENABLE_LOUDNORM = bool(CONFIG.get("enable_loudnorm", True))
ENABLE_DYNAMIC_EXPANDER = bool(CONFIG.get("enable_dynamic_expander", True))
ENABLE_LINEAR_AIR = bool(CONFIG.get("enable_linear_air", True))
APL_ENABLE_SPECTRAL_DENOISE = bool(CONFIG.get("apl_enable_spectral_denoise", True))
APL_SPECTRAL_ALPHA = float(CONFIG["apl_spectral_alpha"])
APL_SPECTRAL_ALPHA_TONAL = float(CONFIG.get("apl_spectral_alpha_tonal", 2.0))
APL_TONAL_FLATNESS_MAX = float(CONFIG.get("apl_tonal_flatness_max", 0.035))
AUTO_CATHAR_FLATNESS_MAX = float(CONFIG.get("auto_cathar_flatness_max", 0.04))
AUTO_CATHAR_PROBE_SIMILARITY = float(CONFIG.get("auto_cathar_probe_similarity", 0.9))
AUTO_CATHAR_TONAL = bool(CONFIG.get("auto_cathar_tonal", True))
APL_NOISEPRINT_TONAL_S = float(CONFIG.get("apl_noiseprint_tonal_s", 4.0))
APL_TONAL_SKIP_NEURAL = bool(CONFIG.get("apl_tonal_skip_neural", False))
APL_SPECTRAL_MARGIN_DB = float(CONFIG["apl_spectral_margin_db"])
APL_NOISEPRINT_DURATION_S = float(CONFIG.get("apl_noiseprint_duration_s", 4.0))
APL_MUTE_SILENCE_DB = float(CONFIG.get("apl_mute_silence_db", -45.0))
APL_MUTE_MIN_MS = float(CONFIG.get("apl_mute_min_ms", 15.0))
APL_MUTE_MAX_MS = float(CONFIG.get("apl_mute_max_ms", 200.0))
APL_ENABLE_PHYSICAL_REPAIR = bool(CONFIG.get("apl_enable_physical_repair", True))
APL_DEPOP_THRESHOLD = float(CONFIG.get("apl_depop_threshold", 7.0))
APL_USE_DEEPFILTERNET = bool(CONFIG.get("apl_use_deepfilternet", False))
APL_ENABLE_DEHUM = bool(CONFIG.get("apl_enable_dehum", False))
APL_HUM_MIN_EXCESS_DB = float(CONFIG.get("apl_hum_min_excess_db", 6.0))
APL_ENABLE_TONAL_CLEANUP = bool(CONFIG.get("apl_enable_tonal_cleanup", False))
APL_ENABLE_LEARNED_BLEND = bool(CONFIG.get("apl_enable_learned_blend", True))
APL_ENABLE_HUM_CANCEL = bool(CONFIG.get("apl_enable_hum_cancel", True))
APL_SURGICAL_MAINS_NOTCH = bool(CONFIG.get("apl_surgical_mains_notch", True))
APL_HUM_SKIP_NOTCHED = bool(CONFIG.get("apl_hum_skip_notched", False))
APL_HUM_MAX_HARMONICS = int(CONFIG.get("apl_hum_max_harmonics", 40))
APL_HUM_BANDWIDTH_HZ = float(CONFIG.get("apl_hum_bandwidth_hz", 1.5))
APL_USE_NATIVE_SUPPRESS = bool(CONFIG.get("apl_use_native_suppress", False))
APL_SUPPRESS_NOISE_BIAS = float(CONFIG.get("apl_suppress_noise_bias", 1.0))
APL_SUPPRESS_GAIN_FLOOR_DB = float(CONFIG.get("apl_suppress_gain_floor_db", -20.0))
APL_SUPPRESS_DD_ALPHA = float(CONFIG.get("apl_suppress_dd_alpha", 0.96))
APL_ENABLE_PLOSIVE_TAMER = bool(CONFIG.get("apl_enable_plosive_tamer", True))
APL_PLOSIVE_EXCESS_DB = float(CONFIG.get("apl_plosive_excess_db", 12.0))
APL_ENABLE_TONE_CANCEL = bool(CONFIG.get("apl_enable_tone_cancel", False))
APL_USE_RESEMBLE_DENOISE = bool(CONFIG.get("apl_use_resemble_denoise", False))
APL_NEURAL_MODEL = _model_filename("apl_neural_model", CONFIG.get("apl_neural_model"))
LINEAR_AIR_GAIN_DB = float(CONFIG.get("linear_air_gain_db", 2.0))
PRESERVE_ORIGINAL_AUDIO_TRACK = bool(CONFIG.get("preserve_original_audio_track", False))
