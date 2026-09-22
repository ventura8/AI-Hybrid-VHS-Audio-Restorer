"""auto_pure_linear's stem path on music: the chain runs on the voice, the music passes through.

Measured on twelve Internet Archive music clips (`experiments/autotune_music`), the full-mix
chain loses the non-vocal stem where it removes the most noise: the worst octave of the
music stem -16..-30 dB on the mixed clips (cathar's music profile -2.6..-8.5), 8-16 kHz
-15..-30 dB, while the pauses pump by up to +21 dB. A mask-based denoiser and a spectral
subtraction both treat sustained partials as noise once the print is learned from music.

So on material the scanner reads as music (held partials at or above
`apl_music_persistence_min`, the same reading cathar's music profile keys on), the mode
separates the mix with the app's BS-RoFormer, runs its whole chain on the vocal stem only,
sends the background through the mains and CRT notches and a bounded suppressor
(`apl_music_bg_floor_db`: the MMSE-LSA estimator's gain floor, a per-bin removal that lets
sustained partials through; 0 is a pure pass-through), and remixes the two with the shared
stem-mix step. Off by default (`apl_music_stem_path`); the separator failing falls back to
the single-track path.
"""

from pathlib import Path

from .config import APL_MUSIC_BG_FLOOR_DB, APL_MUSIC_PERSISTENCE_MIN, APL_MUSIC_STEM_PATH, APL_NOISEPRINT_TONAL_S, APL_SUPPRESS_DD_ALPHA
from .utils import log_msg


def wanted(strategy):
    """Whether the scanner read the tape as music and the stem path is switched on."""
    persistence = (strategy or {}).get("profile", {}).get("tonal_persistence")
    return bool(APL_MUSIC_STEM_PATH and persistence is not None and float(persistence) >= APL_MUSIC_PERSISTENCE_MIN)


def _background_pass(background_wav, work_dir, total_duration, strategy):
    """The background through the notches and, when a floor is set, the bounded suppressor."""
    from . import processing, spectral_suppress

    notched = processing._post_denoise_cleanup_step(background_wav, work_dir, total_duration=total_duration, strategy=strategy)
    if APL_MUSIC_BG_FLOOR_DB >= 0.0:
        log_msg("    [Stem Path] Background passed through the notches only.")
        return notched
    target = work_dir / f"suppressed_{Path(notched).name}"
    settings = {
        "noise_bias": 1.0,
        "gain_floor_db": APL_MUSIC_BG_FLOOR_DB,
        "dd_alpha": APL_SUPPRESS_DD_ALPHA,
        "probe_s": APL_NOISEPRINT_TONAL_S,
    }
    suppressed = spectral_suppress.suppress_or_none(notched, target, **settings)
    if suppressed is None:
        log_msg("    [Stem Path] Background suppressor found no probe; the notched background is used.")
        return notched
    log_msg(f"    [Stem Path] Background suppressed with a {APL_MUSIC_BG_FLOOR_DB:g} dB floor.")
    return suppressed


def execute(work_dir, clean_wav, original_wav, video_path, final_output_video, video_dur, strategy, denoise_step):
    """Runs the stem path; True when it delivered the output, False when the single-track path must run."""
    from . import processing

    log_msg(f"    [Stem Path] Music: held partials {strategy['profile']['tonal_persistence']:.4f}; the chain runs on the vocal stem.")
    try:
        vocals, background = processing._separate_stems_step(clean_wav, work_dir / "separation", total_duration=video_dur)
    except Exception as exc:  # the separator raises plain Exception
        log_msg(f"    [Stem Path] Separation failed ({exc}); falling back to the single-track path.", is_error=True)
        return False
    vocal_dir, background_dir = work_dir / "denoised_vocals", work_dir / "background"
    vocal_dir.mkdir(parents=True, exist_ok=True)
    background_dir.mkdir(parents=True, exist_ok=True)
    restored_vocals = denoise_step(Path(vocals), vocal_dir, total_duration=video_dur)
    restored_background = _background_pass(Path(background), background_dir, video_dur, strategy)
    processing._align_and_mix_stems(
        work_dir, original_wav, Path(restored_vocals), Path(restored_background), video_path, final_output_video, video_dur, strategy
    )
    return True
