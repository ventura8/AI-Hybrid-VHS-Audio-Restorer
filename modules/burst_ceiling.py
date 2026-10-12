"""After the polish: a per-band ceiling on the hiss the neural stage leaves next to speech.

The Mel-RoFormer takes `auto_pure_linear`'s true pauses to about -93 dB under the source, but
not every frame near speech: on the hiss diagnosis's stage WAVs (2026-10-11, the pause floor
off) 2-11 % of the true-pause frames carry 80 % of the pause's 3-16 kHz energy, and the energy
mean of the pauses stands 50-75 dB over their median. Read per (frame, ERB band) cell against
the source's own band floor (2026-10-12), most of that energy is not hiss: on Tele7abc the
four frames that carry it are a word's tail, the source 10.9 dB over its pause level there and
10-13 dB over its own floor in the 3.2-3.6 kHz bands the output keeps; on abc-primetime the
file's first frame, where the source starts on programme. The hiss the RoFormer passes sits in
the cells where the source holds nothing but its floor: beside that tail (its 4.5-7 kHz bands,
9-15 dB under the source's floor), in the short gaps between words, at the word edges and
inside the pauses.

This stage holds every cell of the restored audio, in the scope `apl_burst_ceiling` names, to
a ceiling read on the source (the pre-conditioned audio the chain started from), at the
restored programme's level:

- where the source holds only hiss, `apl_burst_ceiling_depth_db` under the source's band floor
  (the 10th percentile of the band's energy over +-1 s);
- where the source surely carries programme over its hiss, the source's own envelope plus
  HEADROOM_DB, so the restored cell keeps all of that programme;
- between the two, in power, by how surely it carries programme: its envelope averaged over three
  neighbouring bands from PROGRAMME_DB over the floor, a cell on its own from
  PROGRAMME_CELL_DB, each surely so RAMP_DB further on; a sure programme cell's tail, the cells
  joined to it along time in its band that stand over TAIL_DB, from TAIL_DB. The envelope is
  closed over three blocks (a one-block dip is filled; a word's edge is not moved).

A cell under its ceiling passes untouched, so a restored cell already in the floor is never
lifted and a word's tail keeps what the source shows it carried. 'gaps' acts in the source's
non-speech blocks (the four-band activity detector of `modules/hiss_floor.py`), 'edges' also in
the speech blocks within EDGE_REACH_S of a gap. A band acts only where the neural stage
emptied the gaps: its restored energy over the source's floor, the median over the non-speech
blocks, at or under -`apl_burst_ceiling_depth_db` (a tape the RoFormer passed whole, an advert
or music, keeps every sample), and only where the source's floor stands over 16-bit dither.

Everything is read and applied on one STFT of about 23 ms (a power of two, periodic Hann, hop a
quarter), the readings summed over blocks of four frames (one STFT length), so a burst between
two frames is read whole. The cuts are interpolated from the blocks to the frames, never deeper
than a frame's own block, and spread over three bins; the output is the restored audio plus the
overlap-added STFT of `X * (gain - 1)`: every sample no capped frame reaches is the input's
exactly.

The CPU replay on the diagnosis's stage WAVs is in `docs/configuration.md` ("auto_pure_linear
Burst Ceiling"). Every decision is logged, and written to the event log
(`AI_RESTORE_EVENT_LOG`) as `burst_ceiling__...json` with the blocks capped. With
`apl_burst_ceiling` 'none' (the default) nothing is read and the audio is untouched.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.ndimage
import scipy.signal
import soundfile as sf

from . import event_log, hiss_floor
from .config import APL_BURST_CEILING, APL_BURST_CEILING_DEPTH_DB
from .hygiene import atomic_target
from .impulse_repair import _spans
from .utils import is_valid_audio, log_msg

STAGE = "burst_ceiling"
SCOPES = ("gaps", "edges")
# The source's band floor: the 10th percentile of the band's block energy over +-1 s.
FLOOR_HALF_S = 1.0
FLOOR_PERCENTILE = 10.0
# Where the source carries programme over its hiss. A cell's envelope averaged over BAND_SMOOTH
# neighbouring bands reads 1.5-1.8 / 2.6-2.7 / 3.8-6.6 dB over the p10 floor at the median / p90
# / p99 in the Tata excerpts' gaps (the p99 holds word tails), a single cell 6-7 dB at the p99:
# the neighbourhood counts as programme from PROGRAMME_DB, a cell on its own from
# PROGRAMME_CELL_DB, each surely so RAMP_DB further on (read 4 dB, one hiss cell in ten to a
# hundred passed as programme and kept the burst over it).
PROGRAMME_DB = 6.0
PROGRAMME_CELL_DB = 10.0
RAMP_DB = 3.0
# A word's tail decays through PROGRAMME_DB: a cell whose three-band average stands over TAIL_DB
# above the floor and that joins a sure programme cell along time in its band, through such
# cells, is that programme's tail, its share ramping from TAIL_DB to PROGRAMME_DB. TAIL_DB is the
# hiss's own p95 in the Tata excerpts' gaps away from speech (2.9-3.3 dB). Read as hiss, the tails
# the RoFormer kept at the source's level were cut to the depth (abc-primetime at 3.88 s: five
# 3-5.3 kHz bands 5.4-7.1 dB over the floor, the three whose neighbourhood read under 6 dB cut
# 34-37 dB): 22-48 % of the energy the stage removed on five of the six replayed clips it acted
# on, and the word-edge cells 3-6 dB over the floor lost 3.7-6.1 dB.
TAIL_DB = 3.0
# Joins cells along time only (axis 0), never across bands.
ALONG_TIME = ((0, 1, 0), (0, 1, 0), (0, 1, 0))
# A cell that surely carries programme may keep the source's closed envelope plus this.
HEADROOM_DB = 3.0
# The source's envelope is closed (dilated, then eroded) over this many blocks.
CLOSING_BLOCKS = 3
# ... and averaged over this many neighbouring ERB bands before the programme is read off it.
BAND_SMOOTH = 3
# 'edges' reaches this far into the speech on either side of a gap.
EDGE_REACH_S = 0.15
# The scope's edge is smoothed over this many blocks.
SCOPE_SMOOTH_BLOCKS = 3
# A cell of hiss alone takes the deepest cut over this many blocks (its own and one either side).
HOLD_BLOCKS = 3
# A band's gap background needs at least this much non-speech (a 15 s clip of dense speech
# holds 0.2-0.6 s).
MIN_GAP_S = 0.2
MIN_BLOCKS = 10
# A block counts as capped (the log line, the event spans) when a band loses more than this.
REPORT_CUT_DB = 1.0
# The STFT: about 23 ms (1024 samples at 44.1 kHz), hop a quarter; readings summed over blocks
# of four frames.
STFT_S = 0.023
HOP_DIVISOR = 4
BLOCK_FRAMES = 4
# Blocks read per chunk while streaming (about 24 s at 44.1 kHz), and samples written per block.
CHUNK_BLOCKS = 1024
BLOCK_SAMPLES = 1 << 20
MAX_LENGTH_DIFF_S = 0.5
# Every filter along time or bands repeats its edge value outward, but the floor's percentile,
# which mirrors the blocks at the file's ends: repeated, the last block (partly past the end,
# 1-8 dB under the band's median on the unit test's tape) held the floor up to 6 dB low over the
# last 0.8 s, so the hiss there read as programme and its bursts were kept.
EDGE_MODE = "nearest"
FLOOR_EDGE_MODE = "mirror"
STAGE_FAILURES = (OSError, RuntimeError, ValueError, MemoryError)


class Stft(NamedTuple):
    """The STFT the stage reads and writes in: its size, hop, window and WOLA norm, the ERB edges and the bin layouts.

    `band` is each bin's ERB band (-1 outside 3-16 kHz); `hiss`, `vad` and `programme` sum
    a power spectrum into the ERB bands, the activity detector's four bands and 100-4000 Hz.
    """

    size: int
    hop: int
    window: np.ndarray
    norm: float
    edges: np.ndarray
    band: np.ndarray
    hiss: np.ndarray
    vad: np.ndarray
    programme: np.ndarray


class Plan(NamedTuple):
    """What the stage cuts: `cut_db` per block and band (0 or negative), with the readings behind it."""

    cut_db: np.ndarray
    stft: Stft
    rate: int
    active: np.ndarray
    background_db: np.ndarray
    programme_gain_db: float


def db(power):
    """Power in dB, floored so silence reads as a number."""
    return 10.0 * np.log10(np.asarray(power, dtype=np.float64) + hiss_floor.EPS)


def stft_for(rate):
    """The STFT at `rate`: about STFT_S long (a power of two), periodic Hann, hop a quarter, and its bin layouts."""
    size = int(2 ** round(np.log2(STFT_S * rate)))
    window = scipy.signal.get_window("hann", size)
    hop = size // HOP_DIVISOR
    freqs = np.fft.rfftfreq(size, 1.0 / rate)
    edges = hiss_floor.erb_band_edges(hiss_floor.HISS_LO_HZ, min(hiss_floor.HISS_HI_HZ, rate / 2.0 - hiss_floor.NYQUIST_MARGIN_HZ))
    band = np.searchsorted(edges, freqs, side="right") - 1
    band[(freqs < edges[0]) | (freqs >= edges[-1])] = -1
    hiss = (band[:, None] == np.arange(len(edges) - 1)[None, :]).astype(np.float64)
    vad = np.stack([(freqs >= lo) & (freqs < hi) for lo, hi in hiss_floor.VAD_BANDS_HZ], axis=1).astype(np.float64)
    programme = (freqs >= hiss_floor.PROGRAMME_BAND_HZ[0]) & (freqs < hiss_floor.PROGRAMME_BAND_HZ[1])
    return Stft(size, hop, window, float(np.sum(window**2) / hop), edges, band, hiss, vad, programme)


def block_samples(stft):
    """Samples per analysis block: BLOCK_FRAMES hops (one STFT length)."""
    return BLOCK_FRAMES * stft.hop


def padded(handle, start, stop):
    """Samples [start, stop) of an open file as float64, zeros outside the file."""
    out = np.zeros((stop - start, handle.channels))
    lo, hi = max(start, 0), min(stop, handle.frames)
    if hi > lo:
        handle.seek(lo)
        out[slice(lo - start, hi - start)] = handle.read(hi - lo, dtype="float32", always_2d=True)
    return out


def spectra(audio, centres, base, stft):
    """The windowed spectra of the channel mean at frame `centres` (samples); `audio` starts at sample `base`."""
    mono = audio.mean(axis=1)
    index = (centres - stft.size // 2 - base)[:, None] + np.arange(stft.size)[None, :]
    return np.fft.rfft(mono[index] * stft.window, axis=1)


def chunk_stats(handles, first, count, stft):
    """Block readings for `count` blocks from block `first` of the (reference, restored) pair, summed over their frames.

    Both files' ERB band powers, the source's detector bands, its energy (the loud-block
    pick) and the 100-4000 Hz projection terms `programme_match` reads.
    """
    reference, restored = handles
    centres = (first * BLOCK_FRAMES + np.arange(count * BLOCK_FRAMES)) * stft.hop
    base, stop = int(centres[0]) - stft.size // 2, int(centres[-1]) + stft.size // 2
    src = spectra(padded(reference, base, stop), centres, base, stft)
    out = spectra(padded(restored, base, stop), centres, base, stft)
    src_power, out_power = np.abs(src) ** 2, np.abs(out) ** 2
    band = stft.programme
    frames = {
        "src_bands": src_power @ stft.hiss,
        "out_bands": out_power @ stft.hiss,
        "vad": src_power @ stft.vad,
        "level": src_power.sum(axis=1),
        "cross": np.sum((out[:, band] * np.conj(src[:, band])).real, axis=1),
        "out_power": out_power[:, band].sum(axis=1),
        "src_power": src_power[:, band].sum(axis=1),
    }
    return {key: value.reshape(count, BLOCK_FRAMES, *value.shape[1:]).sum(axis=1) for key, value in frames.items()}


def pair_problem(reference, restored):
    """Why the two open files cannot be read together, or None."""
    if reference.samplerate != restored.samplerate:
        return "the restored audio's rate differs from the source's"
    if abs(reference.frames - restored.frames) > MAX_LENGTH_DIFF_S * restored.samplerate:
        return "reference and restored audio differ in length"
    return None


def block_stats(reference_wav, restored_wav):
    """Per-block readings of the pair over the restored file's length, with the STFT; or a string saying why there are none."""
    with sf.SoundFile(str(reference_wav)) as reference, sf.SoundFile(str(restored_wav)) as restored:
        reason = pair_problem(reference, restored)
        if reason is not None:
            return reason
        stft = stft_for(restored.samplerate)
        blocks = -(-restored.frames // block_samples(stft))
        if blocks < MIN_BLOCKS:
            return "too short to read"
        return {**read_blocks((reference, restored), blocks, stft), "rate": restored.samplerate, "stft": stft}


def read_blocks(handles, blocks, stft):
    """`chunk_stats` over the first `blocks` blocks of the open pair, CHUNK_BLOCKS at a time, joined."""
    parts = [chunk_stats(handles, first, min(CHUNK_BLOCKS, blocks - first), stft) for first in range(0, blocks, CHUNK_BLOCKS)]
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def band_floor(src_bands, block_s):
    """The source's floor per block and band: the FLOOR_PERCENTILE of the band over +-FLOOR_HALF_S."""
    half = max(1, int(round(FLOOR_HALF_S / block_s)))
    return scipy.ndimage.percentile_filter(src_bands, FLOOR_PERCENTILE, size=(2 * half + 1, 1), mode=FLOOR_EDGE_MODE)


def closed_envelope(src_bands):
    """The source's band energy closed over CLOSING_BLOCKS along time: one-block dips filled, edges kept."""
    size = (CLOSING_BLOCKS, 1)
    return scipy.ndimage.minimum_filter(scipy.ndimage.maximum_filter(src_bands, size=size, mode=EDGE_MODE), size=size, mode=EDGE_MODE)


def programme_share(envelope, floor):
    """Per cell, 0 to 1: how surely the source carries programme over its hiss there.

    Read on the closed envelope twice: averaged over BAND_SMOOTH neighbouring bands (the floor
    averaged alike) from PROGRAMME_DB, and on the cell alone from PROGRAMME_CELL_DB, each rising
    to 1 over RAMP_DB; the larger counts. A cell's hiss fluctuates on its own while a tail or a
    breath spans its neighbours, so one band's chance peak in a gap is not programme, and a
    narrow partial the neighbourhood would dilute still is. The programme's tails keep a share
    down to TAIL_DB (`tail_share`).
    """
    spread = db(scipy.ndimage.uniform_filter1d(envelope, size=BAND_SMOOTH, axis=1, mode=EDGE_MODE))
    spread -= db(scipy.ndimage.uniform_filter1d(floor, size=BAND_SMOOTH, axis=1, mode=EDGE_MODE))
    alone = db(envelope) - db(floor)
    share = np.clip(np.maximum(spread - PROGRAMME_DB, alone - PROGRAMME_CELL_DB) / RAMP_DB, 0.0, 1.0)
    return tail_share(share, spread)


def tail_share(share, spread):
    """`share` with its programme's tails: the cells over TAIL_DB joined along time, in their band, to a sure programme cell.

    Such a cell takes at least its spread's ramp from TAIL_DB (0) to PROGRAMME_DB (1); a cell of
    the same level that no sure programme cell (share 1) reaches through its band stays hiss, so a
    chance peak of the hiss (the p99 of a gap's spread stands at 3.8-6.6 dB) starts no tail.
    """
    seed = share >= 1.0
    labels, _count = scipy.ndimage.label(seed | (spread > TAIL_DB), structure=ALONG_TIME)
    joined = np.isin(labels, labels[seed])
    ramp = np.clip((spread - TAIL_DB) / (PROGRAMME_DB - TAIL_DB), 0.0, 1.0)
    return np.where(joined, np.maximum(share, ramp), share)


def ceiling_db(src_bands, floor, depth_db):
    """Per cell, `(ceiling, share)`: the most the restored cell may hold (dB) and the source's programme share.

    depth_db under the source's floor where the source holds only hiss, its closed envelope plus
    HEADROOM_DB where it surely carries programme, and between the two by the share, in power: a
    cell half sure of its programme keeps half of it (3 dB under), not a level halfway in dB
    (which cut the 10-15 dB tails of a replayed clip by 16-24 dB).
    """
    envelope = closed_envelope(src_bands)
    share = programme_share(envelope, floor)
    hiss = floor * 10.0 ** (-depth_db / 10.0)
    return db(hiss + share * (envelope * 10.0 ** (HEADROOM_DB / 10.0) - hiss)), share


def scope_weight(speech, scope, block_s):
    """Per block, how much the stage acts: 1 in the scope's blocks, smoothed over SCOPE_SMOOTH_BLOCKS."""
    gaps = ~speech
    if scope == "edges":
        reach = max(1, int(round(EDGE_REACH_S / block_s)))
        gaps = scipy.ndimage.maximum_filter1d(gaps.astype(np.uint8), size=2 * reach + 1) > 0
    kernel = np.ones(SCOPE_SMOOTH_BLOCKS) / SCOPE_SMOOTH_BLOCKS
    return np.convolve(gaps.astype(np.float64), kernel, mode="same")


def gap_background_db(restored_bands, floor, speech, block_s):
    """Per band, the restored energy over the source's floor in dB, the median over the non-speech blocks; None with too few."""
    gaps = ~speech
    if gaps.sum() * block_s < MIN_GAP_S:
        return None
    return db(np.median(restored_bands[gaps] / (floor[gaps] + hiss_floor.EPS), axis=0))


def readable_db(stft):
    """Per band, the block energy the source's floor must stand over: 16-bit TPDF dither in the band plus the margin."""
    dither = (hiss_floor.QUANT_LSB**2 / 4.0) * float(np.sum(stft.window**2)) * BLOCK_FRAMES
    return db(dither * stft.hiss.sum(axis=0)) + hiss_floor.DITHER_MARGIN_DB


def active_bands(background_db, floor, stft, depth_db):
    """The bands the stage acts in: the gaps emptied at least depth_db under the floor, the floor over 16-bit dither."""
    readable = db(np.median(floor, axis=0)) > readable_db(stft)
    return readable & (background_db <= -depth_db)


def cut_db(restored_bands, ceiling, share):
    """Per block and band, the dB that brings the restored cell to its ceiling (0 where it is under).

    A cell where the source holds only hiss (share 0) takes the deepest cut of its block and the
    blocks either side (HOLD_BLOCKS), so a burst's onset and tail, which share a block with
    silence, are cut with it; a cell with programme keeps its own cut.
    """
    over = np.minimum(0.0, ceiling - db(restored_bands))
    held = scipy.ndimage.minimum_filter1d(over, size=HOLD_BLOCKS, axis=0, mode=EDGE_MODE)
    return np.where(share > 0.0, over, held)


def matched_gain(stats):
    """The amplitude gain that puts the restored programme at the source's level, or None when the pair does not match."""
    gain, correlation = hiss_floor.programme_match(stats)
    if correlation < hiss_floor.MIN_CORRELATION or gain <= 0.0:
        return None
    return gain


def plan_cuts(stats, gain, scope, depth_db):
    """The plan from a matched pair's block readings, or the reason there is none."""
    stft, rate = stats["stft"], stats["rate"]
    block_s = block_samples(stft) / rate
    speech = hiss_floor.programme_frames(stats["vad"], rate, block_samples(stft))
    restored = stats["out_bands"] * gain**2
    source = stats["src_bands"]
    floor = band_floor(source, block_s)
    background = gap_background_db(restored, floor, speech, block_s)
    if background is None:
        return "too few gaps to read the restored floor in"
    active = active_bands(background, floor, stft, depth_db)
    if not active.any():
        return f"the neural stage left every band's gaps within {depth_db:g} dB of the source's floor"
    cuts = cut_db(restored, *ceiling_db(source, floor, depth_db))
    scoped = cuts * scope_weight(speech, scope, block_s)[:, None] * active[None, :]
    return Plan(scoped, stft, rate, active, background, float(20.0 * np.log10(gain)))


def plan(reference_wav, restored_wav, scope, depth_db):
    """The cuts for one restored file against the source it was restored from, or a string saying why there are none."""
    stats = block_stats(reference_wav, restored_wav)
    if isinstance(stats, str):
        return stats
    gain = matched_gain(stats)
    if gain is None:
        return "the restored programme does not follow the source's"
    result = plan_cuts(stats, gain, scope, depth_db)
    if isinstance(result, Plan) and not (result.cut_db < -REPORT_CUT_DB).any():
        return "nothing stands over the ceiling"
    return result


def frame_cuts(the_plan, centres):
    """The band cuts (dB) at STFT frame centres (samples), interpolated between the blocks' centres.

    No frame is cut deeper than the block its reading belongs to: interpolated toward a deep
    neighbour, the last frame of a word's block lost 0.375 of that neighbour's cut (13 dB beside
    a 35 dB cut) of the programme it read.
    """
    hop = the_plan.stft.hop
    blocks = the_plan.cut_db.shape[0]
    block_centres = (np.arange(blocks) * BLOCK_FRAMES + (BLOCK_FRAMES - 1) / 2.0) * hop
    columns = [np.interp(centres, block_centres, the_plan.cut_db[:, band]) for band in range(the_plan.cut_db.shape[1])]
    own = the_plan.cut_db[np.clip(np.asarray(centres) // (BLOCK_FRAMES * hop), 0, blocks - 1)]
    return np.maximum(np.stack(columns, axis=1), own)


def bin_gains(cuts, stft):
    """Per frame and bin, the linear gain minus one: the band's cut, spread over three bins inside 3-16 kHz, 0 outside."""
    inside = stft.band >= 0
    per_bin = np.zeros((cuts.shape[0], len(stft.band)))
    per_bin[:, inside] = scipy.ndimage.uniform_filter1d(cuts[:, stft.band[inside]], size=3, axis=1, mode=EDGE_MODE)
    return 10.0 ** (per_bin / 20.0) - 1.0


def frames_over(start, stop, stft):
    """The STFT frame centres (samples) whose windows reach into [start, stop)."""
    half = stft.size // 2
    first = -(-(start - half + 1) // stft.hop)
    last = (stop + half - 1) // stft.hop
    return np.arange(first, last + 1) * stft.hop


def correction(handle, span, the_plan):
    """The overlap-added correction for samples [start, stop): each capped frame's spectrum times (gain - 1), resynthesised."""
    start, stop = span
    stft = the_plan.stft
    centres = frames_over(start, stop, stft)
    gains = bin_gains(frame_cuts(the_plan, centres), stft)
    # A frame needs a correction only where some bin's gain moved; |g| > 0 is that test for finite gains
    # without an equality on floats (Sonar S1244).
    capped = np.flatnonzero(np.any(np.abs(gains) > 0.0, axis=1))
    base = start - stft.size
    out = np.zeros((stop - start + 2 * stft.size, handle.channels))
    if capped.size:
        audio = padded(handle, base, stop + stft.size)
        for index in capped:
            frame = slice(int(centres[index]) - stft.size // 2 - base, int(centres[index]) + stft.size // 2 - base)
            spectrum = np.fft.rfft(audio[frame] * stft.window[:, None], axis=0) * gains[index][:, None]
            out[frame] += np.fft.irfft(spectrum, n=stft.size, axis=0) * stft.window[:, None] / stft.norm
    return out[slice(stft.size, stft.size + stop - start)]


def write_capped(restored_wav, target_wav, the_plan):
    """Writes `restored + correction` block by block; samples no capped frame reaches are copied exactly."""
    with (
        atomic_target(target_wav) as partial,
        sf.SoundFile(str(restored_wav)) as restored,
        sf.SoundFile(str(restored_wav)) as context,
        sf.SoundFile(str(partial), "w", samplerate=restored.samplerate, channels=restored.channels, subtype="FLOAT") as out,
    ):
        for start in range(0, restored.frames, BLOCK_SAMPLES):
            block = restored.read(BLOCK_SAMPLES, dtype="float32", always_2d=True)
            fix = correction(context, (start, start + len(block)), the_plan)
            out.write((block.astype(np.float64) + fix).astype(np.float32))
    return Path(target_wav)


def _target_for(restored_wav, audio_dir):
    """Where the capped audio goes: the scope and depth in the name, so a render at other settings is never reused."""
    output_dir = Path(audio_dir) / STAGE
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"capped_{APL_BURST_CEILING}{APL_BURST_CEILING_DEPTH_DB:g}_{Path(restored_wav).name}"


def _settings():
    """The settings every record of the stage carries."""
    return {
        "scope": APL_BURST_CEILING,
        "depth_db": APL_BURST_CEILING_DEPTH_DB,
        "programme_db": PROGRAMME_DB,
        "programme_cell_db": PROGRAMME_CELL_DB,
        "headroom_db": HEADROOM_DB,
    }


def capped_spans(the_plan):
    """The sample spans of the blocks that lose more than REPORT_CUT_DB in some band."""
    size = block_samples(the_plan.stft)
    capped = np.any(the_plan.cut_db < -REPORT_CUT_DB, axis=1)
    return [(start * size, stop * size) for start, stop in _spans(capped)]


def _record(restored_wav, the_plan, spans):
    """The cuts and their readings in the event log, when the log is on."""
    edges = the_plan.stft.edges
    bands = [[float(lo), float(hi)] for lo, hi, on in zip(edges[:-1], edges[1:], the_plan.active) if on]
    details = {
        **_settings(),
        "active_bands_hz": bands,
        "background_db": [round(float(value), 1) for value in the_plan.background_db],
        "programme_gain_db": the_plan.programme_gain_db,
    }
    event_log.write_spans(STAGE, restored_wav, spans, the_plan.rate, **details)


def cap(reference_wav, restored_wav, audio_dir):
    """Runs the stage; returns the capped audio, or `restored_wav` with the reason logged."""
    the_plan = plan(reference_wav, restored_wav, APL_BURST_CEILING, APL_BURST_CEILING_DEPTH_DB)
    if isinstance(the_plan, str):
        log_msg(f"    [Burst Ceiling] Skipped: {the_plan}.")
        event_log.write_skip(STAGE, restored_wav, the_plan, **_settings())
        return Path(restored_wav)
    spans = capped_spans(the_plan)
    _record(restored_wav, the_plan, spans)
    target = _target_for(restored_wav, audio_dir)
    if is_valid_audio(target):
        log_msg(f"    [Burst Ceiling] Reusing {target.name}.")
        return target
    seconds = sum(stop - start for start, stop in spans) / the_plan.rate
    log_msg(
        f"    [Burst Ceiling] Capping {seconds:.2f} s ({APL_BURST_CEILING}) in {int(the_plan.active.sum())} bands"
        f" at {APL_BURST_CEILING_DEPTH_DB:g} dB under the source's floor..."
    )
    return write_capped(restored_wav, target, the_plan)


def apply_when_needed(reference_wav, restored_wav, audio_dir, stages):
    """The restored audio after the ceiling when the caller opted in and `apl_burst_ceiling` names a scope; else untouched.

    `reference_wav` is the pre-conditioned audio the chain started from; `stages` the caller's
    post-neural switches (`burst_ceiling`). With the key at 'none' nothing is read.
    """
    if not stages.get("burst_ceiling") or APL_BURST_CEILING not in SCOPES:
        return restored_wav
    try:
        return cap(reference_wav, restored_wav, audio_dir)
    except STAGE_FAILURES as exc:
        log_msg(f"    [Burst Ceiling] Skipped after failure: {exc}")
        event_log.write_skip(STAGE, restored_wav, f"failed: {exc}", **_settings())
        return restored_wav
