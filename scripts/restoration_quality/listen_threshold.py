"""The absolute-threshold run: a listener's thresholds in dBFS at the session's device and volume, under a hard level cap.

Why: the audibility check places its threshold in quiet by assuming the incumbent's loud frames
play at 65 dB SPL (`auditory.PLAYBACK_SPL_LOUD_FRAMES`), and Session 0's listener heard none of
the restoration differences the check called audible (docs/ear_v3_round0.md). The level is the
first suspect: a pause-hiss change of about 5 dB under the threshold in quiet is not heard at
any switching speed. This run measures where this listener's threshold sits in dBFS at the
recorded playback, so the check can stop assuming.

How: a three-interval forced choice. Each trial plays A, B and C, one second each and looped
on the page (keys 1, 2, 3 switch between them), one of them holding a 300 ms tone burst (20 ms
raised-cosine ramps) at 1, 2, 4, 8 or 12.5 kHz; the listener says which, guessing when unsure:
the trial offers no "can't tell". The first run (2026-10-25, the session1_v9 listener) offered
one, and 45 of its 60 trials were "can't tell" at the first level not heard: such an answer
leaves the posterior where it is, so the next trial played the same level and the run stalled
one step under the last tone heard, every 90% band about 45 dB wide. The psychometric
function's 1/3 guess rate is what a forced guess gives, so a guess, not an abstention, is what
moves the posterior towards the threshold. Per frequency a QUEST
posterior over the threshold (Weibull in dB with slope 3.5, guess 1/3, lapse 0.02, the blend
mode's shape) places each trial at its median, run on the server like `listen_modes.BlendMode`
so a simulated observer tests it. The prior is the check's own assumption (speech at the
session level playing at 65 dB SPL, Terhardt's threshold in quiet), 20 dB wide. 12 trials per
frequency by default (60 in all, the frequencies interleaved), after `listen_protocol`'s
familiarisation (each tone once at the cap, with feedback) and with its catch trials (a tone at
the cap). Three intervals, not two: simulated 2026-10-12 (400 runs each, the prior right or
15 dB off), 12 trials per frequency leave a median error of 1.1-1.6 dB with 90% of runs within
4.4-5.4 dB, where two intervals and 10 trials leave 1.7-2.2 dB and 8.3-11.8 dB: at a guess rate
of 1/2 a run of lucky guesses below threshold reads as hearing (one simulated run put a
threshold 25 dB too low). A tone the listener cannot hear under the cap read above the cap in
81% of three-interval runs and 45% of two-interval ones; the at-cap rule below catches more.
The level the check takes is the median over the five frequencies, which one stray frequency
does not move.

Hearing safety: no tone passes `cap_dbfs(level)`, -33 dBFS RMS or 10 dB under the speech level
the session's blocks play at, whichever is lower. If the speech is comfortable, no tone is
louder than speech 10 dB down. The posterior may place a threshold above the cap; the trial is
then played at the cap. A frequency is reported `censored` (not heard reliably at the cap) when
its 75% point lies above the cap, or when at least `CENSOR_MIN_TRIALS` trials played at the cap
were answered right less than 75% of the time: above the cap the answers cannot move the
posterior (every level there is guessed alike), so its median alone can stay just under the cap
for a listener who hears nothing.

What it gives: `sl_map`, the dBFS of 0 dB sensation level per frequency at this device and
volume (a sound at L dBFS sits L - sl_map[hz] dB above threshold), and one playback level for the
check: assuming the listener's threshold is Terhardt's, a 0 dBFS tone plays at
`threshold_in_quiet_db(hz) - threshold_dbfs` dB SPL; the median over the uncensored frequencies
is `spl_full_scale_db`, and the loud frames play at that plus the session level plus
`LOUD_OVER_ACTIVE_DB` (loud frames read 1.1-2.1 dB over the speech-active level on the Session 0
and 1 restorations, measured 2026-10-12). The error budget is 7-10 dB (Masalski et al. 2014,
JMIR 16(1):e11: thresholds on a biologically calibrated PC against clinical audiometry,
SD 7.27-10.38 dB), dominated by the normal-hearing assumption; the per-frequency map itself
repeats within about 4-5 dB (their test-retest SD 3.87-4.97 dB). Thresholds near -96 dBFS meet
a 16-bit output's floor, so the device's format belongs in the record (`--connection`).
"""

import math

import numpy as np

from scripts.restoration_quality import audio_io, auditory, listen_modes

FREQUENCIES_HZ = (1000.0, 2000.0, 4000.0, 8000.0, 12500.0)
TONE_CAP_DBFS = -33.0
TONE_HEADROOM_DB = 10.0
GRID_DBFS = np.arange(-140.0, -9.75, 0.5)
SLOPE = 3.5
GUESS = 1.0 / 3.0
INTERVALS = ("A", "B", "C")
LAPSE = 0.02
TARGET_P = 0.75
PRIOR_SD_DB = 20.0
BAND_QUANTILES = (0.05, 0.95)
BURST_S = 0.3
RAMP_S = 0.02
PERIOD_S = 1.0
RATE = audio_io.NATIVE_RATE
LOUD_OVER_ACTIVE_DB = 1.5
ERROR_BUDGET_DB = (7.0, 10.0)
CENSOR_MIN_TRIALS = 3
CATCH_TONE = "catch:tone"
PROMPT = "Which of A, B and C holds the tone? When you hear none, guess: every answer moves the run."
# The 75%-correct point lies this far from the Weibull's threshold parameter.
OFFSET_75_DB = 20.0 / SLOPE * math.log10(-math.log(1.0 - (TARGET_P - GUESS) / (1.0 - GUESS - LAPSE)))


def cap_dbfs(level_dbfs):
    """The hard cap on every tone: -33 dBFS RMS, or 10 dB under the session's speech level when that is lower."""
    return min(TONE_CAP_DBFS, float(level_dbfs) - TONE_HEADROOM_DB)


def p_right(level_dbfs, threshold_dbfs):
    """Probability of a right answer to a tone at `level_dbfs` for a listener whose threshold parameter is `threshold_dbfs`."""
    detect = 1.0 - np.exp(-(10.0 ** (SLOPE * (level_dbfs - np.asarray(threshold_dbfs)) / 20.0)))
    return GUESS + (1.0 - GUESS - LAPSE) * detect


def burst_envelope(rate=RATE):
    """The burst's envelope: BURST_S long, RAMP_S raised-cosine ramps at both ends."""
    rise = 0.5 - 0.5 * np.cos(np.pi * np.arange(int(RAMP_S * rate)) / int(RAMP_S * rate))
    flat = np.ones(int(BURST_S * rate) - 2 * len(rise))
    return np.concatenate([rise, flat, rise[::-1]])


def tone_clip(freq_hz, level_dbfs, cap, rate=RATE):
    """A PERIOD_S stereo clip with one BURST_S tone at `level_dbfs` RMS (never above `cap`) in its middle."""
    amplitude = math.sqrt(2.0) * 10.0 ** (min(float(level_dbfs), float(cap)) / 20.0)
    envelope = burst_envelope(rate)
    burst = amplitude * envelope * np.sin(2.0 * np.pi * freq_hz * np.arange(len(envelope)) / rate)
    lead = (int(PERIOD_S * rate) - len(envelope)) // 2
    tail = int(PERIOD_S * rate) - len(envelope) - lead
    mono = np.concatenate([np.zeros(lead), burst, np.zeros(tail)]).astype(np.float32)
    return np.repeat(mono[:, None], 2, axis=1)


def silent_clip(rate=RATE):
    """The interval without the tone."""
    return np.zeros((int(PERIOD_S * rate), 2), dtype=np.float32)


def prior_threshold_dbfs(freq_hz, level_dbfs):
    """Where the check's own assumption puts the 75% point: speech at `level_dbfs` playing at 65 dB SPL, Terhardt's threshold."""
    full_scale_spl = auditory.DEFAULT_PLAYBACK_SPL_LOUD_FRAMES - float(level_dbfs) - LOUD_OVER_ACTIVE_DB
    return float(auditory.threshold_in_quiet_db(freq_hz)) - full_scale_spl


def prior_log_posterior(freq_hz, level_dbfs):
    """The log prior over the threshold parameter at one frequency: Gaussian, PRIOR_SD_DB wide, on the check's assumption."""
    mean = prior_threshold_dbfs(freq_hz, level_dbfs) - OFFSET_75_DB
    return -0.5 * ((GRID_DBFS - mean) / PRIOR_SD_DB) ** 2


def frequency_label(freq_hz):
    """A frequency as the record names it (`1000Hz`)."""
    return f"{float(freq_hz):g}Hz"


def placed_level(entries, level_dbfs):
    """`(spl_full_scale_db, loud_frames_spl)` from the uncensored entries' SPL offsets; `(None, None)` when every one is censored."""
    heard = [entry["spl_offset_db"] for entry in entries if not entry["censored"]]
    if not heard:
        return None, None
    full_scale = float(np.median(heard))
    return full_scale, full_scale + float(level_dbfs) + LOUD_OVER_ACTIVE_DB


def run_context(entries, full_scale):
    """The calibration record's `context` from the run itself: the tone, the prior, the model and each frequency's residual."""
    residuals = {} if full_scale is None else {f"{e['hz']:g}": e["spl_offset_db"] - full_scale for e in entries if not e["censored"]}
    return {
        "tone": {"burst_s": BURST_S, "ramp_s": RAMP_S, "period_s": PERIOD_S, "rms": True},
        "prior": {"sd_db": PRIOR_SD_DB, "assumed_loud_frames_spl": auditory.DEFAULT_PLAYBACK_SPL_LOUD_FRAMES},
        "psychometric": {"slope": SLOPE, "guess": GUESS, "lapse": LAPSE, "target_p": TARGET_P},
        "loud_over_active_db": LOUD_OVER_ACTIVE_DB,
        "error_budget_db": list(ERROR_BUDGET_DB),
        "residuals_db": residuals,
    }


def sl_map(entries):
    """`{hz: dBFS at 0 dB sensation level}`, null where the cap was not heard reliably."""
    return {f"{entry['hz']:g}": None if entry["censored"] else entry["threshold_dbfs"] for entry in entries}


def censored(point, responses, cap):
    """Not heard reliably at the cap: the 75% point above it, or CENSOR_MIN_TRIALS or more trials there under 75% right."""
    at_cap = [right for level, right in responses if level >= cap]
    return point > cap or (len(at_cap) >= CENSOR_MIN_TRIALS and sum(at_cap) < TARGET_P * len(at_cap))


def split_answers(answers):
    """`(responses [[level, right], ...], "can't tell" levels)` of one frequency's answers."""
    responses, unsure = [], []
    for answer in answers:
        if answer["choice"] == listen_modes.CANT_TELL:
            unsure.append(answer["x"])
        else:
            responses.append([answer["x"], answer["choice"] == answer["key"]])
    return responses, unsure


def interleaved(count, trials, rng):
    """Frequency indices for `trials` trials: each round a fresh permutation of the `count` frequencies."""
    rounds = math.ceil(trials / count)
    return [int(index) for _ in range(rounds) for index in rng.permutation(count)][:trials]


class ThresholdMode(listen_modes.Mode):
    """Three-interval forced-choice tone detection at each frequency, QUEST-placed under the hard cap."""

    kind = "threshold"
    keys = INTERVALS

    def __init__(self, trials, rng, level_dbfs, frequencies=FREQUENCIES_HZ):
        self.frequencies = list(map(float, frequencies))
        super().__init__(list(map(frequency_label, self.frequencies)), trials)
        self.level_dbfs = float(level_dbfs)
        self.cap = cap_dbfs(level_dbfs)
        self.order = interleaved(len(self.frequencies), trials, rng)
        self.slots = [str(key) for key in rng.choice(INTERVALS, trials)]
        self.log_posterior = [prior_log_posterior(hz, level_dbfs) for hz in self.frequencies]

    def quantile(self, index, share):
        """The threshold parameter at `share` of frequency `index`'s posterior."""
        posterior = np.exp(self.log_posterior[index] - self.log_posterior[index].max())
        cdf = np.cumsum(posterior) / posterior.sum()
        return float(GRID_DBFS[min(int(np.searchsorted(cdf, share)), len(GRID_DBFS) - 1)])

    def next_level(self, index):
        """The posterior median of frequency `index`, never above the cap."""
        return min(self.cap, self.quantile(index, 0.5))

    def tone_trial(self, index, level, key):
        """A trial at frequency `index`: the tone in the interval `key` names, silence in the other two."""
        hz = self.frequencies[index]
        tone, silence = tone_clip(hz, level, self.cap), silent_clip()
        play = [(name, tone if name == key else silence) for name in INTERVALS]
        choices = list(INTERVALS)
        return {
            "prompt": PROMPT,
            "play": play,
            "choices": choices,
            "key": key,
            "x": min(level, self.cap),
            "hz": hz,
            "stimuli": [self.labels[index]],
        }

    def trial(self, index):
        """Scored trial `index`."""
        frequency = self.order[index]
        return self.tone_trial(frequency, self.next_level(frequency), self.slots[index])

    def familiar_trial(self, index, key):
        """Familiarisation: each frequency in turn at the cap."""
        return self.tone_trial(index % len(self.frequencies), self.cap, key)

    def catch_trial(self, _label, key):
        """A catch trial: the cap at the frequency whose threshold sits lowest so far."""
        lowest = int(np.argmin([self.quantile(index, 0.5) for index in range(len(self.frequencies))]))
        return self.tone_trial(lowest, self.cap, key)

    def record(self, trial, choice):
        """Keeps the answer and moves its frequency's posterior; a "can't tell" (offered by no trial since 2026-10-25, a
        record written before still parses) leaves it."""
        super().record(trial, choice)
        self.answers[-1]["hz"] = trial["hz"]
        if choice == listen_modes.CANT_TELL:
            return
        index = self.frequencies.index(trial["hz"])
        right = p_right(trial["x"], GRID_DBFS)
        self.log_posterior[index] += np.log(right if choice == trial["key"] else 1.0 - right)

    def threshold(self, index):
        """One frequency's entry: the 75% point, its 90% band, censored or not, the SPL offset and every response."""
        hz = self.frequencies[index]
        point = self.quantile(index, 0.5) + OFFSET_75_DB
        band = [self.quantile(index, share) + OFFSET_75_DB for share in BAND_QUANTILES]
        responses, cant_tell = split_answers([answer for answer in self.answers if answer["hz"] == hz])
        return {
            "hz": hz,
            "threshold_dbfs": point,
            "band_dbfs": band,
            "censored": censored(point, responses, self.cap),
            "spl_offset_db": float(auditory.threshold_in_quiet_db(hz)) - point,
            "responses": responses,
            "cant_tell": cant_tell,
        }

    def results(self):
        """The calibration record's body: the map, the thresholds and the playback level they place."""
        entries = [self.threshold(index) for index in range(len(self.frequencies))]
        full_scale, loud = placed_level(entries, self.level_dbfs)
        body = {"level_dbfs": self.level_dbfs, "cap_dbfs": self.cap, "sl_map": sl_map(entries), "thresholds": entries}
        body.update(spl_full_scale_db=full_scale, loud_frames_spl=loud, context=run_context(entries, full_scale))
        return [body]

    def summary(self):
        """One line: each threshold (or above the cap) and the playback level they place."""
        body = self.results()[0]
        parts = [threshold_text(entry) for entry in body["thresholds"]]
        level = body["loud_frames_spl"]
        placed = "no level placed (no tone heard under the cap)" if level is None else f"loud frames at {level:.1f} dB SPL (+-7-10 dB)"
        return f"thresholds: {', '.join(parts)} | {placed}"


def threshold_text(entry):
    """One frequency's threshold for the summary: its level, or "above the cap"."""
    where = "above the cap" if entry["censored"] else f"{entry['threshold_dbfs']:.1f} dBFS"
    return f"{entry['hz'] / 1000.0:g} kHz {where}"
