"""Ten shipped-cathar hunt scores re-gated offline: the hard gates must not refuse the shipped defaults' denoising itself.

`tests/fixtures/shipped_cathar_scores.json` freezes ten 15 s IA excerpts of the 2026-10-11 hunt
(`experiments/tune_ia_hunt_v3/runs/cathar__baseline`), trimmed to the statistics the gates read. The
hunt's own verdicts failed them 28 times, 13 of those on the raw stem octave, CER and Audiobox PC;
re-gated under today's gates they fail 16 times. A gate change that makes the shipped outputs fail
more, or one gate fail most of them, breaks this test before it reaches a loop.
"""

import json
from pathlib import Path

import pytest

from scripts.restoration_quality import gates as g

FROZEN = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "shipped_cathar_scores.json").read_text(encoding="utf-8"))
CLIPS = FROZEN["clips"]
# The drivers' default --gates (experiments/quality_calibration/gates.json, 2026-09-22; experiments/ is not in git).
V2_FILE = {
    "dsp.hf_4k8k": {"threshold": -8.0, "severity": "hard", "source": "floor"},
    "dsp.pause_pumping": {"threshold": 12.0, "severity": "soft", "source": "floor"},
    "dsp.hf_8k16k": {"threshold": -12.0, "severity": "hard", "source": "floor"},
    "dsp.lkr": {"threshold": 1.74687879261231, "severity": "hard", "source": "known good bound"},
    "dsp.hum": {"threshold": 3.0, "severity": "hard", "source": "floor"},
    "dsp.clicks": {"threshold": 0.7933333333333334, "severity": "hard", "source": "known good bound"},
    "dsp.dropouts": {"threshold": 57.300000000000004, "severity": "hard", "source": "known good bound"},
    "dsp.whistle": {"threshold": 3.0, "severity": "hard", "source": "floor"},
    "file.lra": {"threshold": -3.0, "severity": "hard", "source": "floor"},
    "file.lufs": {"threshold": 11.390237319916348, "severity": "soft", "source": "known good bound"},
    "dsp.output_silent": {"threshold": 0.0, "severity": "hard", "source": "floor"},
    "listener.hiss": {"threshold": -19.95167635035534, "severity": "flag", "source": "known ordering"},
    "listener.dead_air": {"threshold": -24.565719395055254, "severity": "flag", "source": "known ordering"},
}
# A "manifest" round (the v3 and v3b files): every entry the floor rule, resting on no verdict.
MANIFEST_FILE = {
    "dsp.lkr": {"threshold": 0.3, "severity": "hard", "source": "floor", "n_verdicts": 0},
    "dsp.dropouts": {"threshold": 1.0, "severity": "hard", "source": "floor", "n_verdicts": 0},
    "dsp.clicks": {"threshold": 0.5, "severity": "hard", "source": "floor", "n_verdicts": 0},
}
EXPECTED = {
    "16-april-2009-cnn-weather-update": [],
    "1994-6-ntsc20241207": ["speech.logprob"],
    "1996-commercial-for-a-joan-armatrading-concert-2": ["mos.sigmos_disc"],
    "1plus2lvcontinuity06092022": ["mos.sigmos_noise"],
    "2003-commercial-for-toshiba": ["mos.sigmos_col"],
    "2004-nickelodeon-promo-break-33-late-night": ["mos.sigmos_col", "mos.sigmos_disc"],
    "22-april-2009-bloomberg-promo-break-40": ["mos.sigmos_disc", "mos.sigmos_noise"],
    "andrew-and-sarah-royal-wedding-l": ["dsp.hf_4k8k", "dsp.hf_8k16k"],
    # file.dropouts reports only (soft) since the integration review of 2026-10-12, so it no longer fails this clip.
    "bbc-london-last-news-intro-outro-bloopers-cctv-p": [
        "file.lra",
        "mos.sigmos_col",
        "mos.sigmos_disc",
        "mos.sigmos_noise",
    ],
    "bbciplayerbbcnewsspecialtheresamayconfirmsexitop": ["dsp.lkr"],
}
DENOISING_GATES = ("stems.octave", "speech.cer_median", "mos.audiobox_pc")


def _failures(gates):
    """`{clip: sorted hard failures}` of every frozen clip under `gates`."""
    return {clip: sorted(g.hard_failures(g.evaluate_gates(doc["aggregate"], gates, doc["speaker_floor"]))) for clip, doc in CLIPS.items()}


def _loaded(tmp_path, entries):
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return g.load_gates(path)


def _per_gate(failures):
    counts = {}
    for names in failures.values():
        for name in names:
            counts[name] = counts.get(name, 0) + 1
    return counts


def test_the_hunts_own_verdicts_leaned_on_the_gates_that_read_denoising():
    """28 stored failures, 13 of them on the raw octave, CER and Audiobox PC (the hand-set guesses E1 retired or softened)."""
    stored = {clip: doc["hard_failures_stored"] for clip, doc in CLIPS.items()}
    assert sum(len(names) for names in stored.values()) == 28
    assert sum(count for name, count in _per_gate(stored).items() if name in DENOISING_GATES) == 13


def test_todays_gates_fail_the_shipped_outputs_on_their_own_readings_only():
    """Under the gates in code the frozen clips fail 16 times, never on a gate that reads the denoising itself."""
    failures = _failures(g.GATES)
    assert failures == EXPECTED
    assert not set(_per_gate(failures)) & set(DENOISING_GATES)


@pytest.mark.parametrize("entries", [V2_FILE, MANIFEST_FILE], ids=["v2", "manifest"])
def test_a_stored_gates_file_cannot_make_them_fail_more(tmp_path, entries):
    """The drivers' default file and a manifest-round file give the same verdicts: no retired bound, no floor that tightens."""
    failures = _failures(_loaded(tmp_path, entries))
    assert sum(len(names) for names in failures.values()) <= 16
    assert max(_per_gate(failures).values()) <= len(CLIPS) // 2
    assert failures == EXPECTED
