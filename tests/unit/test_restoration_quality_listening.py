"""Worst-window selection and rendering of listening picks from files."""

import numpy as np
import soundfile as sf

from scripts.restoration_quality import listening

RATE = 44100


def _result(rows_by_label):
    variants = {}
    for label, deltas in rows_by_label.items():
        rows = [{"window": i, "start_s": i * 7.5, "end_s": i * 7.5 + 15.0, "delta": {"mos.sigmos_col": d}} for i, d in enumerate(deltas)]
        variants[label] = {"rows": rows}
    return {"variants": variants}


def test_select_worst_reads_the_bad_direction_across_outputs():
    result = _result({"a": [0.5, -0.9, 0.1], "b": [-0.2, 0.3, None]})
    picks = listening.select_worst(result, "mos.sigmos_col", 2)
    assert [(p.label, p.window) for p in picks] == [("a", 1), ("b", 0)]
    assert picks[0].value == -0.9


def test_select_worst_handles_lower_is_better_metrics():
    result = _result({"a": [0.5, -0.9]})
    for row in result["variants"]["a"]["rows"]:
        row["delta"] = {"speech.cer": row["delta"]["mos.sigmos_col"]}
    assert listening.select_worst(result, "speech.cer", 1)[0].window == 0


def test_render_from_files_cuts_source_and_aligned_output(tmp_path):
    rng = np.random.default_rng(3)
    source = rng.standard_normal(20 * RATE).astype(np.float32) * 0.1
    output = np.concatenate([np.zeros(200, dtype=np.float32), source * 0.5])
    sf.write(str(tmp_path / "src.wav"), source, RATE, subtype="FLOAT")
    sf.write(str(tmp_path / "out.wav"), output, RATE, subtype="FLOAT")
    pick = listening.Pick("mos.sigmos_col", "v", 0, 2.0, 5.0, -0.4)
    lines = listening.render_from_files([pick], tmp_path / "src.wav", {"v": tmp_path / "out.wav"}, tmp_path / "listen", prefix="x_")
    cut_source, _rate = sf.read(str(tmp_path / "listen" / "x_mos_sigmos_col_1_source.wav"), dtype="float32")
    cut_output, _rate = sf.read(str(tmp_path / "listen" / "x_mos_sigmos_col_1_v.wav"), dtype="float32")
    assert len(cut_source) == len(cut_output) == 3 * RATE
    assert np.allclose(cut_source, cut_output, atol=1e-3)
    assert lines[0].startswith("- `x_mos_sigmos_col_1`")
