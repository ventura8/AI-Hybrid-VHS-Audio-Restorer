"""The v3 listening set: APL's sibilance plateau on the four Tata tapes, beside the v2 APL plateau.

The user's second listening round (2026-10-05) asked for one thing: APL's 's' is thin. The
sibilance loop (`experiments/run_autotune_v3_apl_sibilance.cmd`) restored the full tapes, so its
final candidate's WAVs are the listening outputs. This copies them into
``D:/Tata/New folder/variants/v3`` as ``<slug>__final3_apl.wav`` next to ``<slug>__final2_apl.wav``
(what the user heard) and ``<slug>__source.wav``, and writes ``index.md`` with the settings the
loop changed and the sibilance readings of both plateaus. Reuses the v2 builder's helpers.
"""

import importlib.util
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("build_listen_v2", HERE / "build_listen_v2.py")
v2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v2)

V3_DIR = v2.REPO / "experiments" / "autotune_v3"
OUT = Path(r"D:\Tata\New folder\variants\v3")
SIB_READINGS = [
    ("gates.listener_flags", "flags"),
    ("gates.hard_failures", "hard"),
    ("dsp.sib_centroid_hz.delta.median", "sib centroid d Hz"),
    ("dsp.gap_air_db.output.median", "gap air dB"),
    ("dsp.pause_depth_db.delta.median", "pause depth d"),
    ("speech.cer.output.median", "CER"),
]


def _cell(scores, key):
    value = scores.get(key)
    return "-" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


def _row(slug, label, score_path):
    scores = v2.json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else {}
    return f"| {slug} | {label} | " + " | ".join(_cell(scores, key) for key, _ in SIB_READINGS) + " |"


def main():
    v2.SETS["v3"] = {"dir": V3_DIR, "tapes": {"apl": "experiments/autotune/tapes_apl.json"}}
    cid3, final3 = v2.final_of("v3", "apl")
    cid2, _ = v2.final_of("tata", "apl")
    if cid3 is None:
        raise SystemExit("the sibilance loop has no final.json yet")
    OUT.mkdir(parents=True, exist_ok=True)
    tapes = v2.json.loads((v2.REPO / "experiments/autotune/tapes_apl.json").read_text(encoding="utf-8"))
    rows = []
    for slug, source in tapes.items():
        for tag, root, cid in (("final3_apl", V3_DIR, cid3), ("final2_apl", v2.SETS["tata"]["dir"], cid2)):
            target, wav = OUT / f"{slug}__{tag}.wav", root / "apl" / "cands" / cid / f"{slug}.wav"
            if not target.exists() and wav.exists():
                shutil.copy2(wav, target)
            rows.append(_row(slug, tag, root / "apl" / "cands" / cid / f"{slug}.score.json"))
        v2.OUT = OUT
        v2.ensure_source("tata", slug, Path(source))
    diff = v2.settings_diff("v3", "apl", final3)
    text = ["# Listening set v3: APL's 's' (2026-10-05)", ""]
    text += ["Per tape: `__source`, `__final2_apl` (what you heard in round two), `__final3_apl` (the", "sibilance loop's plateau).", ""]
    text += [f"Changed by the loop (start -> final `{cid3}`):", ""] + [f"- `{k}`: {a} -> {b}" for k, (a, b) in diff.items()] + [""]
    text += ["| tape | file | " + " | ".join(label for _, label in SIB_READINGS) + " |", "|---|---|" + "---|" * len(SIB_READINGS)]
    text += rows + ["", "A positive sibilant centroid shift is a thinner 's'; `sibilance_thin` flags above +300 Hz."]
    (OUT / "index.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"v3 listening set: {OUT}")


if __name__ == "__main__":
    main()
