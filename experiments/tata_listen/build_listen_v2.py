"""The v2 listening set: each engine's plateau outputs, straight from the loops' final candidates.

The Tata loops restore the whole tapes (their `tapes.json` points at the full captures), so the
final candidate's `<slug>.wav` under `experiments/autotune_v2/<engine>/cands/<final>/` *is* the
listening output; the music loops do the same for the 12 clips. Nothing is re-rendered: this
script copies those WAVs into ``D:/Tata/New folder/variants/v2`` as ``<slug>__final2_<engine>.wav``,
extracts each Tata source once as ``<slug>__source.wav``, links the music sources from the v1
music set, and writes ``index.md`` with the settings that changed and the harness readings the
loop recorded for every output (flags, hard failures, the listener readings). Idempotent: run it
again when a later final appears (the APL music loop) and only the missing files are added.
"""

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\Users\ventu\Projects\AI-Hybrid-VHS-Audio-Restorer")
sys.path.insert(0, str(REPO))
candidate_id = importlib.import_module("scripts.autotune_restoration").candidate_id

OUT = Path(r"D:\Tata\New folder\variants\v2")
MUSIC_V1 = Path(r"D:\Tata\New folder\variants\music")
FFMPEG = str(REPO / ".venv" / "Scripts" / "ffmpeg.exe")
SETS = {
    "tata": {
        "dir": REPO / "experiments" / "autotune_v2",
        "tapes": {"cathar": "experiments/autotune/tapes.json", "apl": "experiments/autotune/tapes_apl.json"},
    },
    "music": {
        "dir": REPO / "experiments" / "autotune_music_v2",
        "tapes": {"cathar": "experiments/autotune_music/tapes_music.json", "apl": "experiments/autotune_music/tapes_music_apl.json"},
    },
}
READINGS = [
    ("gates.hard_failures", "hard"),
    ("gates.listener_flags", "flags"),
    ("dsp.gap_air_db.output.median", "gap air dB"),
    ("dsp.pause_depth_db.delta.median", "pause depth d"),
    ("dsp.pause_pumping_db.delta.median", "pumping d"),
    ("dsp.sib_centroid_hz.delta.median", "sib centroid d Hz"),
    ("dsp.hf_4k8k.delta.median", "4-8k d dB"),
    ("dsp.residual_noise_db.delta.median", "residual d dB"),
    ("mos.sigmos_col.delta.median", "COL d"),
    ("speech.cer.output.median", "CER"),
]


def final_of(set_name, engine):
    path = SETS[set_name]["dir"] / engine / "final.json"
    if not path.exists():
        return None, None
    overrides = json.loads(path.read_text(encoding="utf-8"))
    return candidate_id(overrides), overrides


def copy_outputs(set_name, engine, cid, lines):
    tapes = json.loads((REPO / SETS[set_name]["tapes"][engine]).read_text(encoding="utf-8"))
    cand_dir = SETS[set_name]["dir"] / engine / "cands" / cid
    for slug, source in tapes.items():
        target = OUT / f"{slug}__final2_{engine}.wav"
        wav = cand_dir / f"{slug}.wav"
        if not target.exists() and wav.exists():
            shutil.copy2(wav, target)
        ensure_source(set_name, slug, Path(source))
        lines.append((set_name, slug, engine, cand_dir / f"{slug}.score.json", target.exists()))


def ensure_source(set_name, slug, source):
    target = OUT / f"{slug}__source.wav"
    if target.exists():
        return
    v1 = MUSIC_V1 / f"{slug}__source.wav"
    if set_name == "music" and v1.exists():
        shutil.copy2(v1, target)
        return
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-i", str(source), "-map", "0:a:0", "-vn", "-acodec", "pcm_f32le", str(target)], check=True
    )


def round_one_incumbent(set_name, engine):
    """The settings the loop started from (the v1 final seeded with the app's defaults), from its run log."""
    log = SETS[set_name]["dir"] / f"{engine}_run.log"
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("round 1: incumbent "):
            return json.JSONDecoder().raw_decode(line[line.index("{") :])[0]
    return {}


def settings_diff(set_name, engine, overrides):
    start = round_one_incumbent(set_name, engine)
    return {k: (start.get(k), overrides.get(k)) for k in sorted(set(start) | set(overrides)) if start.get(k) != overrides.get(k)}


def reading_row(score_path):
    if not score_path.exists():
        return "| " + " | ".join("-" for _ in READINGS) + " |"
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    cells = []
    for key, _label in READINGS:
        value = scores.get(key)
        cells.append("-" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value))
    return "| " + " | ".join(cells) + " |"


def write_index(finals, lines):
    text = ["# Listening set v2 (2026-09-25)", ""]
    text.append("Each engine's plateau from the listener round (harness v2: gap air, pause depth, sibilance, attack readings and the")
    text.append("listener flags), on the full Tata tapes and the 12 music clips. `__source` is the tape; `__final2_<engine>` the v2")
    text.append("plateau. The v1 plateaus are the `__final_<engine>.mov` files one folder up (Tata) and `music/*__apl.wav`,")
    text.append("`music/*__cathar_music.wav` (music).")
    text.append("")
    for (set_name, engine), (cid, overrides) in finals.items():
        text.append(f"## {engine} {set_name} final `{cid}`")
        text.append("")
        diff = settings_diff(set_name, engine, overrides)
        text.append("Changed against the loop's start (the v1 final with the app defaults filled in; start -> final):")
        text.append("")
        for key, (old, new) in diff.items():
            text.append(f"- `{key}`: {old} -> {new}")
        text.append("")
    text.append("## Harness readings of every output (delta = output minus source, median over 15 s windows)")
    text.append("")
    text.append("| set | clip | engine | file | " + " | ".join(label for _key, label in READINGS) + " |")
    text.append("|---|---|---|---|" + "---|" * len(READINGS))
    for set_name, slug, engine, score_path, present in lines:
        text.append(f"| {set_name} | {slug} | {engine} | {'ok' if present else 'MISSING'} " + reading_row(score_path))
    text.append("")
    text.append("Flags are the listener gates (hiss, dead air, pause collapse, sibilance thin/dull, attack); hard = hard gates failed.")
    (OUT / "index.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    finals, lines = {}, []
    for set_name in SETS:
        for engine in ("cathar", "apl"):
            cid, overrides = final_of(set_name, engine)
            if cid is None:
                print(f"{set_name}/{engine}: no final yet", flush=True)
                continue
            finals[(set_name, engine)] = (cid, overrides)
            copy_outputs(set_name, engine, cid, lines)
            print(f"{set_name}/{engine}: {cid}", flush=True)
    write_index(finals, lines)
    print(f"{sum(1 for line in lines if line[4])}/{len(lines)} outputs present, index written to {OUT / 'index.md'}", flush=True)


if __name__ == "__main__":
    main()
