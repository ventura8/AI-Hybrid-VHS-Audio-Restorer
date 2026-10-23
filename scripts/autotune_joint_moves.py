"""Joint moves: settings the self-driving loop cannot reach one knob at a time, proposed whole.

The loop's neighbourhood is one knob per candidate (plus the combination of the moves that won
on their own). A pair whose halves each lose alone, or whose second half is inert until the first
is on, is never proposed. cathar's round 3 (2026-10-20) hit one: its reference suppressor emptied
the pauses (the room band 16-19 dB down on every Tata tape, a v4 guard) because cathar's round 1
had switched the pause fill off (its cluster fill raised the pause hiss), and with the fill off
the per-frame target and the shelf are inert. APL reached the same pair step by step only because
the shelf won on its own first (v9 round 1, then the suppressor with the per-frame fill in round
2, `docs/configuration.md`, "Reference-SNR Suppressor").

Each joint move is proposed, on top of the incumbent, while the incumbent does not already carry
it; it is judged like any candidate and joins the combination of winners.

cathar's round 4 (2026-10-21) rendered the pair: the hiss under the programme and in the pauses read
0 on every Tata tape, but Gaudeamus5's loud highs fell 2.35 dB (`dsp.hiss_prog_kept_hf_db`, past its
1.0 move), so the guard held it back. The second move adds the programme release at 6 dB against the
pause print, which on CPU replay of APL's plateau brought the loud highs back on all four tapes
(Gaudeamus5 -4.05 to -3.49) with every v4 guard passing (`docs/configuration.md`).
"""

SUPPRESSED_FILL = {
    "cathar_reference_suppress": True,
    "enable_pause_floor": True,
    "pause_floor_target": "frame",
    "pause_floor_shape": "shelf",
    "pause_floor_shelf_hz": 2000.0,
    "pause_floor_fill_db": 18.0,
}

JOINT_MOVES = {
    "apl": (),
    "cathar": (
        # The reference suppressor with the fill that refills what it lowers: per frame, the source's hiss
        # shelved at 2 kHz, 18 dB under each source frame (APL v9's round-2 winner).
        SUPPRESSED_FILL,
        # The same with the programme release at 6 dB against the pause print (round 4 held the pair back on
        # Gaudeamus5's loud highs).
        {**SUPPRESSED_FILL, "cathar_reference_release_db": 6.0},
    ),
}


def joint_moves(engine, incumbent, candidate_id):
    """`{id: overrides}` for every joint move of `engine` the incumbent does not already carry."""
    proposals = {}
    for move in JOINT_MOVES.get(engine, ()):
        overrides = {**incumbent, **move}
        if overrides != incumbent:
            proposals[candidate_id(overrides)] = overrides
    return proposals
