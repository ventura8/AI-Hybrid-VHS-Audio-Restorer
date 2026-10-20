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
"""

JOINT_MOVES = {
    "apl": (),
    "cathar": (
        # The reference suppressor with the fill that refills what it lowers: per frame, the source's hiss
        # shelved at 2 kHz, 18 dB under each source frame (APL v9's round-2 winner).
        {
            "cathar_reference_suppress": True,
            "enable_pause_floor": True,
            "pause_floor_target": "frame",
            "pause_floor_shape": "shelf",
            "pause_floor_shelf_hz": 2000.0,
            "pause_floor_fill_db": 18.0,
        },
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
