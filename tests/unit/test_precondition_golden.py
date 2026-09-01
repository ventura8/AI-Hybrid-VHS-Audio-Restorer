"""Golden snapshot of the analog pre-conditioning filter graph.

Every restoration mode is pre-conditioned through
`modules.filters._precondition_filter_from_config`, so this graph is the single point at
which a scanner change can silently move an already-validated mode's audio -- `cathar`
above all, whose published PAL/NTSC benchmark results depend on it.

The strings below were captured from the shipped implementation. FFmpeg and the Cathar
Rust CLI are both deterministic, so an unchanged graph means unchanged audio and
unchanged metrics: this asserts that guarantee directly, rather than inferring it from a
benchmark average that could hide a regression inside its own measurement noise.

A deliberate change to pre-conditioning must update these strings in the same commit that
re-runs the affected benchmarks. A change that arrives here unexplained is a regression.
"""

import pytest

from modules.filters import _precondition_filter_from_config

GOLDEN_CASES = {
    "pal_typical": (
        {"highpass_hz": 80, "notch_hz": 50.0, "crt_notch_hz": 15625.0},
        "highpass=f=80,adeclick,bandreject=f=50.0:width_type=q:w=15,"
        "bandreject=f=100.0:width_type=q:w=15,bandreject=f=15625.0:width_type=q:w=30",
    ),
    "ntsc_typical": (
        {"highpass_hz": 60, "notch_hz": 60.0, "crt_notch_hz": 15734.0},
        "highpass=f=60,adeclick,bandreject=f=60.0:width_type=q:w=15,"
        "bandreject=f=120.0:width_type=q:w=15,bandreject=f=15734.0:width_type=q:w=30",
    ),
    "mains_detected_as_harmonic": (
        {"highpass_hz": 45, "notch_hz": 100.0, "crt_notch_hz": 15625.0},
        "highpass=f=45,adeclick,bandreject=f=50.0:width_type=q:w=15,"
        "bandreject=f=100.0:width_type=q:w=15,bandreject=f=15625.0:width_type=q:w=30",
    ),
    "nothing_detected": (
        {"highpass_hz": 0, "notch_hz": 0.0, "crt_notch_hz": 0.0, "enable_adeclick": False},
        "anull",
    ),
    "defaults_only": ({}, "highpass=f=80,adeclick"),
    "full_house": (
        {
            "highpass_hz": 75,
            "notch_hz": 60.0,
            "crt_notch_hz": 15734.0,
            "enable_adeclip": True,
            "azimuth_delay_ms": 0.35,
            "enable_dc_block": True,
            "balance_db": 3.5,
            "resonance_hz": 220.0,
        },
        "highpass=f=2,pan=stereo|c0=0.668*c0|c1=c1,adeclip,adelay=15S|0,highpass=f=75,adeclick,"
        "bandreject=f=60.0:width_type=q:w=15,bandreject=f=120.0:width_type=q:w=15,"
        "bandreject=f=15734.0:width_type=q:w=30,bandreject=f=220.0:width_type=q:w=12",
    ),
    "negative_azimuth": (
        {"highpass_hz": 80, "notch_hz": 50.0, "crt_notch_hz": 15625.0, "azimuth_delay_ms": -0.42},
        "adelay=0|19S,highpass=f=80,adeclick,bandreject=f=50.0:width_type=q:w=15,"
        "bandreject=f=100.0:width_type=q:w=15,bandreject=f=15625.0:width_type=q:w=30",
    ),
    "dead_channel": (
        {"highpass_hz": 80, "notch_hz": 50.0, "crt_notch_hz": 15625.0, "balance_db": 41.0},
        "pan=stereo|c0=c0|c1=c0,highpass=f=80,adeclick,bandreject=f=50.0:width_type=q:w=15,"
        "bandreject=f=100.0:width_type=q:w=15,bandreject=f=15625.0:width_type=q:w=30",
    ),
}


@pytest.mark.parametrize("case_name", sorted(GOLDEN_CASES))
def test_precondition_graph_is_unchanged(case_name):
    """The pre-conditioning graph for each profile shape must be byte-identical."""
    precond_config, expected = GOLDEN_CASES[case_name]
    assert _precondition_filter_from_config(precond_config) == expected
