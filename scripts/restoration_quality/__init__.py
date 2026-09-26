"""Reference-free validation of a restored track against its own source.

Every reading is a paired delta (output minus source) over 15 s windows of the
same timeline, aggregated as a median and a "tail" that always points at the
bad end. Learned predictors (SIGMOS, DNSMOS, UTMOS, Audiobox) are guardrails,
never objectives: in the URGENT 2024 challenge the systems that topped
DNSMOS/NISQA ranked at the bottom with human listeners. Hard gates veto an
output the way a listener would: words changed, timbre changed, highs gone,
musical noise, background stripped.
"""

from scripts.restoration_quality.gates import GATES, evaluate_gates, load_gates
from scripts.restoration_quality.scorecard import METRICS, MetricSpec, ScoreCard, WindowRow, aggregate, paired_deltas

__all__ = [
    "GATES",
    "METRICS",
    "MetricSpec",
    "ScoreCard",
    "WindowRow",
    "aggregate",
    "evaluate_gates",
    "load_gates",
    "paired_deltas",
]
