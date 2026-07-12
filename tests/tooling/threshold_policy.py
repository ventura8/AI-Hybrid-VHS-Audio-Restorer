import math
import os

DEFAULT_COVERAGE_THRESHOLD = 90.0
ENV_COVERAGE_THRESHOLD = "COVERAGE_THRESHOLD"


def get_coverage_threshold() -> float:
    raw = os.getenv(ENV_COVERAGE_THRESHOLD)
    if raw is None:
        return DEFAULT_COVERAGE_THRESHOLD

    try:
        value = float(raw)
        if not math.isfinite(value) or value < 0.0 or value > 100.0:
            return DEFAULT_COVERAGE_THRESHOLD
        return value
    except ValueError:
        return DEFAULT_COVERAGE_THRESHOLD
