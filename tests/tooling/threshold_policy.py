import math
import os

DEFAULT_COVERAGE_THRESHOLD = 90.0
ENV_COVERAGE_THRESHOLD = "COVERAGE_THRESHOLD"


def is_valid_threshold(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 100.0


def get_coverage_threshold() -> float:
    raw = os.getenv(ENV_COVERAGE_THRESHOLD)
    if raw is None:
        return DEFAULT_COVERAGE_THRESHOLD

    try:
        value = float(raw)
        if not is_valid_threshold(value):
            return DEFAULT_COVERAGE_THRESHOLD
        return value
    except ValueError:
        return DEFAULT_COVERAGE_THRESHOLD
