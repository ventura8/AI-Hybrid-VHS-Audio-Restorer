"""Timeline assembly helpers for reproducible multi-duration fixtures."""

import numpy as np


def repeat_to_duration(samples, sample_rate, duration_seconds):
    """Repeat speech with one-second quiet pauses until target duration."""
    target = int(sample_rate * duration_seconds)
    pause = np.zeros((sample_rate, samples.shape[1]), dtype=np.float32)
    block = np.concatenate((samples, pause))
    repetitions = int(np.ceil(target / len(block)))
    return np.tile(block, (repetitions, 1))[:target]
