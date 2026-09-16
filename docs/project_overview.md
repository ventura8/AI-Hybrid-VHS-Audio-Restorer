# Project Overview

The **AI-Hybrid VHS Audio Restorer** is a specialized tool optimized for
restoring low-fidelity VHS audio using a multi-stage AI pipeline. It is designed
to run on high-end hardware (e.g., RTX 5090) but includes auto-tuning for
lower-spec configurations.

## Directory Structure

- `input/`: Source video files (MP4, MKV, etc.).
- Restored videos are written next to their source with a mode-specific
  `*_Cleaned` suffix; the launch directory only receives `session_log.txt`.
- `.temp_work_<video>/`: Hidden work directory created next to each source
  video, holding every intermediate track, sidecar and library scratch file for
  that restoration. It is removed once the output is valid, also when a rerun
  finds the output already there, and kept for a resume otherwise.
- `venv/`: Local Python virtual environment.
- `assets/`: UI assets like logos.
