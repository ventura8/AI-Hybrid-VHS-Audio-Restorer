# mert-v1-95m

Fetched by `scripts/download_quality_models.py` from <https://huggingface.co/m-a-p/MERT-v1-95M>
(revision `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`).

Licence: CC-BY-NC-4.0 (m-a-p/MERT-v1-95M): research use, no commercial use.

MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training,
Li et al. 2023, <https://arxiv.org/abs/2306.00107>. A 95M-parameter HuBERT-style
encoder for music, 24 kHz input, 75 Hz frame rate, 12 transformer layers.

Loaded with `trust_remote_code=True` from the two pinned `.py` files beside the
config; `scripts/restoration_quality/judges.py` reads `stems.mert_dist`, one minus
the cosine of the time-averaged layer-12 states of the source and the output.
This checkpoint has `feature_extractor_cqt: false`, so `nnAudio` is imported but
not exercised. The `MERT-v1-95M_fairseq.pt` of the repository is not fetched.

`MANIFEST.json` holds the sha256 of every file as observed on first fetch; a later
fetch that hashes differently is refused.
