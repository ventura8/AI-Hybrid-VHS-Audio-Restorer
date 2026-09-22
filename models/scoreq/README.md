# scoreq

Fetched by `scripts/download_quality_models.py` from <https://zenodo.org/records/15739280>
(revision `https://zenodo.org/records/15739280/files/`).

Licence: MIT code (alessandroragano/scoreq); CC-BY-4.0 weights (Zenodo record 15739280).

SCOREQ: Speech Quality Assessment with Contrastive Regression, Ragano, Skoglund and
Hines, NeurIPS 2024, <https://arxiv.org/abs/2410.06675>; code at
<https://github.com/alessandroragano/scoreq> (its `onnx-scripts/export_to_onnx.py`
made these files from the PyTorch weights of Zenodo record 13860326).

- `adapt_nr_telephone.onnx`: no-reference MOS, the "natural" data domain (the
  package's `Scoreq(data_domain='natural', mode='nr')`); read by
  `scripts/restoration_quality/mos_models.py` as `mos.scoreq_nr`.
- `fixed_nmr_telephone.onnx`: the non-matching-reference embedding of the same
  domain (`mode='ref'`); the score is the L2 distance of two embeddings. Kept
  beside the NR file, not read by the harness yet.

Input: mono float32 at 16 kHz, shape `(1, samples)`, zero-padded to a multiple
of 320 samples (the wav2vec 2.0 feature-encoder stride), no normalisation; one
MOS-like scalar out. Trained on segments up to 4 s, evaluated up to 15 s.

`MANIFEST.json` holds the sha256 of every file as observed on first fetch; a later
fetch that hashes differently is refused.
