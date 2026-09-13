# DNSMOS P.835 models: provenance and licensing

The two ONNX files in this directory are the DNSMOS P.835 estimator from Microsoft's
Deep Noise Suppression Challenge repository, fetched by `scripts/download_dnsmos.py`:

- `sig_bak_ovr.onnx` -- SIG, BAK and OVRL (ITU-T P.835), 16 kHz waveform input
- `model_v8.onnx` -- P.808 overall MOS, 120-band log-mel input

Upstream: <https://github.com/microsoft/DNS-Challenge>, path `DNSMOS/DNSMOS/`.
License: Creative Commons Attribution 4.0 International (CC BY 4.0), which requires
attribution to Microsoft and the DNS Challenge authors when the files are shared.

Reference: Chandan K. A. Reddy, Vishak Gopal, Ross Cutler, "DNSMOS P.835: A
Non-Intrusive Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors",
ICASSP 2022.

SHA-256 as fetched:

- `sig_bak_ovr.onnx`: `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd`
- `model_v8.onnx`: `9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91`
