"""Tests for the residual CRT flyback notch that auto_pure_linear applies after denoising.

Deliberately a single stage. Measured in isolation, a second cascaded Q=80 stage takes a
strong whistle from 445x-26244x attenuation to 9170x-41990x, which looked worth having.
Measured end to end over the 174-clip Internet Archive corpus it was not: the earlier
stages already remove the tone, so a second stage moved 63 clips for a median CRT delta of
+0.05 while sending one clip to -1908. The isolated gain does not survive the full chain.
"""

from modules.filters import _precondition_filter_from_config, build_post_denoise_cleanup_filter


def test_residual_crt_notch_is_applied_once():
    """One stage is enough; the chain ahead of it has already removed the tone."""
    strategy = {"profile": {"crt_notch_hz": 15625.0, "notch_hz": 0.0}}
    cleanup = build_post_denoise_cleanup_filter(strategy)
    assert cleanup.count("bandreject=f=15625.0:width_type=q:w=80") == 1


def test_residual_notch_is_skipped_without_a_detected_whistle():
    """No detected line rate means no notch, so clips without a whistle are left alone."""
    assert build_post_denoise_cleanup_filter({"profile": {"crt_notch_hz": 0.0, "notch_hz": 0.0}}) is None


def test_residual_notch_does_not_reach_the_shared_preconditioning_graph():
    """The residual notch is post-denoise only, so pre-conditioning is untouched.

    `cathar` is built from the shared pre-conditioning graph and never calls the
    post-denoise cleanup builder, so its audio cannot move when this stage changes.
    """
    graph = _precondition_filter_from_config({"highpass_hz": 80, "notch_hz": 50.0, "crt_notch_hz": 15625.0})
    assert graph.count("bandreject=f=15625.0:width_type=q:w=30") == 1
    assert "w=80" not in graph
