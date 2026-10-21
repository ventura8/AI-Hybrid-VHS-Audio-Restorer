"""Stage-output names stay under the bound: the chain a long render builds no longer breaks the write (2026-10-20)."""

from pathlib import Path

import pytest

from modules import burst_ceiling, pause_floor, post_neural_suppress, reference_suppress, stage_names

# The sync stage's input on Tele7abc with the burst ceiling, the reference suppressor and a surgical notch on: its
# `aligned_` output was 261 characters with `.tmp`, past NTFS's 255, and the render failed writing it.
FAILED_CHAIN = (
    "paused_18db_p15_frame_shelf2000c18_capped_gaps30_polished_bd27d863ff55_referenced_eda2ba96d7e2_guarded_cleaned_"
    "d045b0b054ff_decrackled_depopped_surgical_baab63780080_preconditioned_2c934dbf8079_audio_(dry)_"
    "denoise_mel_band_roformer_aufr33_sdr_27.wav"
)


def test_a_short_chain_keeps_its_whole_name():
    assert stage_names.stage_name("guarded", "cleaned_ab12_audio.wav") == "guarded_cleaned_ab12_audio.wav"


def test_the_chain_that_failed_on_tele7abc_collapses_under_the_bound():
    name = stage_names.stage_name("aligned", FAILED_CHAIN)
    assert len(f"aligned_{FAILED_CHAIN}") > 250
    assert (name.startswith("aligned_"), name.endswith(".wav"), len(name) <= stage_names.NAME_LIMIT) == (True, True, True)


def test_a_collapsed_name_is_the_same_for_the_same_chain_and_differs_for_another():
    again = stage_names.stage_name("aligned", FAILED_CHAIN)
    other = stage_names.stage_name("aligned", FAILED_CHAIN.replace("gaps30", "edges30"))
    assert (again == stage_names.stage_name("aligned", FAILED_CHAIN), again != other) == (True, True)


def test_the_bound_is_inclusive():
    source = "x" * (stage_names.NAME_LIMIT - len("tag_") - len(".wav")) + ".wav"
    assert stage_names.stage_name("tag", source) == f"tag_{source}"
    assert stage_names.stage_name("tag", "y" + source) != f"tag_y{source}"


def test_stage_output_joins_the_folder_and_takes_a_string_path(tmp_path):
    produced = stage_names.stage_output(tmp_path, "paused_18db", str(tmp_path / "x" / "restored.wav"))
    assert produced == tmp_path / "paused_18db_restored.wav"


TARGETS = [
    lambda src, folder: burst_ceiling._target_for(src, folder),
    lambda src, folder: pause_floor._target_for(src, folder, pause_floor.configured()),
    lambda src, folder: post_neural_suppress._target_for(src, folder),
    lambda src, folder: reference_suppress.target_for(src, folder, reference_suppress.CATHAR_SETTINGS),
]


@pytest.mark.parametrize("target", TARGETS)
def test_the_post_neural_stages_name_their_output_within_the_bound(tmp_path, target):
    produced = Path(target(tmp_path / FAILED_CHAIN, tmp_path))
    assert (len(produced.name) <= stage_names.NAME_LIMIT, produced.suffix) == (True, ".wav")
