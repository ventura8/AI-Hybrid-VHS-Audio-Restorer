"""Names of stage outputs: each stage puts its tag in front of its input's name, within a bound.

A stage writes `<tag>_<input name>`, so a name carries the chain that made it
(`aligned_paused_..._polished_<fp>_referenced_<fp>_guarded_cleaned_<fp>_...`). The chain only
grows: with the burst ceiling, the reference suppressor and a surgical notch on, the sync
stage's output on a Tata tape reached 261 characters (2026-10-20), past the 255 NTFS allows in
one name, and the render failed writing it ("System error"). A Windows install without long
paths also refuses any path over 260 characters, which a long chain under a long work folder
reaches sooner.

Past `NAME_LIMIT` the inherited part becomes a digest of it: the name still starts with this
stage's tag and ends with the input's suffix, the same chain always gets the same name, and
two different chains never share one.
"""

import hashlib
from pathlib import Path

# The longest stage-output name kept whole. A Tata work folder is about 100 characters, so a
# name this long keeps the path under Windows' 260 and leaves room for what is derived from
# it (`<name>.tmp.wav` while it is written, cathar's `noise_<stem>_..._4s.np.json`).
NAME_LIMIT = 120
# Hex digits of the SHA-256 of the inherited name that stand in for it past the limit.
DIGEST_CHARS = 16


def stage_name(tag, source_name):
    """`<tag>_<source_name>`, or `<tag>_<digest><suffix>` when that is longer than `NAME_LIMIT`."""
    name = f"{tag}_{source_name}"
    if len(name) <= NAME_LIMIT:
        return name
    digest = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:DIGEST_CHARS]
    return f"{tag}_{digest}{Path(source_name).suffix}"


def stage_output(directory, tag, source):
    """The path a stage tagged `tag` writes in `directory` for the input `source` (`stage_name`)."""
    return Path(directory) / stage_name(tag, Path(source).name)
