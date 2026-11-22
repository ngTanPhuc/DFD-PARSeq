"""The label string written next to each crop, per the frozen tone-placement decision.

Train labels are tone-normalized to the new placement (``hoà``, ``khoẻ``, ``thuỷ``) when the
training data is built; validation and test labels stay as shipped, and only the metrics normalize
them, so a scored word is never silently rewritten. Every label is NFC, case-sensitive, and
untouched apart from the tone mark's position.

Agreed with the training side (Phase 2, step 16): the LMDB stores this plain UTF-8 string. Any
sub-character decomposition the three-head model needs is the tokenizer's job, not a second field.
"""

from __future__ import annotations

import unicodedata

from dfstr.labels.orthography import tone_normalize

#: Splits whose labels are tone-normalized when the training data is built.
NORMALIZED_SPLITS = frozenset({"train"})


def training_label(transcript: str, split: str) -> str:
    # Input: a transcript as shipped and the split it belongs to; Output: the label to store
    # How: NFC for every split; tone placement rewritten for train only
    text = unicodedata.normalize("NFC", transcript)
    return tone_normalize(text) if split in NORMALIZED_SPLITS else text
