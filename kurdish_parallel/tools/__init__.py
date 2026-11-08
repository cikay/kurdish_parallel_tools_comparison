"""Alignment tools and embedding models for the kurdish-theses-yok aligner.

Two concerns, two modules:
  * ``embeddings`` - sentence-embedding models (LaBSE, SONAR) and the neutral
    LaBSE scorer that makes every tool/embed combo comparable.
  * ``tools`` - the alignment algorithms (Bertalign, Vecalign) and the
    ``align_pair`` dispatcher.

Re-exported here so callers can do
``from kurdish_parallel.tools import align_pair, score_pairs``.
"""

from .embeddings import (
    EMBEDS,
    labse_embed_fn,
    score_pairs,
    sonar_embed_fn,
)
from .tools import (
    TOOLS,
    align_bertalign,
    align_pair,
    align_vecalign,
)

__all__ = [
    "EMBEDS",
    "TOOLS",
    "align_bertalign",
    "align_pair",
    "align_vecalign",
    "labse_embed_fn",
    "score_pairs",
    "sonar_embed_fn",
]
