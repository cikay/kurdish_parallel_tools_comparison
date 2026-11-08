"""Sentence-embedding models and the neutral LaBSE scorer.

Vecalign is embedding-agnostic: it only needs sentence vectors in a shared
multilingual space. This module provides the per-side embed functions that feed
it (LaBSE for both sides, or a SONAR pipeline per language) and, separately, the
LaBSE-based ``score_pairs`` used to score EVERY alignment regardless of which
embeddings produced it -- so ``mean_score`` is comparable across combos.

NOTE on the scorer's limits: scoring with LaBSE is only a fair *ranking* metric
between configs that share the embedding (e.g. bertalign/labse vs
vecalign/labse). Across different embeddings (vs vecalign/sonar) LaBSE
structurally favours LaBSE-built alignments, so a lower score there does not by
itself mean a worse alignment -- judge those by the beads or a gold set.
"""

from __future__ import annotations

from functools import lru_cache

# Embedding models available for Vecalign. Bertalign is hard-wired to LaBSE.
# (LASER3 was evaluated but is not installable here: laser_encoders -> fairseq
# is incompatible with Python 3.12 / torch 2.x. See README.)
EMBEDS = ("labse", "sonar")


def labse_embed_fn(labse_model):
    """Return an embed function: list[str] -> L2-normalized float32 LaBSE vecs."""
    import numpy as np

    def embed(lines: list[str]):
        return np.asarray(
            labse_model.encode(lines, normalize_embeddings=True), dtype=np.float32
        )

    return embed


@lru_cache(maxsize=None)
def _sonar_pipeline(device: str):
    """Build (and cache) the SONAR text encoder pipeline on ``device``.

    One pipeline covers all 200 NLLB languages; the language is given per call
    via ``source_lang``. SONAR natively covers kmr_Latn and ckb_Arab and beats
    LaBSE/LASER3 on multilingual similarity (CC-BY-NC, non-commercial).
    """
    import torch
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    return TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device(device),
    )


def sonar_embed_fn(lang: str, device: str = "cpu"):
    """Return an embed function backed by SONAR for FLORES-200 code ``lang``."""
    import numpy as np

    pipeline = _sonar_pipeline(device)

    def embed(lines: list[str]):
        vecs = pipeline.predict(lines, source_lang=lang).detach().cpu().numpy()
        vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-8)

    return embed


def score_pairs(labse_model, pairs: list[tuple[str, str]]) -> list[float | None]:
    """Cosine similarity for each aligned bead, using LaBSE as a neutral scorer.

    The aligner discards (Bertalign) or doesn't expose a comparable per-bead
    score, so we recover an interpretable one that is the SAME regardless of
    which tool/embeddings produced the alignment: re-encode each bead's joined
    source and target segment with LaBSE (L2-normalized) and take the dot product
    (= cosine, in [-1, 1]; good matches are typically ~0.6-0.95). Insertion/
    deletion beads (one side empty) have no counterpart, so their score is None.

    Args:
        labse_model: A ``sentence_transformers.SentenceTransformer`` LaBSE model
            (e.g. a bertalign ``Encoder.model`` or a freshly loaded one).
        pairs: The aligner's (src, tgt) text beads.

    Returns:
        One score (or None) per bead, aligned with ``pairs``.
    """
    import numpy as np

    scores: list[float | None] = [None] * len(pairs)
    srcs, tgts, idxs = [], [], []
    for i, (src, tgt) in enumerate(pairs):
        if src.strip() and tgt.strip():
            srcs.append(src)
            tgts.append(tgt)
            idxs.append(i)
    if srcs:
        src_vecs = labse_model.encode(srcs, normalize_embeddings=True)
        tgt_vecs = labse_model.encode(tgts, normalize_embeddings=True)
        cos = (np.asarray(src_vecs) * np.asarray(tgt_vecs)).sum(axis=1)
        for j, i in enumerate(idxs):
            scores[i] = round(float(cos[j]), 4)
    return scores
