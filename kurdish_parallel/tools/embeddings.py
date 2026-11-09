"""Sentence-embedding models for the aligners.

Vecalign is embedding-agnostic: it only needs sentence vectors in a shared
multilingual space. This module provides the per-side embed functions that feed
it (LaBSE for both sides, or a SONAR pipeline per language). Alignment quality is
evaluated against an external gold set, not an intrinsic score.
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
