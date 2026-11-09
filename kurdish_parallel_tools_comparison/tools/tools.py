"""Alignment algorithms and the ``align_pair`` dispatcher.

Each aligner returns the SAME shape -- ``list[(src_text, tgt_text)]`` -- so the
scorer and output code don't care which one ran. Keep that contract if you add
a tool. Embeddings come from ``tools.embeddings``; this module is only the DP.
"""

from __future__ import annotations

from .embeddings import EMBEDS, labse_embed_fn, laser_embed_fn, sonar_embed_fn

TOOLS = ("bertalign", "vecalign")


def align_bertalign(encoder, src_sents: list[str], tgt_sents: list[str],
                    max_align: int = 5) -> list[tuple[str, str]]:
    """Align with Bertalign (two-pass LaBSE DP). Returns (src, tgt) text beads."""
    from bertalign import Bertalign

    aligner = Bertalign(
        "\n".join(src_sents),
        "\n".join(tgt_sents),
        is_split=True,
        model=encoder,
        max_align=max_align,
    )
    aligner.align_sents()
    return aligner.sent_pairs()


def align_vecalign(src_sents: list[str], tgt_sents: list[str],
                   src_embed_fn, tgt_embed_fn,
                   max_align: int = 4) -> list[tuple[str, str]]:
    """Align with Vecalign, using the given per-side embed functions.

    Vecalign only needs sentence vectors in a shared multilingual space; the
    embed functions supply them (LaBSE for both sides, or a SONAR pipeline per
    language). We build the overlap embeddings in memory (no temp files) and
    call Vecalign's own DP, so this is the real Vecalign algorithm.

    Returns (src, tgt) text beads; an insertion/deletion has one side empty.
    """
    import random
    from math import ceil

    import numpy as np
    from vecalign.dp_utils import (
        make_alignment_types,
        make_doc_embedding,
        vecalign as _vecalign,
        yield_overlaps,
    )

    # Vecalign samples randomly to normalize costs; seed for reproducibility.
    random.seed(42)
    np.random.seed(42)

    def overlap_embeddings(sents: list[str], embed_fn):
        # Unique, sorted overlap lines (concatenations of up to max_align
        # consecutive sentences), mirroring Vecalign's overlap.py.
        uniq = sorted(set(yield_overlaps(sents, max_align)))
        sent2line = {line: i for i, line in enumerate(uniq)}
        return sent2line, np.asarray(embed_fn(uniq), dtype=np.float32)

    src_s2l, src_emb = overlap_embeddings(src_sents, src_embed_fn)
    tgt_s2l, tgt_emb = overlap_embeddings(tgt_sents, tgt_embed_fn)
    vecs0 = make_doc_embedding(src_s2l, src_emb, src_sents, max_align)
    vecs1 = make_doc_embedding(tgt_s2l, tgt_emb, tgt_sents, max_align)

    width_over2 = ceil(max_align / 2.0) + 5
    stack = _vecalign(
        vecs0=vecs0,
        vecs1=vecs1,
        final_alignment_types=make_alignment_types(max_align),
        del_percentile_frac=0.2,
        width_over2=width_over2,
        max_size_full_dp=300,
        costs_sample_size=20000,
        num_samps_for_norm=100,
    )
    alignments = stack[0]["final_alignments"]

    pairs = []
    for src_idx, tgt_idx in alignments:
        src_text = " ".join(src_sents[i] for i in src_idx)
        tgt_text = " ".join(tgt_sents[j] for j in tgt_idx)
        pairs.append((src_text, tgt_text))
    return pairs


def align_pair(tool: str, embed: str, labse_model, src_sents: list[str],
               tgt_sents: list[str], src_lang: str, tgt_lang: str,
               device: str = "cpu", bert_encoder=None) -> list[tuple[str, str]]:
    """Dispatch to the selected aligner; all return the same (src, tgt) shape.

    Args:
        tool: "bertalign" or "vecalign".
        embed: Vecalign's embeddings, "labse", "sonar", or "laser" (LASER3,
            legacy image only). Bertalign is LaBSE-only.
        labse_model: SentenceTransformer LaBSE model (for the labse embed option).
        src_lang/tgt_lang: FLORES-200 codes (dataset keys), used by SONAR.
        device: torch device string, used by SONAR.
        bert_encoder: a bertalign ``Encoder`` (required only for tool=bertalign).
    """
    if tool == "bertalign":
        if embed != "labse":
            raise ValueError("bertalign only supports --embed labse")
        return align_bertalign(bert_encoder, src_sents, tgt_sents)
    if tool == "vecalign":
        if embed == "labse":
            src_fn = tgt_fn = labse_embed_fn(labse_model)
        elif embed == "sonar":
            src_fn = sonar_embed_fn(src_lang, device)
            tgt_fn = sonar_embed_fn(tgt_lang, device)
        elif embed == "laser":
            src_fn = laser_embed_fn(src_lang)
            tgt_fn = laser_embed_fn(tgt_lang)
        else:
            raise ValueError(f"unknown embed: {embed!r} (expected one of {EMBEDS})")
        return align_vecalign(src_sents, tgt_sents, src_fn, tgt_fn)
    raise ValueError(f"unknown tool: {tool!r} (expected one of {TOOLS})")
