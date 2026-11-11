"""Alignment algorithms and the ``align_pair`` dispatcher.

Each aligner returns the SAME shape -- ``list[(src_text, tgt_text)]`` -- so the
scorer and output code don't care which one ran. Keep that contract if you add
a tool. Embeddings come from ``tools.embeddings``; this module is only the DP.
"""

from __future__ import annotations

from .embeddings import EMBEDS, labse_embed_fn, laser_embed_fn, sonar_embed_fn

TOOLS = ("bertalign", "vecalign", "sentalign")


def align_bertalign(encoder, src_sents: list[str], tgt_sents: list[str],
                    max_align: int = 4) -> list[tuple[str, str]]:
    """Align with Bertalign (two-pass LaBSE DP). Returns (src, tgt) text beads.

    ``max_align=4`` (Bertalign's own default is 5) matches Vecalign's cap so the
    two DPs are compared under the same max bead size, not different ones.
    """
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


def align_sentalign(labse_model, src_sents: list[str], tgt_sents: list[str],
                    max_concats: int = 4, free_concats: int = 2,
                    score_cutoff: float = 0.4, min_words: int = 1,
                    max_words: int = 110, penalty_after_words: int = 80,
                    penalty_per_word: float = 0.01) -> list[tuple[str, str]]:
    """Align with SentAlign's grid-DP core (LaBSE), driven in-process.

    SentAlign (Steingrímsson et al., 2023) is LaBSE + a grid DP with a
    length-penalty scoring function and a null/merge re-evaluation pass. For a
    document small enough to skip its Gale-Church anchoring -- all our
    ~14-sentence abstracts -- the whole alignment is a single DP block, so we
    call its vendored pure-Python core (``_sentalign_core.align_anchors``)
    directly on one block ``((-1,-1),(n_src,n_tgt))``, exactly as SentAlign
    constructs it. No subprocess, no temp files, no Cython. Defaults mirror
    SentAlign's CLI.

    LaBSE-only, like Bertalign. The DP scores with a plain dot product, so the
    embeddings must be L2-normalized (``labse_embed_fn`` already is).

    Returns (src, tgt) text beads; an insertion/deletion has one side empty.
    """
    from ._sentalign_core import align_anchors

    embed = labse_embed_fn(labse_model)
    source_dict = {i: s for i, s in enumerate(src_sents)}
    target_dict = {i: s for i, s in enumerate(tgt_sents)}

    def build_emb_dict(sents: list[str]) -> dict:
        # Every space-joined concatenation of 1..max_concats consecutive
        # sentences -- the exact strings the DP's create_concats will look up.
        n = len(sents)
        strings = [" ".join(sents[i:i + length])
                   for i in range(n)
                   for length in range(1, max_concats + 1)
                   if i + length <= n]
        uniq = list(dict.fromkeys(strings))  # dedup, keep first occurrence
        vecs = embed(uniq)
        return {s: v for s, v in zip(uniq, vecs)}

    src_emb_dict = build_emb_dict(src_sents)
    trg_emb_dict = build_emb_dict(tgt_sents)

    class _ProcInfo:
        """Minimal stand-in for SentAlign's ReportInfo (progress hooks only)."""

        def set_total_calculations(self, n):
            pass

        def add_nodes(self, n):
            pass

    anchor = ((-1, -1), (len(src_sents), len(tgt_sents)))
    _, path, _scores = align_anchors(
        anchor, source_dict, target_dict, src_emb_dict, trg_emb_dict,
        score_cutoff, max_concats, min_words, max_words,
        penalty_after_words, penalty_per_word, free_concats, _ProcInfo(),
    )

    pairs: list[tuple[str, str]] = []
    for line in path.strip().split("\n"):
        if not line:
            continue
        s, t = line.replace("[", "").replace("]", "").split(":")
        s_nums = [x for x in s.split(",") if x != ""]
        t_nums = [x for x in t.split(",") if x != ""]
        src_text = " ".join(source_dict[int(x)] for x in s_nums)
        tgt_text = " ".join(target_dict[int(x)] for x in t_nums)
        pairs.append((src_text, tgt_text))
    return pairs


def align_pair(tool: str, embed: str, labse_model, src_sents: list[str],
               tgt_sents: list[str], src_lang: str, tgt_lang: str,
               device: str = "cpu", bert_encoder=None) -> list[tuple[str, str]]:
    """Dispatch to the selected aligner; all return the same (src, tgt) shape.

    Args:
        tool: "bertalign", "vecalign", or "sentalign".
        embed: Vecalign's embeddings, "labse", "sonar", or "laser" (LASER3,
            legacy image only). Bertalign and SentAlign are LaBSE-only.
        labse_model: SentenceTransformer LaBSE model (for the labse embed option).
        src_lang/tgt_lang: FLORES-200 codes (dataset keys), used by SONAR.
        device: torch device string, used by SONAR.
        bert_encoder: a bertalign ``Encoder`` (required only for tool=bertalign).
    """
    if tool == "bertalign":
        if embed != "labse":
            raise ValueError("bertalign only supports --embed labse")
        return align_bertalign(bert_encoder, src_sents, tgt_sents)
    if tool == "sentalign":
        if embed != "labse":
            raise ValueError("sentalign only supports --embed labse")
        return align_sentalign(labse_model, src_sents, tgt_sents)
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
