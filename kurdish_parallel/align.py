#!/usr/bin/env python
"""Align parallel sentences from the kurdish-theses-yok summaries.

The dataset stores abstracts in a ``summaries`` struct with one key per
language (kmr_Latn, zza_Latn, ckb_Arab, tur_Latn, eng_Latn); any key may be
null. This script pairs two of those languages, splits each abstract into
sentences, and aligns them with the selected tool/embedding.

Why we pre-split and pass is_split=True (Bertalign): its bundled language
detector and sentence splitter only know ~25 European languages, so Kurdish
would be mislabelled and mis-split. LaBSE itself is multilingual and embeds
Kurdish fine, so we do the segmentation ourselves (Turkish splitter as a proxy
for the Latin-script Kurdish variants) and let the tool do only the alignment.

This module is the data + CLI layer. The alignment algorithms and embedding
models live in the ``tools`` package and are re-exported below so existing
callers (``import kurdish_parallel.align as A``) keep working:
    * tools.tools      - align_bertalign, align_vecalign, align_pair (+ TOOLS)
    * tools.embeddings - labse/sonar embed fns, score_pairs (+ EMBEDS)

Source is always kmr_Latn; every other summary language present in a row is a
target (null ones are skipped). Run as a module inside the pipenv shell:
    pipenv run python -m kurdish_parallel.align --n 3
    pipenv run python -m kurdish_parallel.align --tool vecalign --embed sonar --n 5 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

# Re-exported so ``import kurdish_parallel.align as A`` still exposes the aligners/scorer that
# now live in the tools package (used by modal_app/app.py and the local CLI).
from .tools.embeddings import (  # noqa: F401
    EMBEDS,
    labse_embed_fn,
    score_pairs,
    sonar_embed_fn,
)
from .tools.tools import (  # noqa: F401
    TOOLS,
    align_bertalign,
    align_pair,
    align_vecalign,
)

REPO_ID = "muzaffercky/kurdish-theses-yok"

# Dataset summary key -> sentence-splitter language code.
# Kurdish variants have no dedicated splitter; Turkish ("tr") is the closest
# Latin-script proxy (shared punctuation and abbreviation conventions).
SPLITTER_LANG = {
    "kmr_Latn": "tr",   # Kurmanji (Latin)
    "zza_Latn": "tr",   # Zazaki (Latin)
    "tur_Latn": "tr",   # Turkish
    "eng_Latn": "en",   # English
    # ckb_Arab / *_Arab handled by a regex fallback (no splitter model).
}

# Fixed CLI pairing: Kurmanji is the source, every OTHER summary language is a
# target. Rows missing a given abstract are skipped for that target.
SRC_LANG = "kmr_Latn"
TARGET_LANGS = ("zza_Latn", "ckb_Arab", "tur_Latn", "eng_Latn")

# Local outputs live in results/ (repo root is the package's parent).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def resolve_device(requested: str) -> str:
    """Pick the torch device: explicit request, else MPS (Apple), CUDA, or CPU."""
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def split_sentences(text: str, key: str) -> list[str]:
    """Split one abstract into sentences based on its language key."""
    text = text.strip()
    if not text:
        return []
    if key.endswith("_Arab"):
        # No splitter model for Arabic-script text; split on sentence-final
        # punctuation (Latin + Arabic question mark/full stop) and newlines.
        parts = re.split(r"(?<=[.!?؟۔])\s+|\n+", text)
        return [p.strip() for p in parts if p.strip()]

    from sentence_splitter import SentenceSplitter

    lang = SPLITTER_LANG.get(key, "en")
    splitter = SentenceSplitter(language=lang)
    return [s.strip() for s in splitter.split(text) if s.strip()]


def load_rows(subset: str):
    """Download the subset parquet from the Hub and return it as a DataFrame."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{subset}/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    return pd.read_parquet(path)


def get_summary(row, key: str):
    """Return the abstract for ``key`` from a row's summaries struct, or None."""
    summaries = row["summaries"]
    if summaries is None:
        return None
    # pyarrow struct -> dict; guard against odd types.
    value = summaries.get(key) if hasattr(summaries, "get") else None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI. Source/targets/device are fixed; see module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="kmr_Latn",
                        help="Which parquet subset to read (default: kmr_Latn).")
    parser.add_argument("--tool", default="bertalign", choices=TOOLS,
                        help="Alignment tool to use (default: bertalign).")
    parser.add_argument("--embed", default="labse", choices=EMBEDS,
                        help="Vecalign embedding model (default: labse). "
                             "bertalign is LaBSE-only.")
    parser.add_argument("--n", type=int, default=None,
                        help="How many theses to align. Required unless --is-full.")
    parser.add_argument("--is-full", action="store_true", default=False,
                        help="Align the whole subset (ignores --n).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show Bertalign's INFO logging.")
    return parser


def load_labse_and_encoder(tool: str, device: str):
    """Load the LaBSE scorer (and a bertalign Encoder when tool==bertalign).

    LaBSE is always needed as the neutral scorer and as the --embed labse model.
    For bertalign we load it through the Encoder and reuse that model for
    scoring too; otherwise a plain SentenceTransformer. Call ONCE and reuse the
    result across every thesis.

    Returns:
        (labse_model, bert_encoder) where bert_encoder is None unless bertalign.
    """
    if tool == "bertalign":
        from bertalign import Encoder

        bert_encoder = Encoder("LaBSE", device=device)
        return bert_encoder.model, bert_encoder

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("LaBSE", device=device), None


def align_one(tool: str, embed: str, labse_model, bert_encoder, device: str,
              src_sents: list[str], tgt_sents: list[str],
              tez_id, title, tgt_lang: str) -> dict:
    """Align one Kurmanji/target sentence-list pair into a JSONL record dict."""
    t0 = time.perf_counter()
    pairs = align_pair(tool, embed, labse_model, src_sents, tgt_sents,
                       SRC_LANG, tgt_lang, device=device, bert_encoder=bert_encoder)
    scores = score_pairs(labse_model, pairs)
    align_seconds = time.perf_counter() - t0

    matched = [s for s in scores if s is not None]
    mean_score = round(sum(matched) / len(matched), 4) if matched else None

    return {
        "tez_id": tez_id,
        "title": title,
        "tool": tool,
        "embed": embed,
        "src_lang": SRC_LANG,
        "tgt_lang": tgt_lang,
        "n_src_sents": len(src_sents),
        "n_tgt_sents": len(tgt_sents),
        "align_seconds": round(align_seconds, 3),
        "mean_score": mean_score,
        "pairs": [{"src": s, "tgt": t, "score": sc}
                  for (s, t), sc in zip(pairs, scores)],
    }


def align_row(row, tool: str, embed: str, labse_model, bert_encoder,
              device: str) -> list[dict]:
    """Align Kurmanji against every target language present in one row.

    Returns one record per target that has a usable abstract; an empty list if
    the row has no Kurmanji abstract or it doesn't segment.
    """
    src_text = get_summary(row, SRC_LANG)
    if not src_text:
        return []
    src_sents = split_sentences(src_text, SRC_LANG)
    if not src_sents:
        return []

    tez_id = row.get("tez_id")
    title = row.get("title")
    records = []
    for tgt in TARGET_LANGS:
        tgt_text = get_summary(row, tgt)
        if not tgt_text:
            continue
        tgt_sents = split_sentences(tgt_text, tgt)
        if not tgt_sents:
            continue
        records.append(align_one(tool, embed, labse_model, bert_encoder, device,
                                 src_sents, tgt_sents, tez_id, title, tgt))
    return records


def print_record(record: dict, show_beads: bool) -> None:
    """Print a one-line summary of a record, plus every bead when show_beads."""
    print(f"=== tez_id={record['tez_id']} | {record['tool']}/{record['embed']} | "
          f"{record['src_lang']}->{record['tgt_lang']} | "
          f"{record['n_src_sents']}->{record['n_tgt_sents']} sents | "
          f"{len(record['pairs'])} beads | mean cos={record['mean_score']} | "
          f"{record['align_seconds']}s ===")
    if not show_beads:
        return
    print(f"    {record['title']}")
    for pair in record["pairs"]:
        score = pair["score"]
        tag = f"{score:.3f}" if score is not None else "  -  "
        print(f"  [{tag}] [{record['src_lang']}] {pair['src']}")
        print(f"          [{record['tgt_lang']}] {pair['tgt']}")
        print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.is_full and args.n is None:
        parser.error("--n is required unless --is-full is set")
    if args.tool == "bertalign" and args.embed != "labse":
        parser.error("bertalign only supports --embed labse")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    device = resolve_device("auto")
    print(f"Device: {device} | tool: {args.tool} | embed: {args.embed}")

    t0 = time.perf_counter()
    labse_model, bert_encoder = load_labse_and_encoder(args.tool, device)
    print(f"Loaded LaBSE in {time.perf_counter() - t0:.1f}s\n")

    df = load_rows(args.subset)

    suffix = "full" if args.is_full else str(args.n)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.tool}_{args.embed}_{suffix}.jsonl"

    theses_done = 0
    records_written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            if not args.is_full and theses_done >= args.n:
                break
            records = align_row(row, args.tool, args.embed, labse_model,
                                bert_encoder, device)
            if not records:
                continue
            for record in records:
                print_record(record, show_beads=not args.is_full)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1
            theses_done += 1

    print(f"Wrote {records_written} records "
          f"({theses_done} theses x up to {len(TARGET_LANGS)} targets) "
          f"to {out_path.resolve()}")


if __name__ == "__main__":
    main()
