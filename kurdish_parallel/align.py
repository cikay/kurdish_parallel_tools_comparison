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

Usage (inside the pipenv shell):
    pipenv run python align.py --n 3
    pipenv run python align.py --src kmr_Latn --tgt eng_Latn --n 5 --verbose
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="kmr_Latn",
                        help="Which parquet subset to read (default: kmr_Latn).")
    parser.add_argument("--src", default="kmr_Latn",
                        help="Source summary language key (default: kmr_Latn).")
    parser.add_argument("--tgt", default="tur_Latn",
                        help="Target summary language key (default: tur_Latn).")
    parser.add_argument("--tool", default="bertalign", choices=TOOLS,
                        help="Alignment tool to use (default: bertalign).")
    parser.add_argument("--embed", default="labse", choices=EMBEDS,
                        help="Vecalign embedding model (default: labse). "
                             "bertalign is LaBSE-only.")
    parser.add_argument("--n", type=int, default=None,
                        help="How many theses to align; omit for the full subset.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "mps", "cuda", "cpu"],
                        help="Torch device (default: auto -> MPS on Apple Silicon).")
    parser.add_argument("--out", default=None,
                        help="Output JSONL path (default: {tool}_{embed}_{src}_"
                             "{tgt}_{n|full}.jsonl).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show Bertalign's INFO logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    if args.tool == "bertalign" and args.embed != "labse":
        parser.error("bertalign only supports --embed labse")

    device = resolve_device(args.device)
    print(f"Device: {device} | tool: {args.tool} | embed: {args.embed}")

    # LaBSE is always needed as the neutral scorer (and as the embed model for
    # --embed labse). Load it via bertalign's Encoder when we need the bertalign
    # tool, else directly via sentence-transformers. Loaded ONCE and reused.
    t0 = time.perf_counter()
    bert_encoder = None
    if args.tool == "bertalign":
        from bertalign import Encoder

        bert_encoder = Encoder("LaBSE", device=device)
        labse_model = bert_encoder.model
    else:
        from sentence_transformers import SentenceTransformer

        labse_model = SentenceTransformer("LaBSE", device=device)
    print(f"Loaded LaBSE in {time.perf_counter() - t0:.1f}s\n")

    df = load_rows(args.subset)

    suffix = str(args.n) if args.n else "full"
    out_path = Path(
        args.out
        or f"{args.tool}_{args.embed}_{args.src}_{args.tgt}_{suffix}.jsonl"
    )
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            if args.n and written >= args.n:
                break
            src_text = get_summary(row, args.src)
            tgt_text = get_summary(row, args.tgt)
            if not src_text or not tgt_text:
                continue

            src_sents = split_sentences(src_text, args.src)
            tgt_sents = split_sentences(tgt_text, args.tgt)
            if not src_sents or not tgt_sents:
                continue

            tez_id = row.get("tez_id")
            title = row.get("title")

            t1 = time.perf_counter()
            pairs = align_pair(args.tool, args.embed, labse_model, src_sents,
                               tgt_sents, args.src, args.tgt,
                               device=device, bert_encoder=bert_encoder)
            scores = score_pairs(labse_model, pairs)
            dt = time.perf_counter() - t1
            matched = [s for s in scores if s is not None]
            mean_score = round(sum(matched) / len(matched), 4) if matched else None

            print(f"=== tez_id={tez_id} | {args.tool}/{args.embed} | "
                  f"{args.src}->{args.tgt} | "
                  f"{len(src_sents)}->{len(tgt_sents)} sents | "
                  f"{len(pairs)} beads | mean cos={mean_score} | {dt:.2f}s ===")
            # Print every bead only for a small smoke run; stay quiet for full runs.
            if args.n:
                print(f"    {title}")
                for (src_line, tgt_line), score in zip(pairs, scores):
                    tag = f"{score:.3f}" if score is not None else "  -  "
                    print(f"  [{tag}] [{args.src}] {src_line}")
                    print(f"          [{args.tgt}] {tgt_line}")
                    print()

            record = {
                "tez_id": tez_id,
                "title": title,
                "tool": args.tool,
                "embed": args.embed,
                "src_lang": args.src,
                "tgt_lang": args.tgt,
                "n_src_sents": len(src_sents),
                "n_tgt_sents": len(tgt_sents),
                "align_seconds": round(dt, 3),
                "mean_score": mean_score,
                "pairs": [{"src": s, "tgt": t, "score": sc}
                          for (s, t), sc in zip(pairs, scores)],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} aligned theses to {out_path.resolve()}")


if __name__ == "__main__":
    main()
