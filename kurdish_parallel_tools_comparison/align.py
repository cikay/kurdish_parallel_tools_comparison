#!/usr/bin/env python
"""Align parallel sentences from the kurdish-theses-yok summaries.

The dataset stores abstracts in a ``summaries`` struct with one key per
language (kmr_Latn, zza_Latn, ckb_Arab, tur_Latn, eng_Latn); any key may be
null. This script pairs two of those languages, splits each abstract into
sentences, and aligns them with the selected tool/embedding.

Why we pre-split and pass is_split=True (Bertalign): its bundled language
detector and sentence splitter only know ~25 European languages, so Kurdish
would be mislabelled and mis-split. LaBSE itself is multilingual and embeds
Kurdish fine, so we do the segmentation ourselves -- with wtpsplit's SaT, a
neural multilingual segmenter that is language-agnostic (no per-language model,
handles Latin- and Arabic-script Kurdish alike) -- and let the tool do only the
alignment. Segmentation is done ONCE per run (see ``prepare_theses``) and the
sentence lists are reused by every tool/embedding, so the comparison is exactly
deterministic.

This module is the data + CLI layer. The alignment algorithms and embedding
models live in the ``tools`` package and are re-exported below so existing
callers (``import kurdish_parallel_tools_comparison.align as A``) keep working:
    * tools.tools      - align_bertalign, align_vecalign, align_pair (+ TOOLS)
    * tools.embeddings - labse/sonar embed fns (+ EMBEDS)

Source is always kmr_Latn; every other summary language present in a row is a
target (null ones are skipped). Run as a module inside the pipenv shell:
    pipenv run python -m kurdish_parallel_tools_comparison.align --n 3
    pipenv run python -m kurdish_parallel_tools_comparison.align --tool vecalign --embed sonar --n 5 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

# Re-exported so ``import kurdish_parallel_tools_comparison.align as A`` still exposes the aligners
# that now live in the tools package (used by modal_app/app.py and the local CLI).
from .tools.embeddings import (  # noqa: F401
    EMBEDS,
    labse_embed_fn,
    laser_embed_fn,
    sonar_embed_fn,
)
from .tools.tools import (  # noqa: F401
    TOOLS,
    align_bertalign,
    align_pair,
    align_sentalign,
    align_vecalign,
)

REPO_ID = "muzaffercky/kurdish-theses-yok"

# wtpsplit SaT model used for segmentation. "sat-12l-sm" is the largest
# general-purpose checkpoint (12 layers + limited lookahead): the most accurate
# of the SaT family, still fast enough to run on CPU for abstract-sized text.
# (The smaller "sat-3l-sm" over-split some Kurmanji sentences; 12l segments them
# cleanly.) It is language-agnostic -- no per-language code is needed -- which is
# exactly why it fits Kurdish, for which no dedicated sentence splitter exists.
SAT_MODEL_NAME = "sat-12l-sm"

# Fixed CLI pairing: Kurmanji is the source, every OTHER summary language is a
# target. Rows missing a given abstract are skipped for that target.
SRC_LANG = "kmr_Latn"
TARGET_LANGS = ("zza_Latn", "ckb_Arab", "tur_Latn", "eng_Latn")

# Local outputs live in data/ (repo root is the package's parent).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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


_SAT_MODEL = None  # cached wtpsplit SaT instance (load once, reuse everywhere)


def get_sat_model(device: str | None = None):
    """Load (once) and return the shared SaT segmenter.

    Cached at module scope so the model loads a single time per process. Moved to
    CUDA (half precision) only when a CUDA device is given; on CPU/MPS it stays on
    CPU -- the small model segments abstract-sized text quickly there, and
    wtpsplit does not target MPS.
    """
    global _SAT_MODEL
    if _SAT_MODEL is None:
        from wtpsplit import SaT

        _SAT_MODEL = SaT(SAT_MODEL_NAME)
        if device == "cuda":
            _SAT_MODEL.half().to("cuda")
    return _SAT_MODEL


def _clean_sentence(s: str) -> str:
    """Collapse every whitespace run (incl. newlines) to a single space.

    Guarantees no sentence carries an embedded line break. Besides tidying the
    text, this is what makes it safe to round-trip sentences through Bertalign's
    ``is_split=True`` path (``"\\n".join(sents)`` then ``str.splitlines()``):
    ``splitlines`` breaks on more than ``\\n`` (``\\r``, ``\\v``, ``\\f``, NEL,
    ``\\u2028``/``\\u2029`` ...), so an embedded one would silently desync
    Bertalign's segmentation from the shared one.
    """
    return " ".join(s.split())


def split_sentences(text: str, key: str | None = None, model=None) -> list[str]:
    """Split one abstract into sentences with wtpsplit SaT.

    SaT is language-agnostic, so ``key`` is accepted for call-site compatibility
    but no longer selects a model. ``model`` lets callers pass an already-loaded
    SaT instance (e.g. one placed on a GPU); it defaults to the shared cached one.
    """
    if not text or not text.strip():
        return []
    sat = model or get_sat_model()
    return [c for s in sat.split(text) if (c := _clean_sentence(s))]


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


def prepare_theses(subset: str, src: str, targets: list[str], n: int | None,
                   require_all_targets: bool = True, model=None) -> list[dict]:
    """Segment the first ``n`` usable theses ONCE and return their sentences.

    This is the single shared segmentation step: it loads the subset, selects
    theses, and splits the Kurmanji source plus each target abstract, so every
    tool/embedding downstream aligns the *identical* sentence lists. The result
    is a plain, serializable structure (safe to ship to remote workers):

        [{"tez_id", "title", "src_summary", "src_sents",
          "targets": {tgt_lang: {"tgt_summary", "tgt_sents"}}}, ...]

    A thesis needs a Kurmanji abstract that segments. With
    ``require_all_targets`` (the comparison default) a thesis is kept only when
    every requested target is present and segments; otherwise targets are kept
    individually and theses with at least one usable target are included.
    ``n=None`` means no limit (the whole subset).
    """
    df = load_rows(subset)
    sat = model or get_sat_model()
    prepared: list[dict] = []
    for _, row in df.iterrows():
        if n is not None and len(prepared) >= n:
            break
        src_text = get_summary(row, src)
        if not src_text:
            continue
        target_texts = {t: get_summary(row, t) for t in targets}
        if require_all_targets and any(not v for v in target_texts.values()):
            continue
        src_sents = split_sentences(src_text, src, model=sat)
        if not src_sents:
            continue
        tgt_entries: dict = {}
        for t in targets:
            tgt_text = target_texts[t]
            if not tgt_text:
                continue
            tgt_sents = split_sentences(tgt_text, t, model=sat)
            if not tgt_sents:
                continue
            tgt_entries[t] = {"tgt_summary": tgt_text, "tgt_sents": tgt_sents}
        if not tgt_entries:
            continue
        prepared.append({
            "tez_id": row.get("tez_id"),
            "title": row.get("title"),
            "src_summary": src_text,
            "src_sents": src_sents,
            "targets": tgt_entries,
        })
    return prepared


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


def load_labse_and_encoder(tool: str, embed: str, device: str):
    """Load LaBSE only when it is the aligning embedding.

    LaBSE is needed as the ``--embed labse`` model (and, for bertalign, via its
    Encoder). With ``--embed sonar`` nothing here is needed, so we skip the load
    entirely. Call ONCE and reuse the result across every thesis.

    Returns:
        (labse_model, bert_encoder); either may be None when LaBSE isn't needed.
    """
    if tool == "bertalign":
        from bertalign import Encoder

        bert_encoder = Encoder("LaBSE", device=device)
        return bert_encoder.model, bert_encoder

    if embed == "labse":
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("LaBSE", device=device), None

    return None, None


def align_one(tool: str, embed: str, labse_model, bert_encoder, device: str,
              src_sents: list[str], tgt_sents: list[str],
              tez_id, title, tgt_lang: str) -> dict:
    """Align one Kurmanji/target sentence-list pair into a JSONL record dict."""
    t0 = time.perf_counter()
    pairs = align_pair(tool, embed, labse_model, src_sents, tgt_sents,
                       SRC_LANG, tgt_lang, device=device, bert_encoder=bert_encoder)
    align_seconds = time.perf_counter() - t0

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
        "pairs": [{"src": s, "tgt": t} for s, t in pairs],
    }


def align_thesis(thesis: dict, tool: str, embed: str, labse_model, bert_encoder,
                 device: str) -> list[dict]:
    """Align one already-segmented thesis (from ``prepare_theses``) against each
    of its target languages. Returns one record per target."""
    src_sents = thesis["src_sents"]
    tez_id = thesis.get("tez_id")
    title = thesis.get("title")
    records = []
    for tgt, info in thesis["targets"].items():
        records.append(align_one(tool, embed, labse_model, bert_encoder, device,
                                 src_sents, info["tgt_sents"], tez_id, title, tgt))
    return records


def print_record(record: dict, show_beads: bool) -> None:
    """Print a one-line summary of a record, plus every bead when show_beads."""
    print(f"=== tez_id={record['tez_id']} | {record['tool']}/{record['embed']} | "
          f"{record['src_lang']}->{record['tgt_lang']} | "
          f"{record['n_src_sents']}->{record['n_tgt_sents']} sents | "
          f"{len(record['pairs'])} beads | {record['align_seconds']}s ===")
    if not show_beads:
        return
    print(f"    {record['title']}")
    for pair in record["pairs"]:
        print(f"  [{record['src_lang']}] {pair['src']}")
        print(f"  [{record['tgt_lang']}] {pair['tgt']}")
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

    # Single shared segmentation step: split every thesis ONCE (SaT), up front.
    t0 = time.perf_counter()
    n = None if args.is_full else args.n
    prepared = prepare_theses(args.subset, SRC_LANG, list(TARGET_LANGS), n,
                              require_all_targets=False,
                              model=get_sat_model(device))
    print(f"Segmented {len(prepared)} theses in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    labse_model, bert_encoder = load_labse_and_encoder(args.tool, args.embed, device)
    print(f"Loaded models in {time.perf_counter() - t0:.1f}s\n")

    suffix = "full" if args.is_full else str(args.n)
    # Outputs are nested under the SaT model name, so results from different
    # segmenters don't collide (segmentation is the one shared input to every
    # tool/embedding, so it defines the run).
    out_dir = DATA_DIR / SAT_MODEL_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tool}_{args.embed}_{suffix}.jsonl"

    records_written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for thesis in prepared:
            for record in align_thesis(thesis, args.tool, args.embed, labse_model,
                                       bert_encoder, device):
                print_record(record, show_beads=not args.is_full)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1

    print(f"Wrote {records_written} records "
          f"({len(prepared)} theses x up to {len(TARGET_LANGS)} targets) "
          f"to {out_path.resolve()}")


if __name__ == "__main__":
    main()
