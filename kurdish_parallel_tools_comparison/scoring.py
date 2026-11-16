#!/usr/bin/env python
"""Score alignment outputs against the hand-annotated gold set (P / R / F1).

This module is the COMPARISON layer and is deliberately **decoupled from Modal**
(and from the aligners): it depends only on the standard library, so it runs
anywhere -- local CLI, a notebook, or inside a thin Modal wrapper
(``modal_app/score_app.py``). It reads the JSONL the alignment run writes plus
``gold_set.json`` and reports precision / recall / F1 per option.

Why precision / recall / F1
---------------------------
There is no intrinsic score for an alignment; quality is judged against a gold
set. The standard sentence-alignment metric (Bertalign, Vecalign, SentAlign all
report it) compares the *set of alignment links* a tool proposes with the gold
links:

    precision = |pred_links  ∩  gold_links| / |pred_links|
    recall    = |pred_links  ∩  gold_links| / |gold_links|
    F1        = 2 * precision * recall / (precision + recall)

Two granularities are reported, as is conventional:

* **lax** (the headline): every bead ``(src_idx_set, tgt_idx_set)`` is expanded
  into its sentence-to-sentence links (the Cartesian product of the two index
  sets). A 1→[1,2] bead becomes links {(1,1), (1,2)}. Partial overlaps get
  partial credit, so a tool that gets 1→1 right but misses the 1→2 half of a
  gold 1→[1,2] still scores on the (1,1) link. Null beads (one side empty --
  an insertion/deletion) contribute no links and simply drop out.
* **strict**: a predicted bead counts as correct only if it equals a gold bead
  exactly (same src index set AND same tgt index set). Null beads are excluded
  on both sides (the gold set annotates only real alignments).

Metrics are **micro-averaged**: links are pooled across every scored thesis so
each link weighs the same, which is the standard for these corpora. A per-target
(tur/eng) breakdown is also produced.

Gold format (``gold_set.json``)
-------------------------------
    [{"tez_id": "1003999",
      "alignments": [{"tgt_lang": "tur_Latn",
                      "links": [{"src": [0], "tgt": [0]},
                                {"src": [1], "tgt": [1, 2]}, ...]}]}]

Indices are 0-based and refer to the shared segmentation (the same indices the
Modal ``result_{n}.json`` prints via ``numbered(sents)``). The gold set may be
incomplete -- only the (thesis, target) pairs it actually contains are scored,
and everything else is reported as skipped, so a partial gold set still works.

Prediction format
-----------------
One JSONL file per option (``{tool}_{embed}_{n}.jsonl``), one record per
(thesis, target). A record carries the text beads in ``pairs`` plus the shared
``src_sents`` / ``tgt_sents``; this module reconstructs each bead's 0-based index
sets from them (``reconstruct_beads``). If a record already carries an explicit
``links`` list (same shape as the gold ``links``), that is used directly and no
reconstruction happens -- so the scorer keeps working if the aligners start
emitting indices themselves.

CLI
---
    python -m kurdish_parallel_tools_comparison.scoring --n 20
    python -m kurdish_parallel_tools_comparison.scoring --n 20 --gold gold_set.json --verbose
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# The five options the Modal comparison run produces, in report order.
CANONICAL_OPTIONS = (
    "bertalign_labse",
    "vecalign_labse",
    "sentalign_labse",
    "vecalign_sonar",
    "vecalign_laser",
)

# Repo root is the package's parent; default gold + data locations live there.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = REPO_ROOT / "gold_set.json"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
# Segmentation defines a run, so outputs are nested under the SaT model name.
# Imported lazily-ish to avoid a hard dependency when only scoring.
try:  # pragma: no cover - trivial
    from .align import SAT_MODEL_NAME
except Exception:  # pragma: no cover - keep scoring importable standalone
    SAT_MODEL_NAME = "sat-12l-sm"


# A bead is a pair of sorted index tuples: (src_indices, tgt_indices).
Bead = tuple[tuple[int, ...], tuple[int, ...]]


def normalize_ws(s: str) -> str:
    """Collapse every whitespace run to a single space and strip the ends.

    Bead text is a space-join of ``_clean_sentence``'d sentences, so comparing
    under this normalization matches the reconstruction exactly regardless of
    stray whitespace differences.
    """
    return re.sub(r"\s+", " ", s or "").strip()


def _as_bead(link: dict) -> Bead:
    """Turn a gold/record ``{"src": [...], "tgt": [...]}`` link into a Bead."""
    src = tuple(sorted(int(i) for i in link.get("src", [])))
    tgt = tuple(sorted(int(i) for i in link.get("tgt", [])))
    return (src, tgt)


def _find_windows(text: str, sents: list[str], max_bead: int = 16) -> list[list[int]]:
    """Return every contiguous sentence window whose space-join equals ``text``.

    Each bead's text is ``" ".join(sents[i:i+k])`` for some contiguous run, so we
    recover its indices by scanning for the run(s) that reproduce the text under
    whitespace normalization. Returns all matching windows (ambiguity is resolved
    by the caller); an empty / whitespace text returns ``[[]]`` (the null bead).
    """
    t = normalize_ws(text)
    if t == "":
        return [[]]
    windows: list[list[int]] = []
    for i in range(len(sents)):
        for k in range(1, max_bead + 1):
            if i + k > len(sents):
                break
            if normalize_ws(" ".join(sents[i:i + k])) == t:
                windows.append(list(range(i, i + k)))
    return windows


def _pick_window(windows: list[list[int]], cursor: int) -> list[int]:
    """Choose one window, preferring the first starting at/after ``cursor``.

    Alignments are usually monotonic, so preferring a window at/after the running
    cursor keeps the natural order; but SentAlign can emit overlapping beads
    (a sentence reused across two beads), so we fall back to the globally first
    window when none starts at/after the cursor. Deterministic either way.
    """
    if not windows:
        return []
    for w in windows:
        if w and w[0] >= cursor:
            return w
    return windows[0]


def reconstruct_beads(record: dict) -> list[Bead]:
    """Index-based beads for one prediction record.

    Uses the record's explicit ``links`` when present (forward-compatible with
    aligners that emit indices); otherwise reconstructs each bead's 0-based index
    sets from the text ``pairs`` and the shared ``src_sents`` / ``tgt_sents`` by
    matching each bead's text back to its contiguous sentence window.
    """
    if record.get("links"):
        return [_as_bead(link) for link in record["links"]]

    src_sents = record["src_sents"]
    tgt_sents = record["tgt_sents"]
    beads: list[Bead] = []
    src_cursor = tgt_cursor = 0
    for pair in record["pairs"]:
        s_win = _pick_window(_find_windows(pair["src"], src_sents), src_cursor)
        t_win = _pick_window(_find_windows(pair["tgt"], tgt_sents), tgt_cursor)
        if s_win:
            src_cursor = s_win[-1] + 1
        if t_win:
            tgt_cursor = t_win[-1] + 1
        beads.append((tuple(s_win), tuple(t_win)))
    return beads


def bead_link_set(beads: list[Bead]) -> set[tuple[int, int]]:
    """Expand beads into the set of (src_idx, tgt_idx) links (lax granularity).

    A bead contributes the Cartesian product of its two index sets; null beads
    (one side empty) contribute nothing.
    """
    links: set[tuple[int, int]] = set()
    for src, tgt in beads:
        for i in src:
            for j in tgt:
                links.add((i, j))
    return links


def strict_bead_set(beads: list[Bead]) -> set[Bead]:
    """Beads as a set, for strict (exact-bead) scoring; null beads dropped.

    This only normalizes the beads into a comparable set and excludes null beads
    (one side empty). The exact match itself is the set intersection the caller
    takes (``pred_set & gold_set``): because each bead is a pair of *sorted*
    index tuples, two beads are equal -- and so intersect -- iff they have the
    identical src index set AND the identical tgt index set.
    """
    return {(src, tgt) for src, tgt in beads if src and tgt}


def _prf(tp: int, n_pred: int, n_gold: int) -> dict:
    """Precision / recall / F1 from true-positive, predicted, and gold counts."""
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "n_pred": n_pred,
        "n_gold": n_gold,
    }


class _Counts:
    """Running micro-average tallies for one slice (an option, or option×lang)."""

    def __init__(self) -> None:
        self.lax_tp = self.lax_pred = self.lax_gold = 0
        self.strict_tp = self.strict_pred = self.strict_gold = 0
        self.n_pairs = 0  # scored (thesis, target) pairs

    def add(self, pred_beads: list[Bead], gold_beads: list[Bead]) -> None:
        self.n_pairs += 1

        pred_links = bead_link_set(pred_beads)
        gold_links = bead_link_set(gold_beads)
        self.lax_tp += len(pred_links & gold_links)
        self.lax_pred += len(pred_links)
        self.lax_gold += len(gold_links)

        pred_strict = strict_bead_set(pred_beads)
        gold_strict = strict_bead_set(gold_beads)
        self.strict_tp += len(pred_strict & gold_strict)
        self.strict_pred += len(pred_strict)
        self.strict_gold += len(gold_strict)

    def result(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "lax": _prf(self.lax_tp, self.lax_pred, self.lax_gold),
            "strict": _prf(self.strict_tp, self.strict_pred, self.strict_gold),
        }


def load_gold(gold: str | Path | list) -> dict[tuple[str, str], list[Bead]]:
    """Load the gold set into ``{(tez_id, tgt_lang): [bead, ...]}``.

    Accepts a path or an already-parsed list. Unfinished annotations are skipped
    so a partial gold set scores fairly: an alignment is kept only when it has a
    non-empty ``tgt_lang`` AND at least one link. An entry with ``links: []`` is
    a thesis queued for annotation, not a claim that it has no alignments, so
    scoring it would count every predicted link as a false positive and wrongly
    crater precision.
    """
    data = gold if isinstance(gold, list) else json.loads(Path(gold).read_text("utf-8"))
    out: dict[tuple[str, str], list[Bead]] = {}
    for thesis in data:
        tez_id = str(thesis["tez_id"])
        for alignment in thesis.get("alignments", []):
            tgt_lang = alignment.get("tgt_lang") or ""
            links = alignment.get("links") or []
            if not tgt_lang or not links:
                continue  # placeholder / not-yet-annotated entry
            out[(tez_id, tgt_lang)] = [_as_bead(l) for l in links]
    return out


def load_predictions(path: str | Path) -> dict[tuple[str, str], dict]:
    """Load one option's JSONL into ``{(tez_id, tgt_lang): record}``."""
    out: dict[tuple[str, str], dict] = {}
    for line in Path(path).read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[(str(rec["tez_id"]), rec["tgt_lang"])] = rec
    return out


def score_option(records: dict[tuple[str, str], dict],
                 gold: dict[tuple[str, str], list[Bead]]) -> dict:
    """Score one option's records against the gold set.

    Only (thesis, target) pairs present in BOTH the gold set and this option's
    records are scored. Returns the overall micro-averaged result, a per-target
    breakdown, and the keys that were scored vs. missing from the predictions.
    """
    overall = _Counts()
    per_lang: dict[str, _Counts] = {}
    scored: list[tuple[str, str]] = []
    missing_pred: list[tuple[str, str]] = []

    for key, gold_beads in sorted(gold.items()):
        rec = records.get(key)
        if rec is None:
            missing_pred.append(key)
            continue
        pred_beads = reconstruct_beads(rec)
        overall.add(pred_beads, gold_beads)
        per_lang.setdefault(key[1], _Counts()).add(pred_beads, gold_beads)
        scored.append(key)

    return {
        "overall": overall.result(),
        "per_lang": {lang: c.result() for lang, c in sorted(per_lang.items())},
        "scored": scored,
        "missing_pred": missing_pred,
    }


def compare(gold: str | Path | list,
            data_dir: str | Path,
            n: int | None = None,
            suffix: str | None = None,
            options: tuple[str, ...] = CANONICAL_OPTIONS) -> dict:
    """Score every available option against the gold set.

    ``data_dir`` holds the ``{option}_{suffix}.jsonl`` files (``suffix`` defaults
    to ``str(n)``). Options whose file is absent are skipped and noted. Returns a
    summary dict (also what the CLI prints and writes to ``scores_{suffix}.json``).
    """
    if suffix is None:
        if n is None:
            raise ValueError("compare needs either n or an explicit suffix")
        suffix = str(n)
    data_dir = Path(data_dir)
    gold_map = load_gold(gold)

    results: dict[str, dict] = {}
    missing_files: list[str] = []
    for option in options:
        path = data_dir / f"{option}_{suffix}.jsonl"
        if not path.exists():
            missing_files.append(option)
            continue
        results[option] = score_option(load_predictions(path), gold_map)

    return {
        "n": n,
        "suffix": suffix,
        "data_dir": str(data_dir),
        "n_gold_pairs": len(gold_map),
        "options": list(results.keys()),
        "missing_option_files": missing_files,
        "results": results,
    }


def compare_records(gold: str | Path | list,
                    option_records: dict[str, list[dict]],
                    n: int | None = None,
                    suffix: str | None = None) -> dict:
    """Same as :func:`compare`, but from in-memory records instead of files.

    ``option_records`` maps an option name to its list of record dicts (each the
    JSON object one JSONL line holds). This is the filesystem-free entry point
    the Modal wrapper calls, so the comparison logic stays here (decoupled from
    Modal) and the Modal file only moves data. Returns the same summary shape.
    """
    if suffix is None:
        suffix = str(n) if n is not None else "mem"
    gold_map = load_gold(gold)
    results: dict[str, dict] = {}
    for option in CANONICAL_OPTIONS:
        records = option_records.get(option)
        if not records:
            continue
        keyed = {(str(r["tez_id"]), r["tgt_lang"]): r for r in records}
        results[option] = score_option(keyed, gold_map)
    return {
        "n": n,
        "suffix": suffix,
        "data_dir": None,
        "n_gold_pairs": len(gold_map),
        "options": list(results.keys()),
        "missing_option_files": [o for o in CANONICAL_OPTIONS
                                 if o not in option_records],
        "results": results,
    }


def _fmt_row(name: str, res: dict) -> str:
    lax, strict = res["lax"], res["strict"]
    return (f"{name:<18} {res['n_pairs']:>5} "
            f"{lax['precision']:>7.3f} {lax['recall']:>7.3f} {lax['f1']:>7.3f} "
            f"   {strict['precision']:>7.3f} {strict['recall']:>7.3f} {strict['f1']:>7.3f}")


def format_report(summary: dict, verbose: bool = False) -> str:
    """Render a human-readable report (returned as a string so callers choose)."""
    lines: list[str] = []
    lines.append(f"Gold set: {summary['n_gold_pairs']} (thesis, target) pairs "
                 f"| suffix: {summary['suffix']} "
                 f"| {summary['data_dir'] or '(in-memory)'}")
    if summary["missing_option_files"]:
        lines.append("Missing option files (skipped): "
                     + ", ".join(summary["missing_option_files"]))
    lines.append("")
    header = (f"{'option':<18} {'pairs':>5} "
              f"{'P':>7} {'R':>7} {'F1':>7}    {'P':>7} {'R':>7} {'F1':>7}")
    lines.append(f"{'':<24}{'------ lax (per-link) ------':^23}   "
                 f"{'----- strict (bead) ------':^23}")
    lines.append(header)
    lines.append("-" * len(header))

    # Rank options by lax F1 (the headline metric), best first.
    ranked = sorted(summary["results"].items(),
                    key=lambda kv: kv[1]["overall"]["lax"]["f1"], reverse=True)
    for option, res in ranked:
        lines.append(_fmt_row(option, res["overall"]))
        if verbose:
            for lang, langres in res["per_lang"].items():
                lines.append(_fmt_row(f"  {lang}", langres))
            if res["missing_pred"]:
                miss = ", ".join(f"{t}/{l}" for t, l in res["missing_pred"])
                lines.append(f"  (no prediction for: {miss})")
    lines.append("")
    lines.append("Columns: P=precision R=recall F1 (micro-averaged over links). "
                 "lax = sentence-link overlap; strict = exact bead match.")
    if ranked:
        best = ranked[0]
        lines.append(f"Best lax F1: {best[0]} ({best[1]['overall']['lax']['f1']:.3f})")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=None,
                        help="Run size: scores {option}_{n}.jsonl. Omit with --suffix.")
    parser.add_argument("--suffix", default=None,
                        help="Explicit file suffix (e.g. 'full'); overrides --n.")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD),
                        help=f"Gold set JSON (default: {DEFAULT_GOLD}).")
    parser.add_argument("--no-write", action="store_true",
                        help="Print the report but don't write scores_{suffix}.json.")
    parser.add_argument("--verbose", action="store_true",
                        help="Also show the per-target (tur/eng) breakdown.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.n is None and args.suffix is None:
        parser.error("provide --n (or --suffix)")

    data_dir = DEFAULT_DATA_DIR / SAT_MODEL_NAME
    summary = compare(args.gold, data_dir, n=args.n, suffix=args.suffix)

    print(format_report(summary, verbose=args.verbose))

    if not args.no_write:
        out_path = data_dir / f"scores_{summary['suffix']}.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
