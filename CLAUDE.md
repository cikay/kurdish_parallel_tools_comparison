# CLAUDE.md

Guidance for working in this repo with Claude Code.

## What this is

Sentence alignment of Kurdish parallel data from the Hugging Face dataset
`muzaffercky/kurdish-theses-yok` (thesis abstracts), using the `bertalign==2.0.1`
PyPI fork (LaBSE embeddings). See `README.md` for the full picture.

## Key facts about the data

- Dataset is Parquet, one subset per Kurdish variant. Kurmanji =
  `kmr_Latn/train-00000-of-00001.parquet` (408 rows).
- Abstracts live in a `summaries` **struct** column with keys `kmr_Latn`,
  `zza_Latn`, `ckb_Arab`, `tur_Latn`, `eng_Latn`. **Any key may be `null`** —
  always guard for it (`align.py:get_summary`).
- Abstracts are short (~200–280 words). Parallel pairs we care about:
  **kmr_Latn → tur_Latn** and **kmr_Latn → eng_Latn**.

## Core design decision (do not undo without reason)

Bertalign's bundled language detector (lingua) and splitter (sentence-splitter)
only know ~25 European languages — **Kurdish is not among them**. But LaBSE
embeds Kurdish fine. So:

- We **pre-split sentences ourselves** and call `Bertalign(..., is_split=True)`.
- Kurdish (Latin) is split with the **Turkish** sentence-splitter as a proxy;
  `*_Arab` keys use a regex fallback. See `align.py:split_sentences` and
  `SPLITTER_LANG`.
- This means language detection quality is irrelevant to results — only
  segmentation matters.

The `align.py` functions (`resolve_device`, `load_rows`, `get_summary`,
`split_sentences`) are the single source of truth; the Modal app imports them
rather than duplicating logic.

## Running

Local (pipenv — pulls torch locally, uses Apple MPS):
```bash
pipenv install
pipenv run python align.py --n 3                 # kmr_Latn -> tur_Latn
```

Modal smoke test (recommended; no local torch, runs on a GPU, both pairs):
```bash
modal run modal_app/app.py --n 3
```
- One `local_entrypoint` (`main`), so no `::name` needed.
- Source/targets are **fixed** in the app (`SRC_LANG`, `TARGET_LANGS`); the CLI
  only takes `--n`, `--gpu`, `--device`.
- `modal` lives in the user's project virtualenvs, not on the system Python.
  One known path: `~/.local/share/virtualenvs/sorjin-modal-ZLvY6A0z/bin/modal`.
  The Modal token is at `~/.modal.toml` (already configured).

## Conventions

- Output files go in `results/`.
  - Local CLI: `{src}_{tgt}_{n}.jsonl`.
  - Modal smoke test: `smoke_{n}.jsonl` (one record per thesis-and-target).
- Device: `resolve_device("auto")` → MPS on Apple Silicon, CUDA on a Modal GPU,
  else CPU. The LaBSE encoder is loaded **once** and reused across summaries.
- The Modal image caches HF downloads on a Volume (`HF_HOME` → `/cache`); call
  `cache_volume.commit()` after new downloads.

## Gotchas

- First Modal run builds the image (installs torch) and downloads LaBSE
  (~1.8 GB) — several minutes. Reruns reuse the image + Volume and are fast.
- Don't add `faiss`/`googletrans` — the whole reason for the fork is to avoid
  them. Don't switch to the original `bfsujason/bertalign` repo (pins
  `faiss-gpu`).
- Downloaded dataset files are untrusted; the loaders pass paths as arguments
  and never execute anything from the data.

## Attribution for commits/PRs (when asked to create them)

- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- PR footer: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
