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
  always guard for it (`kurdish_parallel/align.py:get_summary`).
- Abstracts are short (~200–280 words). Parallel pairs we care about:
  **kmr_Latn → tur_Latn** and **kmr_Latn → eng_Latn**.

## Core design decision (do not undo without reason)

Bertalign's bundled language detector (lingua) and splitter (sentence-splitter)
only know ~25 European languages — **Kurdish is not among them**. But LaBSE
embeds Kurdish fine. So:

- We **pre-split sentences ourselves** and call `Bertalign(..., is_split=True)`.
- Kurdish (Latin) is split with the **Turkish** sentence-splitter as a proxy;
  `*_Arab` keys use a regex fallback. See
  `kurdish_parallel/align.py:split_sentences` and `SPLITTER_LANG`.
- This means language detection quality is irrelevant to results — only
  segmentation matters.

### Package layout (all code in `kurdish_parallel/`, except Modal)

All library code lives in the `kurdish_parallel/` package; only the Modal
orchestration sits outside it (in `modal_app/`, which imports the package).

| Module | Holds |
|---|---|
| `kurdish_parallel/align.py` | Data load (`load_rows`, `get_summary`), segmentation (`split_sentences`, `SPLITTER_LANG`), `resolve_device`, and the CLI `main`. Re-exports the aligners/scorer so `import kurdish_parallel.align as A` still exposes everything. |
| `kurdish_parallel/tools/tools.py` | Alignment algorithms (`align_bertalign`, `align_vecalign`) + `align_pair` dispatch + `TOOLS`. |
| `kurdish_parallel/tools/embeddings.py` | Embed fns (`labse_embed_fn`, `sonar_embed_fn`, `_sonar_pipeline`) + neutral `score_pairs` + `EMBEDS`. |

`align.py` is the single source of truth for the data/CLI layer; the Modal app
imports the package rather than duplicating logic. `align.py` uses package
relative imports, so run the CLI as a module: `python -m kurdish_parallel.align`
(not `python align.py`).

## Tools (`--tool`) and embeddings (`--embed`)

`kurdish_parallel/tools/tools.py:align_pair(tool, embed, encoder, src_sents,
tgt_sents, src_lang, tgt_lang)` dispatches to:

- `align_bertalign` — `Bertalign(..., is_split=True, model=encoder)`. LaBSE-only
  (Bertalign wraps sentence-transformers); `--embed` must be `labse`.
- `align_vecalign` — drives vecalign's own DP (`vecalign.dp_utils.vecalign`)
  **in-process**: build unique overlap lines (`yield_overlaps`), embed them,
  `make_doc_embedding`, then `vecalign(...)`. No temp files, no subprocess.
  Embeddings come from per-side embed functions:
  - `--embed labse` → `labse_embed_fn(labse_model)` (one LaBSE model, both sides).
  - `--embed sonar` → `sonar_embed_fn(lang, device)` per side, one cached SONAR
    pipeline (`_sonar_pipeline`); `lang` is the dataset key (a FLORES-200 code)
    passed as SONAR's `source_lang`. SONAR natively covers kmr_Latn + ckb_Arab.

`score_pairs(labse_model, pairs)` ALWAYS uses LaBSE as a **neutral scorer**, so
`mean_score` is comparable across every tool/embed combo regardless of which
embeddings drove the alignment. The scorer is a plain SentenceTransformer
(bertalign's `Encoder.model`, or a freshly loaded `SentenceTransformer("LaBSE")`
in the SONAR image) — intentionally decoupled from the aligner's embeddings.

All aligners return the same `list[(src_text, tgt_text)]`. Keep that contract if
you add a tool.

### Two Modal images (don't merge them)

- `labse_image`: bertalign (`Python>=3.12`) + vecalign + sentence-transformers.
- `sonar_image`: vecalign + `sonar-space` (fairseq2) + sentence-transformers,
  **no bertalign**. SONAR's fairseq2 asset cache goes to the volume via
  `XDG_CACHE_HOME`. The entrypoint dispatches `--embed sonar` to
  `align_sonar_remote`, everything else to `align_labse_remote`; both call the
  shared `_run_alignment`.

Installs: vecalign from git (`@main`) compiles a Cython extension (`dp_core`) →
images need `build-essential` + `git`, local needs Xcode CLT. **LASER3 is NOT
usable** (`laser_encoders` → fairseq v1, incompatible with Python 3.12/torch 2.x
and bertalign). SONAR is the replacement but is **CC-BY-NC** (non-commercial).

## Running

Local (pipenv — pulls torch locally, uses Apple MPS):
```bash
pipenv install
pipenv run python -m kurdish_parallel.align --n 3   # kmr_Latn -> tur_Latn
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
  - Local CLI: `{tool}_{embed}_{src}_{tgt}_{n}.jsonl`, or `…_full.jsonl` when
    `--n` is omitted.
  - Modal smoke test: `{tool}_{embed}_smoke_{n}.jsonl` when `--n` > 0, else
    `{tool}_{embed}_full.jsonl` (one record per thesis-and-target). `--n 0` /
    omitted means the full subset. (`embed` is in the name so vecalign/labse and
    vecalign/laser don't collide.)
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
