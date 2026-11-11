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
  always guard for it (`kurdish_parallel_tools_comparison/align.py:get_summary`).
- Abstracts are short (~200–280 words). Parallel pairs we care about:
  **kmr_Latn → tur_Latn** and **kmr_Latn → eng_Latn**.

## Core design decision (do not undo without reason)

Bertalign's bundled language detector (lingua) and splitter (sentence-splitter)
only know ~25 European languages — **Kurdish is not among them**. But LaBSE
embeds Kurdish fine. So:

- We **pre-split sentences ourselves** and call `Bertalign(..., is_split=True)`.
- All languages are split with **wtpsplit SaT** (`sat-12l-sm`), a neural
  multilingual segmenter that is language-agnostic — it covers Kurdish in both
  Latin and Arabic script, so there is no Turkish-proxy and no `*_Arab` regex
  fallback any more. See `kurdish_parallel_tools_comparison/align.py:split_sentences`,
  `get_sat_model`, and `SAT_MODEL_NAME`.
- Segmentation is a **single shared step**: `align.py:prepare_theses` splits each
  thesis once and every tool/embedding aligns the *identical* sentence lists
  (the Modal run does this once in `segment_remote` and ships the result to all
  three alignment images). Each returned sentence has its whitespace collapsed
  (`_clean_sentence`), which also keeps bertalign's `is_split=True` round-trip
  (`"\n".join(sents)` → `splitlines()`) exact.
- This means language detection quality is irrelevant to results — only
  segmentation matters.

### Package layout (all code in `kurdish_parallel_tools_comparison/`, except Modal)

All library code lives in the `kurdish_parallel_tools_comparison/` package; only the Modal
orchestration sits outside it (in `modal_app/`, which imports the package).

| Module | Holds |
|---|---|
| `kurdish_parallel_tools_comparison/align.py` | Data load (`load_rows`, `get_summary`), the single shared segmentation step (`split_sentences`, `get_sat_model`, `prepare_theses`), `resolve_device`, and the CLI `main`. Re-exports the aligners so `import kurdish_parallel_tools_comparison.align as A` still exposes everything. |
| `kurdish_parallel_tools_comparison/tools/tools.py` | Alignment algorithms (`align_bertalign`, `align_vecalign`, `align_sentalign`) + `align_pair` dispatch + `TOOLS`. |
| `kurdish_parallel_tools_comparison/tools/_sentalign_core.py` | SentAlign's pure-Python grid-DP core, **vendored unmodified** from `steinst/SentAlign` (`align_anchors.py`, Apache-2.0). Called in-process by `align_sentalign`; no Cython/Gale-Church anchoring (not needed at abstract size). |
| `kurdish_parallel_tools_comparison/tools/embeddings.py` | Embed fns (`labse_embed_fn`, `sonar_embed_fn`, `_sonar_pipeline`) + `EMBEDS`. |

`align.py` is the single source of truth for the data/CLI layer; the Modal app
imports the package rather than duplicating logic. `align.py` uses package
relative imports, so run the CLI as a module: `python -m kurdish_parallel_tools_comparison.align`
(not `python align.py`).

## Tools (`--tool`) and embeddings (`--embed`)

`kurdish_parallel_tools_comparison/tools/tools.py:align_pair(tool, embed, encoder, src_sents,
tgt_sents, src_lang, tgt_lang)` dispatches to:

- `align_bertalign` — `Bertalign(..., is_split=True, model=encoder)`. LaBSE-only
  (Bertalign wraps sentence-transformers); `--embed` must be `labse`.
- `align_sentalign` — SentAlign's grid-DP core (`_sentalign_core.align_anchors`)
  run **in-process** on a single block `((-1,-1),(n_src,n_tgt))`, which is how
  SentAlign aligns any document small enough to skip Gale-Church anchoring (all
  our ~14-sentence abstracts). Builds the exact concatenation→vector dicts the DP
  looks up from `labse_embed_fn` (L2-normalized, so the DP's dot product is
  cosine). LaBSE-only, like Bertalign; `--embed` must be `labse`.
- `align_vecalign` — drives vecalign's own DP (`vecalign.dp_utils.vecalign`)
  **in-process**: build unique overlap lines (`yield_overlaps`), embed them,
  `make_doc_embedding`, then `vecalign(...)`. No temp files, no subprocess.
  Embeddings come from per-side embed functions:
  - `--embed labse` → `labse_embed_fn(labse_model)` (one LaBSE model, both sides).
  - `--embed sonar` → `sonar_embed_fn(lang, device)` per side, one cached SONAR
    pipeline (`_sonar_pipeline`); `lang` is the dataset key (a FLORES-200 code)
    passed as SONAR's `source_lang`. SONAR natively covers kmr_Latn + ckb_Arab.
  - `--embed laser` → `laser_embed_fn(lang)` per side, one cached LASER3
    pipeline (`_laser_pipeline`); `lang` is the dataset key (FLORES-200 code).
    LASER3 has a dedicated kmr_Latn encoder; tur/eng use the LASER2 base.
    `encode_sentences(..., normalize_embeddings=True)` returns unit vectors.
    **Legacy stack — Modal `laser_image` only** (fairseq v1); see below.

There is **no intrinsic scoring** of alignments. Output records carry only the
aligned `pairs` (plus metadata); quality is judged against an external gold set.
LaBSE is loaded only when it is the aligning embedding (`--embed labse`, or
bertalign); `--embed sonar` loads no LaBSE at all.

All aligners return the same `list[(src_text, tgt_text)]`. Keep that contract if
you add a tool.

### SentAlign (vendored DP core, not the pip package)

SentAlign (Steingrímsson et al., 2023) **is** in the comparison as the
`sentalign` tool — it's a third LaBSE aligner whose *DP algorithm* differs from
Bertalign and vecalign (length-penalty scoring + null/merge re-evaluation). It
adds no new embedding signal (it's LaBSE, like Bertalign), so it tests whether a
different DP over the same LaBSE vectors helps Kurdish; the embedding axis is
still carried by SONAR and LASER.

We do **not** install the pip/git package: it pins Python 3.8 / transformers
4.22.2 / numpy 1.22.1 (won't coexist with our Python 3.13 + torch 2.x) and its
entrypoint does argparse + file I/O at import. Instead we **vendor its
pure-Python core** (`_sentalign_core.py`, copied unmodified from `align_anchors.py`)
and call `align_anchors` directly. Its Cython anchoring / Gale-Church modules are
not vendored and not needed at abstract size. Keep `_sentalign_core.py`
byte-for-byte as upstream (only the header comment is ours); put any adaptation
in `align_sentalign`.

### Four Modal images (don't merge them)

- `seg_image`: **wtpsplit SaT only**, on CPU. The single shared segmentation
  step (`segment_remote` → `align.prepare_theses`): it splits the `n` theses
  once and the result is handed to the three alignment images, so SaT never has
  to coexist with bertalign / fairseq2 / fairseq-v1, and no alignment image
  re-splits. The SaT model caches on the volume via `HF_HOME`.
- `labse_image`: bertalign (`Python>=3.12`) + vecalign + sentence-transformers.
  Also runs `sentalign` (pure-Python `_sentalign_core`, no extra deps). The three
  LaBSE tools share one container so LaBSE loads once.
- `sonar_image`: vecalign + `sonar-space` (fairseq2) + sentence-transformers,
  **no bertalign**. SONAR's fairseq2 asset cache goes to the volume via
  `XDG_CACHE_HOME`.
- `laser_image`: vecalign + `laser_encoders` (LASER3) on a **legacy base**
  (`Python 3.10` + `torch==1.13.1` + `fairseq==0.12.2`), **no bertalign**.
  sentence-transformers `2.2.2` loads LaBSE for scoring only. LASER3 model
  downloads cache on the volume via `LASER_MODEL_DIR`. `torch==1.13.1` is the
  CUDA 11.7 build and LASER3 self-moves to GPU when one is attached, so run it
  on a **T4** (`--gpu T4`); an **L4** (Ada) needs CUDA >=11.8 and won't work with
  this torch. This is the fragile image — its pins are a self-consistent set
  (fairseq v1 needs `numpy<1.24`; s-t 2.2.2 needs `huggingface-hub<0.26`); don't
  bump them casually.

The entrypoint `main(n, gpu, device)` first runs `segment_remote` **once**
(blocking) to segment the `n` theses, then **spawns the three alignment images
at once** and waits on them concurrently, passing each the shared `prepared`
sentence lists: `align_labse_remote` (runs bertalign + vecalign + sentalign),
`align_sonar_remote` (vecalign/sonar), `align_laser_remote` (vecalign/laser) —
five options total. All call the shared `_run_alignment`, which takes
`(prepared, src, device, runs)` — `runs: list[(tool, embed)]` — and aligns the
identical pre-segmented theses, returning one result dict per run. It no longer
loads the dataset or splits anything. One image failing (laser is the fragile
one) is caught per-image so the others still write their output.

Installs: vecalign from git (`@main`) compiles a Cython extension (`dp_core`) →
images need `build-essential` + `git`, local needs Xcode CLT. **LASER3 can't
share the other two images** (`laser_encoders` → fairseq v1 is incompatible with
Python 3.12/torch 2.x, bertalign, and fairseq2), so it runs only in its own
legacy `laser_image`. SONAR remains the modern replacement but is **CC-BY-NC**
(non-commercial); LASER3 (BSD) is the commercially-usable embedding option.

## Running

Local (pipenv — pulls torch locally, uses Apple MPS):
```bash
pipenv install
pipenv run python -m kurdish_parallel_tools_comparison.align --n 3   # kmr_Latn -> every other language
```
- CLI args: `--subset`, `--tool`, `--embed`, `--n`, `--is-full`, `--verbose`.
  `--n` is required unless `--is-full` is passed (full subset, ignores `--n`).
  Source is fixed (`SRC_LANG = kmr_Latn`) and every other summary language
  (`TARGET_LANGS`) is a target; device is always `resolve_device("auto")`. (No
  `--src/--tgt/--device/--out` — removed; outputs always go to `data/`.)

Modal comparison run (recommended; no local torch, runs on a GPU, both pairs).
**One command runs all five options concurrently on the same `n` theses** — the
pool you then hand-annotate into a gold set and score each option against:
```bash
modal run modal_app/app.py --n 20
```
- One `local_entrypoint` (`main`), so no `::name` needed.
- **`--n` is REQUIRED** (must be ≥ 1) and is the number of theses **each** option
  aligns — not a total split across options. Every option aligns the identical
  `n` theses with identical segmentation, so the comparison is deterministic.
  There is no `--tool`/`--embed` (removed) and no full-subset mode here: all five
  options always run. Writes one file per option; a failing image is reported but
  doesn't abort the rest.
- Source/targets are **fixed** in the app (`SRC_LANG`, `TARGET_LANGS`); the CLI
  only takes `--n`, `--gpu`, `--device`. `--gpu` drives every option and defaults
  to `T4` (LASER3 needs it; an L4 won't work with its torch 1.13).
- `modal` lives in the user's project virtualenvs, not on the system Python
  (Homebrew Python blocks `pip install` via PEP 668). It is installed in THIS
  project's pipenv venv (`kurdish_parallel_tools_comparison-*`), so `modal run …` works with the
  venv active. Another copy is at
  `~/.local/share/virtualenvs/sorjin-modal-ZLvY6A0z/bin/modal`. Token at
  `~/.modal.toml` (already configured).

## Conventions

- Output files go in `data/`.
  - Local CLI: `{tool}_{embed}_{n}.jsonl`, or `{tool}_{embed}_full.jsonl` with
    `--is-full`. Source is fixed (kmr_Latn) and all targets go in one file,
    so src/tgt are no longer in the name. NOTE: the `…_full.jsonl` form matches
    the Modal full name — local covers all `TARGET_LANGS`, Modal only tur+eng, so
    don't mix them in the same `data/`.
  - Modal comparison run: `{tool}_{embed}_{n}.jsonl`, one file per option (five
    files: `bertalign_labse_`, `vecalign_labse_`, `sentalign_labse_`,
    `vecalign_sonar_`, `vecalign_laser_`), one record per thesis-and-target.
    `tool` and `embed` are both in the name so the five never collide. These are
    tur+eng only; the local files share the same `{tool}_{embed}_{n}.jsonl` form
    but cover all targets — keep them straight.
  - Also `result_{n}.json`: the (thesis, target) pairs whose alignment **differs
    across ≥ 2 options**, each with the options grouped by identical alignment —
    the shortlist to hand-annotate for the gold set. The entrypoint prints only a
    short summary (per-option record counts + how many pairs differ); the detail
    lives in these files.
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
