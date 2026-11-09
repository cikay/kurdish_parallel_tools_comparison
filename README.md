# kurdish_parallel

Sentence alignment of Kurdish parallel data from the
[`muzaffercky/kurdish-theses-yok`](https://huggingface.co/datasets/muzaffercky/kurdish-theses-yok)
thesis summaries, using [Bertalign](https://github.com/bfsujason/bertalign)
(LaBSE sentence embeddings).

## What gets aligned

Each thesis row has a `summaries` struct with one abstract per language:
`kmr_Latn` (Kurmanji), `zza_Latn` (Zazaki), `ckb_Arab` (Sorani), `tur_Latn`
(Turkish), `eng_Latn` (English); any key may be `null`. Abstracts are short
(~200–280 words). The natural parallel pairs are **Kurmanji ↔ Turkish** and
**Kurmanji ↔ English** (the two best-covered targets).

## Why Bertalign (this fork)

We use the PyPI `bertalign==2.0.1` fork (`chikingsley/bertalign`), not the
original repo: it drops `faiss-gpu` and `googletrans` in favour of
`lingua-language-detector` + `sentence-splitter` + `sentence-transformers`, so
it installs cleanly on macOS / Python 3.12+.

**Important for Kurdish:** Bertalign's bundled language detector and sentence
splitter only cover ~25 European languages, so Kurdish would be mislabelled and
mis-split. LaBSE itself is fully multilingual and embeds Kurdish fine — the only
weak spot is **sentence segmentation**. So we segment ourselves (Turkish
splitter as a proxy for the Latin-script Kurdish variants) and pass
`is_split=True`, leaving Bertalign to do only embedding + alignment. See
`kurdish_parallel/align.py:split_sentences`.

## Layout

| Path | Purpose |
|---|---|
| `kurdish_parallel/` | The code package (all library code lives here). |
| `kurdish_parallel/align.py` | Data load, sentence split, and the local CLI (`python -m kurdish_parallel.align`). |
| `kurdish_parallel/tools/tools.py` | Alignment algorithms (Bertalign, Vecalign) + `align_pair` dispatch. |
| `kurdish_parallel/tools/embeddings.py` | Embedding models (LaBSE, SONAR). |
| `modal_app/app.py` | Modal smoke test — runs both pairs on a GPU, no local torch. Imports the package. |
| `Pipfile` | Local dependencies (bertalign, vecalign, pandas, pyarrow, huggingface-hub). |
| `results/` | Output JSONL files. |

## Two aligners, two embedding models

`--tool` picks the alignment algorithm; `--embed` picks the sentence-embedding
model Vecalign uses:

- **`--tool bertalign`** (default) — two-pass LaBSE DP, many-to-many beads.
  LaBSE-only. See `kurdish_parallel/tools/tools.py:align_bertalign`.
- **`--tool vecalign`** — Thompson & Koehn's vecalign (overlap embeddings +
  FastDTW DP), driven in-process. Installs from git (Cython build). See
  `kurdish_parallel/tools/tools.py:align_vecalign`.
  - **`--embed labse`** (default) — LaBSE, same as Bertalign.
  - **`--embed sonar`** — **SONAR** (`sonar-space`, fairseq2), which natively
    covers `kmr_Latn` and `ckb_Arab` and beats LaBSE/LASER3 on multilingual
    similarity. Dataset keys are FLORES-200 codes, passed as SONAR's
    `source_lang`. **License: CC-BY-NC-4.0 (non-commercial).**

The embedding model matters more than the DP algorithm for a low-resource
language, so `vecalign --embed sonar` is the strongest setup for Kurdish.
Alignments are not scored intrinsically; quality is judged against an external
gold set.

> **Why not LASER3?** It was the obvious choice, but `laser_encoders` requires
> `fairseq` (v1), whose wheels stop at Python 3.8 and which conflicts with
> `torch 2.x` and with bertalign's `Python>=3.12` pin. SONAR (fairseq**2**) is
> the modern, installable replacement.

## Run it on Modal (quick test — recommended)

Aligns Kurmanji against **both** Turkish and English on the same theses, on a
cheap GPU. No multi-GB install on your machine; LaBSE + dataset are cached on a
Modal Volume so reruns are fast.

```bash
modal run modal_app/app.py                              # bertalign/labse, 3 theses
modal run modal_app/app.py --tool vecalign              # vecalign/labse
modal run modal_app/app.py --tool vecalign --embed sonar  # vecalign/SONAR
modal run modal_app/app.py --n 50                       # smoke on 50 theses
modal run modal_app/app.py --tool vecalign --embed sonar --n 0  # FULL subset
modal run modal_app/app.py --gpu L4                     # bigger GPU
modal run modal_app/app.py --gpu none                   # CPU only
```

Output file: `results/{tool}_{embed}_smoke_{n}.jsonl` when `--n` is given
(n > 0), otherwise `results/{tool}_{embed}_full.jsonl`. One record per
thesis-and-target, with aligned `pairs` and timing.

First run builds the remote image (installs torch) and downloads LaBSE
(~1.8 GB) — a few minutes. After that, alignment itself is sub-second per
abstract pair.

## Run it locally (pipenv)

Source is always `kmr_Latn`; every other summary language present in a row
(`zza_Latn`, `ckb_Arab`, `tur_Latn`, `eng_Latn`) is aligned as a target.

```bash
pipenv install                                            # one-time (pulls torch etc.)
pipenv run python -m kurdish_parallel.align --n 3                         # bertalign/labse, 3 theses
pipenv run python -m kurdish_parallel.align --tool vecalign --embed sonar --n 3  # vecalign/SONAR (needs sonar-space)
pipenv run python -m kurdish_parallel.align --tool vecalign --n 3         # vecalign/labse
pipenv run python -m kurdish_parallel.align --is-full                     # the full subset
```

`--n` is **required** unless you pass `--is-full` (which aligns the whole subset
and ignores `--n`). On Apple Silicon the device is auto-selected as **MPS**
(Metal); the first line of output prints `Device: … | tool: … | embed: …`.
Output goes to `results/{tool}_{embed}_{n}.jsonl`, or
`results/{tool}_{embed}_full.jsonl` with `--is-full` — one record per
thesis-and-target.

### Local timing (Apple M3, 16 GB, MPS)

| Step | When | Time |
|---|---|---|
| `pipenv install` | once | 3–6 min |
| LaBSE download (~1.8 GB) | first run | 1–3 min |
| LaBSE load into MPS | every run | 5–12 s |
| Align one abstract pair | per summary | ~0.3–1 s (MPS) |

The model loads once and is reused, so 3 summaries ≈ the wall time of 1.

## Output format

Each JSONL record:

```json
{
  "tez_id": "...",
  "title": "...",
  "src_lang": "kmr_Latn",
  "tgt_lang": "tur_Latn",
  "n_src_sents": 14,
  "n_tgt_sents": 15,
  "align_seconds": 0.42,
  "pairs": [{"src": "…", "tgt": "…"}, ...]
}
```

A `pair` with an empty `src` or `tgt` is an insertion/deletion (a sentence on
one side with no counterpart).

### Evaluating alignment quality

Records carry no intrinsic score. Alignment quality is evaluated against an
external gold set rather than a self-reported similarity.

## Quality notes

- Kurmanji is split with the **Turkish** splitter (closest Latin-script proxy).
  Watch for abbreviations (`vb.`, `h.d.`) causing bad splits. The upgrade is
  Bertalign's optional `sat` extra (neural multilingual splitter).
- `eng_Latn` has the widest coverage (394 abstracts), `tur_Latn` next (365),
  Kurmanji abstracts ~361.
