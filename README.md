# kurdish_parallel_tools_comparison

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
weak spot is **sentence segmentation**. So we segment ourselves with
[wtpsplit](https://github.com/segment-any-text/wtpsplit)'s **SaT** (`sat-12l-sm`),
a neural multilingual segmenter that is language-agnostic and handles Kurdish in
both Latin and Arabic script, then pass `is_split=True`, leaving Bertalign to do
only embedding + alignment. Segmentation is a **single shared step** — every
tool/embedding aligns the identical sentence lists. See
`kurdish_parallel_tools_comparison/align.py:split_sentences` and `prepare_theses`.

## Layout

| Path | Purpose |
|---|---|
| `kurdish_parallel_tools_comparison/` | The code package (all library code lives here). |
| `kurdish_parallel_tools_comparison/align.py` | Data load, sentence split, and the local CLI (`python -m kurdish_parallel_tools_comparison.align`). |
| `kurdish_parallel_tools_comparison/tools/tools.py` | Alignment algorithms (Bertalign, Vecalign, SentAlign) + `align_pair` dispatch. |
| `kurdish_parallel_tools_comparison/tools/_sentalign_core.py` | SentAlign's grid-DP core, vendored unmodified from `steinst/SentAlign` (Apache-2.0). |
| `kurdish_parallel_tools_comparison/tools/embeddings.py` | Embedding models (LaBSE, SONAR, LASER3). |
| `kurdish_parallel_tools_comparison/scoring.py` | Scores the outputs against `gold_set.json` (precision/recall/F1). Pure stdlib, no Modal/torch. |
| `modal_app/app.py` | Modal run — runs all five options on a GPU, no local torch. Imports the package. |
| `modal_app/score_app.py` | Thin Modal wrapper around `scoring.py` (same metrics, on Modal). |
| `gold_set.json` | Hand-annotated gold alignments (0-based sentence-index links) to score against. |
| `Pipfile` | Local dependencies (bertalign, vecalign, pandas, pyarrow, huggingface-hub). |
| `data/{sat_model}/` | Output JSONL files, nested under the SaT model used to segment. |

## Three aligners, three embedding models (five options)

`--tool` picks the alignment algorithm; `--embed` picks the sentence-embedding
model. The comparison runs **five** tool/embed options:

- **`--tool bertalign`** (default) — two-pass LaBSE DP, many-to-many beads.
  LaBSE-only. See `kurdish_parallel_tools_comparison/tools/tools.py:align_bertalign`.
- **`--tool sentalign`** — SentAlign's grid-DP (length-penalty scoring + null/merge
  re-evaluation), its pure-Python core vendored and driven in-process. LaBSE-only.
  See `kurdish_parallel_tools_comparison/tools/tools.py:align_sentalign`.
- **`--tool vecalign`** — Thompson & Koehn's vecalign (overlap embeddings +
  FastDTW DP), driven in-process. Installs from git (Cython build). See
  `kurdish_parallel_tools_comparison/tools/tools.py:align_vecalign`.
  - **`--embed labse`** (default) — LaBSE, same as Bertalign/SentAlign.
  - **`--embed sonar`** — **SONAR** (`sonar-space`, fairseq2), which natively
    covers `kmr_Latn` and `ckb_Arab` and beats LaBSE/LASER3 on multilingual
    similarity. Dataset keys are FLORES-200 codes, passed as SONAR's
    `source_lang`. **License: CC-BY-NC-4.0 (non-commercial).**
  - **`--embed laser`** — **LASER3** (`laser_encoders`), which has a dedicated
    `kmr_Latn` encoder. Needs legacy `fairseq` v1, so it runs only in a dedicated
    legacy Modal image. **License: BSD (commercially usable).**

So the five options are: `bertalign/labse`, `sentalign/labse`, `vecalign/labse`,
`vecalign/sonar`, `vecalign/laser`. Bertalign and SentAlign vary the *DP* over
the same LaBSE vectors; SONAR and LASER vary the *embedding*. The embedding model
matters more than the DP for a low-resource language, so `vecalign --embed sonar`
is expected to be the strongest setup for Kurdish. Alignments are not scored
intrinsically; quality is judged against an external gold set.

## Run it on Modal (recommended)

**One command runs all five options concurrently on the same `n` theses**, so
their outputs are directly comparable: pick `n`, build a gold set over exactly
those `n` theses, then score each option against it. Aligns Kurmanji against
**both** Turkish and English on a cheap GPU. No multi-GB install on your machine;
LaBSE + dataset are cached on a Modal Volume so reruns are fast.

```bash
modal run modal_app/app.py --n 20            # all 5 options on 20 theses
modal run modal_app/app.py --n 50 --gpu L4   # ...on a bigger GPU
modal run modal_app/app.py --n 10 --gpu none # ...on CPU (slow)
```

`--n` is **required** (≥ 1) and is the number of theses **each** option aligns —
the pool you hand-annotate into the gold set — not a total split across options.
Every option aligns the identical `n` theses with identical segmentation, so the
comparison is deterministic. `--gpu` drives every option and defaults to `T4`
(LASER3's torch 1.13 needs it; an L4 won't work).

Output: one file per option, `data/{sat_model}/{tool}_{embed}_{n}.jsonl` (five
files), each one record per thesis-and-target with aligned `pairs` and timing,
plus `data/{sat_model}/result_{n}.json` — the (thesis, target) pairs whose alignment **differs
across ≥ 2 options** (options grouped by identical alignment), i.e. the shortlist
to hand-annotate for the gold set. The command prints only a short summary; a
failing image (LASER is the fragile one) is reported but doesn't abort the rest.

First run builds the remote image (installs torch) and downloads LaBSE
(~1.8 GB) — a few minutes. After that, alignment itself is sub-second per
abstract pair.

## Run it locally (pipenv)

Source is always `kmr_Latn`; every other summary language present in a row
(`zza_Latn`, `ckb_Arab`, `tur_Latn`, `eng_Latn`) is aligned as a target.

```bash
pipenv install                                            # one-time (pulls torch etc.)
pipenv run python -m kurdish_parallel_tools_comparison.align --n 3                         # bertalign/labse, 3 theses
pipenv run python -m kurdish_parallel_tools_comparison.align --tool vecalign --embed sonar --n 3  # vecalign/SONAR (needs sonar-space)
pipenv run python -m kurdish_parallel_tools_comparison.align --tool vecalign --n 3         # vecalign/labse
pipenv run python -m kurdish_parallel_tools_comparison.align --is-full                     # the full subset
```

`--n` is **required** unless you pass `--is-full` (which aligns the whole subset
and ignores `--n`). On Apple Silicon the device is auto-selected as **MPS**
(Metal); the first line of output prints `Device: … | tool: … | embed: …`.
Output goes to `data/{sat_model}/{tool}_{embed}_{n}.jsonl`, or
`data/{sat_model}/{tool}_{embed}_full.jsonl` with `--is-full` — one record per
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

## Scoring against the gold set

Records carry no intrinsic score — quality is judged against a hand-annotated
gold set (`gold_set.json`) with **precision / recall / F1**. Because every option
aligns the identical segmentation, the comparison is apples-to-apples.

```bash
# Local (pure stdlib, no torch/Modal needed):
python -m kurdish_parallel_tools_comparison.scoring --n 20            # scores the *_20.jsonl files
python -m kurdish_parallel_tools_comparison.scoring --n 20 --verbose  # + per-target (tur/eng) breakdown

# Same metrics, through Modal:
modal run modal_app/score_app.py --n 20
```

`--n` selects which run to score (`{option}_{n}.jsonl` under `data/{sat_model}/`).
The report is printed and written to `data/{sat_model}/scores_{n}.json`; pass
`--no-write` to skip the file. Scoring logic lives entirely in `scoring.py`
(decoupled from Modal); `score_app.py` only ships it to a CPU container.

**Gold format** — a list of theses, each with 0-based sentence-index links per
target language (the same indices `result_{n}.json` prints via `numbered(sents)`):

```json
[{"tez_id": "1003999",
  "alignments": [{"tgt_lang": "tur_Latn",
                  "links": [{"src": [0], "tgt": [0]},
                            {"src": [1], "tgt": [1, 2]}]}]}]
```

The gold set can be **partial**: an entry with empty `links` (or empty
`tgt_lang`) is treated as not-yet-annotated and skipped, and only (thesis,
target) pairs present in both the gold set and a prediction file are scored. So a
half-finished gold set still gives a fair comparison over what it does cover.

**How predictions are scored** — the aligners emit text beads; the scorer
reconstructs each bead's 0-based sentence-index sets from `pairs` +
`src_sents`/`tgt_sents` (or uses an explicit `links` field if a record carries
one). Metrics are **micro-averaged** (links pooled across theses) in two
granularities:

- **lax** (headline) — each bead is expanded into its sentence-to-sentence links
  (the cross product of its index sets); partial overlaps get partial credit. A
  gold `[3] → [4,5]` becomes links `(3,4)` and `(3,5)`, so a prediction that gets
  only `[3] → [4]` still scores the `(3,4)` link.
- **strict** — a predicted bead counts only if it matches a gold bead exactly
  (same src *and* tgt index sets). All-or-nothing per bead.

Null beads (one side empty — an insertion/deletion) contribute no links and are
excluded from both metrics, since the gold set annotates only real alignments.

### Current results (`--n 20`, gold = 9 theses / 17 thesis-target pairs)

Micro-averaged over the annotated pairs (9 `tur_Latn` + 8 `eng_Latn`), ranked by
lax F1:

| option | lax P | lax R | **lax F1** | strict P | strict R | strict F1 |
|---|---|---|---|---|---|---|
| **bertalign_labse** | 0.940 | 0.984 | **0.961** | 0.942 | 0.939 | 0.940 |
| sentalign_labse | 0.961 | 0.943 | **0.952** | 0.953 | 0.939 | 0.946 |
| vecalign_labse | 0.910 | 0.959 | **0.934** | 0.885 | 0.889 | 0.887 |
| vecalign_sonar | 0.925 | 0.940 | **0.932** | 0.876 | 0.897 | 0.886 |
| vecalign_laser | 0.915 | 0.924 | **0.920** | 0.852 | 0.885 | 0.868 |

Reading this (small gold set — treat as directional, not final):

- The two **LaBSE grid-DP** tools (bertalign, sentalign) lead. Their gap is
  within noise at this size — bertalign edges lax F1, sentalign edges strict F1 —
  so they're effectively tied; pick between them on secondary criteria (bertalign
  is a cleaner dependency, sentalign is vendored core you own).
- Swapping the **embedding** under vecalign (SONAR, LASER) did **not** help here:
  both land at or below `vecalign_labse`. So on tur/eng the *DP* matters more than
  the embedding. This runs against the prior hypothesis that `vecalign --embed
  sonar` would be strongest for Kurdish — but the gold set covers only tur_Latn
  and eng_Latn, and SONAR/LASER's dedicated Kurdish coverage is most likely to pay
  off on **ckb_Arab / zza_Latn**, which aren't annotated yet.
- Numbers will move as the gold set grows; re-run the scorer to refresh them.

## Quality notes

- All languages are split with wtpsplit **SaT** (`sat-12l-sm`), a neural
  multilingual segmenter — language-agnostic, so Kurmanji/Zazaki (Latin) and
  ckb (Arabic script) no longer rely on a Turkish-splitter proxy or a regex.
  (We moved up from `sat-3l-sm`, which over-split some Kurmanji sentences.)
- `eng_Latn` has the widest coverage (394 abstracts), `tur_Latn` next (365),
  Kurmanji abstracts ~361.
