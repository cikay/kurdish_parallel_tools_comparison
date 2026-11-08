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
`align.py:split_sentences`.

## Layout

| Path | Purpose |
|---|---|
| `align.py` | Core logic + a local CLI (data load, sentence split, align). |
| `modal_app/app.py` | Modal smoke test — runs both pairs on a GPU, no local torch. |
| `Pipfile` | Local dependencies (bertalign, pandas, pyarrow, huggingface-hub). |
| `results/` | Output JSONL files. |

## Run it on Modal (quick test — recommended)

Aligns Kurmanji against **both** Turkish and English on the same theses, on a
cheap GPU. No multi-GB install on your machine; LaBSE + dataset are cached on a
Modal Volume so reruns are fast.

```bash
modal run modal_app/app.py            # 3 theses, T4
modal run modal_app/app.py --n 5      # 5 theses
modal run modal_app/app.py --gpu L4   # bigger GPU
modal run modal_app/app.py --gpu none # CPU only
```

Output: `results/smoke_{n}.jsonl` (one record per thesis-and-target, with
aligned `pairs` and per-alignment timing).

First run builds the remote image (installs torch) and downloads LaBSE
(~1.8 GB) — a few minutes. After that, alignment itself is sub-second per
abstract pair.

## Run it locally (pipenv)

```bash
pipenv install                                       # one-time (pulls torch etc.)
pipenv run python align.py --n 3                      # kmr_Latn -> tur_Latn
pipenv run python align.py --src kmr_Latn --tgt eng_Latn --n 3
```

On Apple Silicon the script auto-selects the **MPS** (Metal) device; the first
line of output prints `Device: mps`. Output: `results/{src}_{tgt}_{n}.jsonl`
(or `--out`). Default output name is `{src}_{tgt}_{n}.jsonl` in the CWD.

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
  "mean_score": 0.83,
  "pairs": [{"src": "…", "tgt": "…", "score": 0.87}, ...]
}
```

A `pair` with an empty `src` or `tgt` is an insertion/deletion (a sentence on
one side with no counterpart); its `score` is `null`.

### Alignment score

Bertalign computes a per-bead similarity internally but discards it, so we
recover our own: `score` is the **cosine similarity** of the aligned segment
pair (re-encoded with LaBSE, L2-normalized), in `[-1, 1]`. Good matches are
typically ~0.6–0.95; a low score flags a dubious pair. `mean_score` averages
the scored (non-insertion) beads — a quick per-thesis quality signal and a
natural threshold for filtering. See `align.py:score_pairs`.

## Quality notes

- Kurmanji is split with the **Turkish** splitter (closest Latin-script proxy).
  Watch for abbreviations (`vb.`, `h.d.`) causing bad splits. The upgrade is
  Bertalign's optional `sat` extra (neural multilingual splitter).
- `eng_Latn` has the widest coverage (394 abstracts), `tur_Latn` next (365),
  Kurmanji abstracts ~361.
