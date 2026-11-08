"""
Modal smoke test for Bertalign on the kurdish-theses-yok summaries.

It aligns Kurmanji against BOTH Turkish and English, on the same theses (those
that carry all three abstracts), so you can eyeball both pairs from one run. It
reuses the local ``align.py`` logic (shipped into the image) and runs on a cheap
GPU, so there is no multi-GB torch install on your Mac. The LaBSE model and the
dataset parquet are cached on a Modal Volume, so only the first run pays the
download cost.

    # one entry point -> no ::name needed
    modal run modal_app/app.py                 # 3 theses, T4
    modal run modal_app/app.py --n 5           # 5 theses
    modal run modal_app/app.py --gpu L4 --n 10 # bigger GPU
    modal run modal_app/app.py --gpu none      # CPU only

Both pairs print to your terminal and are saved to results/smoke_{n}.jsonl
(one record per thesis-and-target). Requires modal installed and authenticated
locally (you already use Modal).
"""

import json
import time
from pathlib import Path

import modal

APP_NAME = "kurdish-align"
CACHE_VOLUME_NAME = "kurdish-align-cache"
CACHE_PATH = "/cache"

# Fixed smoke-test pairing: Kurmanji source, aligned to Turkish and English.
SRC_LANG = "kmr_Latn"
TARGET_LANGS = ["tur_Latn", "eng_Latn"]
SUBSET = "kmr_Latn"

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_ALIGN = REPO_ROOT / "align.py"
REMOTE_ALIGN_DIR = "/root/src"

app = modal.App(APP_NAME)

cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "bertalign==2.0.1",
        "pandas",
        "pyarrow",
        "huggingface-hub",
    )
    # Cache Hugging Face downloads (LaBSE model + dataset parquet) on the volume.
    .env({"HF_HOME": f"{CACHE_PATH}/huggingface"})
    # Ship the local align.py so the remote logic is identical to the local one.
    .add_local_file(str(LOCAL_ALIGN), f"{REMOTE_ALIGN_DIR}/align.py")
)


@app.function(image=image, volumes={CACHE_PATH: cache_volume}, timeout=30 * 60)
def align_remote(src: str, targets: list[str], subset: str, n: int, device: str) -> dict:
    import sys

    sys.path.insert(0, REMOTE_ALIGN_DIR)
    import align as A  # the shipped local module
    from bertalign import Bertalign, Encoder

    dev = A.resolve_device(device)  # "auto" -> cuda when a GPU is attached

    t0 = time.perf_counter()
    encoder = Encoder("LaBSE", device=dev)
    load_seconds = time.perf_counter() - t0

    df = A.load_rows(subset)

    results = []
    theses_done = 0
    for _, row in df.iterrows():
        if theses_done >= n:
            break
        src_text = A.get_summary(row, src)
        target_texts = {t: A.get_summary(row, t) for t in targets}
        # Only use theses that have the source and ALL targets, so both pairs
        # cover the same documents.
        if not src_text or any(not v for v in target_texts.values()):
            continue
        src_sents = A.split_sentences(src_text, src)
        if not src_sents:
            continue

        for tgt in targets:
            tgt_sents = A.split_sentences(target_texts[tgt], tgt)
            if not tgt_sents:
                continue

            t1 = time.perf_counter()
            aligner = Bertalign(
                "\n".join(src_sents),
                "\n".join(tgt_sents),
                is_split=True,
                model=encoder,
            )
            aligner.align_sents()
            pairs = aligner.sent_pairs()
            scores = A.score_pairs(encoder, pairs)
            align_seconds = time.perf_counter() - t1
            matched = [s for s in scores if s is not None]
            mean_score = round(sum(matched) / len(matched), 4) if matched else None

            results.append(
                {
                    "tez_id": row.get("tez_id"),
                    "title": row.get("title"),
                    "src_lang": src,
                    "tgt_lang": tgt,
                    "n_src_sents": len(src_sents),
                    "n_tgt_sents": len(tgt_sents),
                    "align_seconds": round(align_seconds, 3),
                    "mean_score": mean_score,
                    "pairs": [{"src": s, "tgt": t, "score": sc}
                              for (s, t), sc in zip(pairs, scores)],
                }
            )
        theses_done += 1

    # Persist the HF cache (model + parquet) so the next run is fast.
    cache_volume.commit()

    return {
        "device": dev,
        "model_load_seconds": round(load_seconds, 2),
        "n_theses": theses_done,
        "records": results,
    }


@app.local_entrypoint()
def main(n: int = 3, gpu: str = "T4", device: str = "auto"):
    gpu_option = None if gpu.lower() in {"none", "cpu", ""} else gpu
    out = align_remote.with_options(gpu=gpu_option).remote(
        SRC_LANG, TARGET_LANGS, SUBSET, n, device
    )

    print(f"\nDevice: {out['device']} | LaBSE load: {out['model_load_seconds']}s "
          f"| GPU: {gpu_option or 'cpu'} | theses: {out['n_theses']}\n")
    for record in out["records"]:
        print(f"=== tez_id={record['tez_id']} | {record['src_lang']}->{record['tgt_lang']} | "
              f"{record['n_src_sents']}->{record['n_tgt_sents']} sents | "
              f"{len(record['pairs'])} beads | mean cos={record['mean_score']} | "
              f"{record['align_seconds']}s ===")
        print(f"    {record['title']}")
        for pair in record["pairs"]:
            score = pair["score"]
            tag = f"{score:.3f}" if score is not None else "  -  "
            print(f"  [{tag}] [{record['src_lang']}] {pair['src']}")
            print(f"          [{record['tgt_lang']}] {pair['tgt']}")
            print()

    out_path = REPO_ROOT / "results" / f"smoke_{n}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in out["records"]:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(out['records'])} records "
          f"({out['n_theses']} theses x {len(TARGET_LANGS)} targets) to {out_path}")
