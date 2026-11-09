"""
Modal smoke test for sentence alignment on the kurdish-theses-yok summaries.

Aligns Kurmanji against BOTH Turkish and English, on the same theses (those that
carry all three abstracts), with a selectable tool and embedding model. Reuses
the local ``align.py`` logic (shipped into each image). Alignment quality is
evaluated against an external gold set, not an intrinsic score.

Three images, because the dependency stacks don't mix:
  * labse_image  - bertalign (needs Python>=3.12) + vecalign; LaBSE embeddings.
  * sonar_image  - vecalign + SONAR (fairseq2); no bertalign.
  * laser_image  - vecalign + LASER3 (laser_encoders -> fairseq v1) on a LEGACY
                   Python 3.10 / torch 1.13 base; no bertalign. Isolated because
                   fairseq v1 can't coexist with torch 2.x or fairseq2.

    # one entry point -> no ::name needed
    modal run modal_app/app.py                              # bertalign/labse
    modal run modal_app/app.py --tool vecalign              # vecalign/labse
    modal run modal_app/app.py --tool vecalign --embed sonar  # vecalign/SONAR
    modal run modal_app/app.py --tool vecalign --embed laser  # vecalign/LASER3
    modal run modal_app/app.py --n 50                      # smoke on 50 theses
    modal run modal_app/app.py --tool vecalign --embed sonar --n 0  # FULL subset
    modal run modal_app/app.py --gpu L4 --n 10

Output: results/{tool}_{embed}_smoke_{n}.jsonl when --n > 0, else
results/{tool}_{embed}_full.jsonl. One record per thesis-and-target, with the
aligned `pairs`.
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
LOCAL_PKG = REPO_ROOT / "kurdish_parallel"  # the code package (align + tools/)
REMOTE_SRC_DIR = "/root/src"  # put on sys.path; package lands at src/kurdish_parallel

app = modal.App(APP_NAME)

cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

# Vecalign builds a Cython extension -> needs a C compiler and git.
_base = modal.Image.debian_slim(python_version="3.12").apt_install("build-essential", "git")

# LaBSE stack: bertalign (pulls sentence-transformers + torch) + vecalign.
labse_image = (
    _base.pip_install("bertalign==2.0.1", "pandas", "pyarrow", "huggingface-hub")
    .pip_install("git+https://github.com/thompsonb/vecalign.git@main")
    .env({"HF_HOME": f"{CACHE_PATH}/huggingface"})
    .add_local_dir(str(LOCAL_PKG), f"{REMOTE_SRC_DIR}/kurdish_parallel")
)

# SONAR stack: vecalign + sonar-space (fairseq2) + sentence-transformers. No
# bertalign (keeping it out avoids transformers/torch fights with fairseq2).
# sentence-transformers no longer runs any of our code (scoring was removed) but
# stays as part of the PINNED self-consistent set below -- dropping a member lets
# the free resolver backtrack to a broken sonar-space (no fairseq2):
#   fairseq2n 0.8.1 -> torch==2.9.1, numpy~=2.2
#   fairseq2  0.8.1 -> transformers~=4.57
#   sentence-transformers 3.2.1 accepts transformers 4.57 and torch 2.9
# NOTE: XDG_CACHE_HOME (SONAR/fairseq2 asset cache) is set at RUNTIME in
# _run_alignment, not here: setting it at build time makes pip cache into
# /cache, and Modal then refuses to mount the volume on a non-empty /cache.
sonar_image = (
    _base.apt_install("libsndfile1", "sox")
    .pip_install(
        "torch==2.9.1",
        "numpy~=2.2",
        "transformers~=4.57.0",
        "sentence-transformers==3.2.1",
        "fairseq2n==0.8.1",
        "fairseq2==0.8.1",
        "sonar-space==0.5.0",
        "huggingface-hub~=0.32",
        "sentence-splitter",  # align.py:split_sentences (bertalign provides it in the other image)
        "pandas",
        "pyarrow",
    )
    .pip_install("git+https://github.com/thompsonb/vecalign.git@main")
    .env({"HF_HOME": f"{CACHE_PATH}/huggingface"})
    .add_local_dir(str(LOCAL_PKG), f"{REMOTE_SRC_DIR}/kurdish_parallel")
)

# LASER3 stack: laser_encoders -> fairseq v1, which is pinned to legacy torch and
# will NOT coexist with the labse_image (bertalign/torch 2.x) or sonar_image
# (fairseq2). So it gets its own image on a LEGACY base: Python 3.10 + torch 1.13
# + fairseq 0.12.2. This is the fragile corner of the project -- the pins below
# are a self-consistent set, don't bump them casually:
#   fairseq 0.12.2        -> needs torch>=1.10 + torchaudio, numpy<1.24 (np.float)
#   torch/torchaudio 1.13 -> last line that fairseq v1 reliably builds against
#   sentence-transformers 2.2.2 -> needs huggingface-hub<0.26 (it imports the
#     removed `cached_download`) and transformers ~4.30; it no longer runs any of
#     our code (scoring was removed) but stays in the proven pin set.
# LASER3 provides the aligning embeddings; huggingface-hub also fetches the data.
_laser_base = modal.Image.debian_slim(python_version="3.10").apt_install(
    "build-essential", "git"
)
laser_image = (
    # fairseq 0.12.2 needs omegaconf<2.1, but omegaconf 2.0.5/2.0.6 ship legacy
    # metadata (`PyYAML (>=5.1.*)`) that pip>=24.1 rejects, so modern pip discards
    # them and reports a bogus "conflict". Pin pip<24.1 to accept that metadata.
    _laser_base.run_commands("python -m pip install 'pip<24.1'")
    .pip_install(
        "numpy==1.23.5",
        "torch==1.13.1",
        "torchaudio==0.13.1",
        "fairseq==0.12.2",
        "laser_encoders==0.0.2",
        "huggingface-hub==0.16.4",
        "transformers==4.30.2",
        "sentence-transformers==2.2.2",
        "sentence-splitter",  # align.py:split_sentences
        "pandas",
        "pyarrow",
        "setuptools>=74",  # vecalign build backend (see run_commands below)
        "Cython>=3.1",
    )
    # vecalign@main pins requires-python>=3.11 and numpy>=1.26.4, both of which
    # clash with this legacy stack (Python 3.10, numpy<1.24 for fairseq v1). It's
    # the SAME code the other images pip-install; here we clone it, relax the
    # Python floor to 3.10, and build it against the already-installed numpy
    # 1.23.5 (--no-build-isolation --no-deps) so the compiled dp_core matches the
    # runtime numpy and nothing drags numpy back up.
    .run_commands(
        "git clone --depth 1 https://github.com/thompsonb/vecalign.git /opt/vecalign",
        "sed -i 's/requires-python = \">=3.11\"/requires-python = \">=3.10\"/' /opt/vecalign/pyproject.toml",
        "cd /opt/vecalign && python -m pip install --no-deps --no-build-isolation .",
    )
    .env({"HF_HOME": f"{CACHE_PATH}/huggingface"})
    .add_local_dir(str(LOCAL_PKG), f"{REMOTE_SRC_DIR}/kurdish_parallel")
)


def _run_alignment(subset: str, src: str, targets: list[str], n: int,
                   device: str, tool: str, embed: str) -> dict:
    """Shared remote body: load models and align every thesis."""
    import os
    import sys

    # Point SONAR / fairseq2 (and any ~/.cache users) at the volume AT RUNTIME.
    # Must not be a build-time env var, or pip caches into /cache and the volume
    # mount fails. Harmless for the LaBSE image.
    os.environ.setdefault("XDG_CACHE_HOME", f"{CACHE_PATH}/xdg")
    # LASER3 (laser_encoders) downloads models to ~/.cache/laser_encoders by
    # default; point it at the volume so reruns reuse them. Harmless otherwise.
    # It does shutil.move(temp, model_dir/...) WITHOUT creating model_dir, so we
    # must create it ourselves or the move fails with FileNotFoundError.
    os.environ.setdefault("LASER_MODEL_DIR", f"{CACHE_PATH}/laser_encoders")
    os.makedirs(os.environ["LASER_MODEL_DIR"], exist_ok=True)

    sys.path.insert(0, REMOTE_SRC_DIR)
    import kurdish_parallel.align as A  # the shipped local package

    dev = A.resolve_device(device)  # "auto" -> cuda when a GPU is attached

    t0 = time.perf_counter()
    bert_encoder = None
    labse_model = None
    if tool == "bertalign":
        from bertalign import Encoder

        bert_encoder = Encoder("LaBSE", device=dev)
        labse_model = bert_encoder.model
    elif embed == "labse":
        from sentence_transformers import SentenceTransformer

        labse_model = SentenceTransformer("LaBSE", device=dev)
    load_seconds = time.perf_counter() - t0

    df = A.load_rows(subset)

    results = []
    theses_done = 0
    for _, row in df.iterrows():
        if n and theses_done >= n:  # n == 0 means "full subset".
            break
        src_text = A.get_summary(row, src)
        target_texts = {t: A.get_summary(row, t) for t in targets}
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
            pairs = A.align_pair(tool, embed, labse_model, src_sents, tgt_sents,
                                 src, tgt, device=dev, bert_encoder=bert_encoder)
            align_seconds = time.perf_counter() - t1

            results.append({
                "tez_id": row.get("tez_id"),
                "title": row.get("title"),
                "tool": tool,
                "embed": embed,
                "src_lang": src,
                "tgt_lang": tgt,
                "n_src_sents": len(src_sents),
                "n_tgt_sents": len(tgt_sents),
                "align_seconds": round(align_seconds, 3),
                "pairs": [{"src": s, "tgt": t} for s, t in pairs],
            })
        theses_done += 1

    cache_volume.commit()  # persist HF / SONAR caches for fast reruns
    return {
        "device": dev,
        "tool": tool,
        "embed": embed,
        "model_load_seconds": round(load_seconds, 2),
        "n_theses": theses_done,
        "records": results,
    }


@app.function(image=labse_image, volumes={CACHE_PATH: cache_volume}, timeout=60 * 60)
def align_labse_remote(src, targets, subset, n, device, tool) -> dict:
    return _run_alignment(subset, src, targets, n, device, tool, "labse")


@app.function(image=sonar_image, volumes={CACHE_PATH: cache_volume}, timeout=60 * 60)
def align_sonar_remote(src, targets, subset, n, device) -> dict:
    return _run_alignment(subset, src, targets, n, device, "vecalign", "sonar")


@app.function(image=laser_image, volumes={CACHE_PATH: cache_volume}, timeout=60 * 60)
def align_laser_remote(src, targets, subset, n, device) -> dict:
    # torch 1.13.1 here is the CUDA 11.7 build, and LASER3 self-moves to CUDA
    # when a GPU is attached; without one it runs on CPU (slow). Attach a T4
    # (not an L4 -- Ada needs CUDA >=11.8) via --gpu for the fast path.
    return _run_alignment(subset, src, targets, n, device, "vecalign", "laser")


@app.local_entrypoint()
def main(n: int = 0, tool: str = "bertalign", embed: str = "labse",
         gpu: str = "T4", device: str = "auto"):
    if tool not in ("bertalign", "vecalign"):
        raise ValueError("tool must be 'bertalign' or 'vecalign'")
    if embed not in ("labse", "sonar", "laser"):
        raise ValueError("embed must be 'labse', 'sonar', or 'laser'")
    if tool == "bertalign" and embed != "labse":
        raise ValueError("bertalign only supports --embed labse")

    gpu_option = None if gpu.lower() in {"none", "cpu", ""} else gpu
    if embed == "sonar":
        fn = align_sonar_remote.with_options(gpu=gpu_option)
        out = fn.remote(SRC_LANG, TARGET_LANGS, SUBSET, n, device)
    elif embed == "laser":
        # LASER3 auto-uses CUDA when a GPU is visible (laser_encoders'
        # SentenceEncoder: use_cuda = torch.cuda.is_available() and not cpu).
        # torch 1.13.1 here is the CUDA 11.7 build, so use a T4 (compute 7.5);
        # an L4 (Ada) needs CUDA >=11.8 and won't work with this torch.
        fn = align_laser_remote.with_options(gpu=gpu_option)
        out = fn.remote(SRC_LANG, TARGET_LANGS, SUBSET, n, device)
    else:
        fn = align_labse_remote.with_options(gpu=gpu_option)
        out = fn.remote(SRC_LANG, TARGET_LANGS, SUBSET, n, device, tool)

    print(f"\nTool: {out['tool']}/{out['embed']} | Device: {out['device']} | "
          f"model load: {out['model_load_seconds']}s | GPU: {gpu_option or 'cpu'} | "
          f"theses: {out['n_theses']}\n")
    for record in out["records"]:
        print(f"=== tez_id={record['tez_id']} | {record['tool']}/{record['embed']} | "
              f"{record['src_lang']}->{record['tgt_lang']} | "
              f"{record['n_src_sents']}->{record['n_tgt_sents']} sents | "
              f"{len(record['pairs'])} beads | {record['align_seconds']}s ===")
        if n:  # print every bead only for a small smoke run
            print(f"    {record['title']}")
            for pair in record["pairs"]:
                print(f"  [{record['src_lang']}] {pair['src']}")
                print(f"  [{record['tgt_lang']}] {pair['tgt']}")
                print()

    stem = f"{tool}_{embed}_smoke_{n}" if n else f"{tool}_{embed}_full"
    out_path = REPO_ROOT / "results" / f"{stem}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in out["records"]:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(out['records'])} records "
          f"({out['n_theses']} theses x {len(TARGET_LANGS)} targets) to {out_path}")
