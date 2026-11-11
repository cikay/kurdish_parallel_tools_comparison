"""
Modal comparison run for sentence alignment on the kurdish-theses-yok summaries.

Runs ALL FIVE tool/embedding options CONCURRENTLY on the SAME ``n`` theses, so
their outputs are directly comparable. Pick ``n`` theses, build a gold set over
exactly those ``n`` theses by hand, then score each option's output against it
to decide which tool is best for Kurdish -- a deterministic comparison, because
every option aligns the identical ``n`` theses (and identical segmentation).

The five options:
    bertalign / labse      vecalign / labse       sentalign / labse
    vecalign  / sonar      vecalign / laser

Kurmanji is aligned against BOTH Turkish and English, on the same theses (those
that carry all three abstracts). Reuses the local ``align.py`` logic (shipped
into each image). Alignment quality is judged against the external gold set, not
an intrinsic score.

Segmentation is a SINGLE shared step: segment_remote splits the n theses ONCE
with wtpsplit SaT and the identical sentence lists are handed to all three
alignment images, so no image re-splits and the five options are byte-identical
on input. Four images, because the dependency stacks don't mix (one spawn each;
the three LaBSE tools share the one labse_image container, so LaBSE loads once):
  * seg_image    - wtpsplit SaT only (CPU); the shared segmentation step.
  * labse_image  - bertalign (needs Python>=3.12) + vecalign + sentalign; LaBSE.
  * sonar_image  - vecalign + SONAR (fairseq2); no bertalign.
  * laser_image  - vecalign + LASER3 (laser_encoders -> fairseq v1) on a LEGACY
                   Python 3.10 / torch 1.13 base; no bertalign. Isolated because
                   fairseq v1 can't coexist with torch 2.x or fairseq2.

    # one entry point -> no ::name needed; --n is REQUIRED (theses per option)
    modal run modal_app/app.py --n 20            # all 5 options on 20 theses
    modal run modal_app/app.py --n 50 --gpu L4   # ...on a bigger GPU
    modal run modal_app/app.py --n 10 --gpu none # ...on CPU (slow)

Output: one file per option, data/{tool}_{embed}_smoke_{n}.jsonl, one record
per thesis-and-target, with the aligned `pairs`.
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
LOCAL_PKG = REPO_ROOT / "kurdish_parallel_tools_comparison"  # the code package (align + tools/)
REMOTE_SRC_DIR = "/root/src"  # put on sys.path; package lands at src/kurdish_parallel_tools_comparison

app = modal.App(APP_NAME)

cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

# Vecalign builds a Cython extension -> needs a C compiler and git.
_base = modal.Image.debian_slim(python_version="3.12").apt_install("build-essential", "git")

# Segmentation stack: wtpsplit SaT only. Segmentation is a SINGLE shared step run
# once (segment_remote), and its sentence lists are handed to all three alignment
# images -- so SaT lives here alone and never has to coexist with bertalign,
# fairseq2, or the fragile legacy fairseq-v1 (laser) stack. CPU is plenty for the
# small model on abstract-sized text. The SaT model caches to the volume via HF_HOME.
seg_image = (
    modal.Image.debian_slim(python_version="3.12")
    # torch must be installed explicitly: wtpsplit treats it as OPTIONAL (it also
    # has an ONNX path), so neither it nor transformers pulls it in. SaT runs on
    # CPU here, but the default PyPI wheel (CUDA-enabled) is fine on CPU too.
    # transformers is pinned alongside wtpsplit because it governs huggingface-hub.
    .pip_install("wtpsplit==2.2.2", "torch==2.15.0", "transformers==5.22.2",
                 "huggingface-hub==2.4.0", "pandas==2.2.3", "pyarrow==18.1.0")
    .env({"HF_HOME": f"{CACHE_PATH}/huggingface"})
    .add_local_dir(str(LOCAL_PKG), f"{REMOTE_SRC_DIR}/kurdish_parallel_tools_comparison")
)

# LaBSE stack: bertalign (pulls sentence-transformers + torch) + vecalign.
# sentence-transformers is pinned explicitly because it governs huggingface-hub
# (bertalign itself leaves it unpinned); torch comes in via sentence-transformers.
labse_image = (
    _base.pip_install("bertalign==2.0.1", "sentence-transformers==6.2.0",
                      "huggingface-hub==2.4.0", "pandas==2.2.3", "pyarrow==18.1.0")
    .pip_install("git+https://github.com/thompsonb/vecalign.git@main")
    .env({"HF_HOME": f"{CACHE_PATH}/huggingface"})
    .add_local_dir(str(LOCAL_PKG), f"{REMOTE_SRC_DIR}/kurdish_parallel_tools_comparison")
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
        "pandas==2.2.3",
        "pyarrow==18.1.0",
    )
    .pip_install("git+https://github.com/thompsonb/vecalign.git@main")
    .env({"HF_HOME": f"{CACHE_PATH}/huggingface"})
    .add_local_dir(str(LOCAL_PKG), f"{REMOTE_SRC_DIR}/kurdish_parallel_tools_comparison")
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
        "pandas==2.0.3",    # last pandas supporting numpy<1.24 / py3.10
        "pyarrow==12.0.1",  # compatible with numpy 1.23.5 + py3.10
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
    .add_local_dir(str(LOCAL_PKG), f"{REMOTE_SRC_DIR}/kurdish_parallel_tools_comparison")
)


def _run_alignment(prepared: list[dict], src: str, device: str,
                   runs: list[tuple[str, str]]) -> list[dict]:
    """Shared remote body: load models once, align the ALREADY-SEGMENTED theses
    in ``prepared`` with every ``(tool, embed)`` in ``runs``, and return one
    result dict per run.

    ``prepared`` comes from the single shared segmentation step (segment_remote
    -> ``align.prepare_theses``), so every image -- and every run within an image
    -- aligns byte-identical sentence lists. That is the determinism the gold-set
    comparison needs; this body no longer loads the dataset or splits anything.
    Running several runs in one container still loads LaBSE once.
    """
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
    import kurdish_parallel_tools_comparison.align as A  # the shipped local package

    dev = A.resolve_device(device)  # "auto" -> cuda when a GPU is attached

    tools = {tool for tool, _ in runs}
    embeds = {embed for _, embed in runs}

    t0 = time.perf_counter()
    bert_encoder = None
    labse_model = None
    if "bertalign" in tools:
        from bertalign import Encoder

        bert_encoder = Encoder("LaBSE", device=dev)
        labse_model = bert_encoder.model  # reuse for the other LaBSE tools
    if "labse" in embeds and labse_model is None:
        from sentence_transformers import SentenceTransformer

        labse_model = SentenceTransformer("LaBSE", device=dev)
    load_seconds = time.perf_counter() - t0

    per_run = {run: [] for run in runs}
    for thesis in prepared:
        src_sents = thesis["src_sents"]
        for tool, embed in runs:
            for tgt, info in thesis["targets"].items():
                tgt_sents = info["tgt_sents"]

                t1 = time.perf_counter()
                pairs = A.align_pair(tool, embed, labse_model, src_sents, tgt_sents,
                                     src, tgt, device=dev, bert_encoder=bert_encoder)
                align_seconds = time.perf_counter() - t1

                per_run[(tool, embed)].append({
                    "tez_id": thesis.get("tez_id"),
                    "title": thesis.get("title"),
                    "tool": tool,
                    "embed": embed,
                    "src_lang": src,
                    "tgt_lang": tgt,
                    "n_src_sents": len(src_sents),
                    "n_tgt_sents": len(tgt_sents),
                    "src_summary": thesis["src_summary"],
                    "tgt_summary": info["tgt_summary"],
                    "src_sents": src_sents,
                    "tgt_sents": tgt_sents,
                    "align_seconds": round(align_seconds, 3),
                    "pairs": [{"src": s, "tgt": t} for s, t in pairs],
                })

    cache_volume.commit()  # persist HF / SONAR caches for fast reruns
    return [
        {
            "device": dev,
            "tool": tool,
            "embed": embed,
            "model_load_seconds": round(load_seconds, 2),
            "n_theses": len(prepared),
            "records": per_run[(tool, embed)],
        }
        for tool, embed in runs
    ]


@app.function(image=seg_image, volumes={CACHE_PATH: cache_volume}, timeout=60 * 30)
def segment_remote(subset, src, targets, n) -> list[dict]:
    """The single shared segmentation step: split the first ``n`` usable theses
    ONCE with SaT and return their sentence lists for every alignment image to
    reuse. Runs on CPU (no GPU) -- SaT's small model is fast on abstract text."""
    import sys

    sys.path.insert(0, REMOTE_SRC_DIR)
    import kurdish_parallel_tools_comparison.align as A

    prepared = A.prepare_theses(subset, src, list(targets), n,
                                require_all_targets=True)
    cache_volume.commit()  # persist the downloaded SaT model for fast reruns
    return prepared


# The three LaBSE tools share one container so LaBSE loads once; all of them (and
# the other two images) align the identical pre-segmented sentence lists.
LABSE_RUNS = [("bertalign", "labse"), ("vecalign", "labse"), ("sentalign", "labse")]


@app.function(image=labse_image, volumes={CACHE_PATH: cache_volume}, timeout=60 * 60)
def align_labse_remote(prepared, src, device) -> list[dict]:
    return _run_alignment(prepared, src, device, LABSE_RUNS)


@app.function(image=sonar_image, volumes={CACHE_PATH: cache_volume}, timeout=60 * 60)
def align_sonar_remote(prepared, src, device) -> list[dict]:
    return _run_alignment(prepared, src, device, [("vecalign", "sonar")])


@app.function(image=laser_image, volumes={CACHE_PATH: cache_volume}, timeout=60 * 60)
def align_laser_remote(prepared, src, device) -> list[dict]:
    # torch 1.13.1 here is the CUDA 11.7 build, and LASER3 self-moves to CUDA
    # when a GPU is attached; without one it runs on CPU (slow). Attach a T4
    # (not an L4 -- Ada needs CUDA >=11.8) via --gpu for the fast path.
    return _run_alignment(prepared, src, device, [("vecalign", "laser")])


@app.local_entrypoint()
def main(n: int, gpu: str = "T4", device: str = "auto"):
    """Run all five options concurrently on the same ``n`` theses.

    ``n`` (REQUIRED) is the number of theses EACH option aligns -- the pool the
    gold set is built over -- not a total split across options. LASER3 needs a
    T4 (its torch 1.13 is CUDA 11.7; an L4 won't work), so --gpu drives every
    option and defaults to T4.
    """
    if n < 1:
        raise ValueError("--n is required and must be >= 1 (theses per option)")

    gpu_option = None if gpu.lower() in {"none", "cpu", ""} else gpu

    # Single shared segmentation step FIRST (blocking): split the n theses once
    # with SaT, on CPU, and hand the identical sentence lists to every image.
    prepared = segment_remote.remote(SUBSET, SRC_LANG, TARGET_LANGS, n)
    print(f"Segmented {len(prepared)} theses "
          f"({SRC_LANG} -> {', '.join(TARGET_LANGS)}); aligning all five options...")

    # Spawn the three images at once (returns immediately) so they run
    # concurrently; the labse container covers bertalign/vecalign/sentalign.
    calls = [
        ("labse", align_labse_remote.with_options(gpu=gpu_option)
         .spawn(prepared, SRC_LANG, device)),
        ("sonar", align_sonar_remote.with_options(gpu=gpu_option)
         .spawn(prepared, SRC_LANG, device)),
        ("laser", align_laser_remote.with_options(gpu=gpu_option)
         .spawn(prepared, SRC_LANG, device)),
    ]

    # Gather. One image failing (laser is the fragile one) must not lose the
    # others, so collect per-image and report failures without aborting.
    outs: list[dict] = []
    failures: list[str] = []
    for label, call in calls:
        try:
            outs.extend(call.get())
        except Exception as exc:  # noqa: BLE001 - surface, don't abort
            failures.append(f"{label}: {exc}")
            print(f"\n!!! {label} image FAILED: {exc}\n")

    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for out in outs:
        out_path = data_dir / f"{out['tool']}_{out['embed']}_{n}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for record in out["records"]:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Find the theses whose alignment differs across >= 2 options -- the ones
    # worth hand-annotating for the gold set. Segmentation is identical across
    # options, so comparing bead lists directly is valid; an option's alignment
    # of one (thesis, target) is signed by its tuple of (src, tgt) beads.
    def numbered(sents):
        # 0-based index per sentence, matching the indices the gold `links` use.
        return [{"i": i, "text": s} for i, s in enumerate(sents)]

    by_thesis: dict = {}
    meta: dict = {}      # tez_id -> {title, src_summary, src_sents} (Kurmanji, shared)
    tgt_info: dict = {}  # (tez_id, tgt_lang) -> {tgt_summary, tgt_sents}
    for out in outs:
        option = f"{out['tool']}_{out['embed']}"
        for record in out["records"]:
            tez_id, tgt_lang = record["tez_id"], record["tgt_lang"]
            sig = tuple((p["src"], p["tgt"]) for p in record["pairs"])
            by_thesis.setdefault(tez_id, {}).setdefault(tgt_lang, {})[option] = sig
            meta.setdefault(tez_id, {"title": record.get("title"),
                                     "src_summary": record["src_summary"],
                                     "src_sents": record["src_sents"]})
            tgt_info.setdefault((tez_id, tgt_lang),
                                {"tgt_summary": record["tgt_summary"],
                                 "tgt_sents": record["tgt_sents"]})

    differing = []
    n_pairs = n_diff_pairs = 0
    for tez_id, targets_map in by_thesis.items():
        diff_targets = []
        for tgt_lang, opt_sigs in targets_map.items():
            n_pairs += 1
            if len(set(opt_sigs.values())) < 2:
                continue  # every option agreed
            n_diff_pairs += 1
            groups: dict = {}
            for option, sig in opt_sigs.items():
                groups.setdefault(sig, []).append(option)
            info = tgt_info[(tez_id, tgt_lang)]
            diff_targets.append({
                "tgt_lang": tgt_lang,
                "tgt_summary": info["tgt_summary"],
                "tgt_sents": numbered(info["tgt_sents"]),
                "n_distinct_alignments": len(groups),
                "agreeing_options": sorted((sorted(o) for o in groups.values()),
                                           key=len, reverse=True),
            })
        if diff_targets:
            differing.append({
                "tez_id": tez_id,
                "title": meta[tez_id]["title"],
                "src_summary": meta[tez_id]["src_summary"],   # Kurmanji, fixed across options
                "src_sents": numbered(meta[tez_id]["src_sents"]),
                "targets": diff_targets,
            })
    differing.sort(key=lambda d: str(d["tez_id"]))

    result = {
        "n": n,
        "options": [f"{o['tool']}_{o['embed']}" for o in outs],
        "n_thesis_target_pairs": n_pairs,
        "n_differing_pairs": n_diff_pairs,
        "differing": differing,
    }
    result_path = data_dir / f"result_{n}.json"
    with result_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    # Summary only.
    print(f"{len(outs)}/5 options over {n} theses "
          f"({SRC_LANG} -> {', '.join(TARGET_LANGS)}): "
          + ", ".join(f"{o['tool']}/{o['embed']} {len(o['records'])}rec" for o in outs))
    print(f"{n_diff_pairs}/{n_pairs} thesis-target pairs differ across >=2 options "
          f"-> {result_path}")
    if failures:
        print("Failed options: " + "; ".join(failures))
