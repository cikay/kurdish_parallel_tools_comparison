"""
Modal entry point for SCORING the alignment comparison against the gold set.

This is intentionally thin: all the metric logic lives in
``kurdish_parallel_tools_comparison.scoring`` (pure stdlib, no Modal), and this file only ships
that package into a lightweight image and moves data in/out. So the comparison
code stays decoupled from Modal -- it runs identically from the local CLI
(``python -m kurdish_parallel_tools_comparison.scoring --n 20``) and from here.

What it does
------------
The alignment run (``modal_app/app.py``) writes one JSONL per option to
``data/{sat_model}/{tool}_{embed}_{n}.jsonl`` locally, and the gold set lives at
``gold_set.json``. This entry point reads those locally, ships the parsed records
+ gold to a CPU function that computes precision / recall / F1 per option
(``scoring.compare_records``), then prints the report and writes
``data/{sat_model}/scores_{n}.json``.

Scoring is pure arithmetic on tiny data, so the remote function needs no GPU and
no heavy deps -- it exists so the whole comparison workflow (align, then score)
can run through Modal with one mental model. To score purely locally instead,
just use the module CLI; it needs nothing from this file.

    modal run modal_app/score_app.py --n 20
    modal run modal_app/score_app.py --n 20 --verbose
    modal run modal_app/score_app.py --suffix full          # score the *_full.jsonl set
"""

import json
from pathlib import Path

import modal

APP_NAME = "kurdish-align-score"

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PKG = REPO_ROOT / "kurdish_parallel_tools_comparison"
REMOTE_SRC_DIR = "/root/src"

# The five options the comparison run produces; names match the JSONL filenames.
CANONICAL_OPTIONS = (
    "bertalign_labse",
    "vecalign_labse",
    "sentalign_labse",
    "vecalign_sonar",
    "vecalign_laser",
)

app = modal.App(APP_NAME)

# Scoring is pure stdlib, so a plain slim image with the package is enough -- no
# torch, no model downloads, no GPU.
score_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(str(LOCAL_PKG), f"{REMOTE_SRC_DIR}/kurdish_parallel_tools_comparison")
)


@app.function(image=score_image, timeout=60 * 5)
def score_remote(gold: list, option_records: dict, n, suffix) -> dict:
    """Score already-loaded option records against the gold set, remotely.

    ``gold`` is the parsed ``gold_set.json`` list; ``option_records`` maps each
    option name to its list of record dicts. All logic is in the shipped
    ``scoring`` module, so this is a one-line call -- the Modal boundary moves
    data, it does not own the metric.
    """
    import sys

    sys.path.insert(0, REMOTE_SRC_DIR)
    from kurdish_parallel_tools_comparison import scoring

    return scoring.compare_records(gold, option_records, n=n, suffix=suffix)


@app.local_entrypoint()
def main(n: int = 0, suffix: str = "", gold: str = "", verbose: bool = False,
         no_write: bool = False):
    """Read the local gold set + per-option JSONL, score remotely, report.

    ``--n`` selects the ``{option}_{n}.jsonl`` files (and the ``scores_{n}.json``
    output name); ``--suffix`` overrides it (e.g. ``full``). One of the two is
    required.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from kurdish_parallel_tools_comparison import scoring
    from kurdish_parallel_tools_comparison.align import SAT_MODEL_NAME

    file_suffix = suffix or (str(n) if n else "")
    if not file_suffix:
        raise ValueError("provide --n (or --suffix)")

    gold_path = Path(gold) if gold else REPO_ROOT / "gold_set.json"
    gold_data = json.loads(gold_path.read_text("utf-8"))

    data_dir = REPO_ROOT / "data" / SAT_MODEL_NAME
    option_records: dict[str, list] = {}
    for option in CANONICAL_OPTIONS:
        path = data_dir / f"{option}_{file_suffix}.jsonl"
        if not path.exists():
            continue
        option_records[option] = [
            json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()
        ]

    if not option_records:
        raise FileNotFoundError(
            f"no {{option}}_{file_suffix}.jsonl files found in {data_dir}")

    summary = score_remote.remote(gold_data, option_records,
                                  n if n else None, file_suffix)

    print(scoring.format_report(summary, verbose=verbose))

    if not no_write:
        out_path = data_dir / f"scores_{file_suffix}.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
        print(f"\nWrote {out_path}")
