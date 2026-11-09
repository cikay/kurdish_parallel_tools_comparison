"""kurdish_parallel_tools_comparison: sentence alignment of the kurdish-theses-yok summaries.

Layout:
  * ``align``            - data loading, sentence splitting, and the CLI.
  * ``tools.tools``      - alignment algorithms (Bertalign, Vecalign) + dispatch.
  * ``tools.embeddings`` - embedding models (LaBSE, SONAR) + neutral LaBSE scorer.

Modal orchestration lives OUTSIDE this package, in ``modal_app/`` at the repo
root; it imports this package rather than duplicating logic.
"""
