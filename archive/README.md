# Archived drafts — not part of the release

`paper_long_v21.tex` is a **stale draft** of an earlier, longer version of this manuscript, dated
2026-09-23. It was moved here rather than deleted so the history is not lost, but it is **not
maintained and must not be cited, quoted, or used to source any number**.

Why it is archived rather than shipped in the repository root:

* **Nothing references it.** No script, table generator, build step or document in this repository
  reads `paper_long_v21.tex`; it was carried along by hand.
* **It predates the results it appears to contain.** The released manuscript at `../paper/main.tex`
  has since gained sections that do not exist in this draft at all: the market-calibrated scope
  analysis, the architecture-sensitivity sweep, and the reconstruction-order generalisation, along
  with the tables `tab_calib`, `tab_arch` and `tab_order`. Numbers in the draft therefore disagree
  with the released tables.
* **It invites a specific misreading.** Its own header describes it as a longer version of the
  manuscript, so a reviewer who finds it in the repository root would reasonably treat its numbers as
  the paper's. They are not.

**If you need the current manuscript, read [`../paper/main.tex`](../paper/main.tex).** Every number in
it is generated from `../results/*.json` by the scripts in `../src/`, and is checked by
`../src/verify_paper_numbers.py` and `../src/check_table_drift.py`.

To reproduce the released results, ignore this directory entirely.
