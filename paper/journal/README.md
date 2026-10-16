# Journal submission package

Target: **Journal of Computational Physics** (Elsevier). Everything here is either compiled or
checked by `src/build_journal.py`; run that after any change:

```powershell
$env:TEXLIVE_BIN = "<your TeX Live bin directory>"   # only if pdflatex is not on PATH;
                                                     # e.g. the folder containing pdflatex.exe
python src\build_journal.py
```

---

## 1. What is in the package

| item | file | state |
|---|---|---|
| Manuscript (submission PDF) | `paper/main.pdf` | compiled, page-checked, 17 pages, 0 errors |
| Manuscript (source) | `paper/main.tex` + `paper/tables/*.tex` + `paper/refs.bib` + `figures/*` | every table generated from `results/` |
| Highlights | `journal/highlights.txt` | 5 bullets, longest 80 chars (limit 85) |
| Cover letter | `journal/cover_letter.tex` → `.pdf` | compiled, 2 pages |
| Declarations (CRediT, competing interest, data availability, funding, AI use) | `journal/declarations.tex` → `.pdf` | compiled, 2 pages |
| Nomenclature | `journal/nomenclature.tex` → `.pdf` | compiled, 1 page |
| Code and results archive | repository + Zenodo DOI | **author action**, see §4 |

## 2. Why the manuscript is compiled in IEEEtran, not `elsarticle`

Elsevier does not require its own class at initial submission — the guide for authors asks for the
class at revision, and for free formatting before that. `elsarticle.cls` is *not* installed in the
TeX Live on this machine, and it could not be fetched here: every CTAN path tried returned either 404
or an anti-bot page, and `tlmgr` refuses to install it until it has updated itself.

**The conversion is mechanical and takes about fifteen minutes.** When it is wanted:

1. Install the class: `tlmgr update --self` then `tlmgr install elsarticle` (or open the project on
   Overleaf, which ships the class).
2. Change the two class lines at the top of `paper/main.tex`:
   ```latex
   \documentclass[preprint,12pt]{elsarticle}     % was: \documentclass[conference]{IEEEtran}
   % delete \IEEEoverridecommandlockouts
   ```
3. Replace the author block with the Elsevier form (`\author[...]{...}`, `\affiliation{...}`,
   `\cortext`/`\fnref` for the corresponding author), and `\begin{IEEEkeywords}...\end{IEEEkeywords}`
   with `\begin{frontmatter}` + `\begin{keyword}...\end{keyword}`.
4. Switch the bibliography: `\bibliographystyle{elsarticle-num}` (numeric, what JCP uses).
5. Rebuild and re-check the page count and overfull boxes.

Nothing in the body depends on IEEEtran: all figures, tables, equations and section commands are
standard. The float-localisation tweaks in the preamble (`\topfraction` and friends) can be dropped —
they exist because the IEEEtran conference layout is float-starved, not because the paper needs them.

## 3. Submission checklist

| requirement | where it is satisfied | checked by |
|---|---|---|
| Editable source | `paper/main.tex`, `refs.bib`, `tables/*.tex` | `src/check_tex.py` |
| Compiles with no undefined references or citations | 0 errors, 0 undefined | `src/build_paper.py` |
| Highlights: 3–5 bullets, ≤ 85 characters | 5 bullets, longest 80 | `src/build_journal.py` |
| CRediT authorship statement | `journal/declarations.tex` §1 | needs author confirmation |
| Declaration of competing interest | `journal/declarations.tex` §2 | — |
| Data availability statement | `journal/declarations.tex` §3 | — |
| Funding statement | `journal/declarations.tex` §4 | **author action** |
| Acknowledgements | `journal/declarations.tex` §5 | **author action** |
| Generative-AI use disclosed | `journal/declarations.tex` §6 | **author action** |
| Reproducibility: every number traceable | `results/`, generated tables | `src/verify_paper_numbers.py` (65 assertions), `src/check_table_drift.py` |
| Code and data released | repository, archived with DOI | **author action**, §4 |
| Figures at publication resolution | PDF (vector) + 300 dpi PNG | `figures/` |
| Nomenclature / notation | `journal/nomenclature.tex` | — |
| Suggested referees | not prepared | **author action** (optional but helpful) |

## 4. Archiving the code so it has a DOI

Elsevier accepts a repository link, but a DOI is what a journal wants for a reproducibility
statement, and a personal GitHub URL can be renamed or deleted after publication. The steps:

1. Make a release in the repository at the commit the paper's numbers were produced with, tagged
   `v1.0-jcp` (the numbers and the tag must match; that is the whole point).
2. In Zenodo, connect the GitHub account and enable the repository; Zenodo then mints a DOI per
   release automatically.
3. Put that DOI in the manuscript's `\repourl` macro (`paper/main.tex`), replacing the GitHub URL,
   and add it to the data availability statement.
4. Record the DOI in `STATUS.md` so the version the paper was compiled against is unambiguous.

The repository already passes a pre-release scan: no credentials, no absolute machine paths, no build
artefacts (`src/check_release.py` → 0 credentials, 0 absolute paths).

## 5. What only the authors can do

These are deliberately left blank rather than guessed:

* confirm the CRediT roles and mark the corresponding author;
* fill in funding and acknowledgements (or state that there are none);
* make the generative-AI disclosure in the wording the publisher requires — it must be true, and only
  the authors know what is true;
* create the Zenodo release and insert the DOI;
* upload, and optionally suggest referees.

## 6. Rebuilding everything from scratch

```powershell
cd <repository root>
$env:TEXLIVE_BIN = "<your TeX Live bin directory>"
$env:PYTHONPATH   = "$PWD\src"
python src\verify_all.py            # 18 checks: environment, units, core numerics
python src\make_tables.py           # regenerates every table from results/
python src\make_journal_figures.py  # regenerates the two journal figures
python src\build_paper.py           # compiles and page-checks the manuscript
python src\build_journal.py         # cover letter, declarations, nomenclature, highlights check
python src\verify_paper_numbers.py  # 65 prose assertions against results/
```

The table and figure generators are byte-reproducible: re-running them produces identical files.
