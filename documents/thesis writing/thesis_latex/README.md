# Thesis LaTeX project

Uva Wellassa University, Faculty of Applied Sciences,
Department of Computer Science and Informatics.

## Compiling

Upload the whole folder to Overleaf and set `main.tex` as the main document.
Overleaf's default LaTeX + BibTeX recipe works, but **the glossaries package
needs a `makeglossaries` pass**, so in Overleaf go to
*Menu → Compiler → LaTeX* and let it run the full sequence. Locally:

```
pdflatex main
bibtex   main
makeglossaries main
pdflatex main
pdflatex main
```

Two `pdflatex` passes after BibTeX are required for the table of contents,
the List of Figures, the List of Tables and all cross-references to resolve.

> This project has **not been compiled** — there is no LaTeX toolchain on the
> machine it was written on. Citations, labels, environments and acronym keys
> were validated statically and are consistent, but the first compile may
> still surface layout warnings (overfull boxes on wide tables in particular).

## Formatting rules implemented

From *SECTION-H Dissertation Preparation Guidelines and Format*:

| Rule | Where |
|---|---|
| A4, 12 pt Times New Roman | `main.tex` — `a4paper, 12pt`, `mathptmx` |
| Left 3.85 cm; right, top, bottom 2.5 cm | `main.tex` — `\geometry` |
| Line spacing 1.5, one line between paragraphs | `\setstretch{1.5}`, `\parskip` |
| Page numbers, bottom centre | `\pagestyle{plain}` |
| Roman numerals for prefatory pages | `\pagenumbering{roman}` before the cover |
| Arabic numerals from Chapter 1 | `\pagenumbering{arabic}` before `chapter1` |
| Roman numerals for appendices | `\renewcommand{\thepage}{A-\roman{page}}` |
| Numbered, captioned tables and figures, all cited in text | `\numberwithin{...}{section}` |
| Author-Year citation system | `natbib` + `agsm` |

**One conflict you must decide on.** The guideline says appendices carry Roman
numerals; the official UWU LaTeX template does not switch back to Roman after
the References. This project follows the *written guideline* and numbers
appendix pages `A-i`, `A-ii`, … If your coordinator prefers the template
behaviour, delete the three lines under `APPENDICES` in `main.tex`.

**Reference style.** The guideline's specimen format
(`Bell, C.H. (1991). Title. Journal Volume, 81-93.`) is not exactly APA and not
exactly `agsm`. The official template ships `agsm`, so `agsm` is used here.
Confirm with your supervisor before final submission; changing it is a
one-line edit to `\bibliographystyle` in `main.tex`.

## Page budgets

- Chapters 1 + 2 together: **maximum 10 pages**. Check this on the first
  compile and trim Chapter 2 first if it runs over.
- Chapters 3 + 4 + 5 together: **maximum 40 pages**.

## Outstanding items

Every place needing your input is marked `\TODO{...}` and renders in bold as
`[TO BE SUPPLIED: ...]`, so nothing can be missed on a read-through. Search
the project for `\TODO` before submitting, and delete the macro definition in
`main.tex` once none remain.

Current count: 46 items — heaviest in `chapter3.tex` (17, mostly protocol
details), `chapter4.tex` (12, awaiting trial data) and `chapter5.tex` (7).

## Two references held back

`references.bib` ends with two commented-out entries, `li2024` and
`nahiduzzaman2025`. Neither PDF was available, and the claims the Word draft
attributed to them are inconsistent with their stated titles. Drop the PDFs
into `documents/references/`, verify what they actually say, then uncomment
and cite. Until then nothing in the text depends on either.
