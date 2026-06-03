# SMART-K CVPR Draft

This folder contains a CVPR-format LaTeX draft for the learned SMART router and
variable-length geometry skill research.

## Files

- `main.tex`: paper draft.
- `refs.bib`: BibTeX references.
- `supplementary.tex`: optional supplementary material draft.
- `cvpr.sty`: CVPR author-kit style file.
- `ieeenat_fullname.bst`: CVPR bibliography style.
- `figures/smart_teaser.png`: copied from `docs/teaser.png`.

## Build

With a full TeX Live or MacTeX install:

```bash
cd paper/learned_router_cvpr
make
```

To build the supplementary PDF:

```bash
make supp
```

If `latexmk` is unavailable but `pdflatex` and `bibtex` exist:

```bash
make fallback
make supp-fallback
```

For Overleaf, upload this directory and set `main.tex` as the main document.

## Review vs. Final Mode

The draft currently uses:

```tex
\usepackage[review]{cvpr}
```

For camera-ready formatting, change it to:

```tex
\usepackage[final]{cvpr}
```

## What Still Needs Manual Editing

- Replace TODO affiliation/email fields in `main.tex`.
- Decide whether the target venue/year should remain `CVPR 2027`.
- Replace validation-only wording once larger held-out experiments are complete.
- Add any new figures/tables generated from final experiments.
