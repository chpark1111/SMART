LATEXMK ?= latexmk
PDFLATEX ?= pdflatex
BIBTEX ?= bibtex

.PHONY: all supp clean distclean fallback supp-fallback

all:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

supp:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error supplementary.tex

fallback:
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(BIBTEX) main
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex

supp-fallback:
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary.tex
	$(BIBTEX) supplementary
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error supplementary.tex

clean:
	$(LATEXMK) -c main.tex || true
	$(LATEXMK) -c supplementary.tex || true
	rm -f main.bbl main.blg main.out main.log main.aux main.fdb_latexmk main.fls
	rm -f supplementary.bbl supplementary.blg supplementary.out supplementary.log supplementary.aux supplementary.fdb_latexmk supplementary.fls

distclean: clean
	rm -f main.pdf supplementary.pdf
