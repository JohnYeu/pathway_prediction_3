# PathwayML-Ath

This workspace contains the reproducible analysis and the manuscript source:

- `code/`: reproducible Python code, input/cache data, generated tables, generated figures, and benchmark outputs.
- `latex/`: manuscript LaTeX source, references, and manuscript figure assets.

Run the analysis from the code directory:

```bash
cd code
python3 reproducible_pipeline.py --sections all --full
```

The command regenerates machine-readable results and scientific figures under
`code/generated/`. Document conversion and upload-package utilities are kept
outside this repository.

To compile the paper, run LaTeX from the manuscript folder:

```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The manuscript reports the current generated results. For manual verification,
use `code/generated/tables/` and `code/generated/tables/latex/` as the
authoritative numerical outputs.
