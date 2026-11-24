"""Build the paper from docs/paper/latex/main.tex: a PDF and the arXiv upload package.

Usage: python3 -I scripts/build_paper.py        (needs tectonic on PATH; it runs the TeX and BibTeX passes itself)
Writes into docs/paper/latex/:
  beyond-refusal-rates.pdf   the paper
  arxiv_source.tar.gz        main.tex, the compiled bibliography (main.bbl) and the figures, for an arXiv upload
Figures are copied from docs/paper/figures/ (made by scripts/v2/make_figures.py). The source stays main.tex so arXiv finds its main file.
"""
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/paper/latex"
FIGURES = ("fig1_what_replaces_refusal", "fig2_projection", "fig3_repairs", "fig7_gate")


def main():
    (D / "figures").mkdir(exist_ok=True)
    for f in FIGURES:
        shutil.copyfile(ROOT / "docs/paper/figures" / f"{f}.pdf", D / "figures" / f"{f}.pdf")
    subprocess.run(["tectonic", "-X", "compile", "--keep-intermediates", "main.tex"], cwd=D, check=True)
    shutil.move(D / "main.pdf", D / "beyond-refusal-rates.pdf")
    with tarfile.open(D / "arxiv_source.tar.gz", "w:gz") as tf:  # arXiv wants the .bbl, not the .bib
        tf.add(D / "main.tex", arcname="main.tex")
        tf.add(D / "main.bbl", arcname="main.bbl")
        for f in sorted((D / "figures").glob("*.pdf")):
            tf.add(f, arcname=f"figures/{f.name}")
    print("built", D / "beyond-refusal-rates.pdf", "and arxiv_source.tar.gz")


if __name__ == "__main__":
    main()
