# Project: Numerical Analysis

Finite element method experiments for beam bending, static response, and modal
analysis.

## Project Structure

```text
.
+-- docs/                  # Reference PDFs and assignment notes
+-- outputs/
|   +-- figures/           # Generated plots and images
+-- scripts/               # Runnable analysis scripts
|   +-- my_plot_fun.py     # Hermite basis and plotting helper
|   +-- static_beam.py     # Static beam FEM analysis
|   +-- modal_analysis.py  # Modal analysis scaffold
+-- src/                   # Package source code
+-- pyproject.toml         # Project and Pixi configuration
+-- pixi.lock              # Locked environment
```

## Setup

Install [Pixi](https://pixi.sh/latest/installation/), then run:

```bash
git clone https://github.com/heyjiacheng/Project-Numerical-Analysis.git
cd Project-Numerical-Analysis
pixi install
pixi shell
```

## Run Scripts

Plot the Hermite basis interpolation example:

```bash
python scripts/my_plot_fun.py
```

Run the static beam analysis:

```bash
python scripts/static_beam.py
```

Run the modal analysis scaffold:

```bash
python scripts/modal_analysis.py
```

## Notes

- Put generated figures in `outputs/figures/`.
- Put reference material and reports in `docs/`.
- Keep reusable Python package code in `src/`; keep one-off runnable examples in
  `scripts/`.
