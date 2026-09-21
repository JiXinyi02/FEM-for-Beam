# FEM for Beam

Finite-element static, modal, and dynamic analysis of Euler-Bernoulli beams
and planar frames.

## Project Layout

```text
beam_fem/
    parameters.py          Beam/frame parameters, nodes, and connectivity
    results.py             Static, modal, and dynamic result containers
    elements.py            Element matrices and beam matrix assembly
    interpolation.py       Hermite shape functions and beam interpolation
    constraints.py         Boundary conditions and nullspace construction
    beam_loads.py          Load models, load assembly, and analytic tip values
    static_beam.py         Static beam solver
    extended_matrix.py     Augmented-matrix beam solver
    eigenvalue.py          Beam modal analysis and free vibration
    newmark.py             Beam Newmark time integration
    framework.py           Planar-frame static/modal/dynamic solvers
    plotting/              Plotting, animations, and output generation
examples/                  The seven original executable demonstrations
code/                      Compatibility imports and legacy script entry points
tests/                     Numerical, API, and visualization regression tests
docs/                      Existing project documents
figs/                      Existing plots and animations
```

Edit numerical implementations in `beam_fem/`, figure generation in
`beam_fem/plotting/`, and demo parameters/workflows in `examples/`.
`code/*.py` contains forwarding wrappers, not a second implementation.

## Setup and Execution

Use Python 3.10 or later. From the project root:

```sh
python -m pip install -e .
python -m examples.static_beam
python -m examples.extended_matrix
python -m examples.newmark
python -m examples.eigenvalue
python -m examples.framework
python -m examples.newmark_visualization
python -m examples.my_plot_fun
```

All seven example scripts also support direct execution without an editable
installation. For example, from the `examples/` directory:

```sh
python eigenvalue.py
python framework.py
```

From the project root, `python examples/eigenvalue.py` also works. Each script
locates the project root from its own file path before importing `beam_fem`.
This only sets the import path; it does not change the working directory.

Installation is optional when running from the project root if NumPy,
Matplotlib, and Pillow are already available. All original demo load cases,
time steps, output filenames, and output paths are retained. Relative output
paths are resolved from the current working directory, as before. Run from the
project root for the expected project output locations. Some examples open
Matplotlib windows; use the `Agg` backend for noninteractive runs.

Legacy commands such as `python code/static_beam.py` and
`python code/framework.py` still run the corresponding examples. Existing
imports from a `code/` directory on `sys.path` remain supported, including
`from static_beam import BeamParameters, FiniteElementMatrices, BeamAnalysis`.

## Library Usage

```python
from beam_fem.parameters import BeamParameters
from beam_fem.elements import FiniteElementMatrices
from beam_fem.static_beam import BeamAnalysis
from beam_fem.framework import FrameworkEigenvalueAnalysis

parameters = BeamParameters(length=10.0, num_elements=25)
stiffness, mass = FiniteElementMatrices(parameters).assemble_global_matrices()

beam = BeamAnalysis(load_case="point_load_end", P=1000.0)
displacement = beam.solve()
beam.plot_results()

frame = FrameworkEigenvalueAnalysis(num_modes=6)
modes = frame.solve()
response = frame.solve_newmark_release_from_point_load(node="p3", fy=-1e5)
```

Plotting methods on analysis objects remain available and delegate to functions
in `beam_fem.plotting`. For example, `beam.plot_results()` and
`beam_fem.plotting.static_beam.plot_results(beam)` use the same implementation.
Imports do not run demonstrations or generate output files.

## Validation

From the project root:

```sh
python -m unittest discover -s tests -v
python tests/snapshot_calculations.py --compare-ref b8cb0e4
```

Individual test files can also be run directly from the `tests/` directory:

```sh
python test_layout.py
python test_numerics.py
```

These entry points locate the project root relative to their own files, so
an editable installation is not required for these test commands.

Tests cover analytic static deflections, augmented-system agreement, modal
residuals and normalization, constrained responses, release energy conservation,
legacy imports/entry points, and PNG/GIF generation. Temporary output directories
keep existing project figures unchanged.

During the reorganization, all 150 original function/method implementations and
all seven demo bodies were checked for preservation, and 199 numerical arrays
or scalars were compared against the pre-refactor implementation with exact
equality. This reorganization does not change the numerical models or correct
pre-existing numerical assumptions.

The snapshot command requires the original commit `b8cb0e4` in local Git
history. It loads that revision in a separate process and compares it with the
current package; it does not check out or modify any tracked files.
