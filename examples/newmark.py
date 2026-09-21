"""Original newmark demonstration, separated from reusable library code."""

from pathlib import Path
import sys

# Direct script execution needs the project root on the import path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from beam_fem.beam_loads import load_case_label
from beam_fem.newmark import NewmarkBeamAnalysis


def main():
    cases = [
        ("uniform", {"q0": 10.0}),
        ("point_load_end", {"P": 1000.0}),
        ("end_moment", {"M": 1000.0}),
    ]

    for load_case, load_params in cases:
        analysis = NewmarkBeamAnalysis(load_case=load_case, **load_params)
        result = analysis.solve()

        print("\n" + "=" * 60)
        print(load_case_label(load_case, **load_params))
        print(f"Final tip displacement: {result.tip_displacement[-1]:.6e} m")
        print(f"Maximum tip displacement: {np.max(np.abs(result.tip_displacement)):.6e} m")
        print(f"Final total energy: {result.energy[-1]:.6e}")


if __name__ == "__main__":
    main()
