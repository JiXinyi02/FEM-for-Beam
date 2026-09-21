"""Original extended_matrix demonstration, separated from reusable library code."""

from pathlib import Path
import sys
import time

# Direct script execution needs the project root on the import path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beam_fem.beam_loads import load_case_label
from beam_fem.extended_matrix import BeamFEM


def main():
    cases = [
        ("uniform", {"q0": 10.0}),
        ("point_load_end", {"P": 1000.0}),
        ("end_moment", {"M": 1000.0}),
    ]

    for load_case, load_params in cases:
        print("\n" + "=" * 60)
        print(load_case_label(load_case, **load_params))

        beam_extended = BeamFEM(load_case=load_case, **load_params)
        t0 = time.perf_counter()
        beam_extended.solve_static_extended()
        extended_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        w_tip_extended, w_tip_static, abs_difference, rel_difference = (
            beam_extended.compare_with_static_beam()
        )
        static_time = time.perf_counter() - t0

        print(f"Tip deflection (extended matrix): {w_tip_extended:.6e} m")
        print(f"Tip deflection (static_beam.py):  {w_tip_static:.6e} m")
        print(f"Absolute difference:              {abs_difference:.6e} m")
        print(f"Relative difference:              {rel_difference:.6e}")
        print(f"Extended matrix time:             {extended_time:.6e} s")
        print(f"static_beam.py time:              {static_time:.6e} s")
        beam_extended.plot_piecewise_polynomial()


if __name__ == "__main__":
    main()
