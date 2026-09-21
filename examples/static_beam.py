"""Original static_beam demonstration, separated from reusable library code."""

from pathlib import Path
import sys

# Direct script execution needs the project root on the import path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beam_fem.beam_loads import analytic_cantilever_tip
from beam_fem.static_beam import BeamParameters, BeamAnalysis


def main():
    BEAM_TYPE = "cantilever"

    # Three load cases from script1_bending_and_fem.pdf (Section 9, Step 4)
    cases = [
        ("uniform", {"q0": 10.0}),
        ("point_load_end", {"P": 1000.0}),
        ("end_moment", {"M": 1000.0}),
    ]

    params = BeamParameters()
    E, I, L = params.youngs_modulus, params.moment_inertia, params.length

    for load_case, load_params in cases:
        print("\n" + "=" * 60)
        analysis = BeamAnalysis(beam_type=BEAM_TYPE, load_case=load_case, **load_params)
        displacement = analysis.solve()

        w_tip_fem = displacement[-1]
        if BEAM_TYPE == "cantilever":
            w_tip_exact = analytic_cantilever_tip(E, I, L, load_case, **load_params)
            rel_err = abs(w_tip_fem - w_tip_exact) / abs(w_tip_exact)
            print(f"Tip deflection (FEM):    {w_tip_fem:.6e} m")
            print(f"Tip deflection (exact):  {w_tip_exact:.6e} m")
            print(f"Relative error at tip:   {rel_err:.6e}")

        analysis.plot_results()


if __name__ == "__main__":
    main()
