"""Physical and algebraic checks independent of the source-file layout."""

import contextlib
import io
from pathlib import Path
import sys
import unittest

# Direct script execution puts tests/, not the project root, on sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from beam_fem.beam_loads import analytic_cantilever_tip
from beam_fem.eigenvalue import EigenvalueBeamAnalysis
from beam_fem.extended_matrix import BeamFEM
from beam_fem.framework import FrameworkEigenvalueAnalysis
from beam_fem.newmark import NewmarkBeamAnalysis
from beam_fem.static_beam import BeamAnalysis


class NumericalTests(unittest.TestCase):
    def test_static_load_cases_and_extended_solver(self):
        for case, params in (("uniform", {"q0": 10.0}), ("point_load_end", {"P": 1000.0}), ("end_moment", {"M": 1000.0})):
            with self.subTest(case=case), contextlib.redirect_stdout(io.StringIO()):
                solver = BeamAnalysis(load_case=case, **params)
                displacement = solver.solve()
                beam = solver.beam_params
                expected = analytic_cantilever_tip(beam.youngs_modulus, beam.moment_inertia, beam.length, case, **params)
                np.testing.assert_allclose(displacement[-1], expected, rtol=1e-8, atol=0)
                extended, _ = BeamFEM(load_case=case, **params).solve_static_extended()
                np.testing.assert_allclose(extended, solver.full_dof, rtol=1e-8, atol=1e-15)

    def test_modal_equation_and_mass_normalization(self):
        solvers = [EigenvalueBeamAnalysis(beam_type=bc, num_elements=8, num_modes=4) for bc in ("cantilever", "simply_supported")]
        solvers += [FrameworkEigenvalueAnalysis(constrain_p4_rotation=fixed) for fixed in (False, True)]
        for solver in solvers:
            with self.subTest(solver=type(solver).__name__):
                result = solver.solve()
                modes = result.modes_reduced
                stiffness, mass = solver.reduced_stiffness, solver.reduced_mass
                residual = stiffness @ modes - (mass @ modes) * result.eigenvalues
                self.assertLess(np.linalg.norm(residual) / np.linalg.norm(stiffness @ modes), 1e-10)
                np.testing.assert_allclose(modes.T @ mass @ modes, np.eye(modes.shape[1]), atol=1e-10)
                np.testing.assert_allclose(result.constraint_matrix @ result.modes_full, 0, atol=1e-12)

    def test_framework_static_equilibrium_and_release_energy(self):
        solver = FrameworkEigenvalueAnalysis()
        static = solver.solve_static_point_load()
        residual = solver.global_stiffness @ static.displacement - static.force
        np.testing.assert_allclose(static.reactions, residual)
        np.testing.assert_allclose(solver.nullspace_basis.T @ residual, 0, atol=1e-7)
        dynamic = solver.solve_newmark_release_from_point_load(total_time=0.01, time_step=0.0002)
        np.testing.assert_allclose(dynamic.displacement[0], static.displacement, atol=1e-14)
        np.testing.assert_allclose(dynamic.force, 0)
        np.testing.assert_allclose(dynamic.energy, dynamic.energy[0], rtol=1e-10)
        np.testing.assert_allclose(dynamic.displacement @ dynamic.constraint_matrix.T, 0, atol=1e-12)

    def test_newmark_zero_load_and_supports(self):
        for bc in ("cantilever", "simply_supported"):
            with self.subTest(bc=bc):
                solver = NewmarkBeamAnalysis(beam_type=bc, load_case="uniform", q0=0, total_time=0.002)
                result = solver.solve()
                np.testing.assert_array_equal(result.displacement, 0)
                np.testing.assert_array_equal(result.velocity, 0)
                np.testing.assert_array_equal(result.energy, 0)


if __name__ == "__main__":
    unittest.main()
