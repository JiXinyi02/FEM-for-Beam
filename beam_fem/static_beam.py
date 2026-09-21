"""
=============================================================================
STATIC BEAM ANALYSIS USING FINITE ELEMENT METHOD
=============================================================================
This module performs static analysis of beams using the finite element method.
Supports both cantilever and simply supported beams under uniform loading.
"""

import numpy as np

from .beam_loads import LoadApplication, load_case_label
from .constraints import BoundaryConditions
from .elements import FiniteElementMatrices
from .parameters import BeamParameters
from .plotting import static_beam as _plotting


class BeamAnalysis:
    """Main class for beam analysis."""

    def __init__(self, beam_type="cantilever", load_case="uniform", **load_params):
        """
        Initialize beam analysis.

        Args:
            beam_type (str): "cantilever" or "simply_supported"
            load_case (str): "uniform", "point_load_end", or "end_moment"
            **load_params: q0 / P / M for the selected load case
        """
        self.beam_type = beam_type
        self.load_case = load_case
        self.load_params = load_params

        # Initialize components
        self.beam_params = BeamParameters()
        self.fe_matrices = FiniteElementMatrices(self.beam_params)
        self.load_handler = LoadApplication(self.beam_params)

        # Analysis results
        self.displacement = None
        self.full_dof = None
        self.global_stiffness = None
        self.global_mass = None

    def solve(self):
        """Perform the complete beam analysis."""
        print(
            f"Analyzing {self.beam_type} beam, "
            f"{load_case_label(self.load_case, **self.load_params)}"
        )

        # 1. Compute global matrices
        self.global_stiffness, self.global_mass = self.fe_matrices.assemble_global_matrices()

        # 2. Apply loads
        force_vector = self.load_handler.apply_load(self.load_case, **self.load_params)

        # 3. Apply boundary conditions and solve
        self._solve_system(force_vector)
        self.displacement = self.full_dof[::2]

        print("Analysis completed successfully!")
        return self.displacement

    def _solve_system(self, force_vector):
        """Solve the system of equations with appropriate boundary conditions."""
        if self.beam_type == "cantilever":
            return self._solve_cantilever(force_vector)
        elif self.beam_type == "simply_supported":
            return self._solve_simply_supported(force_vector)
        else:
            raise ValueError(f"Unknown beam type: {self.beam_type}")

    def _solve_cantilever(self, force_vector):
        """Solve cantilever beam system."""
        K_reduced, F_reduced = BoundaryConditions.apply_cantilever_bc(
            self.global_stiffness, force_vector
        )

        # Solve reduced system
        displacement_reduced = np.linalg.solve(K_reduced, F_reduced)

        # Reconstruct full DOF vector (displacement and rotation at each node)
        self.full_dof = np.zeros(2 * self.beam_params.num_nodes)
        self.full_dof[2:] = displacement_reduced

        return self.full_dof[::2]

    def _solve_simply_supported(self, force_vector):
        """Solve simply supported beam system."""
        K_reduced, F_reduced = BoundaryConditions.apply_simply_supported_bc(
            self.global_stiffness, force_vector, self.beam_params.num_nodes
        )

        # Solve reduced system
        displacement_reduced = np.linalg.solve(K_reduced, F_reduced)

        # Reconstruct full DOF vector (fixed end displacements are zero)
        num_nodes = self.beam_params.num_nodes
        fixed_dofs = [0, 2 * (num_nodes - 1)]
        free_dofs = [i for i in range(2 * num_nodes) if i not in fixed_dofs]
        self.full_dof = np.zeros(2 * num_nodes)
        self.full_dof[free_dofs] = displacement_reduced

        return self.full_dof[::2]

    def plot_results(self):
        """Plot displacement using Hermite interpolation from my_plot_fun."""
        return _plotting.plot_results(self)
