"""Result containers shared by solvers and visualization functions."""

from dataclasses import dataclass

import numpy as np


@dataclass
class NewmarkResult:
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    energy: np.ndarray
    free_dofs: np.ndarray

    @property
    def tip_displacement(self):
        return self.displacement[:, -2]


@dataclass
class EigenvalueResult:
    eigenvalues: np.ndarray
    angular_frequencies: np.ndarray
    frequencies_hz: np.ndarray
    periods: np.ndarray
    modes_reduced: np.ndarray
    modes_full: np.ndarray
    modal_mass: np.ndarray
    modal_stiffness: np.ndarray
    free_dofs: np.ndarray
    constraint_matrix: np.ndarray
    nullspace_basis: np.ndarray


@dataclass
class FreeVibrationResult:
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    modal_coordinates: np.ndarray
    energy: np.ndarray
    free_dofs: np.ndarray
    constraint_matrix: np.ndarray
    nullspace_basis: np.ndarray

    @property
    def tip_displacement(self):
        return self.displacement[:, -2]


@dataclass
class FrameworkEigenvalueResult:
    eigenvalues: np.ndarray
    angular_frequencies: np.ndarray
    frequencies_hz: np.ndarray
    periods: np.ndarray
    modes_reduced: np.ndarray
    modes_full: np.ndarray
    modal_mass: np.ndarray
    modal_stiffness: np.ndarray
    constraint_matrix: np.ndarray
    nullspace_basis: np.ndarray


@dataclass
class FrameworkFreeVibrationResult:
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    modal_coordinates: np.ndarray
    energy: np.ndarray
    constraint_matrix: np.ndarray
    nullspace_basis: np.ndarray


@dataclass
class FrameworkStaticResult:
    displacement: np.ndarray
    displacement_reduced: np.ndarray
    force: np.ndarray
    reactions: np.ndarray
    strain_energy: float
    constraint_matrix: np.ndarray
    nullspace_basis: np.ndarray


@dataclass
class FrameworkNewmarkResult:
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    energy: np.ndarray
    force: np.ndarray
    constraint_matrix: np.ndarray
    nullspace_basis: np.ndarray

    def node_displacement(self, node, component=None):
        """Return one node displacement history by integer node index."""
        node_index = int(node)
        values = self.displacement[:, 3 * node_index:3 * node_index + 3]
        if component is None:
            return values
        component_index = {"ux": 0, "uy": 1, "theta": 2}.get(component, component)
        return values[:, component_index]
