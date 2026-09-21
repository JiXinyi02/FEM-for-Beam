"""Beam and planar-frame element matrices and global beam assembly."""

import numpy as np


class FiniteElementMatrices:
    """Handles computation of local and global finite element matrices."""

    def __init__(self, beam_params):
        self.beam = beam_params
        self.dofs_per_node = 2  # Displacement and rotation per node
        self.total_dofs = self.dofs_per_node * self.beam.num_nodes

    def compute_local_matrices(self):
        """
        Compute local stiffness and mass matrices for beam elements.

        Stiffness uses the standard Euler-Bernoulli element matrix (h = element length).
        Mass uses the closed-form consistent Hermite element matrix.
        """
        h_val = self.beam.element_length
        E_val = self.beam.youngs_modulus
        I_val = self.beam.moment_inertia
        rho_val = self.beam.density

        k = E_val * I_val / h_val ** 3
        local_stiffness = k * np.array([
            [12, 6 * h_val, -12, 6 * h_val],
            [6 * h_val, 4 * h_val ** 2, -6 * h_val, 2 * h_val ** 2],
            [-12, -6 * h_val, 12, -6 * h_val],
            [6 * h_val, 2 * h_val ** 2, -6 * h_val, 4 * h_val ** 2],
        ])

        # Cosed-form consistent mass matrix for Hermite beam element
        local_mass = rho_val * h_val / 420.0 * np.array([
            [156.0, 22.0 * h_val, 54.0, -13.0 * h_val],
            [22.0 * h_val, 4.0 * h_val ** 2, 13.0 * h_val, -3.0 * h_val ** 2],
            [54.0, 13.0 * h_val, 156.0, -22.0 * h_val],
            [-13.0 * h_val, -3.0 * h_val ** 2, -22.0 * h_val, 4.0 * h_val ** 2],
        ])
        return local_stiffness, local_mass

    def assemble_global_matrices(self):
        """
        Assemble global stiffness and mass matrices from local matrices.

        Returns:
            tuple: (global_stiffness_matrix, global_mass_matrix)
        """
        local_stiffness, local_mass = self.compute_local_matrices()

        # Initialize global matrices
        global_stiffness = np.zeros((self.total_dofs, self.total_dofs))
        global_mass = np.zeros((self.total_dofs, self.total_dofs))

        # Assemble element contributions
        for element in self.beam.elements:
            start_node = element[0]
            start_dof = self.dofs_per_node * start_node
            end_dof = start_dof + 4

            # Add element matrices to global matrices
            global_stiffness[start_dof:end_dof, start_dof:end_dof] += local_stiffness
            global_mass[start_dof:end_dof, start_dof:end_dof] += local_mass

        return global_stiffness, global_mass


def frame_element_transformation(cosine, sine):
    """Return T such that local_dofs = T @ global_dofs."""
    # Local x is along the element axis; local y is perpendicular to it.
    # w.r.t. the local frame, each beam has 6 DOFs:
    #   [u_x1, u_y1, theta_1, u_x2, u_y2, theta_2].
    return np.array([
        [cosine, sine, 0.0, 0.0, 0.0, 0.0],
        [-sine, cosine, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, cosine, sine, 0.0],
        [0.0, 0.0, 0.0, -sine, cosine, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])


def local_frame_stiffness(length, youngs_modulus, area, moment_inertia):
    """Local 2D frame element stiffness matrix."""
    # Axial terms use EA/L; bending terms use the Euler-Bernoulli beam matrix.
    ea_l = youngs_modulus * area / length
    ei = youngs_modulus * moment_inertia
    l2 = length ** 2
    l3 = length ** 3

    return np.array([
        [ea_l, 0.0, 0.0, -ea_l, 0.0, 0.0],
        [0.0, 12.0 * ei / l3, 6.0 * ei / l2, 0.0, -12.0 * ei / l3, 6.0 * ei / l2],
        [0.0, 6.0 * ei / l2, 4.0 * ei / length, 0.0, -6.0 * ei / l2, 2.0 * ei / length],
        [-ea_l, 0.0, 0.0, ea_l, 0.0, 0.0],
        [0.0, -12.0 * ei / l3, -6.0 * ei / l2, 0.0, 12.0 * ei / l3, -6.0 * ei / l2],
        [0.0, 6.0 * ei / l2, 2.0 * ei / length, 0.0, -6.0 * ei / l2, 4.0 * ei / length],
    ])


def local_frame_mass(length, density, area):
    """
    Local consistent mass matrix.

    Axial motion uses the bar consistent mass, and transverse/rotational motion
    uses the Euler-Bernoulli beam consistent mass.
    """
    mass_per_length = density * area
    return mass_per_length * length / 420.0 * np.array([
        [140.0, 0.0, 0.0, 70.0, 0.0, 0.0],
        [0.0, 156.0, 22.0 * length, 0.0, 54.0, -13.0 * length],
        [0.0, 22.0 * length, 4.0 * length ** 2, 0.0, 13.0 * length, -3.0 * length ** 2],
        [70.0, 0.0, 0.0, 140.0, 0.0, 0.0],
        [0.0, 54.0, 13.0 * length, 0.0, 156.0, -22.0 * length],
        [0.0, -13.0 * length, -3.0 * length ** 2, 0.0, -22.0 * length, 4.0 * length ** 2],
    ])
