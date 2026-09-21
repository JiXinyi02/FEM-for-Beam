import numpy as np


def fixed_dof_constraint_matrix(total_dofs, fixed_dofs):
    """Build C for homogeneous constraints u[dof] = 0."""
    fixed_dofs = np.asarray(fixed_dofs, dtype=int)
    constraints = np.zeros((len(fixed_dofs), total_dofs))
    for row_index, dof in enumerate(fixed_dofs):
        constraints[row_index, dof] = 1.0
    return constraints


def nullspace_basis(constraint_matrix, tolerance=None):
    """
    Return N such that C @ N = 0.

    Columns of N form an orthonormal basis of the nullspace.  Writing u = N q
    guarantees that every represented displacement satisfies C u = 0.
    """
    constraint_matrix = np.asarray(constraint_matrix, dtype=float)
    total_dofs = constraint_matrix.shape[1]
    if constraint_matrix.shape[0] == 0:
        return np.eye(total_dofs)

    _, singular_values, vh = np.linalg.svd(constraint_matrix, full_matrices=True)
    if tolerance is None:
        scale = singular_values[0] if len(singular_values) else 1.0
        tolerance = scale * max(constraint_matrix.shape) * np.finfo(float).eps

    rank = np.sum(singular_values > tolerance)
    return vh[rank:].T


class BoundaryConditions:
    """Handles different types of boundary conditions."""

    @staticmethod
    def apply_cantilever_bc(stiffness_matrix, force_vector):
        """
        Apply cantilever boundary conditions (fixed at one end).

        Args:
            stiffness_matrix (np.ndarray): Global stiffness matrix
            force_vector (np.ndarray): Force vector

        Returns:
            tuple: (modified_stiffness, modified_force)
        """
        # Remove first two DOFs (displacement and rotation at fixed end)
        fixed_dofs = [0, 1]

        K_modified = np.delete(stiffness_matrix, fixed_dofs, axis=0)
        K_modified = np.delete(K_modified, fixed_dofs, axis=1)
        F_modified = np.delete(force_vector, fixed_dofs)

        return K_modified, F_modified

    @staticmethod
    def apply_simply_supported_bc(stiffness_matrix, force_vector, num_nodes):
        """
        Apply simply supported boundary conditions.

        Args:
            stiffness_matrix (np.ndarray): Global stiffness matrix
            force_vector (np.ndarray): Force vector
            num_nodes (int): Number of nodes

        Returns:
            tuple: (modified_stiffness, modified_force)
        """
        # Remove displacement DOFs at both ends (nodes 0 and n)
        fixed_dofs = [0, 2 * num_nodes - 2]  # First and last displacement DOFs

        K_modified = np.delete(stiffness_matrix, fixed_dofs, axis=0)
        K_modified = np.delete(K_modified, fixed_dofs, axis=1)
        F_modified = np.delete(force_vector, fixed_dofs)

        return K_modified, F_modified
