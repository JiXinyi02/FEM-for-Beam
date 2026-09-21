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
