"""Hermite shape functions and beam displacement interpolation."""

import numpy as np


def form(x) -> np.ndarray:
    """Cubic Hermite beam shape functions on the reference interval [0, 1]."""
    form_1 = 1 - 3*x**2 + 2*x**3
    form_2 = x * ((x - 1)**2)
    form_3 = 3*x**2 - 2*x**3
    form_4 = x**2 * (x - 1)
    return np.array([form_1, form_2, form_3, form_4])


def sample_beam_shape(dof_vector, x_nodes, points_per_element=40, displacement_scale=1.0):
    """Evaluate the Hermite beam displacement curve from a full DOF vector."""
    x_all = []
    w_all = []

    for i in range(len(x_nodes) - 1):
        x_left = x_nodes[i]
        x_right = x_nodes[i + 1]
        h = x_right - x_left

        x_local = np.linspace(x_left, x_right, points_per_element)
        xi = (x_local - x_left) / h
        phi = form(xi)

        u1 = dof_vector[2 * i]
        u2 = dof_vector[2 * i + 1]
        u3 = dof_vector[2 * i + 2]
        u4 = dof_vector[2 * i + 3]

        w_local = u1 * phi[0] + h * u2 * phi[1] + u3 * phi[2] + h * u4 * phi[3]
        x_all.extend(x_local)
        w_all.extend(displacement_scale * w_local)

    return np.asarray(x_all), np.asarray(w_all)
