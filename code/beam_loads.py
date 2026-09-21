"""
Load models for Bernoulli beam FEM (script1_bending_and_fem.pdf, Section 9, Step 4).

Three standard static cases for a left-clamped cantilever:
  1. uniform      - constant load density q(x) = q0
  2. point_load_end - concentrated force P at x = L  (contribution Q_L * e_L)
  3. end_moment   - concentrated moment M at x = L   (contribution M_L * d_L)

Global load vector entries follow the weak form  S w = q + Q_L e_L + M_L d_L  (PDF eq. 12).
"""

import numpy as np
from my_plot_fun import form

LOAD_CASES = ("uniform", "point_load_end", "end_moment")


# ---------------------------------------------------------------------------
# Load density q(x)  (for plotting / documentation)
# ---------------------------------------------------------------------------

def q_uniform(x, q0):
    """Case 1: q(x) = q0."""
    return np.full_like(np.asarray(x, dtype=float), float(q0))


def q_point_load(x, P, x_load):
    """Dirac-like visualization: P at x_load, zero elsewhere."""
    x = np.asarray(x, dtype=float)
    q = np.zeros_like(x)
    idx = np.argmin(np.abs(x - x_load))
    q[idx] = P / (x[1] - x[0]) if len(x) > 1 else P
    return q


def q_end_moment(x):
    """Case 3 has no distributed load; q(x) = 0."""
    return np.zeros_like(np.asarray(x, dtype=float))


def load_density(load_case, x, **params):
    """Return q(x) for the selected load case."""
    if load_case == "uniform":
        return q_uniform(x, params.get("q0", params.get("load_intensity", 1.0)))
    if load_case == "point_load_end":
        x_load = params.get("x_load", params.get("length", x[-1]))
        return q_point_load(x, params.get("P", 1.0), x_load)
    if load_case == "end_moment":
        return q_end_moment(x)
    raise ValueError(f"Unknown load case: {load_case}. Choose from {LOAD_CASES}.")


# ---------------------------------------------------------------------------
# FEM load vector assembly  F_k = integral q phi_k  (+ boundary terms)
# ---------------------------------------------------------------------------

def _element_load_uniform(q0, h):
    """Consistent element load for q(x) = q0 on [0, h]."""
    return q0 * h / 12.0 * np.array([6.0, h, 6.0, -h])


def uniform_load_vector(num_elements, element_length, q0):
    """Assemble global F for uniform distributed load."""
    n_dofs = 2 * (num_elements + 1)
    F = np.zeros(n_dofs)
    fe = _element_load_uniform(q0, element_length)
    for e in range(num_elements):
        dof = 2 * e
        F[dof:dof + 4] += fe
    return F


def point_load_end_vector(num_nodes, P):
    """
    Concentrated force P at the free end x = L.
    Enters the weak form as Q_L * e_L (last displacement DOF).
    """
    F = np.zeros(2 * num_nodes)
    F[2 * (num_nodes - 1)] = P
    return F


def end_moment_vector(num_nodes, M):
    """
    Concentrated moment M at the free end x = L.
    Enters the weak form as M_L * d_L (last rotation DOF).
    """
    F = np.zeros(2 * num_nodes)
    F[2 * (num_nodes - 1) + 1] = M
    return F


def point_load_vector(node_positions, P, x_load):
    """Consistent nodal loads for a point force P at position x_load."""
    n_nodes = len(node_positions)
    F = np.zeros(2 * n_nodes)

    if x_load < node_positions[0] or x_load > node_positions[-1]:
        raise ValueError(f"x_load={x_load} is outside [0, L].")

    for e in range(n_nodes - 1):
        x_left, x_right = node_positions[e], node_positions[e + 1]
        if x_left <= x_load <= x_right:
            h = x_right - x_left
            xi = (x_load - x_left) / h
            phi = form(xi)
            fe = np.array([P * phi[0], P * h * phi[1], P * phi[2], P * h * phi[3]])
            dof = 2 * e
            F[dof:dof + 4] += fe
            break

    return F


def assemble_load_vector(beam_params, load_case, **params):
    """
    Build the global load vector F for a given load case.

    Args:
        beam_params: object with num_elements, element_length, num_nodes, node_positions, length
        load_case: "uniform" | "point_load_end" | "end_moment"
        **params:
            uniform:        q0 or load_intensity
            point_load_end: P (default 1000 N)
            end_moment:     M (default 1000 N*m)
    """
    if load_case == "uniform":
        q0 = params.get("q0", params.get("load_intensity", 10.0))
        return uniform_load_vector(beam_params.num_elements, beam_params.element_length, q0)

    if load_case == "point_load_end":
        P = params.get("P", 1000.0)
        return point_load_end_vector(beam_params.num_nodes, P)

    if load_case == "end_moment":
        M = params.get("M", 1000.0)
        return end_moment_vector(beam_params.num_nodes, M)

    raise ValueError(f"Unknown load case: {load_case}. Choose from {LOAD_CASES}.")


def load_case_label(load_case, **params):
    """Human-readable description for plots."""
    if load_case == "uniform":
        q0 = params.get("q0", params.get("load_intensity", 10.0))
        return f"Uniform load q0 = {q0} N/m"
    if load_case == "point_load_end":
        return f"Point load P = {params.get('P', 1000.0)} N at x = L"
    if load_case == "end_moment":
        return f"End moment M = {params.get('M', 1000.0)} N*m at x = L"
    return load_case


# ---------------------------------------------------------------------------
# Analytic cantilever solutions (for comparison, PDF Step 4)
# ---------------------------------------------------------------------------

def analytic_cantilever_tip(E, I, L, load_case, **params):
    """Tip deflection w(L) for a cantilever with constant EI."""
    if load_case == "uniform":
        q0 = params.get("q0", params.get("load_intensity", 10.0))
        return q0 * L ** 4 / (8 * E * I)
    if load_case == "point_load_end":
        P = params.get("P", 1000.0)
        return P * L ** 3 / (3 * E * I)
    if load_case == "end_moment":
        M = params.get("M", 1000.0)
        return M * L ** 2 / (2 * E * I)
    raise ValueError(f"No analytic solution for load case: {load_case}")
