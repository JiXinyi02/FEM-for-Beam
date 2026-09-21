import numpy as np
import time
from matplotlib import pyplot as plt
from static_beam import BeamAnalysis
from beam_loads import (
    LOAD_CASES,
    assemble_load_vector,
    analytic_cantilever_tip,
    load_case_label,
)
from my_plot_fun import form, plot_piecewise_polynomial

"""
In static_beam.py, boundary conditions are applied by deleting rows/columns (direct elimination).
This file solves the same FEM system using the augmented-matrix method.
The configuration is the same as in static_beam.py.
"""

class BeamFEM:
    """
    DOF order:
        u = [w0, theta0, w1, theta1, ..., wn, thetan]
    """
    def __init__(
            self,
            length=10.0,
            num_elements=25,
            youngs_modulus=200e9,
            moment_inertia=1.0,
            density=7800.0,
            load_intensity=10.0,
            load_case="uniform",
            **load_params
    ):
        # Geometry
        self.length = length                    # Total beam length (m)
        self.num_elements = num_elements                # Number of finite elements
        self.num_nodes = self.num_elements + 1
        self.element_length = self.length / self.num_elements
        self.num_dofs = 2 * self.num_nodes

        # Material properties
        self.youngs_modulus = youngs_modulus           # Young's modulus (Pa)
        self.moment_inertia = moment_inertia             # Moment of inertia (m^4)
        self.density = density                 # Material density (kg/m^3)
        
        # Derived properties
        self.node_positions = np.linspace(0, self.length, self.num_nodes)
        self.S = None
        self.F = None
        self.u = None
        self.mu = None
        self.load_intensity = load_intensity
        self.load_case = load_case
        self.load_params = load_params
        if self.load_case == "uniform" and "q0" not in self.load_params:
            self.load_params["q0"] = self.load_intensity
        if self.load_case == "uniform":
            self.load_intensity = self.load_params["q0"]

    def local_stiffness(self):
        h = self.element_length
        factor = self.youngs_modulus * self.moment_inertia / h**3

        return factor * np.array([
            [12,     6*h,   -12,     6*h],
            [6*h,  4*h**2,  -6*h,  2*h**2],
            [-12,   -6*h,    12,    -6*h],
            [6*h,  2*h**2,  -6*h,  4*h**2]
        ])

    def local_load(self):
        h = self.element_length
        q0 = self.load_intensity

        return q0 * h / 12.0 * np.array([6.0, h, 6.0, -h])


    def assemble_global_matrices(self):
        S = np.zeros((self.num_dofs, self.num_dofs))
        F = np.zeros(self.num_dofs)

        ke = self.local_stiffness()
        fe = self.local_load()

        for i in range(self.num_elements):
            dof = 2 * i
            idx = slice(dof, dof + 4)

            S[idx, idx] += ke
            F[idx] += fe
        if self.load_case != "uniform":
            F = assemble_load_vector(self, self.load_case, **self.load_params)
        self.S = S
        self.F = F

        return S, F

    # Boundary conditions:
    def build_constraints(self, bc_type="cantilever"):
        constraints = []

        if bc_type == "cantilever":
            # w(0) = 0, theta(0) = 0
            constraints.append((0, 0.0))
            constraints.append((1, 0.0))
        elif bc_type == "simply_supported":
            # w(0) = 0, w(L) = 0
            constraints.append((0, 0.0))
            constraints.append((self.num_dofs - 1, 0.0))
        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

        r = len(constraints)
        C = np.zeros((self.num_dofs, r))
        a = np.zeros(r)

        for i, (dof, value) in enumerate(constraints):
            C[dof, i] = 1.0
            a[i] = value

        return C, a


    # Solve the extended matrix:
    def solve_static_extended(self, bc_type="cantilever"):

        if self.S is None or self.F is None:
            self.assemble_global_matrices()
        
        C, a = self.build_constraints(bc_type)
        r = len(a)

        S_ext = np.block([[self.S, C], [C.T, np.zeros((r, r))]])
        rhs_ext = np.concatenate([self.F, a])

        sol = np.linalg.solve(S_ext, rhs_ext)
        u_free = sol[:self.num_dofs]
        u_constrained = sol[self.num_dofs:]

        self.u = u_free
        self.mu = u_constrained

        return u_free, u_constrained
    
    # Compare the results with the static_beam.py:

    def compare_with_analytic_solution(self):
        if self.u is None:
            self.solve_static_extended()

        w_tip_extended = self.u[-2]
        w_tip_exact = analytic_cantilever_tip(
            self.youngs_modulus,
            self.moment_inertia,
            self.length,
            self.load_case,
            **self.load_params,
        )
        abs_error = abs(w_tip_extended - w_tip_exact)
        rel_error = abs_error / abs(w_tip_exact)

        return w_tip_extended, w_tip_exact, abs_error, rel_error

    def compare_with_static_beam(self):
        if self.u is None:
            self.solve_static_extended()

        analysis = BeamAnalysis(
            beam_type="cantilever",
            load_case=self.load_case,
            **self.load_params
        )
        static_displacement = analysis.solve()

        w_tip_extended = self.u[-2]
        w_tip_static = static_displacement[-1]
        abs_difference = abs(w_tip_extended - w_tip_static)
        rel_difference = abs_difference / abs(w_tip_static)

        return w_tip_extended, w_tip_static, abs_difference, rel_difference

    def plot_piecewise_polynomial(self):
        if self.u is None:
            self.solve_static_extended()

        plt.figure(figsize=(10, 6))
        plot_piecewise_polynomial(self.u, self.node_positions)
        plt.plot(self.node_positions, self.u[::2], "ro", markersize=4, label="Nodes")
        plt.xlabel("Position along beam (m)")
        plt.ylabel("Displacement (m)")
        plt.title(f"Left-clamped beam - {load_case_label(self.load_case, **self.load_params)}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
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



