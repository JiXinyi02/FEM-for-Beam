import numpy as np
import matplotlib.pyplot as plt
from my_plot_fun import form, plot_piecewise_polynomial
from beam_loads import (
    LOAD_CASES,
    assemble_load_vector,
    analytic_cantilever_tip,
    load_case_label,
    load_density,
)

"""
=============================================================================
STATIC BEAM ANALYSIS USING FINITE ELEMENT METHOD
=============================================================================
This module performs static analysis of beams using the finite element method.
Supports both cantilever and simply supported beams under uniform loading.
"""

# =============================================================================
# BEAM PARAMETERS AND MATERIAL PROPERTIES
# =============================================================================

class BeamParameters:
    """Container for beam geometry and material properties."""
    
    def __init__(self, length=10.0, num_elements=25, youngs_modulus=200e9, moment_inertia=1.0, density=7800.0):
        # Geometry
        self.length = length                    # Total beam length (m)
        self.num_elements = num_elements                # Number of finite elements
        self.num_nodes = self.num_elements + 1
        self.element_length = self.length / self.num_elements
        
        # Material properties
        self.youngs_modulus = youngs_modulus           # Young's modulus (Pa)
        self.moment_inertia = moment_inertia             # Moment of inertia (m^4)
        self.density = density                 # Material density (kg/m^3)
        
        # Derived properties
        self.node_positions = np.linspace(0, self.length, self.num_nodes)
        self.elements = self._create_element_connectivity()
    
    def _create_element_connectivity(self):
        """Create element connectivity matrix."""
        elements = []
        for i in range(self.num_elements):
            elements.append([i, i + 1])  # [start_node, end_node]
        return elements

# =============================================================================
# FINITE ELEMENT MATRIX COMPUTATION
# =============================================================================

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

# =============================================================================
# LOAD APPLICATION
# =============================================================================

class LoadApplication:
    """Handles application of loads to the beam structure."""

    def __init__(self, beam_params):
        self.beam = beam_params
        self.total_dofs = 2 * beam_params.num_nodes

    def apply_load(self, load_case="uniform", **params):
        """
        Assemble the global load vector for one of the three standard cases.

        Args:
            load_case: "uniform", "point_load_end", or "end_moment"
            **params: q0/P/M depending on the case
        """
        return assemble_load_vector(self.beam, load_case, **params)

    def apply_uniform_load(self, load_intensity):
        """Backward-compatible wrapper for uniform load."""
        return self.apply_load("uniform", q0=load_intensity)

# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

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

# =============================================================================
# SOLVER AND ANALYSIS
# =============================================================================

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
        if self.full_dof is None:
            print("No results to plot. Run solve() first.")
            return
        
        plt.figure(figsize=(10, 6))
        plot_piecewise_polynomial(self.full_dof, self.beam_params.node_positions)
        # plt.gca().get_lines()[0].set_label('Displacement')
        # plt.plot(
        #     self.beam_params.node_positions,
        #     self.full_dof[::2],
        #     'ro',
        #     markersize=6,
        #     label='Nodes',
        # )
        
        plt.xlabel('Position along beam (m)', fontsize=12)
        plt.ylabel('Displacement (m)', fontsize=12)
        plt.title(
            f'{self.beam_type.replace("_", " ").title()} Beam - '
            f'{load_case_label(self.load_case, **self.load_params)}',
            fontsize=14,
        )
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
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
