import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from my_plot_fun import form

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
        self.youngs_modulus = youngs_modulus           # Young's modulus (Pa)
        self.moment_inertia = moment_inertia             # Moment of inertia (m^4)
        self.density = density                 # Material density (kg/m^3)
        
        # Create element connectivity and node positions
        self.elements = self._create_element_connectivity()
        self.node_positions = np.linspace(0, self.length, self.num_nodes)
    
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
        
        Returns:
            tuple: (local_stiffness_matrix, local_mass_matrix)
        """
        # Define symbolic variable for integration
        x = sp.symbols('x')
        
        # Get shape functions from external module
        shape_functions = form(x)
        
        # Initialize matrices
        stiffness_matrix = sp.Matrix.zeros(4, 4)
        mass_matrix = sp.Matrix.zeros(4, 4)
        
        # Get numerical values
        h_val = self.beam.element_length
        E_val = self.beam.youngs_modulus
        I_val = self.beam.moment_inertia
        rho_val = self.beam.density
        
        # Compute matrix elements
        for i in range(4):
            for j in range(i, 4):
                # Stiffness matrix: K_ij = ∫ EI * d²N_i/dx² * d²N_j/dx² dx
                d2_shape_i = sp.diff(shape_functions[i], x, 2)
                d2_shape_j = sp.diff(shape_functions[j], x, 2)
                
                # Compute integral symbolically, then substitute values
                integral_stiffness = sp.integrate(d2_shape_i * d2_shape_j, (x, 0, h_val))
                stiffness_matrix[i, j] = E_val * I_val * float(integral_stiffness)
                stiffness_matrix[j, i] = stiffness_matrix[i, j]  # Symmetry
                
                # Mass matrix: M_ij = ∫ ρ * N_i * N_j dx
                integral_mass = sp.integrate(shape_functions[i] * shape_functions[j], (x, 0, h_val))
                mass_matrix[i, j] = rho_val * float(integral_mass)
                mass_matrix[j, i] = mass_matrix[i, j]  # Symmetry
        
        # Convert to NumPy arrays
        local_stiffness = np.array(stiffness_matrix).astype(np.float64)
        local_mass = np.array(mass_matrix).astype(np.float64)
        
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
    
    def apply_uniform_load(self, load_intensity):
        """
        Apply uniform distributed load to the beam.
        
        Args:
            load_intensity (float): Load intensity (N/m)
            
        Returns:
            np.ndarray: Force vector
        """
        force_vector = np.zeros(self.total_dofs)
        
        # Distribute load to nodes (simplified approach)
        force_per_node = load_intensity * self.beam.element_length / 2
        
        # Apply forces to displacement DOFs only (not rotation DOFs)
        for i in range(self.beam.num_elements):
            start_node_dof = 2 * i      # Displacement DOF of start node
            end_node_dof = 2 * (i + 1)  # Displacement DOF of end node
            
            force_vector[start_node_dof] += force_per_node
            force_vector[end_node_dof] += force_per_node
        
        return force_vector

# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

class BoundaryConditions:
    """Handles different types of boundary conditions by extended matrix method."""
    
    @staticmethod
    def apply_cantilever_bc(stiffness_matrix, force_vector):
        """
        Apply cantilever boundary conditions (fixed at one end).
        
        Args:
            stiffness_matrix (np.ndarray): Global stiffness matrix S (n×n)
            force_vector (np.ndarray): Force vector f (n×1)
            
        Returns:
            tuple: (extended_matrix, extended_force, num_constraints)
                - extended_matrix: extended stiffness matrix S_e ((n+m)×(n+m))
                - extended_force: extended force vector ((n+m)×1)
                - num_constraints: number of constraints m
        """
        n = len(force_vector)  # 原始DOF数量
        
        # 约束的自由度 (displacement and rotation at node 0)
        constrained_dofs = [0, 1]  # DOF indices: [w0, θ0]
        m = len(constrained_dofs)  # 约束数量
        
        # 构造约束矩阵 C (n×m)
        # C[i,j] = 1 if DOF i is constrained by constraint j
        C = np.zeros((n, m))
        for j, dof in enumerate(constrained_dofs):
            C[dof, j] = 1.0
        
        # 构造扩展矩阵 S_e = [[S, C], [C^T, 0]]
        # 维度: (n+m) × (n+m)
        S_extended = np.zeros((n + m, n + m))
        S_extended[:n, :n] = stiffness_matrix      # 左上：S
        S_extended[:n, n:] = C                      # 右上：C
        S_extended[n:, :n] = C.T                    # 左下：C^T
        # 右下角已经是0
        
        # 构造扩展力向量 f_e = [f, a]
        # 维度: (n+m) × 1
        f_extended = np.zeros(n + m)
        f_extended[:n] = force_vector               # 上部：f
        f_extended[n:] = 0.0                        # 下部：a (约束值，这里都是0)
        
        return S_extended, f_extended, m
    
    @staticmethod
    def apply_simply_supported_bc(stiffness_matrix, force_vector, num_nodes):
        """
        Apply simply supported boundary conditions using extended matrix method.
        The features of simply supported boundary conditions are:
        free rotation at both ends, but displacement is constrained.
        
        constraints:
        - w[0] = 0      (displacement constraint at left end)
        - w[n-1] = 0    (displacement constraint at right end)
        
        Args:
            stiffness_matrix (np.ndarray): Global stiffness matrix S
            force_vector (np.ndarray): Force vector f
            num_nodes (int): Number of nodes
            
        Returns:
            tuple: (extended_matrix, extended_force, num_constraints)
        """
        n = len(force_vector)  # 原始DOF数量
        
        # 约束的自由度 (displacement at both ends)
        # DOF顺序：[w0, θ0, w1, θ1, ..., w_{n-1}, θ_{n-1}]
        constrained_dofs = [0, 2 * (num_nodes - 1)]  # [w0, w_{last}]
        m = len(constrained_dofs)  # 约束数量
        
        # 构造约束矩阵 C (n×m)
        C = np.zeros((n, m))
        for j, dof in enumerate(constrained_dofs):
            C[dof, j] = 1.0
        
        # 构造扩展矩阵 S_e = [[S, C], [C^T, 0]]
        S_extended = np.zeros((n + m, n + m))
        S_extended[:n, :n] = stiffness_matrix
        S_extended[:n, n:] = C
        S_extended[n:, :n] = C.T
        
        # 构造扩展力向量 f_e = [f, a]
        f_extended = np.zeros(n + m)
        f_extended[:n] = force_vector
        f_extended[n:] = 0.0  # 约束值都是0
        
        return S_extended, f_extended, m

# =============================================================================
# SOLVER AND ANALYSIS
# =============================================================================

class BeamAnalysis:
    """Main class for beam analysis."""
    
    def __init__(self, beam_type="cantilever", load_intensity=10.0):
        """
        Initialize beam analysis.
        
        Args:
            beam_type (str): Type of beam ("cantilever" or "simply_supported")
            load_intensity (float): Uniform load intensity (N/m)
        """
        self.beam_type = beam_type
        self.load_intensity = load_intensity
        
        # Initialize components
        self.beam_params = BeamParameters()
        self.fe_matrices = FiniteElementMatrices(self.beam_params)
        self.load_handler = LoadApplication(self.beam_params)
        
        # Analysis results
        self.displacement = None
        self.rotation = None  # 添加转角存储
        self.global_stiffness = None
        self.global_mass = None
    
    def solve(self):
        """Perform the complete beam analysis."""
        print(f"Analyzing {self.beam_type} beam with load intensity {self.load_intensity} N/m")
        
        # 1. Compute global matrices
        print("Step 1: Assembling global stiffness matrix...")
        self.global_stiffness, self.global_mass = self.fe_matrices.assemble_global_matrices()
        print(f"  - Global stiffness matrix shape: {self.global_stiffness.shape}\n")
        
        # 2. Apply loads
        print("Step 2: Applying uniform distributed load...")
        force_vector = self.load_handler.apply_uniform_load(self.load_intensity)
        print(f"  - Force vector shape: {force_vector.shape}\n")
        
        # 3. Apply boundary conditions and solve using extended matrix
        print("Step 3: Constructing extended matrix and solving...")
        self.displacement = self._solve_system(force_vector)
        
        print("\n✓ Analysis completed successfully using extended matrix method!")
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
        S_extended, f_extended, m = BoundaryConditions.apply_cantilever_bc(
            self.global_stiffness, force_vector
        )
        
        n = len(force_vector)  # 原始DOF数
        
        # 求解扩展系统 S_extended · [x, μ]^T = f_extended
        solution = np.linalg.solve(S_extended, f_extended)
        
        # 提取位移和拉格朗日乘子
        displacement_all_dofs = solution[:n]  # 前n个是位移和转角
        lagrange_multipliers = solution[n:]   # 后m个是拉格朗日乘子（约束反力）
        
        # 存储拉格朗日乘子用于后续分析
        self.constraint_forces = lagrange_multipliers
        
        # 提取位移和转角DOF
        # DOF顺序：[w0, θ0, w1, θ1, w2, θ2, ...]
        full_displacement = displacement_all_dofs[::2]  # 偶数索引：w[2i]
        full_rotation = displacement_all_dofs[1::2]     # 奇数索引：θ[2i+1]
        
        # 存储转角数据
        self.rotation = full_rotation
        
        return full_displacement
    
    def _solve_simply_supported(self, force_vector):
        
        S_extended, f_extended, m = BoundaryConditions.apply_simply_supported_bc(
            self.global_stiffness, force_vector, self.beam_params.num_nodes
        )
        
        n = len(force_vector)
        
        # 求解扩展系统
        solution = np.linalg.solve(S_extended, f_extended)
        
        # 提取位移和拉格朗日乘子
        displacement_all_dofs = solution[:n]
        lagrange_multipliers = solution[n:]
        
        # 存储拉格朗日乘子
        self.constraint_forces = lagrange_multipliers
        
        # 提取位移和转角DOF
        # DOF顺序：[w0, θ0, w1, θ1, w2, θ2, ...]
        full_displacement = displacement_all_dofs[::2]  # 偶数索引：w[2i]
        full_rotation = displacement_all_dofs[1::2]     # 奇数索引：θ[2i+1]
        
        # 存储转角数据
        self.rotation = full_rotation
        
        print(f"  - Constraint forces (Lagrange multipliers): {lagrange_multipliers}")
        
        return full_displacement
    
    def show_extended_matrix_structure(self, max_display=8):
        """
        Display the structure of the extended matrix (for educational purposes).
        显示扩展矩阵的结构（教学用途）
        """
        if self.global_stiffness is None:
            print("请先运行 solve() 方法")
            return
        
        # 创建载荷向量
        force_vector = self.load_handler.apply_uniform_load(self.load_intensity)
        
        # 应用边界条件得到扩展矩阵
        if self.beam_type == "cantilever":
            S_extended, f_extended, m = BoundaryConditions.apply_cantilever_bc(
                self.global_stiffness, force_vector
            )
            constrained_dofs = [0, 1]
        elif self.beam_type == "simply_supported":
            S_extended, f_extended, m = BoundaryConditions.apply_simply_supported_bc(
                self.global_stiffness, force_vector, self.beam_params.num_nodes
            )
            constrained_dofs = [0, 2 * (self.beam_params.num_nodes - 1)]
        else:
            return
        
        n = len(force_vector)  # 原始DOF数量
        
        print("\n" + "="*80)
        print("Extended Matrix Structure (Lagrange Multiplier Method)")
        print("="*80)
        print(f"Original system size: {n}×{n}")
        print(f"Number of constraints: {m}")
        print(f"Extended system size: {n+m}×{n+m}")
        print(f"Constrained DOFs: {constrained_dofs}")
        
        # 显示扩展矩阵的一部分
        n_display = min(max_display, n + m)
        print(f"\nExtended stiffness matrix S_extended (first {n_display}×{n_display}):")
        print(np.round(S_extended[:n_display, :n_display], 2))
        
        print(f"\nExtended force vector f_extended (first {n_display} elements):")
        print(np.round(f_extended[:n_display], 4))
        
        # 显示约束矩阵C
        C = S_extended[:n, n:]
        print(f"\nConstraint matrix C ({n}×{m}):")
        print(C[:min(max_display, n), :])
        
        print("\n" + "="*80 + "\n")
    
    def plot_results(self):
        """Plot the displacement results."""
        if self.displacement is None:
            print("No results to plot. Run solve() first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.beam_params.node_positions, self.displacement, 
                'b-o', linewidth=2, markersize=6, label='Displacement')
        
        plt.xlabel('Position along beam (m)', fontsize=12)
        plt.ylabel('Displacement (m)', fontsize=12)
        plt.title(f'{self.beam_type.replace("_", " ").title()} Beam - '
                 f'Load = {self.load_intensity} N/m', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration
    BEAM_TYPE = "simply_supported"  # Options: "cantilever" or "simply_supported"
    LOAD_INTENSITY = 10.0     # Load intensity in N/m
    
    # Perform analysis
    analysis = BeamAnalysis(beam_type=BEAM_TYPE, load_intensity=LOAD_INTENSITY)
    displacement_results = analysis.solve()
    
    # Display extended matrix structure (educational)
    analysis.show_extended_matrix_structure(max_display=8)
    
    # Display results
    print("\n" + "="*80)
    print("结果 (Results)")
    print("="*80)
    
    # Displacement results
    print("\n位移 (Displacement):")
    print(f"  Maximum displacement: {np.max(np.abs(displacement_results)):.6e} m")
    print(f"  Displacement at end: {displacement_results[-1]:.6e} m")
    print(f"  Displacement at start: {displacement_results[0]:.6e} m")
    
    # Rotation results (转角自由度 θ[2i+1])
    if hasattr(analysis, 'rotation') and analysis.rotation is not None:
        print("\n转角 (Rotation θ[2i+1]):")
        print(f"  Maximum rotation: {np.max(np.abs(analysis.rotation)):.6e} rad")
        print(f"  Rotation at end: {analysis.rotation[-1]:.6e} rad")
        print(f"  Rotation at start: {analysis.rotation[0]:.6e} rad")
    
    # Display constraint forces (Lagrange multipliers)
    if hasattr(analysis, 'constraint_forces'):
        print(f"\n约束反力 (Constraint Forces / Lagrange Multipliers):")
        for i, force in enumerate(analysis.constraint_forces):
            print(f"  λ[{i}] = {force:.6e} N")
    
    print("="*80 + "\n")
    
    # Plot results
    analysis.plot_results()