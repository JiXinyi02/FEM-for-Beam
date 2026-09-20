import numpy as np
from static_beam import BeamParameters, FiniteElementMatrices

"""
In the file static_beam.py, we have already implemented the global stiffness matrix and mass matrix.
Now we will take the inertial force into consideration, and the equation in the form of matrix can be extended.
"""

class ModalAnalysis:

    """对给定的 (M, S) 做模态分析：求特征值、特征向量、整理结果。"""

    def __init__(self, M, S):
        self.M = M
        self.S = S
        self.eigvals = None      # ω_j^2
        self.eigvecs = None      # φ_j

    def solve_eigenproblem(self, num_modes: int):
        """
        解广义特征值问题 S φ = λ M φ
        - 只取最小的 num_modes 个 λ（低阶模态）
        - 保存 λ 和 φ
        """
        pass

    def get_natural_frequencies(self):
        """
        从 λ_j 计算频率 ω_j = sqrt(λ_j)，返回数组。
        """
        pass

    def mass_normalize_modes(self):
        """
        对每个模态做 M-正交归一化，使 φ_i^T M φ_j = δ_ij。
        方便后续用 φ^T M w0 直接做投影。
        """
        pass

if __name__ == "__main__":
    
    test_beam = BeamParameters()
    fe_matrices = FiniteElementMatrices(test_beam)
    global_stiffness, global_mass = fe_matrices.assemble_global_matrices()

    