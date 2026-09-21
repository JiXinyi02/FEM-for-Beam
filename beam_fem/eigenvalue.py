"""
Free-vibration analysis with the eigenvalue method.

For an undamped beam with no external load, the finite element equations are

    M u''(t) + K u(t) = 0.

Using the modal ansatz u(t) = phi exp(i omega t) gives the generalized
eigenvalue problem

    K phi = omega^2 M phi.

The numerical modes below use the same Euler-Bernoulli Hermite beam element
matrices as static_beam.py/newmark.py.  Analytic comparisons follow the
standard Bernoulli beam results:

    omega_n = beta_n^2 * sqrt(EI / rho)

where beta_n L = n pi for a simply supported beam, and beta_n L solves
cosh(beta_n L) cos(beta_n L) + 1 = 0 for a cantilever.  For the
cantilever roots we use the equivalent condition

    cos(x) = -1 / cosh(x)

with the starting approximation x_j = (j - 0.5) pi.
Here, the solution is refined with Newton's method.
"""

import numpy as np

from .constraints import fixed_dof_constraint_matrix, nullspace_basis
from .elements import FiniteElementMatrices
from .parameters import BeamParameters
from .plotting import eigenvalue as _plotting
from .plotting.beam import aligned_analytic_mode, normalized_mode
from .results import EigenvalueResult, FreeVibrationResult


def cantilever_characteristic_initial_values(num_modes):
    """
    Return the vector x_j = (j - 0.5) pi from the lecture approximation.

    This avoids explicitly listing the cantilever roots.  The approximation is
    already very accurate from moderate mode numbers because 1 / cosh(x) is
    close to zero.
    """
    if num_modes <= 0:
        return np.array([], dtype=float)

    mode_numbers = np.arange(1, num_modes + 1, dtype=float)
    return (mode_numbers - 0.5) * np.pi


def cantilever_characteristic_residual(x):
    """
    Evaluate cos(x) + 1 / cosh(x).

    The positive zeros of this function are the cantilever characteristic
    values alpha_j = beta_j L.
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore"):
        sech = 1.0 / np.cosh(x)
    return np.cos(x) + sech


def cantilever_characteristic_residual_derivative(x):
    """Derivative of cos(x) + 1 / cosh(x)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore"):
        sech = 1.0 / np.cosh(x)
    return -np.sin(x) - sech * np.tanh(x)


def cantilever_characteristic_values(num_modes, newton_steps=8, tolerance=1e-14):
    """
    Return the first num_modes values alpha_n = beta_n L.

    The roots are computed as one vector.  We start from the lecture
    approximation x_j = (j - 0.5) pi and refine it with Newton's method applied
    to cos(x) + 1 / cosh(x) = 0.  Set newton_steps=0 to use only the
    approximation.
    """
    values = cantilever_characteristic_initial_values(num_modes)
    if len(values) == 0:
        return values

    for _ in range(newton_steps):
        residual = cantilever_characteristic_residual(values)
        derivative = cantilever_characteristic_residual_derivative(values)
        correction = residual / derivative
        values = values - correction
        if np.max(np.abs(correction)) < tolerance:
            break
    return values


def simply_supported_characteristic_values(num_modes):
    """Return the first num_modes values alpha_n = beta_n L."""
    return np.arange(1, num_modes + 1, dtype=float) * np.pi


class EigenvalueBeamAnalysis:
    """Eigenvalue/modal analysis for the beam FEM model used in this project."""

    def __init__(
            self,
            beam_type="cantilever",
            num_modes=6,
            length=10.0,
            num_elements=25,
            youngs_modulus=200e9,
            moment_inertia=1.0,
            density=7800.0,
            constraint_matrix=None,
    ):
        self.beam_type = beam_type
        self.num_modes = num_modes
        self.user_constraint_matrix = constraint_matrix

        self.beam_params = BeamParameters(
            length=length,
            num_elements=num_elements,
            youngs_modulus=youngs_modulus,
            moment_inertia=moment_inertia,
            density=density,
        )
        self.fe_matrices = FiniteElementMatrices(self.beam_params)

        self.global_stiffness = None
        self.global_mass = None
        self.free_dofs = None
        self.constraint_matrix = None
        self.nullspace_basis = None
        self.reduced_mass = None
        self.reduced_stiffness = None
        self.result = None

    def _fixed_dofs(self):
        if self.beam_type == "cantilever":
            return np.array([0, 1], dtype=int)
        if self.beam_type == "simply_supported":
            return np.array([0, 2 * (self.beam_params.num_nodes - 1)], dtype=int)
        raise ValueError(f"Unknown beam type: {self.beam_type}")

    def build_constraint_matrix(self):
        """
        Build the homogeneous constraint matrix C for C u = 0.

        Pass constraint_matrix=... to the constructor to use general multi-point
        constraints, for example beam-to-beam connection equations.
        """
        total_dofs = 2 * self.beam_params.num_nodes
        if self.user_constraint_matrix is not None:
            constraints = np.asarray(self.user_constraint_matrix, dtype=float)
            if constraints.ndim != 2 or constraints.shape[1] != total_dofs:
                raise ValueError(
                    "constraint_matrix must have shape "
                    f"(number_of_constraints, {total_dofs})."
                )
            return constraints.copy()

        return fixed_dof_constraint_matrix(total_dofs, self._fixed_dofs())

    def _to_reduced_vector(self, vector, default_value=0.0):
        if self.nullspace_basis is None:
            self.assemble_reduced_system()

        reduced_size = self.nullspace_basis.shape[1]
        total_dofs = self.nullspace_basis.shape[0]
        vector = (
            np.full(reduced_size, default_value, dtype=float)
            if vector is None
            else np.asarray(vector, dtype=float)
        )

        if len(vector) == reduced_size:
            return vector.copy()
        if len(vector) == total_dofs:
            return self.nullspace_basis.T @ vector

        raise ValueError(
            "Initial vectors must have either nullspace-coordinate length "
            f"{reduced_size} or full length {total_dofs}."
        )

    def assemble_reduced_system(self):
        self.global_stiffness, self.global_mass = self.fe_matrices.assemble_global_matrices()
        self.constraint_matrix = self.build_constraint_matrix()
        self.nullspace_basis = nullspace_basis(self.constraint_matrix)
        if self.nullspace_basis.shape[1] == 0:
            raise ValueError("The constraint matrix leaves no admissible displacement DOFs.")
        self.free_dofs = np.arange(self.nullspace_basis.shape[1], dtype=int)

        # Calculate the reduced mass and stiffness matrices(M_r, S_r) in the nullspace coordinates.
        # M_r = N^T MN, S_r = N^T SN
        self.reduced_mass = self.nullspace_basis.T @ self.global_mass @ self.nullspace_basis
        self.reduced_stiffness = (
            self.nullspace_basis.T @ self.global_stiffness @ self.nullspace_basis
        )
        return self.reduced_mass, self.reduced_stiffness

    def expand_modes(self, modes_reduced):
        if self.nullspace_basis is None:
            self.assemble_reduced_system()
        return self.nullspace_basis @ modes_reduced

    def expand_history(self, reduced_history):
        if self.nullspace_basis is None:
            self.assemble_reduced_system()
        return reduced_history @ self.nullspace_basis.T

    def solve(self, num_modes=None):
        """Solve K phi = omega^2 M phi and mass-normalize the selected modes."""
        mass, stiffness = self.assemble_reduced_system()
        num_modes = self.num_modes if num_modes is None else num_modes
        num_modes = min(num_modes, len(self.free_dofs))

        """
        In oder to generalize the method to prepare for future frameworks, we calculate the eigenvalues
        directly from the reduced mass and stiffness matrices.
        """
        # Use Cholesky factorization to convert the generalized eigenvalue problem to a standard one.
        cholesky_mass = np.linalg.cholesky(mass)
        left_scaled = np.linalg.solve(cholesky_mass, stiffness)
        standard_matrix = np.linalg.solve(cholesky_mass, left_scaled.T).T
        standard_matrix = 0.5 * (standard_matrix + standard_matrix.T)

        eigenvalues, transformed_modes = np.linalg.eigh(standard_matrix)
        # Filter out small negative eigenvalues due to numerical errors and sort the modes by ascending eigenvalue.
        positive = eigenvalues > max(np.max(np.abs(eigenvalues)), 1.0) * 1e-12
        eigenvalues = eigenvalues[positive]
        transformed_modes = transformed_modes[:, positive]

        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order][:num_modes]
        transformed_modes = transformed_modes[:, order][:, :num_modes]

        modes_reduced = np.linalg.solve(cholesky_mass.T, transformed_modes)
        for mode_index in range(modes_reduced.shape[1]):
            modal_mass = modes_reduced[:, mode_index] @ mass @ modes_reduced[:, mode_index]
            modes_reduced[:, mode_index] /= np.sqrt(modal_mass)

        modes_full = self.expand_modes(modes_reduced)
        for mode_index in range(modes_full.shape[1]):
            displacement_values = modes_full[::2, mode_index]
            anchor = np.argmax(np.abs(displacement_values))
            if displacement_values[anchor] < 0.0:
                modes_reduced[:, mode_index] *= -1.0
                modes_full[:, mode_index] *= -1.0

        angular_frequencies = np.sqrt(eigenvalues)
        frequencies_hz = angular_frequencies / (2.0 * np.pi)
        periods = 1.0 / frequencies_hz

        modal_mass = np.diag(modes_reduced.T @ mass @ modes_reduced)
        modal_stiffness = np.diag(modes_reduced.T @ stiffness @ modes_reduced)

        self.result = EigenvalueResult(
            eigenvalues=eigenvalues,
            angular_frequencies=angular_frequencies,
            frequencies_hz=frequencies_hz,
            periods=periods,
            modes_reduced=modes_reduced,
            modes_full=modes_full,
            modal_mass=modal_mass,
            modal_stiffness=modal_stiffness,
            free_dofs=self.free_dofs,
            constraint_matrix=self.constraint_matrix,
            nullspace_basis=self.nullspace_basis,
        )
        return self.result

    def analytic_characteristic_values(self, num_modes=None):
        num_modes = self.num_modes if num_modes is None else num_modes
        if self.beam_type == "cantilever":
            return cantilever_characteristic_values(num_modes)
        if self.beam_type == "simply_supported":
            return simply_supported_characteristic_values(num_modes)
        raise ValueError(f"No analytic characteristic values for {self.beam_type}.")

    def analytic_frequencies(self, num_modes=None):
        """Return analytic omega, f, and periods for the current beam."""
        num_modes = self.num_modes if num_modes is None else num_modes
        beam = self.beam_params
        alpha = self.analytic_characteristic_values(num_modes)
        beta = alpha / beam.length
        omega = beta ** 2 * np.sqrt(beam.youngs_modulus * beam.moment_inertia / beam.density)
        frequency_hz = omega / (2.0 * np.pi)
        return omega, frequency_hz, 1.0 / frequency_hz

    def analytic_mode_dofs(self, mode_number):
        """Evaluate an analytic mode shape at the FEM nodes as [w0, theta0, ...]."""
        if mode_number < 1:
            raise ValueError("mode_number is 1-based and must be at least 1.")

        beam = self.beam_params
        x = beam.node_positions
        alpha = self.analytic_characteristic_values(mode_number)[mode_number - 1]
        beta = alpha / beam.length

        if self.beam_type == "cantilever":
            sigma = (np.cosh(alpha) + np.cos(alpha)) / (np.sinh(alpha) + np.sin(alpha))
            displacement = (
                np.cosh(beta * x)
                - np.cos(beta * x)
                - sigma * (np.sinh(beta * x) - np.sin(beta * x))
            )
            rotation = beta * (
                np.sinh(beta * x)
                + np.sin(beta * x)
                - sigma * (np.cosh(beta * x) - np.cos(beta * x))
            )
        elif self.beam_type == "simply_supported":
            displacement = np.sin(beta * x)
            rotation = beta * np.cos(beta * x)
        else:
            raise ValueError(f"No analytic mode shape for {self.beam_type}.")

        scale = np.max(np.abs(displacement))
        if scale != 0.0:
            displacement = displacement / scale
            rotation = rotation / scale

        mode = np.zeros(2 * beam.num_nodes)
        mode[::2] = displacement
        mode[1::2] = rotation
        return mode

    def compare_with_analytic(self, result=None, num_modes=None):
        requested_modes = self.num_modes if num_modes is None else num_modes
        result = result or self._result_with_at_least(requested_modes)
        num_modes = min(requested_modes, len(result.angular_frequencies))
        omega_exact, frequency_exact, period_exact = self.analytic_frequencies(num_modes)

        omega_fem = result.angular_frequencies[:num_modes]
        frequency_fem = result.frequencies_hz[:num_modes]
        period_fem = result.periods[:num_modes]

        return {
            "mode": np.arange(1, num_modes + 1),
            "omega_fem": omega_fem,
            "omega_exact": omega_exact,
            "omega_relative_error": np.abs(omega_fem - omega_exact) / np.abs(omega_exact),
            "frequency_fem": frequency_fem,
            "frequency_exact": frequency_exact,
            "frequency_relative_error": (
                np.abs(frequency_fem - frequency_exact) / np.abs(frequency_exact)
            ),
            "period_fem": period_fem,
            "period_exact": period_exact,
        }

    def free_vibration_response(
            self,
            initial_displacement=None,
            initial_velocity=None,
            total_time=1.0,
            time_step=1e-3,
            num_modes=None,
    ):
        """Compute free response by modal superposition from initial conditions."""
        requested_modes = self.num_modes if num_modes is None else num_modes
        result = self._result_with_at_least(requested_modes)
        mass = self.reduced_mass
        stiffness = self.reduced_stiffness
        num_modes = min(requested_modes, len(result.angular_frequencies))
        modes = result.modes_reduced[:, :num_modes]
        omega = result.angular_frequencies[:num_modes]

        if initial_displacement is None and initial_velocity is None:
            initial_displacement = self._default_initial_displacement(result)

        u0 = self._to_reduced_vector(initial_displacement)
        v0 = self._to_reduced_vector(initial_velocity)

        q0 = modes.T @ mass @ u0
        qdot0 = modes.T @ mass @ v0

        h = float(time_step)
        time = np.arange(0.0, total_time + 0.5 * h, h)
        cos_terms = np.cos(np.outer(time, omega))
        sin_terms = np.sin(np.outer(time, omega))

        modal_coordinates = q0 * cos_terms + (qdot0 / omega) * sin_terms
        modal_velocities = -q0 * omega * sin_terms + qdot0 * cos_terms
        modal_accelerations = -(omega ** 2) * modal_coordinates

        displacement_reduced = modal_coordinates @ modes.T
        velocity_reduced = modal_velocities @ modes.T
        acceleration_reduced = modal_accelerations @ modes.T
        energy = self.energy(displacement_reduced, velocity_reduced, mass, stiffness)

        return FreeVibrationResult(
            time=time,
            displacement=self.expand_history(displacement_reduced),
            velocity=self.expand_history(velocity_reduced),
            acceleration=self.expand_history(acceleration_reduced),
            modal_coordinates=modal_coordinates,
            energy=energy,
            free_dofs=self.free_dofs,
            constraint_matrix=self.constraint_matrix,
            nullspace_basis=self.nullspace_basis,
        )

    def _default_initial_displacement(self, result):
        """Small first-mode displacement for demo/free-run usage."""
        first_mode = result.modes_full[:, 0].copy()
        max_displacement = np.max(np.abs(first_mode[::2]))
        if max_displacement == 0.0:
            return first_mode
        return 1e-3 * first_mode / max_displacement

    def _result_with_at_least(self, num_modes):
        if self.result is None or len(self.result.angular_frequencies) < num_modes:
            return self.solve(num_modes)
        return self.result

    def energy(self, displacement, velocity, mass, stiffness):
        kinetic = np.einsum("ij,jk,ik->i", velocity, mass, velocity)
        elastic = np.einsum("ij,jk,ik->i", displacement, stiffness, displacement)
        return 0.5 * (kinetic + elastic)

    def print_frequency_table(self, result=None, num_modes=None):
        comparison = self.compare_with_analytic(result, num_modes)
        print("mode | FEM omega (rad/s) | exact omega (rad/s) | rel. error | FEM f (Hz)")
        print("-" * 76)
        for i in range(len(comparison["mode"])):
            print(
                f"{comparison['mode'][i]:4d} | "
                f"{comparison['omega_fem'][i]:17.6e} | "
                f"{comparison['omega_exact'][i]:19.6e} | "
                f"{comparison['omega_relative_error'][i]:10.3e} | "
                f"{comparison['frequency_fem'][i]:10.6e}"
            )

    def plot_mode_shapes(self, result=None, num_modes=None, include_analytic=True):
        """Delegate to the eigenvalue visualization function."""
        return _plotting.plot_mode_shapes(self, result, num_modes, include_analytic)

    def plot_tip_response(self, response):
        """Delegate to the eigenvalue visualization function."""
        return _plotting.plot_tip_response(self, response)

    def _normalized_mode(self, dof_vector):
        return normalized_mode(dof_vector)

    def _aligned_analytic_mode(self, numerical_mode, mode_number):
        return aligned_analytic_mode(self, numerical_mode, mode_number)

    def plot_mode_shape_comparison(
        self,
        result=None,
        num_modes=5,
        save_path=None,
        show=True,
    ):
        """Plot FEM and analytic mode shapes side by side for the first modes."""
        return _plotting.plot_mode_shape_comparison(
            self,
            result,
            num_modes,
            save_path,
            show,
        )

    def plot_frequency_errors(
        self,
        result=None,
        num_modes=5,
        save_path=None,
        show=True,
    ):
        """Plot FEM natural-frequency relative errors against analytic values."""
        return _plotting.plot_frequency_errors(self, result, num_modes, save_path, show)

    def save_frequency_error_table(self, result=None, num_modes=5, save_path=None):
        """Write a CSV table with FEM/analytic frequencies and relative errors."""
        return _plotting.save_frequency_error_table(self, result, num_modes, save_path)

    def plot_first_eigenmodes(
        self,
        result=None,
        num_modes=5,
        save_path=None,
        show=True,
    ):
        """Plot the first numerical eigenmodes in one stacked figure."""
        return _plotting.plot_first_eigenmodes(self, result, num_modes, save_path, show)

    def standing_wave_response(
            self,
            result=None,
            mode_number=1,
            amplitude=1e-3,
            periods=2.0,
            frames_per_period=60,
    ):
        """Build a single-mode standing wave u(x, t) = A phi_n(x) cos(omega_n t)."""
        if mode_number < 1:
            raise ValueError("mode_number is 1-based and must be at least 1.")

        result = result or self._result_with_at_least(mode_number)
        mode_index = mode_number - 1
        if mode_index >= len(result.angular_frequencies):
            raise ValueError(f"Mode {mode_number} is not available in the current result.")

        mode = self._normalized_mode(result.modes_full[:, mode_index])
        omega = result.angular_frequencies[mode_index]
        period = 2.0 * np.pi / omega
        n_steps = max(2, int(np.ceil(periods * frames_per_period)) + 1)
        time = np.linspace(0.0, periods * period, n_steps)

        cos_term = np.cos(omega * time)
        sin_term = np.sin(omega * time)
        displacement = amplitude * cos_term[:, None] * mode[None, :]
        velocity = -amplitude * omega * sin_term[:, None] * mode[None, :]
        acceleration = -omega ** 2 * displacement

        if self.global_mass is None or self.global_stiffness is None:
            self.assemble_reduced_system()
        energy = self.energy(displacement, velocity, self.global_mass, self.global_stiffness)

        modal_coordinates = np.zeros((len(time), mode_number))
        modal_coordinates[:, mode_index] = amplitude * cos_term

        return FreeVibrationResult(
            time=time,
            displacement=displacement,
            velocity=velocity,
            acceleration=acceleration,
            modal_coordinates=modal_coordinates,
            energy=energy,
            free_dofs=self.free_dofs,
            constraint_matrix=self.constraint_matrix,
            nullspace_basis=self.nullspace_basis,
        )

    def save_standing_wave_visualizations(
        self,
        result=None,
        output_dir='figs/eigenvalue_visual_output',
        prefix=None,
        mode_number=1,
        amplitude=0.001,
        periods=2.0,
        frames_per_period=60,
        snapshot_time_step=None,
        scale='auto',
        video_extension='.gif',
        fps=30,
    ):
        """Save snapshots and an animation for a single standing-wave mode."""
        return _plotting.save_standing_wave_visualizations(
            self,
            result,
            output_dir,
            prefix,
            mode_number,
            amplitude,
            periods,
            frames_per_period,
            snapshot_time_step,
            scale,
            video_extension,
            fps,
        )

    def save_modal_analysis_outputs(
        self,
        result=None,
        output_dir='figs/eigenvalue_visual_output',
        num_modes=5,
        standing_wave_mode=1,
        video_extension='.gif',
        snapshot_time_step=None,
    ):
        """Save the standard eigenvalue report figures requested by the project."""
        return _plotting.save_modal_analysis_outputs(
            self,
            result,
            output_dir,
            num_modes,
            standing_wave_mode,
            video_extension,
            snapshot_time_step,
        )

    def save_free_vibration_visualizations(
        self,
        response=None,
        output_dir='figs/eigenvalue_visual_output',
        prefix=None,
        scale='auto',
        video_extension='.gif',
        frame_step=1,
        snapshot_time_step=None,
        snapshot_time_start=None,
        snapshot_time_end=None,
        fps=30,
    ):
        """
            Reuse the Newmark visualization helpers for eigenvalue free vibration.

            The eigenvalue FreeVibrationResult has the same plotting fields used by
            newmark_visualization.py: time, displacement, energy, and
            tip_displacement.
            """
        return _plotting.save_free_vibration_visualizations(
            self,
            response,
            output_dir,
            prefix,
            scale,
            video_extension,
            frame_step,
            snapshot_time_step,
            snapshot_time_start,
            snapshot_time_end,
            fps,
        )
