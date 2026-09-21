"""
Dynamic Bernoulli beam analysis with the Newmark method.

The implementation follows script2_newmark.pdf for systems of the form

    M a(t) + D v(t) + S u(t) = p(t)

using the update

    u_star = u_j + h v_j + (1/2 - beta) h^2 a_j
    v_star = v_j + (1 - gamma) h a_j
    (M + gamma h D + beta h^2 S) a_{j+1}
        = p(t_{j+1}) - D v_star - S u_star
    u_{j+1} = u_star + beta h^2 a_{j+1}
    v_{j+1} = v_star + gamma h a_{j+1}

The default parameters beta=1/4 and gamma=1/2 are the stable average
acceleration parameters recommended in the notes.
"""

import numpy as np

from .beam_loads import assemble_load_vector
from .elements import FiniteElementMatrices
from .parameters import BeamParameters
from .plotting import newmark as _plotting
from .results import NewmarkResult


class NewmarkBeamAnalysis:
    """Newmark time integration for the beam FEM model used in static_beam.py."""

    def __init__(
            self,
            beam_type="cantilever",
            load_case="uniform",
            total_time=0.5,
            time_step=5e-4,
            beta=0.25,
            gamma=0.5,
            damping_mass=0.0,
            damping_stiffness=0.0,
            load_time_function=None,
            **load_params  # To pass load parameters like q0, P, M, etc.
    ):
        self.beam_type = beam_type
        self.load_case = load_case
        self.load_params = load_params
        self.total_time = total_time
        self.time_step = time_step
        self.beta = beta
        self.gamma = gamma
        self.damping_mass = damping_mass
        self.damping_stiffness = damping_stiffness
        self.load_time_function = load_time_function or (lambda t: 1.0)

        self.beam_params = BeamParameters()
        self.fe_matrices = FiniteElementMatrices(self.beam_params)

        self.global_stiffness = None
        self.global_mass = None
        self.global_damping = None
        self.free_dofs = None
        self.result = None


    # Newmark method only applies for non-fixed DOFs, so we need to identify the free DOFs:

    def _fixed_dofs(self):
        if self.beam_type == "cantilever":
            return np.array([0, 1], dtype=int)
        if self.beam_type == "simply_supported":
            return np.array([0, 2 * (self.beam_params.num_nodes - 1)], dtype=int)
        raise ValueError(f"Unknown beam type: {self.beam_type}")

    def _free_dofs(self):
        fixed = set(self._fixed_dofs())
        return np.array(
            [dof for dof in range(2 * self.beam_params.num_nodes) if dof not in fixed],
            dtype=int,
        )

    def _to_reduced_vector(self, vector, default_value=0.0):
        if vector is None:
            return np.full(len(self.free_dofs), default_value, dtype=float)

        vector = np.asarray(vector, dtype=float)
        if len(vector) == len(self.free_dofs):
            return vector.copy()
        if len(vector) == 2 * self.beam_params.num_nodes:
            return vector[self.free_dofs].copy()

        raise ValueError(
            "Initial vectors must have either reduced length "
            f"{len(self.free_dofs)} or full length {2 * self.beam_params.num_nodes}."
        )

    def assemble_reduced_system(self):
        self.global_stiffness, self.global_mass = self.fe_matrices.assemble_global_matrices()
        self.global_damping = (
            self.damping_mass * self.global_mass
            + self.damping_stiffness * self.global_stiffness
        )

        self.free_dofs = self._free_dofs()
        idx = np.ix_(self.free_dofs, self.free_dofs)

        mass = self.global_mass[idx]
        damping = self.global_damping[idx]
        stiffness = self.global_stiffness[idx]

        return mass, damping, stiffness

    def base_load_vector(self):
        return assemble_load_vector(self.beam_params, self.load_case, **self.load_params)

    def force_vector(self, time_value):
        load_value = self.load_time_function(time_value)

        if np.isscalar(load_value):
            force_full = float(load_value) * self.base_load_vector()
        else:
            force_full = np.asarray(load_value, dtype=float)
            if len(force_full) != 2 * self.beam_params.num_nodes:
                raise ValueError(
                    "A vector-valued load_time_function must return a full global "
                    f"force vector of length {2 * self.beam_params.num_nodes}."
                )

        return force_full[self.free_dofs]

    def expand_solution(self, reduced_history):
        full_history = np.zeros((len(reduced_history), 2 * self.beam_params.num_nodes))
        full_history[:, self.free_dofs] = reduced_history
        return full_history

    def energy(self, displacement, velocity, mass, stiffness):
        kinetic = np.einsum("ij,jk,ik->i", velocity, mass, velocity)
        elastic = np.einsum("ij,jk,ik->i", displacement, stiffness, displacement)
        return 0.5 * (kinetic + elastic)

    def solve(self, initial_displacement=None, initial_velocity=None):
        mass, damping, stiffness = self.assemble_reduced_system()
        h = self.time_step
        time = np.arange(0.0, self.total_time + 0.5 * h, h)
        n_steps = len(time)
        n_dofs = len(self.free_dofs)


        # Initialize arrays for displacement, velocity, and acceleration
        displacement = np.zeros((n_steps, n_dofs))
        velocity = np.zeros((n_steps, n_dofs))
        acceleration = np.zeros((n_steps, n_dofs))

        # Set initial conditions
        displacement[0] = self._to_reduced_vector(initial_displacement)
        velocity[0] = self._to_reduced_vector(initial_velocity)

        # Compute initial acceleration using the equation of motion:
        # M u'' = F - D u' - K u
        # This step is necessary since Newmark requires u''(0) to start the integration.
        acceleration[0] = np.linalg.solve(
            mass,
            self.force_vector(time[0])
            - damping @ velocity[0]
            - stiffness @ displacement[0],
        )

        effective_matrix = (
            mass
            + self.gamma * h * damping
            + self.beta * h ** 2 * stiffness
        )


        for j in range(n_steps - 1):
            # Compute the mediate values u*, u'*:
            u_star = (
                displacement[j]
                + h * velocity[j]
                + (0.5 - self.beta) * h ** 2 * acceleration[j]
            )
            v_star = velocity[j] + (1.0 - self.gamma) * h * acceleration[j]

            rhs = self.force_vector(time[j + 1]) - damping @ v_star - stiffness @ u_star
            acceleration[j + 1] = np.linalg.solve(effective_matrix, rhs)
            displacement[j + 1] = u_star + self.beta * h ** 2 * acceleration[j + 1]
            velocity[j + 1] = v_star + self.gamma * h * acceleration[j + 1]

        result = NewmarkResult(
            time=time,
            # The solutions only contain the free DOFs, expand them to the full set:
            displacement=self.expand_solution(displacement),
            velocity=self.expand_solution(velocity),
            acceleration=self.expand_solution(acceleration),
            energy=self.energy(displacement, velocity, mass, stiffness),
            free_dofs=self.free_dofs,
        )
        self.result = result
        return result

    def plot_tip_response(self, result=None):
        """Delegate to the newmark visualization function."""
        return _plotting.plot_tip_response(self, result)
