from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection

from constraints import fixed_dof_constraint_matrix, nullspace_basis


"""
Eigenvalue analysis for the simple 2D framework from the lecture.

Each beam is modeled as one planar frame element with six DOFs:

    [u_x1, u_y1, theta_1, u_x2, u_y2, theta_2].

The global nodal DOF order is

    [p1_ux, p1_uy, p1_theta, p2_ux, p2_uy, p2_theta, ...].

Rigid connections at p2 and p3 are represented by sharing the same global
node DOFs between all incident elements.  Supports are enforced by a
homogeneous constraint matrix C u = 0 and the nullspace substitution u = N q.
"""


@dataclass
class FrameworkNode:
    name: str
    x: float
    y: float


@dataclass
class FrameworkElement:
    start: int
    end: int
    youngs_modulus: float
    cross_section_area: float
    moment_inertia: float
    density: float


@dataclass
class FrameworkEigenvalueResult:
    eigenvalues: np.ndarray
    angular_frequencies: np.ndarray
    frequencies_hz: np.ndarray
    periods: np.ndarray
    modes_reduced: np.ndarray
    modes_full: np.ndarray
    modal_mass: np.ndarray
    modal_stiffness: np.ndarray
    constraint_matrix: np.ndarray
    nullspace_basis: np.ndarray


@dataclass
class FrameworkFreeVibrationResult:
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    modal_coordinates: np.ndarray
    energy: np.ndarray
    constraint_matrix: np.ndarray
    nullspace_basis: np.ndarray


@dataclass
class FrameworkStaticResult:
    displacement: np.ndarray
    displacement_reduced: np.ndarray
    force: np.ndarray
    reactions: np.ndarray
    strain_energy: float
    constraint_matrix: np.ndarray
    nullspace_basis: np.ndarray


@dataclass
class FrameworkNewmarkResult:
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    energy: np.ndarray
    force: np.ndarray
    constraint_matrix: np.ndarray
    nullspace_basis: np.ndarray

    def node_displacement(self, node, component=None):
        """Return one node displacement history by integer node index."""
        node_index = int(node)
        values = self.displacement[:, 3 * node_index:3 * node_index + 3]
        if component is None:
            return values
        component_index = {"ux": 0, "uy": 1, "theta": 2}.get(component, component)
        return values[:, component_index]


def default_framework_nodes():
    """
    Coordinates for p1...p4.

    The lecture image is schematic, so these coordinates define a simple
    representative geometry with p1 fixed at the lower left and p4 on the
    right support line.
    """
    return [
        FrameworkNode("p1", 0.0, 0.0),
        FrameworkNode("p2", 3.0, 0.6),
        FrameworkNode("p3", 2.0, 2.4),
        FrameworkNode("p4", 5.0, 3.0),
    ]


def default_framework_connectivity():
    """Five one-element beams in the simple framework."""
    return [
        (0, 1),  # p1-p2
        (0, 2),  # p1-p3
        (1, 2),  # p2-p3
        (1, 3),  # p2-p4
        (2, 3),  # p3-p4
    ]


def frame_element_transformation(cosine, sine):
    """Return T such that local_dofs = T @ global_dofs."""
    # Local x is along the element axis; local y is perpendicular to it.
    # w.r.t. the local frame, each beam has 6 DOFs:
    #   [u_x1, u_y1, theta_1, u_x2, u_y2, theta_2]. 
    return np.array([
        [cosine, sine, 0.0, 0.0, 0.0, 0.0],
        [-sine, cosine, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, cosine, sine, 0.0],
        [0.0, 0.0, 0.0, -sine, cosine, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])


def local_frame_stiffness(length, youngs_modulus, area, moment_inertia):
    """Local 2D frame element stiffness matrix."""
    # Axial terms use EA/L; bending terms use the Euler-Bernoulli beam matrix.
    ea_l = youngs_modulus * area / length
    ei = youngs_modulus * moment_inertia
    l2 = length ** 2
    l3 = length ** 3

    return np.array([
        [ea_l, 0.0, 0.0, -ea_l, 0.0, 0.0],
        [0.0, 12.0 * ei / l3, 6.0 * ei / l2, 0.0, -12.0 * ei / l3, 6.0 * ei / l2],
        [0.0, 6.0 * ei / l2, 4.0 * ei / length, 0.0, -6.0 * ei / l2, 2.0 * ei / length],
        [-ea_l, 0.0, 0.0, ea_l, 0.0, 0.0],
        [0.0, -12.0 * ei / l3, -6.0 * ei / l2, 0.0, 12.0 * ei / l3, -6.0 * ei / l2],
        [0.0, 6.0 * ei / l2, 2.0 * ei / length, 0.0, -6.0 * ei / l2, 4.0 * ei / length],
    ])


def local_frame_mass(length, density, area):
    """
    Local consistent mass matrix.

    Axial motion uses the bar consistent mass, and transverse/rotational motion
    uses the Euler-Bernoulli beam consistent mass.
    """
    mass_per_length = density * area
    return mass_per_length * length / 420.0 * np.array([
        [140.0, 0.0, 0.0, 70.0, 0.0, 0.0],
        [0.0, 156.0, 22.0 * length, 0.0, 54.0, -13.0 * length],
        [0.0, 22.0 * length, 4.0 * length ** 2, 0.0, 13.0 * length, -3.0 * length ** 2],
        [70.0, 0.0, 0.0, 140.0, 0.0, 0.0],
        [0.0, 54.0, 13.0 * length, 0.0, 156.0, -22.0 * length],
        [0.0, -13.0 * length, -3.0 * length ** 2, 0.0, -22.0 * length, 4.0 * length ** 2],
    ])


class FrameworkEigenvalueAnalysis:
    """Modal analysis for the simple framework."""

    def __init__(
            self,
            nodes=None,
            connectivity=None,
            youngs_modulus=200e9,
            cross_section_area=0.02,
            moment_inertia=8.0e-5,
            density=7800.0,
            num_modes=6,
            constrain_p4_rotation=False,
    ):
        self.nodes = nodes or default_framework_nodes()
        self.connectivity = connectivity or default_framework_connectivity()
        self.youngs_modulus = youngs_modulus
        self.cross_section_area = cross_section_area
        self.moment_inertia = moment_inertia
        self.density = density
        self.num_modes = num_modes
        self.constrain_p4_rotation = constrain_p4_rotation

        self.elements = [
            FrameworkElement(
                start=start,
                end=end,
                youngs_modulus=youngs_modulus,
                cross_section_area=cross_section_area,
                moment_inertia=moment_inertia,
                density=density,
            )
            for start, end in self.connectivity
        ]

        self.total_dofs = 3 * len(self.nodes)
        self.global_stiffness = None
        self.global_mass = None
        self.constraint_matrix = None
        self.nullspace_basis = None
        self.reduced_mass = None
        self.reduced_stiffness = None
        self.result = None

    def node_coordinates(self):
        return np.array([[node.x, node.y] for node in self.nodes], dtype=float)

    def node_index(self, node_name):
        for index, node in enumerate(self.nodes):
            if node.name == node_name:
                return index
        raise ValueError(f"Unknown node name: {node_name}")

    def node_dofs(self, node):
        node = self.node_index(node) if isinstance(node, str) else int(node)
        return np.array([3 * node, 3 * node + 1, 3 * node + 2], dtype=int)

    def element_dofs(self, element):
        # Three global DOFs per node: ux, uy, theta.
        return np.array([
            3 * element.start,
            3 * element.start + 1,
            3 * element.start + 2,
            3 * element.end,
            3 * element.end + 1,
            3 * element.end + 2,
        ], dtype=int)

    def element_geometry(self, element):
        coordinates = self.node_coordinates()
        start = coordinates[element.start]
        end = coordinates[element.end]
        dx, dy = end - start
        length = float(np.hypot(dx, dy))
        if length <= 0.0:
            raise ValueError("Frame elements must have positive length.")
        # Return length and sin\phi, cos\phi.
        return length, dx / length, dy / length

    def point_load_vector(self, node="p3", fx=0.0, fy=-1.0e5, moment=0.0):
        """Build a global nodal force vector [Fx, Fy, M] applied at one node."""
        force = np.zeros(self.total_dofs)
        ux_dof, uy_dof, theta_dof = self.node_dofs(node)
        force[ux_dof] = fx
        force[uy_dof] = fy
        force[theta_dof] = moment
        return force

    def assemble_global_matrices(self):
        stiffness = np.zeros((self.total_dofs, self.total_dofs))
        mass = np.zeros((self.total_dofs, self.total_dofs))

        for element in self.elements:
            length, cosine, sine = self.element_geometry(element)
            transformation = frame_element_transformation(cosine, sine)
            local_stiffness = local_frame_stiffness(
                length,
                element.youngs_modulus,
                element.cross_section_area,
                element.moment_inertia,
            )
            local_mass = local_frame_mass(
                length,
                element.density,
                element.cross_section_area,
            )

            # Transform local element matrices into global coordinates:
            # k_g = T.T k_l T, m_g = T.T m_l T.
            element_stiffness = transformation.T @ local_stiffness @ transformation
            element_mass = transformation.T @ local_mass @ transformation
            dofs = self.element_dofs(element)
            index = np.ix_(dofs, dofs)

            # Add this element's contribution into the shared global DOFs.
            # Shared p2/p3 DOFs are what make the internal connections rigid.
            stiffness[index] += element_stiffness
            mass[index] += element_mass

        self.global_stiffness = stiffness
        self.global_mass = mass
        return mass, stiffness

    def fixed_dofs(self):
        """
        Support conditions:
        - p1: ux = uy = theta = 0
        - p4: ux = 0, vertical motion uy remains free.

        Set constrain_p4_rotation=True if the p4 bearing should also suppress
        nodal rotation.
        """
        p1 = 0
        p4 = 3
        fixed = [3 * p1, 3 * p1 + 1, 3 * p1 + 2, 3 * p4]
        if self.constrain_p4_rotation:
            fixed.append(3 * p4 + 2)
        return np.array(fixed, dtype=int)

    def build_constraint_matrix(self):
        return fixed_dof_constraint_matrix(self.total_dofs, self.fixed_dofs())

    def assemble_reduced_system(self):
        mass, stiffness = self.assemble_global_matrices()
        self.constraint_matrix = self.build_constraint_matrix()

        # Constraints C u = 0 are enforced by u = N q, where C N = 0.
        self.nullspace_basis = nullspace_basis(self.constraint_matrix)
        if self.nullspace_basis.shape[1] == 0:
            raise ValueError("The constraint matrix leaves no admissible DOFs.")

        # Reduced matrices in admissible coordinates q.
        self.reduced_mass = self.nullspace_basis.T @ mass @ self.nullspace_basis
        self.reduced_stiffness = self.nullspace_basis.T @ stiffness @ self.nullspace_basis
        return self.reduced_mass, self.reduced_stiffness

    def solve(self, num_modes=None):
        """
        Solve the constrained framework eigenproblem.

        Starting from M u'' + K u = 0 and C u = 0, write u = N q with C N = 0:

            (N.T M N) q'' + (N.T K N) q = 0
            Kc phi = omega^2 Mc phi.
        """
        mass, stiffness = self.assemble_reduced_system()
        num_modes = self.num_modes if num_modes is None else num_modes
        num_modes = min(num_modes, self.nullspace_basis.shape[1])

        # Mc = L L.T converts Kc phi = lambda Mc phi into
        # (L^-1 Kc L^-T) y = lambda y, with phi = L^-T y.
        cholesky_mass = np.linalg.cholesky(mass)
        left_scaled = np.linalg.solve(cholesky_mass, stiffness)
        standard_matrix = np.linalg.solve(cholesky_mass, left_scaled.T).T
        standard_matrix = 0.5 * (standard_matrix + standard_matrix.T)

        eigenvalues, transformed_modes = np.linalg.eigh(standard_matrix)
        positive = eigenvalues > max(np.max(np.abs(eigenvalues)), 1.0) * 1e-12
        eigenvalues = eigenvalues[positive]
        transformed_modes = transformed_modes[:, positive]

        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order][:num_modes]
        transformed_modes = transformed_modes[:, order][:, :num_modes]

        # Convert eigenvectors back to q-coordinates and mass-normalize them.
        modes_reduced = np.linalg.solve(cholesky_mass.T, transformed_modes)
        for mode_index in range(modes_reduced.shape[1]):
            modal_mass = modes_reduced[:, mode_index] @ mass @ modes_reduced[:, mode_index]
            modes_reduced[:, mode_index] /= np.sqrt(modal_mass)

        # Expand modal vectors from q-coordinates to full nodal DOFs.
        modes_full = self.nullspace_basis @ modes_reduced
        for mode_index in range(modes_full.shape[1]):
            translations = modes_full[:, mode_index].reshape(-1, 3)[:, :2]
            anchor = np.unravel_index(np.argmax(np.abs(translations)), translations.shape)
            if translations[anchor] < 0.0:
                modes_reduced[:, mode_index] *= -1.0
                modes_full[:, mode_index] *= -1.0

        angular_frequencies = np.sqrt(eigenvalues)
        frequencies_hz = angular_frequencies / (2.0 * np.pi)
        periods = 1.0 / frequencies_hz

        self.result = FrameworkEigenvalueResult(
            eigenvalues=eigenvalues,
            angular_frequencies=angular_frequencies,
            frequencies_hz=frequencies_hz,
            periods=periods,
            modes_reduced=modes_reduced,
            modes_full=modes_full,
            modal_mass=np.diag(modes_reduced.T @ mass @ modes_reduced),
            modal_stiffness=np.diag(modes_reduced.T @ stiffness @ modes_reduced),
            constraint_matrix=self.constraint_matrix,
            nullspace_basis=self.nullspace_basis,
        )
        return self.result

    def print_frequency_table(self, result=None):
        result = result or self.result or self.solve()
        print("mode | omega (rad/s) | f (Hz) | period (s)")
        print("-" * 52)
        for mode_index, (omega, frequency, period) in enumerate(
                zip(result.angular_frequencies, result.frequencies_hz, result.periods),
                start=1,
        ):
            print(f"{mode_index:4d} | {omega:13.6e} | {frequency:8.4f} | {period:10.6e}")

    def solve_static(self, force):
        """
        Solve K u = F with the same constraints C u = 0.

        Using u = N q gives the reduced static system

            (N.T K N) q = N.T F.
        """
        if self.reduced_stiffness is None:
            self.assemble_reduced_system()

        force = np.asarray(force, dtype=float)
        if len(force) != self.total_dofs:
            raise ValueError(f"force must have length {self.total_dofs}.")

        reduced_force = self.nullspace_basis.T @ force
        displacement_reduced = np.linalg.solve(self.reduced_stiffness, reduced_force)
        displacement = self.nullspace_basis @ displacement_reduced

        internal_force = self.global_stiffness @ displacement
        reactions = internal_force - force
        strain_energy = 0.5 * displacement @ self.global_stiffness @ displacement

        return FrameworkStaticResult(
            displacement=displacement,
            displacement_reduced=displacement_reduced,
            force=force,
            reactions=reactions,
            strain_energy=strain_energy,
            constraint_matrix=self.constraint_matrix,
            nullspace_basis=self.nullspace_basis,
        )

    def solve_static_point_load(self, node="p3", fx=0.0, fy=-1.0e5, moment=0.0):
        """Static response to a point load applied at one framework node."""
        return self.solve_static(self.point_load_vector(node=node, fx=fx, fy=fy, moment=moment))

    def solve_newmark(
            self,
            base_force=None,
            load_time_function=None,
            total_time=0.08,
            time_step=1e-4,
            beta=0.25,
            gamma=0.5,
            damping_mass=0.0,
            damping_stiffness=0.0,
            initial_displacement=None,
            initial_velocity=None,
    ):
        """
        Newmark time integration for the constrained framework system.

        The full equation is M u'' + D u' + K u = F(t).  With constraints
        C u = 0 and u = N q:

            Mr q'' + Dr q' + Kr q = N.T F(t).

        Average acceleration parameters beta=1/4, gamma=1/2 are unconditionally
        stable for linear systems.
        """
        if self.reduced_mass is None:
            self.assemble_reduced_system()

        if base_force is None:
            base_force = np.zeros(self.total_dofs)
        base_force = np.asarray(base_force, dtype=float)
        if len(base_force) != self.total_dofs:
            raise ValueError(f"base_force must have length {self.total_dofs}.")

        load_time_function = load_time_function or (lambda t: 1.0)
        damping = damping_mass * self.reduced_mass + damping_stiffness * self.reduced_stiffness

        h = float(time_step)
        time = np.arange(0.0, total_time + 0.5 * h, h)
        n_steps = len(time)
        reduced_size = self.nullspace_basis.shape[1]

        displacement = np.zeros((n_steps, reduced_size))
        velocity = np.zeros((n_steps, reduced_size))
        acceleration = np.zeros((n_steps, reduced_size))
        force_history = np.zeros((n_steps, self.total_dofs))

        def to_reduced(vector, default_value=0.0):
            if vector is None:
                return np.full(reduced_size, default_value, dtype=float)
            vector = np.asarray(vector, dtype=float)
            if len(vector) == reduced_size:
                return vector.copy()
            if len(vector) == self.total_dofs:
                return self.nullspace_basis.T @ vector
            raise ValueError(f"Vector must have length {reduced_size} or {self.total_dofs}.")

        def force_at(t):
            load_value = load_time_function(t)
            if np.isscalar(load_value):
                force_full = float(load_value) * base_force
            else:
                force_full = np.asarray(load_value, dtype=float)
                if len(force_full) != self.total_dofs:
                    raise ValueError(f"Vector load must have length {self.total_dofs}.")
            return force_full, self.nullspace_basis.T @ force_full

        displacement[0] = to_reduced(initial_displacement)
        velocity[0] = to_reduced(initial_velocity)
        force_history[0], reduced_force = force_at(time[0])
        acceleration[0] = np.linalg.solve(
            self.reduced_mass,
            reduced_force - damping @ velocity[0] - self.reduced_stiffness @ displacement[0],
        )

        effective_matrix = (
            self.reduced_mass
            + gamma * h * damping
            + beta * h ** 2 * self.reduced_stiffness
        )

        for step in range(n_steps - 1):
            u_star = (
                displacement[step]
                + h * velocity[step]
                + (0.5 - beta) * h ** 2 * acceleration[step]
            )
            v_star = velocity[step] + (1.0 - gamma) * h * acceleration[step]

            force_history[step + 1], reduced_force = force_at(time[step + 1])
            rhs = reduced_force - damping @ v_star - self.reduced_stiffness @ u_star
            acceleration[step + 1] = np.linalg.solve(effective_matrix, rhs)
            displacement[step + 1] = u_star + beta * h ** 2 * acceleration[step + 1]
            velocity[step + 1] = v_star + gamma * h * acceleration[step + 1]

        displacement_full = displacement @ self.nullspace_basis.T
        velocity_full = velocity @ self.nullspace_basis.T
        acceleration_full = acceleration @ self.nullspace_basis.T
        energy = self.energy(displacement_full, velocity_full)

        return FrameworkNewmarkResult(
            time=time,
            displacement=displacement_full,
            velocity=velocity_full,
            acceleration=acceleration_full,
            energy=energy,
            force=force_history,
            constraint_matrix=self.constraint_matrix,
            nullspace_basis=self.nullspace_basis,
        )

    def solve_newmark_point_load(
            self,
            node="p3",
            fx=0.0,
            fy=-1.0e5,
            moment=0.0,
            load_time_function=None,
            **newmark_parameters,
    ):
        """Newmark response to a nodal point load that remains active in time."""
        base_force = self.point_load_vector(node=node, fx=fx, fy=fy, moment=moment)
        return self.solve_newmark(
            base_force=base_force,
            load_time_function=load_time_function,
            **newmark_parameters,
        )

    def solve_newmark_release_from_point_load(
            self,
            node="p3",
            fx=0.0,
            fy=-1.0e5,
            moment=0.0,
            **newmark_parameters,
    ):
        """
        Newmark free vibration after releasing a statically loaded framework.

        The nodal load is used only to compute the initial displacement:

            K u0 = F0,   C u0 = 0.

        Then the dynamic step uses F(t) = 0 and D = 0 by default, so the
        framework vibrates freely from that released shape.
        """
        static_result = self.solve_static_point_load(node=node, fx=fx, fy=fy, moment=moment)
        return self.solve_newmark(
            base_force=np.zeros(self.total_dofs),
            load_time_function=lambda t: 0.0,
            initial_displacement=static_result.displacement,
            initial_velocity=np.zeros(self.total_dofs),
            damping_mass=newmark_parameters.pop("damping_mass", 0.0),
            damping_stiffness=newmark_parameters.pop("damping_stiffness", 0.0),
            **newmark_parameters,
        )

    def free_vibration_response(
            self,
            initial_displacement=None,
            initial_velocity=None,
            total_time=1.0,
            time_step=1e-3,
            num_modes=None,
    ):
        result = self.result or self.solve(num_modes)
        num_modes = len(result.angular_frequencies) if num_modes is None else num_modes
        num_modes = min(num_modes, len(result.angular_frequencies))
        modes = result.modes_reduced[:, :num_modes]
        omega = result.angular_frequencies[:num_modes]

        reduced_size = self.nullspace_basis.shape[1]
        if initial_displacement is None:
            first_mode = result.modes_full[:, 0].copy()
            max_translation = np.max(np.linalg.norm(first_mode.reshape(-1, 3)[:, :2], axis=1))
            initial_displacement = 1e-3 * first_mode / max_translation
        if initial_velocity is None:
            initial_velocity = np.zeros(self.total_dofs)

        initial_displacement = np.asarray(initial_displacement, dtype=float)
        initial_velocity = np.asarray(initial_velocity, dtype=float)
        if len(initial_displacement) == self.total_dofs:
            u0 = self.nullspace_basis.T @ initial_displacement
        elif len(initial_displacement) == reduced_size:
            u0 = initial_displacement
        else:
            raise ValueError("initial_displacement must have full or reduced length.")

        if len(initial_velocity) == self.total_dofs:
            v0 = self.nullspace_basis.T @ initial_velocity
        elif len(initial_velocity) == reduced_size:
            v0 = initial_velocity
        else:
            raise ValueError("initial_velocity must have full or reduced length.")

        # Project initial conditions onto the mass-normalized modal basis.
        q0 = modes.T @ self.reduced_mass @ u0
        qdot0 = modes.T @ self.reduced_mass @ v0

        time = np.arange(0.0, total_time + 0.5 * time_step, time_step)
        cos_terms = np.cos(np.outer(time, omega))
        sin_terms = np.sin(np.outer(time, omega))

        modal_coordinates = q0 * cos_terms + (qdot0 / omega) * sin_terms
        modal_velocities = -q0 * omega * sin_terms + qdot0 * cos_terms
        modal_accelerations = -(omega ** 2) * modal_coordinates

        # Rebuild q(t), then expand back to full nodal DOFs u(t) = N q(t).
        displacement_reduced = modal_coordinates @ modes.T
        velocity_reduced = modal_velocities @ modes.T
        acceleration_reduced = modal_accelerations @ modes.T
        displacement = displacement_reduced @ self.nullspace_basis.T
        velocity = velocity_reduced @ self.nullspace_basis.T
        acceleration = acceleration_reduced @ self.nullspace_basis.T
        energy = self.energy(displacement, velocity)

        return FrameworkFreeVibrationResult(
            time=time,
            displacement=displacement,
            velocity=velocity,
            acceleration=acceleration,
            modal_coordinates=modal_coordinates,
            energy=energy,
            constraint_matrix=self.constraint_matrix,
            nullspace_basis=self.nullspace_basis,
        )

    def energy(self, displacement, velocity):
        kinetic = np.einsum("ij,jk,ik->i", velocity, self.global_mass, velocity)
        elastic = np.einsum("ij,jk,ik->i", displacement, self.global_stiffness, displacement)
        return 0.5 * (kinetic + elastic)

    def deformed_coordinates(self, dof_vector, scale=1.0):
        coordinates = self.node_coordinates()
        translations = np.asarray(dof_vector).reshape(-1, 3)[:, :2]
        return coordinates + scale * translations

    def sample_element_displacement_field(self, element, dof_vector, scale=1.0, points_per_element=30):
        """
        Sample the deformed centerline and displacement magnitude of one element.

        Axial displacement is interpolated linearly.  Transverse displacement
        uses the same cubic Hermite interpolation as an Euler-Bernoulli beam.
        """
        length, cosine, sine = self.element_geometry(element)
        transformation = frame_element_transformation(cosine, sine)
        element_global_dofs = np.asarray(dof_vector, dtype=float)[self.element_dofs(element)]
        local_dofs = transformation @ element_global_dofs

        xi = np.linspace(0.0, 1.0, points_per_element)
        axial = (1.0 - xi) * local_dofs[0] + xi * local_dofs[3]
        transverse = (
            (1.0 - 3.0 * xi ** 2 + 2.0 * xi ** 3) * local_dofs[1]
            + length * xi * (xi - 1.0) ** 2 * local_dofs[2]
            + (3.0 * xi ** 2 - 2.0 * xi ** 3) * local_dofs[4]
            + length * xi ** 2 * (xi - 1.0) * local_dofs[5]
        )

        local_x = xi * length
        local_y = np.zeros_like(local_x)
        coordinates = self.node_coordinates()
        start = coordinates[element.start]

        original = np.column_stack((
            start[0] + cosine * local_x - sine * local_y,
            start[1] + sine * local_x + cosine * local_y,
        ))
        global_displacement = np.column_stack((
            cosine * axial - sine * transverse,
            sine * axial + cosine * transverse,
        ))
        deformed = original + scale * global_displacement
        displacement_magnitude = np.linalg.norm(global_displacement, axis=1)
        return original, deformed, displacement_magnitude

    def automatic_mode_scale(self, mode, target_fraction=0.2):
        coordinates = self.node_coordinates()
        span = max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1]))
        max_translation = np.max(np.linalg.norm(mode.reshape(-1, 3)[:, :2], axis=1))
        if max_translation == 0.0:
            return 1.0
        # Make the largest displayed modal displacement a fixed fraction of the structure size.
        return target_fraction * span / max_translation

    def plot_mode_shape(self, result=None, mode_number=1, scale="auto", ax=None, save_path=None, show=True):
        result = result or self.result or self.solve(mode_number)
        mode = result.modes_full[:, mode_number - 1]
        scale_value = self.automatic_mode_scale(mode) if scale == "auto" else float(scale)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        undeformed = self.node_coordinates()
        deformed = self.deformed_coordinates(mode, scale=scale_value)

        for start, end in self.connectivity:
            ax.plot(undeformed[[start, end], 0], undeformed[[start, end], 1], "k--", linewidth=1)
            ax.plot(deformed[[start, end], 0], deformed[[start, end], 1], "r-", linewidth=2)

        for index, node in enumerate(self.nodes):
            ax.text(undeformed[index, 0], undeformed[index, 1], node.name)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Framework mode {mode_number}, f = {result.frequencies_hz[mode_number - 1]:.4f} Hz")
        ax.grid(True, alpha=0.3)
        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return ax

    def automatic_static_scale(self, displacement, target_fraction=0.2):
        coordinates = self.node_coordinates()
        span = max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1]))
        max_translation = np.max(np.linalg.norm(displacement.reshape(-1, 3)[:, :2], axis=1))
        if max_translation == 0.0:
            return 1.0
        return target_fraction * span / max_translation

    def plot_static_deformation(
            self,
            static_result,
            scale="auto",
            ax=None,
            save_path=None,
            show=True,
    ):
        """Visualize the static deformation under a nodal point force."""
        scale_value = (
            self.automatic_static_scale(static_result.displacement)
            if scale == "auto"
            else float(scale)
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        undeformed = self.node_coordinates()
        deformed = self.deformed_coordinates(static_result.displacement, scale=scale_value)

        for start, end in self.connectivity:
            ax.plot(undeformed[[start, end], 0], undeformed[[start, end], 1], "k--", linewidth=1)
            ax.plot(deformed[[start, end], 0], deformed[[start, end], 1], "r-", linewidth=2)

        force_nodes = static_result.force.reshape(-1, 3)[:, :2]
        force_norm = np.linalg.norm(force_nodes, axis=1)
        loaded_nodes = np.where(force_norm > 0.0)[0]
        if len(loaded_nodes) > 0:
            max_force = np.max(force_norm[loaded_nodes])
            arrow_scale = 0.25 * max(np.ptp(undeformed[:, 0]), np.ptp(undeformed[:, 1])) / max_force
            for node_index in loaded_nodes:
                force_vector = force_nodes[node_index] * arrow_scale
                ax.arrow(
                    undeformed[node_index, 0],
                    undeformed[node_index, 1],
                    force_vector[0],
                    force_vector[1],
                    color="tab:blue",
                    width=0.015,
                    length_includes_head=True,
                    label="point force" if node_index == loaded_nodes[0] else None,
                )

        for index, node in enumerate(self.nodes):
            ax.text(undeformed[index, 0], undeformed[index, 1], node.name)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Static deformation, visualization scale = {scale_value:.3e}")
        ax.grid(True, alpha=0.3)
        ax.legend(["undeformed", "deformed", "point force"], loc="best")
        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return ax

    def plot_static_displacement_contour(
            self,
            static_result,
            scale="auto",
            points_per_element=40,
            ax=None,
            save_path=None,
            show=True,
    ):
        """Plot deformed members colored by displacement magnitude."""
        scale_value = (
            self.automatic_static_scale(static_result.displacement)
            if scale == "auto"
            else float(scale)
        )
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        undeformed = self.node_coordinates()
        all_segments = []
        all_values = []

        for start, end in self.connectivity:
            ax.plot(undeformed[[start, end], 0], undeformed[[start, end], 1], "k--", linewidth=0.9)

        for element in self.elements:
            _, deformed, magnitude = self.sample_element_displacement_field(
                element,
                static_result.displacement,
                scale=scale_value,
                points_per_element=points_per_element,
            )
            segments = np.stack([deformed[:-1], deformed[1:]], axis=1)
            segment_values = 0.5 * (magnitude[:-1] + magnitude[1:])
            all_segments.extend(segments)
            all_values.extend(segment_values)

        line_collection = LineCollection(
            all_segments,
            array=np.asarray(all_values),
            cmap="viridis",
            linewidths=3.0,
        )
        ax.add_collection(line_collection)
        colorbar = fig.colorbar(line_collection, ax=ax)
        colorbar.set_label("displacement magnitude (m)")

        force_nodes = static_result.force.reshape(-1, 3)[:, :2]
        force_norm = np.linalg.norm(force_nodes, axis=1)
        loaded_nodes = np.where(force_norm > 0.0)[0]
        if len(loaded_nodes) > 0:
            max_force = np.max(force_norm[loaded_nodes])
            arrow_scale = 0.25 * max(np.ptp(undeformed[:, 0]), np.ptp(undeformed[:, 1])) / max_force
            for node_index in loaded_nodes:
                force_vector = force_nodes[node_index] * arrow_scale
                ax.arrow(
                    undeformed[node_index, 0],
                    undeformed[node_index, 1],
                    force_vector[0],
                    force_vector[1],
                    color="tab:red",
                    width=0.015,
                    length_includes_head=True,
                )

        for index, node in enumerate(self.nodes):
            ax.text(undeformed[index, 0], undeformed[index, 1], node.name)

        margin = 0.25 * max(np.ptp(undeformed[:, 0]), np.ptp(undeformed[:, 1]))
        ax.autoscale()
        ax.set_xlim(min(ax.get_xlim()[0], np.min(undeformed[:, 0]) - margin),
                    max(ax.get_xlim()[1], np.max(undeformed[:, 0]) + margin))
        ax.set_ylim(min(ax.get_ylim()[0], np.min(undeformed[:, 1]) - margin),
                    max(ax.get_ylim()[1], np.max(undeformed[:, 1]) + margin))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Static displacement magnitude, scale = {scale_value:.3e}")
        ax.grid(True, alpha=0.3)

        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        return ax

    def save_static_point_load_visualization(
            self,
            node="p3",
            fx=0.0,
            fy=-1.0e5,
            moment=0.0,
            output_dir="figs/framework_output",
            prefix=None,
            scale="auto",
    ):
        """Solve and save the static deformation for a point load at one node."""
        result = self.solve_static_point_load(node=node, fx=fx, fy=fy, moment=moment)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if prefix is None:
            prefix = f"static_point_load_{node}"

        path = output_dir / f"{prefix}_deformation.png"
        self.plot_static_deformation(result, scale=scale, save_path=path, show=False)
        plt.close()
        return result, path

    def save_static_point_load_visualizations(
            self,
            node="p3",
            fx=0.0,
            fy=-1.0e5,
            moment=0.0,
            output_dir="figs/framework_output",
            prefix=None,
            scale="auto",
    ):
        """Solve and save deformation plus displacement-magnitude contour plots."""
        result = self.solve_static_point_load(node=node, fx=fx, fy=fy, moment=moment)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if prefix is None:
            prefix = f"static_point_load_{node}"

        paths = {
            "deformation": output_dir / f"{prefix}_deformation.png",
            "displacement_contour": output_dir / f"{prefix}_displacement_contour.png",
        }
        self.plot_static_deformation(result, scale=scale, save_path=paths["deformation"], show=False)
        plt.close()
        self.plot_static_displacement_contour(
            result,
            scale=scale,
            save_path=paths["displacement_contour"],
            show=False,
        )
        plt.close()
        return result, paths

    def automatic_dynamic_scale(self, result, target_fraction=0.2):
        coordinates = self.node_coordinates()
        span = max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1]))
        translations = result.displacement.reshape(len(result.time), -1, 3)[:, :, :2]
        max_translation = np.max(np.linalg.norm(translations, axis=2))
        if max_translation == 0.0:
            return 1.0
        return target_fraction * span / max_translation

    def plot_dynamic_node_response(
            self,
            result,
            node="p3",
            component="uy",
            save_path=None,
            show=True,
    ):
        """Plot one nodal displacement component from a Newmark result."""
        node_index = self.node_index(node) if isinstance(node, str) else int(node)
        component_index = {"ux": 0, "uy": 1, "theta": 2}[component]
        values = result.displacement[:, 3 * node_index + component_index]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(result.time, values, linewidth=2)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"{node} {component}")
        ax.set_title(f"Newmark response at {node}: {component}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return ax

    def plot_dynamic_energy(self, result, save_path=None, show=True):
        """Plot mechanical energy from a Newmark result."""
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(result.time, result.energy, linewidth=2, color="tab:red")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("mechanical energy")
        ax.set_title("Framework Newmark mechanical energy")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return ax

    def plot_dynamic_snapshots(
            self,
            result,
            n_snapshots=6,
            frame_indices=None,
            scale="auto",
            save_path=None,
            show=True,
    ):
        """Plot several framework deformation snapshots from a Newmark result."""
        scale_value = self.automatic_dynamic_scale(result) if scale == "auto" else float(scale)
        if frame_indices is None:
            frame_indices = np.linspace(0, len(result.time) - 1, n_snapshots, dtype=int)

        coordinates = self.node_coordinates()
        fig, ax = plt.subplots(figsize=(8, 6))
        for start, end in self.connectivity:
            ax.plot(coordinates[[start, end], 0], coordinates[[start, end], 1], "k--", linewidth=1)

        colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(frame_indices)))
        for color, frame_index in zip(colors, frame_indices):
            deformed = self.deformed_coordinates(result.displacement[frame_index], scale=scale_value)
            for start, end in self.connectivity:
                ax.plot(
                    deformed[[start, end], 0],
                    deformed[[start, end], 1],
                    color=color,
                    linewidth=1.8,
                )
            ax.plot([], [], color=color, label=f"t={result.time[frame_index]:.4f}s")

        for index, node in enumerate(self.nodes):
            ax.text(coordinates[index, 0], coordinates[index, 1], node.name)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Newmark deformation snapshots, scale = {scale_value:.3e}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return ax

    def animate_dynamic_response(
            self,
            result,
            output_path="figs/framework_output/framework_newmark.gif",
            scale="auto",
            frame_step=1,
            fps=30,
    ):
        """Animate the Newmark framework deformation history."""
        scale_value = self.automatic_dynamic_scale(result) if scale == "auto" else float(scale)
        coordinates = self.node_coordinates()
        frame_indices = np.arange(0, len(result.time), frame_step, dtype=int)

        fig, ax = plt.subplots(figsize=(8, 6))
        for start, end in self.connectivity:
            ax.plot(coordinates[[start, end], 0], coordinates[[start, end], 1], "k--", linewidth=1)

        lines = []
        for start, end in self.connectivity:
            line, = ax.plot([], [], "r-", linewidth=2)
            lines.append((line, start, end))

        pad = 0.35 * max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1]))
        ax.set_xlim(np.min(coordinates[:, 0]) - pad, np.max(coordinates[:, 0]) + pad)
        ax.set_ylim(np.min(coordinates[:, 1]) - pad, np.max(coordinates[:, 1]) + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.3)
        title = ax.set_title("")

        def update(frame_number):
            frame_index = frame_indices[frame_number]
            deformed = self.deformed_coordinates(result.displacement[frame_index], scale=scale_value)
            for line, start, end in lines:
                line.set_data(deformed[[start, end], 0], deformed[[start, end], 1])
            title.set_text(f"Framework Newmark response, t = {result.time[frame_index]:.4f} s")
            return [line for line, _, _ in lines] + [title]

        anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), blit=True)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(output_path, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        return output_path

    def save_newmark_visualizations(
            self,
            result,
            output_dir="figs/framework_output",
            prefix="framework_newmark",
            response_node="p3",
            response_component="uy",
            scale="auto",
            frame_step=2,
    ):
        """Save Newmark response plots, snapshots, and animation."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "node_response": output_dir / f"{prefix}_{response_node}_{response_component}.png",
            "energy": output_dir / f"{prefix}_energy.png",
            "snapshots": output_dir / f"{prefix}_snapshots.png",
            "animation": output_dir / f"{prefix}_animation.gif",
        }
        self.plot_dynamic_node_response(
            result,
            node=response_node,
            component=response_component,
            save_path=paths["node_response"],
            show=False,
        )
        self.plot_dynamic_energy(result, save_path=paths["energy"], show=False)
        self.plot_dynamic_snapshots(result, scale=scale, save_path=paths["snapshots"], show=False)
        self.animate_dynamic_response(
            result,
            output_path=paths["animation"],
            scale=scale,
            frame_step=frame_step,
        )
        return paths

    def plot_first_modes(self, result=None, num_modes=5, output_dir="figs/framework_output"):
        result = result or self.result or self.solve(num_modes)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for mode_number in range(1, min(num_modes, len(result.angular_frequencies)) + 1):
            path = output_dir / f"framework_mode_{mode_number}.png"
            self.plot_mode_shape(result, mode_number=mode_number, save_path=path, show=False)
            plt.close()
            paths.append(path)
        return paths

    def animate_mode(
            self,
            result=None,
            mode_number=1,
            output_path="figs/framework_output/framework_mode_1.gif",
            scale="auto",
            periods=2.0,
            frames_per_period=60,
            fps=30,
    ):
        result = result or self.result or self.solve(mode_number)
        mode = result.modes_full[:, mode_number - 1]
        omega = result.angular_frequencies[mode_number - 1]
        period = 2.0 * np.pi / omega
        time = np.linspace(0.0, periods * period, int(periods * frames_per_period) + 1)
        scale_value = self.automatic_mode_scale(mode) if scale == "auto" else float(scale)

        coordinates = self.node_coordinates()
        fig, ax = plt.subplots(figsize=(8, 6))
        for start, end in self.connectivity:
            ax.plot(coordinates[[start, end], 0], coordinates[[start, end], 1], "k--", linewidth=1)

        lines = []
        for start, end in self.connectivity:
            line, = ax.plot([], [], "r-", linewidth=2)
            lines.append((line, start, end))

        ax.set_aspect("equal", adjustable="box")
        pad = 0.4 * max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1]))
        ax.set_xlim(np.min(coordinates[:, 0]) - pad, np.max(coordinates[:, 0]) + pad)
        ax.set_ylim(np.min(coordinates[:, 1]) - pad, np.max(coordinates[:, 1]) + pad)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.3)
        title = ax.set_title("")

        def update(frame_index):
            deformed = self.deformed_coordinates(np.cos(omega * time[frame_index]) * mode, scale_value)
            for line, start, end in lines:
                line.set_data(deformed[[start, end], 0], deformed[[start, end], 1])
            title.set_text(f"Framework mode {mode_number}, t = {time[frame_index]:.4f} s")
            return [line for line, _, _ in lines] + [title]

        anim = animation.FuncAnimation(fig, update, frames=len(time), blit=True)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(output_path, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        return output_path


if __name__ == "__main__":
    analysis = FrameworkEigenvalueAnalysis(num_modes=6)
    output_dir = Path("figs/framework_output")

    static_result, static_paths = analysis.save_static_point_load_visualizations(
        node="p3",
        fy=-1.0e5,
        output_dir=output_dir,
        prefix="p3_vertical_point_load",
    )
    p3 = analysis.node_index("p3")
    p3_ux, p3_uy, p3_theta = static_result.displacement[analysis.node_dofs(p3)]
    print("Static analysis: point load at p3")
    print(f"p3 ux:    {p3_ux:.6e} m")
    print(f"p3 uy:    {p3_uy:.6e} m")
    print(f"p3 theta: {p3_theta:.6e} rad")
    print(f"strain energy: {static_result.strain_energy:.6e}")
    print(f"saved static deformation: {static_paths['deformation']}")
    print(f"saved displacement contour: {static_paths['displacement_contour']}")

    result = analysis.solve()
    analysis.print_frequency_table(result)

    mode_paths = analysis.plot_first_modes(result, num_modes=5, output_dir=output_dir)
    animation_path = analysis.animate_mode(result, mode_number=1, output_path=output_dir / "framework_mode_1.gif")

    response = analysis.free_vibration_response(total_time=0.05, time_step=5e-4)
    print(f"\nInitial energy: {response.energy[0]:.6e}")
    print(f"Final energy:   {response.energy[-1]:.6e}")

    newmark_result = analysis.solve_newmark_release_from_point_load(
        node="p3",
        fy=-1.0e5,
        total_time=0.05,
        time_step=2.0e-4,
    )
    newmark_paths = analysis.save_newmark_visualizations(
        newmark_result,
        output_dir=output_dir,
        prefix="p3_vertical_release_newmark",
        response_node="p3",
        response_component="uy",
        frame_step=3,
    )
    p3_uy_history = newmark_result.displacement[:, analysis.node_dofs("p3")[1]]
    print("\nNewmark dynamic analysis: release from static p3 point load")
    print(f"max |p3 uy|: {np.max(np.abs(p3_uy_history)):.6e} m")
    print(f"final p3 uy: {p3_uy_history[-1]:.6e} m")
    print(f"initial energy: {newmark_result.energy[0]:.6e}")
    print(f"final energy:   {newmark_result.energy[-1]:.6e}")

    print("\nSaved framework outputs:")
    for path in mode_paths:
        print(path)
    print(animation_path)
    for path in newmark_paths.values():
        print(path)
