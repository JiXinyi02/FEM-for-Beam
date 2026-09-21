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

import numpy as np

from .constraints import fixed_dof_constraint_matrix, nullspace_basis
from .elements import (
    frame_element_transformation,
    local_frame_stiffness,
    local_frame_mass,
)
from .parameters import (
    FrameworkNode,
    FrameworkElement,
    default_framework_nodes,
    default_framework_connectivity,
)
from .plotting import framework as _plotting
from .results import (
    FrameworkEigenvalueResult,
    FrameworkFreeVibrationResult,
    FrameworkStaticResult,
    FrameworkNewmarkResult,
)


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
        """Delegate to the framework visualization function."""
        return _plotting.automatic_mode_scale(self, mode, target_fraction)

    def plot_mode_shape(
        self,
        result=None,
        mode_number=1,
        scale='auto',
        ax=None,
        save_path=None,
        show=True,
    ):
        """Delegate to the framework visualization function."""
        return _plotting.plot_mode_shape(
            self,
            result,
            mode_number,
            scale,
            ax,
            save_path,
            show,
        )

    def automatic_static_scale(self, displacement, target_fraction=0.2):
        """Delegate to the framework visualization function."""
        return _plotting.automatic_static_scale(self, displacement, target_fraction)

    def plot_static_deformation(
        self,
        static_result,
        scale='auto',
        ax=None,
        save_path=None,
        show=True,
    ):
        """Visualize the static deformation under a nodal point force."""
        return _plotting.plot_static_deformation(
            self,
            static_result,
            scale,
            ax,
            save_path,
            show,
        )

    def plot_static_displacement_contour(
        self,
        static_result,
        scale='auto',
        points_per_element=40,
        ax=None,
        save_path=None,
        show=True,
    ):
        """Plot deformed members colored by displacement magnitude."""
        return _plotting.plot_static_displacement_contour(
            self,
            static_result,
            scale,
            points_per_element,
            ax,
            save_path,
            show,
        )

    def save_static_point_load_visualization(
        self,
        node='p3',
        fx=0.0,
        fy=-100000.0,
        moment=0.0,
        output_dir='figs/framework_output',
        prefix=None,
        scale='auto',
    ):
        """Solve and save the static deformation for a point load at one node."""
        return _plotting.save_static_point_load_visualization(
            self,
            node,
            fx,
            fy,
            moment,
            output_dir,
            prefix,
            scale,
        )

    def save_static_point_load_visualizations(
        self,
        node='p3',
        fx=0.0,
        fy=-100000.0,
        moment=0.0,
        output_dir='figs/framework_output',
        prefix=None,
        scale='auto',
    ):
        """Solve and save deformation plus displacement-magnitude contour plots."""
        return _plotting.save_static_point_load_visualizations(
            self,
            node,
            fx,
            fy,
            moment,
            output_dir,
            prefix,
            scale,
        )

    def automatic_dynamic_scale(self, result, target_fraction=0.2):
        """Delegate to the framework visualization function."""
        return _plotting.automatic_dynamic_scale(self, result, target_fraction)

    def plot_dynamic_node_response(
        self,
        result,
        node='p3',
        component='uy',
        save_path=None,
        show=True,
    ):
        """Plot one nodal displacement component from a Newmark result."""
        return _plotting.plot_dynamic_node_response(
            self,
            result,
            node,
            component,
            save_path,
            show,
        )

    def plot_dynamic_energy(self, result, save_path=None, show=True):
        """Plot mechanical energy from a Newmark result."""
        return _plotting.plot_dynamic_energy(self, result, save_path, show)

    def plot_dynamic_snapshots(
        self,
        result,
        n_snapshots=6,
        frame_indices=None,
        scale='auto',
        save_path=None,
        show=True,
    ):
        """Plot several framework deformation snapshots from a Newmark result."""
        return _plotting.plot_dynamic_snapshots(
            self,
            result,
            n_snapshots,
            frame_indices,
            scale,
            save_path,
            show,
        )

    def animate_dynamic_response(
        self,
        result,
        output_path='figs/framework_output/framework_newmark.gif',
        scale='auto',
        frame_step=1,
        fps=30,
    ):
        """Animate the Newmark framework deformation history."""
        return _plotting.animate_dynamic_response(
            self,
            result,
            output_path,
            scale,
            frame_step,
            fps,
        )

    def save_newmark_visualizations(
        self,
        result,
        output_dir='figs/framework_output',
        prefix='framework_newmark',
        response_node='p3',
        response_component='uy',
        scale='auto',
        frame_step=2,
    ):
        """Save Newmark response plots, snapshots, and animation."""
        return _plotting.save_newmark_visualizations(
            self,
            result,
            output_dir,
            prefix,
            response_node,
            response_component,
            scale,
            frame_step,
        )

    def plot_first_modes(
        self,
        result=None,
        num_modes=5,
        output_dir='figs/framework_output',
    ):
        """Delegate to the framework visualization function."""
        return _plotting.plot_first_modes(self, result, num_modes, output_dir)

    def animate_mode(
        self,
        result=None,
        mode_number=1,
        output_path='figs/framework_output/framework_mode_1.gif',
        scale='auto',
        periods=2.0,
        frames_per_period=60,
        fps=30,
    ):
        """Delegate to the framework visualization function."""
        return _plotting.animate_mode(
            self,
            result,
            mode_number,
            output_path,
            scale,
            periods,
            frames_per_period,
            fps,
        )
