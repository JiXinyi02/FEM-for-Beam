"""Visualization functions for FrameworkEigenvalueAnalysis."""

from pathlib import Path

from matplotlib import animation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np


def automatic_mode_scale(analysis, mode, target_fraction=0.2):
    coordinates = analysis.node_coordinates()
    span = max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1]))
    max_translation = np.max(np.linalg.norm(mode.reshape(-1, 3)[:, :2], axis=1))
    if max_translation == 0.0:
        return 1.0
    # Make the largest displayed modal displacement a fixed fraction of the structure size.
    return target_fraction * span / max_translation


def plot_mode_shape(analysis, result=None, mode_number=1, scale="auto", ax=None, save_path=None, show=True):
    result = result or analysis.result or analysis.solve(mode_number)
    mode = result.modes_full[:, mode_number - 1]
    scale_value = analysis.automatic_mode_scale(mode) if scale == "auto" else float(scale)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    undeformed = analysis.node_coordinates()
    deformed = analysis.deformed_coordinates(mode, scale=scale_value)

    for start, end in analysis.connectivity:
        ax.plot(undeformed[[start, end], 0], undeformed[[start, end], 1], "k--", linewidth=1)
        ax.plot(deformed[[start, end], 0], deformed[[start, end], 1], "r-", linewidth=2)

    for index, node in enumerate(analysis.nodes):
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


def automatic_static_scale(analysis, displacement, target_fraction=0.2):
    coordinates = analysis.node_coordinates()
    span = max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1]))
    max_translation = np.max(np.linalg.norm(displacement.reshape(-1, 3)[:, :2], axis=1))
    if max_translation == 0.0:
        return 1.0
    return target_fraction * span / max_translation


def plot_static_deformation(
        analysis,
        static_result,
        scale="auto",
        ax=None,
        save_path=None,
        show=True,
):
    """Visualize the static deformation under a nodal point force."""
    scale_value = (
        analysis.automatic_static_scale(static_result.displacement)
        if scale == "auto"
        else float(scale)
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    undeformed = analysis.node_coordinates()
    deformed = analysis.deformed_coordinates(static_result.displacement, scale=scale_value)

    for start, end in analysis.connectivity:
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

    for index, node in enumerate(analysis.nodes):
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
        analysis,
        static_result,
        scale="auto",
        points_per_element=40,
        ax=None,
        save_path=None,
        show=True,
):
    """Plot deformed members colored by displacement magnitude."""
    scale_value = (
        analysis.automatic_static_scale(static_result.displacement)
        if scale == "auto"
        else float(scale)
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    undeformed = analysis.node_coordinates()
    all_segments = []
    all_values = []

    for start, end in analysis.connectivity:
        ax.plot(undeformed[[start, end], 0], undeformed[[start, end], 1], "k--", linewidth=0.9)

    for element in analysis.elements:
        _, deformed, magnitude = analysis.sample_element_displacement_field(
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

    for index, node in enumerate(analysis.nodes):
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
        analysis,
        node="p3",
        fx=0.0,
        fy=-1.0e5,
        moment=0.0,
        output_dir="figs/framework_output",
        prefix=None,
        scale="auto",
):
    """Solve and save the static deformation for a point load at one node."""
    result = analysis.solve_static_point_load(node=node, fx=fx, fy=fy, moment=moment)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = f"static_point_load_{node}"

    path = output_dir / f"{prefix}_deformation.png"
    analysis.plot_static_deformation(result, scale=scale, save_path=path, show=False)
    plt.close()
    return result, path


def save_static_point_load_visualizations(
        analysis,
        node="p3",
        fx=0.0,
        fy=-1.0e5,
        moment=0.0,
        output_dir="figs/framework_output",
        prefix=None,
        scale="auto",
):
    """Solve and save deformation plus displacement-magnitude contour plots."""
    result = analysis.solve_static_point_load(node=node, fx=fx, fy=fy, moment=moment)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = f"static_point_load_{node}"

    paths = {
        "deformation": output_dir / f"{prefix}_deformation.png",
        "displacement_contour": output_dir / f"{prefix}_displacement_contour.png",
    }
    analysis.plot_static_deformation(result, scale=scale, save_path=paths["deformation"], show=False)
    plt.close()
    analysis.plot_static_displacement_contour(
        result,
        scale=scale,
        save_path=paths["displacement_contour"],
        show=False,
    )
    plt.close()
    return result, paths


def automatic_dynamic_scale(analysis, result, target_fraction=0.2):
    coordinates = analysis.node_coordinates()
    span = max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1]))
    translations = result.displacement.reshape(len(result.time), -1, 3)[:, :, :2]
    max_translation = np.max(np.linalg.norm(translations, axis=2))
    if max_translation == 0.0:
        return 1.0
    return target_fraction * span / max_translation


def plot_dynamic_node_response(
        analysis,
        result,
        node="p3",
        component="uy",
        save_path=None,
        show=True,
):
    """Plot one nodal displacement component from a Newmark result."""
    node_index = analysis.node_index(node) if isinstance(node, str) else int(node)
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


def plot_dynamic_energy(analysis, result, save_path=None, show=True):
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
        analysis,
        result,
        n_snapshots=6,
        frame_indices=None,
        scale="auto",
        save_path=None,
        show=True,
):
    """Plot several framework deformation snapshots from a Newmark result."""
    scale_value = analysis.automatic_dynamic_scale(result) if scale == "auto" else float(scale)
    if frame_indices is None:
        frame_indices = np.linspace(0, len(result.time) - 1, n_snapshots, dtype=int)

    coordinates = analysis.node_coordinates()
    fig, ax = plt.subplots(figsize=(8, 6))
    for start, end in analysis.connectivity:
        ax.plot(coordinates[[start, end], 0], coordinates[[start, end], 1], "k--", linewidth=1)

    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(frame_indices)))
    for color, frame_index in zip(colors, frame_indices):
        deformed = analysis.deformed_coordinates(result.displacement[frame_index], scale=scale_value)
        for start, end in analysis.connectivity:
            ax.plot(
                deformed[[start, end], 0],
                deformed[[start, end], 1],
                color=color,
                linewidth=1.8,
            )
        ax.plot([], [], color=color, label=f"t={result.time[frame_index]:.4f}s")

    for index, node in enumerate(analysis.nodes):
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
        analysis,
        result,
        output_path="figs/framework_output/framework_newmark.gif",
        scale="auto",
        frame_step=1,
        fps=30,
):
    """Animate the Newmark framework deformation history."""
    scale_value = analysis.automatic_dynamic_scale(result) if scale == "auto" else float(scale)
    coordinates = analysis.node_coordinates()
    frame_indices = np.arange(0, len(result.time), frame_step, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 6))
    for start, end in analysis.connectivity:
        ax.plot(coordinates[[start, end], 0], coordinates[[start, end], 1], "k--", linewidth=1)

    lines = []
    for start, end in analysis.connectivity:
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
        deformed = analysis.deformed_coordinates(result.displacement[frame_index], scale=scale_value)
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
        analysis,
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
    analysis.plot_dynamic_node_response(
        result,
        node=response_node,
        component=response_component,
        save_path=paths["node_response"],
        show=False,
    )
    analysis.plot_dynamic_energy(result, save_path=paths["energy"], show=False)
    analysis.plot_dynamic_snapshots(result, scale=scale, save_path=paths["snapshots"], show=False)
    analysis.animate_dynamic_response(
        result,
        output_path=paths["animation"],
        scale=scale,
        frame_step=frame_step,
    )
    return paths


def plot_first_modes(analysis, result=None, num_modes=5, output_dir="figs/framework_output"):
    result = result or analysis.result or analysis.solve(num_modes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for mode_number in range(1, min(num_modes, len(result.angular_frequencies)) + 1):
        path = output_dir / f"framework_mode_{mode_number}.png"
        analysis.plot_mode_shape(result, mode_number=mode_number, save_path=path, show=False)
        plt.close()
        paths.append(path)
    return paths


def animate_mode(
        analysis,
        result=None,
        mode_number=1,
        output_path="figs/framework_output/framework_mode_1.gif",
        scale="auto",
        periods=2.0,
        frames_per_period=60,
        fps=30,
):
    result = result or analysis.result or analysis.solve(mode_number)
    mode = result.modes_full[:, mode_number - 1]
    omega = result.angular_frequencies[mode_number - 1]
    period = 2.0 * np.pi / omega
    time = np.linspace(0.0, periods * period, int(periods * frames_per_period) + 1)
    scale_value = analysis.automatic_mode_scale(mode) if scale == "auto" else float(scale)

    coordinates = analysis.node_coordinates()
    fig, ax = plt.subplots(figsize=(8, 6))
    for start, end in analysis.connectivity:
        ax.plot(coordinates[[start, end], 0], coordinates[[start, end], 1], "k--", linewidth=1)

    lines = []
    for start, end in analysis.connectivity:
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
        deformed = analysis.deformed_coordinates(np.cos(omega * time[frame_index]) * mode, scale_value)
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
