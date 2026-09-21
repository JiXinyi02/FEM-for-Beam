from pathlib import Path

from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np

from beam_fem.interpolation import form, sample_beam_shape


def plot_piecewise_polynomial(u: np.ndarray, x_nodes: np.ndarray) -> None:
    x_all, w_all = sample_beam_shape(u, x_nodes, points_per_element=50)
    plt.plot(x_all, w_all)


def automatic_visual_scale(result, x_nodes, target_fraction=0.15):
    """Choose a vertical scale that makes small physical displacements visible."""
    max_displacement = np.max(np.abs(result.displacement[:, ::2]))
    if max_displacement == 0.0:
        return 1.0

    beam_length = x_nodes[-1] - x_nodes[0]
    return target_fraction * beam_length / max_displacement


def resolve_scale(result, x_nodes, scale):
    if scale == "auto":
        return automatic_visual_scale(result, x_nodes)
    return float(scale)


def plot_tip_displacement(result, ax=None, save_path=None, show=True):
    """Plot free-end displacement versus time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(result.time, result.tip_displacement, color="tab:blue", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tip displacement (m)")
    ax.set_title("Tip displacement over time")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def plot_energy(result, ax=None, save_path=None, show=True):
    """Plot total mechanical energy versus time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(result.time, result.energy, color="tab:red", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    ax.set_title("Energy over time")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def plot_beam_snapshots(
        result,
        x_nodes,
        n_snapshots=6,
        frame_indices=None,
        snapshot_time_step=None,
        time_start=None,
        time_end=None,
        scale="auto",
        points_per_element=40,
        ax=None,
        save_path=None,
        show=True,
):
    """
    Plot several deformed beam shapes from a dynamic result.

    If snapshot_time_step is given, snapshots are selected at fixed physical
    times start, start + dt, start + 2dt, ... instead of splitting the whole
    duration into evenly spaced frame indices.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    scale_value = resolve_scale(result, x_nodes, scale)
    if frame_indices is None:
        time = np.asarray(result.time)
        start = time[0] if time_start is None else float(time_start)
        end = time[-1] if time_end is None else float(time_end)
        if end < start:
            raise ValueError("time_end must be greater than or equal to time_start.")

        if snapshot_time_step is None:
            snapshot_times = np.linspace(start, end, n_snapshots)
        else:
            if snapshot_time_step <= 0.0:
                raise ValueError("snapshot_time_step must be positive.")
            snapshot_times = np.arange(start, end + 0.5 * snapshot_time_step, snapshot_time_step)
            if len(snapshot_times) == 0 or snapshot_times[-1] < end:
                snapshot_times = np.append(snapshot_times, end)

        frame_indices = np.searchsorted(time, snapshot_times, side="left")
        frame_indices = np.clip(frame_indices, 0, len(time) - 1)
        left_indices = np.maximum(frame_indices - 1, 0)
        use_left = (
            np.abs(time[left_indices] - snapshot_times)
            < np.abs(time[frame_indices] - snapshot_times)
        )
        frame_indices[use_left] = left_indices[use_left]
        frame_indices = np.unique(frame_indices)

    ax.axhline(0.0, color="0.2", linestyle="--", linewidth=1, label="Undeformed axis")

    for frame_index in frame_indices:
        x_curve, w_curve = sample_beam_shape(
            result.displacement[frame_index],
            x_nodes,
            points_per_element=points_per_element,
            displacement_scale=scale_value,
        )
        ax.plot(x_curve, w_curve, linewidth=1.8, label=f"t = {result.time[frame_index]:.4f} s")

    ax.set_xlabel("Position along beam (m)")
    ax.set_ylabel("Transverse displacement (scaled)")
    ax.set_title(f"Beam snapshots, visualization scale = {scale_value:.3e}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def animate_beam_motion(
        result,
        x_nodes,
        output_path="beam_motion.gif",
        scale="auto",
        points_per_element=40,
        frame_step=1,
        interval=30,
        fps=30,
        show=False,
):
    """Create and optionally save an animation of the beam motion."""
    scale_value = resolve_scale(result, x_nodes, scale)
    frame_indices = np.arange(0, len(result.time), frame_step, dtype=int)
    x_curve, first_shape = sample_beam_shape(
        result.displacement[frame_indices[0]],
        x_nodes,
        points_per_element=points_per_element,
        displacement_scale=scale_value,
    )

    all_shapes = [
        sample_beam_shape(
            result.displacement[frame_index],
            x_nodes,
            points_per_element=points_per_element,
            displacement_scale=scale_value,
        )[1]
        for frame_index in frame_indices
    ]
    y_min = min(np.min(shape) for shape in all_shapes)
    y_max = max(np.max(shape) for shape in all_shapes)
    y_pad = max(0.05 * (x_nodes[-1] - x_nodes[0]), 0.1 * (y_max - y_min), 1e-12)

    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot(x_curve, first_shape, color="tab:blue", linewidth=2)
    nodes = ax.scatter(
        x_nodes,
        scale_value * result.displacement[frame_indices[0], ::2],
        color="tab:orange",
        s=25,
        zorder=3,
    )
    time_text = ax.text(0.02, 0.94, "", transform=ax.transAxes)

    ax.axhline(0.0, color="0.2", linestyle="--", linewidth=1)
    ax.set_xlim(x_nodes[0], x_nodes[-1])
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("Position along beam (m)")
    ax.set_ylabel("Transverse displacement (scaled)")
    ax.set_title(f"Beam motion, visualization scale = {scale_value:.3e}")
    ax.grid(True, alpha=0.3)

    def update(frame_number):
        frame_index = frame_indices[frame_number]
        line.set_ydata(all_shapes[frame_number])
        nodes.set_offsets(
            np.column_stack((
                x_nodes,
                scale_value * result.displacement[frame_index, ::2],
            ))
        )
        time_text.set_text(f"t = {result.time[frame_index]:.4f} s")
        return line, nodes, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=interval,
        blit=True,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()

        if suffix == ".gif":
            anim.save(output_path, writer=animation.PillowWriter(fps=fps))
        elif suffix == ".mp4":
            if "ffmpeg" not in animation.writers.list():
                raise RuntimeError("MP4 output needs ffmpeg. Use .gif or install ffmpeg.")
            anim.save(output_path, writer=animation.FFMpegWriter(fps=fps))
        elif suffix == ".html":
            anim.save(output_path, writer=animation.HTMLWriter(fps=fps))
        else:
            raise ValueError("Use an output path ending in .gif, .mp4, or .html.")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return anim


def normalized_mode(dof_vector):
    """Scale a mode so its largest displacement DOF has absolute value 1."""
    mode = np.asarray(dof_vector, dtype=float).copy()
    max_displacement = np.max(np.abs(mode[::2]))
    if max_displacement != 0.0:
        mode /= max_displacement
    return mode


def aligned_analytic_mode(analysis, numerical_mode, mode_number):
    """Return analytic mode with sign aligned to the numerical mode."""
    analytic_mode = normalized_mode(analysis.analytic_mode_dofs(mode_number))
    if numerical_mode[::2] @ analytic_mode[::2] < 0.0:
        analytic_mode *= -1.0
    return analytic_mode


def plot_mode_shape_comparison(
        analysis,
        result,
        num_modes=5,
        save_path=None,
        show=True,
):
    """Plot FEM and analytic mode shapes side by side for the first modes."""
    num_modes = min(num_modes, len(result.angular_frequencies))
    x_nodes = analysis.beam_params.node_positions
    n_cols = 2
    n_rows = int(np.ceil(num_modes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.2 * n_rows), squeeze=False)

    for mode_index in range(num_modes):
        ax = axes[mode_index // n_cols, mode_index % n_cols]
        numerical_mode = normalized_mode(result.modes_full[:, mode_index])
        analytic_mode = aligned_analytic_mode(analysis, numerical_mode, mode_index + 1)

        x_fem, w_fem = sample_beam_shape(numerical_mode, x_nodes)
        x_exact, w_exact = sample_beam_shape(analytic_mode, x_nodes)
        ax.plot(x_fem, w_fem, linewidth=2, label="FEM")
        ax.plot(x_exact, w_exact, "--", linewidth=1.8, label="Analytic")
        ax.plot(x_nodes, numerical_mode[::2], "o", markersize=3, label="FEM nodes")
        ax.axhline(0.0, color="0.25", linewidth=0.8)
        ax.set_title(f"Mode {mode_index + 1}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("normalized w")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for empty_index in range(num_modes, n_rows * n_cols):
        axes[empty_index // n_cols, empty_index % n_cols].axis("off")

    fig.suptitle(
        f"{analysis.beam_type.replace('_', ' ').title()} beam: analytic vs FEM modes",
        fontsize=14,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_frequency_errors(comparison, save_path=None, show=True):
    """Plot FEM natural-frequency relative errors against analytic values."""
    modes = comparison["mode"]
    errors_percent = 100.0 * comparison["frequency_relative_error"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(modes, errors_percent, "o-", linewidth=2)
    ax.set_xlabel("Mode number")
    ax.set_ylabel("frequency relative error (%)")
    ax.set_title("Eigenfrequency error: FEM vs analytic")
    ax.set_xticks(modes)
    ax.grid(True, which="both", alpha=0.3)

    for mode, error in zip(modes, errors_percent):
        ax.annotate(
            f"{error:.2e}%",
            (mode, error),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def save_frequency_error_table(comparison, save_path):
    """Write a CSV table with FEM/analytic frequencies and relative errors."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack((
        comparison["mode"],
        comparison["omega_fem"],
        comparison["omega_exact"],
        comparison["omega_relative_error"],
        comparison["frequency_fem"],
        comparison["frequency_exact"],
        comparison["frequency_relative_error"],
    ))
    header = (
        "mode,omega_fem_rad_s,omega_analytic_rad_s,omega_relative_error,"
        "frequency_fem_hz,frequency_analytic_hz,frequency_relative_error"
    )
    np.savetxt(save_path, table, delimiter=",", header=header, comments="", fmt="%.12e")
    return save_path


def plot_first_eigenmodes(
        analysis,
        result,
        num_modes=5,
        save_path=None,
        show=True,
):
    """Plot the first numerical eigenmodes in one stacked figure."""
    num_modes = min(num_modes, len(result.angular_frequencies))
    x_nodes = analysis.beam_params.node_positions
    fig, axes = plt.subplots(num_modes, 1, figsize=(10, 2.2 * num_modes), sharex=True)
    axes = np.atleast_1d(axes)

    for mode_index, ax in enumerate(axes[:num_modes]):
        mode = normalized_mode(result.modes_full[:, mode_index])
        x_curve, w_curve = sample_beam_shape(mode, x_nodes)
        ax.plot(x_curve, w_curve, linewidth=2)
        ax.plot(x_nodes, mode[::2], "o", markersize=3)
        ax.axhline(0.0, color="0.25", linewidth=0.8)
        ax.set_ylabel(f"mode {mode_index + 1}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Position along beam (m)")
    fig.suptitle(f"First {num_modes} FEM eigenmodes", fontsize=14)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


"""
Validation using the original function:
y = sinx
"""
