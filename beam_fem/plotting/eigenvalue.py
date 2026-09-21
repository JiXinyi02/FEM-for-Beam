"""Visualization functions for EigenvalueBeamAnalysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beam_fem.plotting.beam import (
    animate_beam_motion,
    plot_beam_snapshots,
    plot_energy,
    plot_first_eigenmodes as draw_first_eigenmodes,
    plot_frequency_errors as draw_frequency_errors,
    plot_mode_shape_comparison as draw_mode_shape_comparison,
    plot_tip_displacement,
    sample_beam_shape,
    save_frequency_error_table as write_frequency_error_table,
)


def plot_mode_shapes(analysis, result=None, num_modes=None, include_analytic=True):
    result = result or analysis.result or analysis.solve(num_modes)
    num_modes = min(num_modes or len(result.angular_frequencies), len(result.angular_frequencies))

    plt.figure(figsize=(10, 6))
    for mode_index in range(num_modes):
        mode = result.modes_full[:, mode_index]
        mode = mode / np.max(np.abs(mode[::2]))
        x_curve, w_curve = sample_beam_shape(mode, analysis.beam_params.node_positions)
        plt.plot(x_curve, w_curve, label=f"FEM mode {mode_index + 1}")

        if include_analytic:
            analytic_mode = analysis.analytic_mode_dofs(mode_index + 1)
            x_exact, w_exact = sample_beam_shape(analytic_mode, analysis.beam_params.node_positions)
            plt.plot(x_exact, w_exact, "--", alpha=0.7, label=f"Analytic mode {mode_index + 1}")

    plt.axhline(0.0, color="0.2", linewidth=1)
    plt.xlabel("Position along beam (m)")
    plt.ylabel("Normalized mode shape")
    plt.title(f"{analysis.beam_type.replace('_', ' ').title()} beam mode shapes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_tip_response(analysis, response):
    plt.figure(figsize=(10, 5))
    plt.plot(response.time, response.tip_displacement, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Tip displacement (m)")
    plt.title("Free-vibration tip response by modal superposition")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_mode_shape_comparison(
        analysis,
        result=None,
        num_modes=5,
        save_path=None,
        show=True,
):
    """Plot FEM and analytic mode shapes side by side for the first modes."""
    result = result or analysis.result or analysis.solve(num_modes)
    return draw_mode_shape_comparison(
        analysis,
        result,
        num_modes=num_modes,
        save_path=save_path,
        show=show,
    )


def plot_frequency_errors(
        analysis,
        result=None,
        num_modes=5,
        save_path=None,
        show=True,
):
    """Plot FEM natural-frequency relative errors against analytic values."""
    result = result or analysis.result or analysis.solve(num_modes)
    comparison = analysis.compare_with_analytic(result, num_modes)
    return draw_frequency_errors(comparison, save_path=save_path, show=show)


def save_frequency_error_table(analysis, result=None, num_modes=5, save_path=None):
    """Write a CSV table with FEM/analytic frequencies and relative errors."""
    result = result or analysis.result or analysis.solve(num_modes)
    comparison = analysis.compare_with_analytic(result, num_modes)
    if save_path is None:
        save_path = Path("figs/eigenvalue_visual_output/frequency_errors.csv")
    return write_frequency_error_table(comparison, save_path)


def plot_first_eigenmodes(
        analysis,
        result=None,
        num_modes=5,
        save_path=None,
        show=True,
):
    """Plot the first numerical eigenmodes in one stacked figure."""
    result = result or analysis.result or analysis.solve(num_modes)
    return draw_first_eigenmodes(
        analysis,
        result,
        num_modes=num_modes,
        save_path=save_path,
        show=show,
    )


def save_standing_wave_visualizations(
        analysis,
        result=None,
        output_dir="figs/eigenvalue_visual_output",
        prefix=None,
        mode_number=1,
        amplitude=1e-3,
        periods=2.0,
        frames_per_period=60,
        snapshot_time_step=None,
        scale="auto",
        video_extension=".gif",
        fps=30,
):
    """Save snapshots and an animation for a single standing-wave mode."""
    result = result or analysis._result_with_at_least(mode_number)
    response = analysis.standing_wave_response(
        result=result,
        mode_number=mode_number,
        amplitude=amplitude,
        periods=periods,
        frames_per_period=frames_per_period,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = f"{analysis.beam_type}_mode_{mode_number}_standing_wave"
    if not video_extension.startswith("."):
        video_extension = f".{video_extension}"

    paths = {
        "snapshots": output_dir / f"{prefix}_snapshots.png",
        "animation": output_dir / f"{prefix}_animation{video_extension}",
    }
    x_nodes = analysis.beam_params.node_positions

    period = 2.0 * np.pi / result.angular_frequencies[mode_number - 1]
    if snapshot_time_step is None:
        snapshot_time_step = period / 8.0

    plot_beam_snapshots(
        response,
        x_nodes,
        snapshot_time_step=snapshot_time_step,
        time_start=0.0,
        time_end=period,
        scale=scale,
        save_path=paths["snapshots"],
        show=False,
    )
    plt.close()
    animate_beam_motion(
        response,
        x_nodes,
        output_path=paths["animation"],
        scale=scale,
        frame_step=1,
        fps=fps,
    )
    return paths


def save_modal_analysis_outputs(
        analysis,
        result=None,
        output_dir="figs/eigenvalue_visual_output",
        num_modes=5,
        standing_wave_mode=1,
        video_extension=".gif",
        snapshot_time_step=None,
):
    """Save the standard eigenvalue report figures requested by the project."""
    result = result or analysis.solve(num_modes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = analysis.beam_type

    paths = {
        "mode_shape_comparison": output_dir / f"{prefix}_mode_shape_comparison.png",
        "frequency_errors": output_dir / f"{prefix}_frequency_errors.png",
        "frequency_error_table": output_dir / f"{prefix}_frequency_errors.csv",
        "first_eigenmodes": output_dir / f"{prefix}_first_{num_modes}_eigenmodes.png",
    }

    analysis.plot_mode_shape_comparison(
        result=result,
        num_modes=num_modes,
        save_path=paths["mode_shape_comparison"],
        show=False,
    )
    analysis.plot_frequency_errors(
        result=result,
        num_modes=num_modes,
        save_path=paths["frequency_errors"],
        show=False,
    )
    analysis.save_frequency_error_table(
        result=result,
        num_modes=num_modes,
        save_path=paths["frequency_error_table"],
    )
    analysis.plot_first_eigenmodes(
        result=result,
        num_modes=num_modes,
        save_path=paths["first_eigenmodes"],
        show=False,
    )

    standing_wave_paths = analysis.save_standing_wave_visualizations(
        result=result,
        output_dir=output_dir,
        prefix=f"{prefix}_mode_{standing_wave_mode}_standing_wave",
        mode_number=standing_wave_mode,
        video_extension=video_extension,
        snapshot_time_step=snapshot_time_step,
    )
    paths.update({
        "standing_wave_snapshots": standing_wave_paths["snapshots"],
        "standing_wave_animation": standing_wave_paths["animation"],
    })
    return paths


def save_free_vibration_visualizations(
        analysis,
        response=None,
        output_dir="figs/eigenvalue_visual_output",
        prefix=None,
        scale="auto",
        video_extension=".gif",
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
    response = response or analysis.free_vibration_response()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = f"{analysis.beam_type}_free_vibration"
    if not video_extension.startswith("."):
        video_extension = f".{video_extension}"

    x_nodes = analysis.beam_params.node_positions
    paths = {
        "tip": output_dir / f"{prefix}_tip_displacement.png",
        "energy": output_dir / f"{prefix}_energy.png",
        "snapshots": output_dir / f"{prefix}_beam_snapshots.png",
        "video": output_dir / f"{prefix}_beam_motion{video_extension}",
    }

    plot_tip_displacement(response, save_path=paths["tip"], show=False)
    plt.close()
    plot_energy(response, save_path=paths["energy"], show=False)
    plt.close()
    plot_beam_snapshots(
        response,
        x_nodes,
        scale=scale,
        snapshot_time_step=snapshot_time_step,
        time_start=snapshot_time_start,
        time_end=snapshot_time_end,
        save_path=paths["snapshots"],
        show=False,
    )
    plt.close()
    animate_beam_motion(
        response,
        x_nodes,
        output_path=paths["video"],
        scale=scale,
        frame_step=frame_step,
        fps=fps,
    )

    return paths
