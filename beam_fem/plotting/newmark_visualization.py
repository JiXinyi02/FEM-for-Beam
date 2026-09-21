from pathlib import Path

import matplotlib.pyplot as plt

from beam_fem.plotting.beam import (
    animate_beam_motion,
    plot_beam_snapshots,
    plot_energy,
    plot_tip_displacement,
)


def create_all_visualizations(
        analysis,
        result=None,
        output_dir="newmark_visual_output",
        prefix=None,
        scale="auto",
        snapshot_time_step=None,
        snapshot_time_start=None,
        snapshot_time_end=None,
):
    """Save tip, energy, snapshots, and beam-motion animation for one analysis."""
    result = result or analysis.result or analysis.solve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = analysis.load_case

    x_nodes = analysis.beam_params.node_positions
    paths = {
        "tip": output_dir / f"{prefix}_tip_displacement.png",
        "energy": output_dir / f"{prefix}_energy.png",
        "snapshots": output_dir / f"{prefix}_beam_snapshots.png",
        "animation": output_dir / f"{prefix}_beam_motion.gif",
    }

    plot_tip_displacement(result, save_path=paths["tip"], show=False)
    plt.close()
    plot_energy(result, save_path=paths["energy"], show=False)
    plt.close()
    plot_beam_snapshots(
        result,
        x_nodes,
        scale=scale,
        snapshot_time_step=snapshot_time_step,
        time_start=snapshot_time_start,
        time_end=snapshot_time_end,
        save_path=paths["snapshots"],
        show=False,
    )
    plt.close()
    animate_beam_motion(result, x_nodes, output_path=paths["animation"], scale=scale)

    return paths
