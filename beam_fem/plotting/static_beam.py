"""Visualization functions for BeamAnalysis."""

import matplotlib.pyplot as plt

from beam_fem.beam_loads import load_case_label
from beam_fem.plotting.beam import plot_piecewise_polynomial


def plot_results(analysis):
    """Plot displacement using Hermite interpolation from my_plot_fun."""
    if analysis.full_dof is None:
        print("No results to plot. Run solve() first.")
        return

    plt.figure(figsize=(10, 6))
    plot_piecewise_polynomial(analysis.full_dof, analysis.beam_params.node_positions)
    # plt.gca().get_lines()[0].set_label('Displacement')
    # plt.plot(
    #     self.beam_params.node_positions,
    #     self.full_dof[::2],
    #     'ro',
    #     markersize=6,
    #     label='Nodes',
    # )

    plt.xlabel('Position along beam (m)', fontsize=12)
    plt.ylabel('Displacement (m)', fontsize=12)
    plt.title(
        f'{analysis.beam_type.replace("_", " ").title()} Beam - '
        f'{load_case_label(analysis.load_case, **analysis.load_params)}',
        fontsize=14,
    )
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
