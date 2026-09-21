"""Visualization functions for BeamFEM."""

from matplotlib import pyplot as plt

from beam_fem.beam_loads import load_case_label
from beam_fem.plotting.beam import plot_piecewise_polynomial as draw_beam_shape


def plot_piecewise_polynomial(analysis):
    if analysis.u is None:
        analysis.solve_static_extended()

    plt.figure(figsize=(10, 6))
    draw_beam_shape(analysis.u, analysis.node_positions)
    plt.plot(analysis.node_positions, analysis.u[::2], "ro", markersize=4, label="Nodes")
    plt.xlabel("Position along beam (m)")
    plt.ylabel("Displacement (m)")
    plt.title(f"Left-clamped beam - {load_case_label(analysis.load_case, **analysis.load_params)}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
