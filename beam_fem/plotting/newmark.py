"""Visualization functions for NewmarkBeamAnalysis."""

import matplotlib.pyplot as plt

from beam_fem.beam_loads import load_case_label


def plot_tip_response(analysis, result=None):
    result = result or analysis.result
    if result is None:
        result = analysis.solve()

    plt.figure(figsize=(10, 6))
    plt.plot(result.time, result.tip_displacement, label="Tip displacement")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title(
        f"Newmark dynamic response - "
        f"{load_case_label(analysis.load_case, **analysis.load_params)}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
