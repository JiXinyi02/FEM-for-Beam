"""Original my_plot_fun demonstration, separated from reusable library code."""

from pathlib import Path
import sys

# Direct script execution needs the project root on the import path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matplotlib import pyplot as plt
import numpy as np

from beam_fem.plotting.beam import (
    form,
    sample_beam_shape,
    plot_piecewise_polynomial,
    automatic_visual_scale,
    resolve_scale,
    plot_tip_displacement,
    plot_energy,
    plot_beam_snapshots,
    animate_beam_motion,
    normalized_mode,
    aligned_analytic_mode,
    plot_mode_shape_comparison,
    plot_frequency_errors,
    save_frequency_error_table,
    plot_first_eigenmodes,
)


def main():
    n = 10
    x_nodes = np.linspace(0, 2 * np.pi, n)

    # build the vector u with length 2*n:
    u = np.zeros(2*n)
    for i in range(n):
        u[2 * i] = np.sin(x_nodes[i])
        u[2 * i + 1] = np.cos(x_nodes[i])

    print(x_nodes)
    print(u)
    plot_piecewise_polynomial(u, x_nodes)
    plt.show()


if __name__ == "__main__":
    main()
