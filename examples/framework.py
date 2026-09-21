"""Original framework demonstration, separated from reusable library code."""

from pathlib import Path
import sys

# Direct script execution needs the project root on the import path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from beam_fem.framework import FrameworkEigenvalueAnalysis


def main():
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


if __name__ == "__main__":
    main()
