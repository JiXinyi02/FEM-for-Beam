"""Original newmark_visualization demonstration, separated from reusable library code."""

from pathlib import Path
import sys

# Direct script execution needs the project root on the import path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beam_fem.newmark import NewmarkBeamAnalysis
from beam_fem.plotting.newmark_visualization import create_all_visualizations


def main():
    analysis = NewmarkBeamAnalysis(
        load_case="point_load_end",
        P=1000.0,
        total_time=0.05,
        time_step=5e-4,
    )
    result = analysis.solve()
    saved_paths = create_all_visualizations(
        analysis,
        result=result,
        output_dir="newmark_visual_output",
        prefix="point_load_end",
    )

    print("Saved Newmark visualizations:")
    for name, path in saved_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
