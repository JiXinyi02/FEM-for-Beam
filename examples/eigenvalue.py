"""Original eigenvalue demonstration, separated from reusable library code."""

from pathlib import Path
import sys

# Direct script execution needs the project root on the import path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beam_fem.eigenvalue import EigenvalueBeamAnalysis


def main():
    NUM_MODES = 5
    OUTPUT_DIR = "figs/eigenvalue_visual_output"

    analysis = EigenvalueBeamAnalysis(beam_type="cantilever", num_modes=NUM_MODES)
    eigen_result = analysis.solve()
    analysis.print_frequency_table(eigen_result)

    saved_paths = analysis.save_modal_analysis_outputs(
        result=eigen_result,
        output_dir=OUTPUT_DIR,
        num_modes=NUM_MODES,
        standing_wave_mode=1,
        video_extension=".gif",
    )

    print("\nSaved eigenvalue visualizations:")
    for name, path in saved_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
