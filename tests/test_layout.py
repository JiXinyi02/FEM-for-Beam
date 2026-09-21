"""Public API, compatibility imports, and visualization smoke tests."""

import ast
import contextlib
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

# Direct script execution puts tests/, not the project root, on sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from beam_fem.eigenvalue import EigenvalueBeamAnalysis
from beam_fem.extended_matrix import BeamFEM
from beam_fem.framework import FrameworkEigenvalueAnalysis
from beam_fem.newmark import NewmarkBeamAnalysis
from beam_fem.plotting.newmark_visualization import create_all_visualizations


class LayoutTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_legacy_imports_share_the_same_classes(self):
        sys.path.insert(0, str(ROOT / "code"))
        try:
            pairs = (
                ("static_beam", "BeamAnalysis"), ("static_beam", "BeamParameters"),
                ("static_beam", "FiniteElementMatrices"), ("static_beam", "BoundaryConditions"),
                ("static_beam", "LoadApplication"), ("framework", "FrameworkNode"),
                ("framework", "FrameworkEigenvalueAnalysis"), ("newmark", "NewmarkResult"),
                ("eigenvalue", "EigenvalueBeamAnalysis"), ("extended_matrix", "BeamFEM"),
            )
            for module, name in pairs:
                with self.subTest(module=module, name=name):
                    self.assertIs(getattr(importlib.import_module(module), name), getattr(importlib.import_module("beam_fem." + module), name))
            self.assertTrue(callable(importlib.import_module("beam_loads")._element_load_uniform))
        finally:
            sys.path.remove(str(ROOT / "code"))

    def test_library_has_no_demo_entry_points(self):
        for path in (ROOT / "beam_fem").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            guards = [n for n in tree.body if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test)]
            self.assertEqual(guards, [], str(path))

    def test_examples_import_without_running(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
            command = "import importlib; from pathlib import Path; " + "; ".join(
                f"importlib.import_module('examples.{p.stem}')"
                for p in (ROOT / "examples").glob("*.py") if p.stem != "__init__"
            ) + "; assert not list(Path('.').iterdir())"
            completed = subprocess.run([sys.executable, "-c", command], cwd=directory, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "")

    def test_static_entry_point_both_old_and_new(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
            commands = [[sys.executable, "-m", "examples.static_beam"], [sys.executable, str(ROOT / "code" / "static_beam.py")]]
            outputs = []
            for command in commands:
                completed = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout.count("Relative error at tip:"), 3)
                outputs.append(completed.stdout)
            self.assertEqual(*outputs)

    def test_all_examples_load_by_path_without_pythonpath(self):
        env = dict(os.environ, MPLBACKEND="Agg")
        env.pop("PYTHONPATH", None)
        command = (
            "import runpy, sys; from pathlib import Path; "
            "namespace = runpy.run_path(sys.argv[1]); "
            "assert callable(namespace['main']); "
            "import beam_fem; "
            "assert Path(beam_fem.__file__).resolve().parent == "
            "Path(sys.argv[1]).resolve().parents[1] / 'beam_fem'"
        )
        with tempfile.TemporaryDirectory() as directory:
            for cwd in (ROOT / "examples", Path(directory)):
                for path in sorted((ROOT / "examples").glob("*.py")):
                    if path.name == "__init__.py":
                        continue
                    with self.subTest(example=path.name, cwd=str(cwd)):
                        completed = subprocess.run(
                            [sys.executable, "-c", command, str(path)],
                            cwd=cwd, env=env, capture_output=True, text=True, timeout=60,
                        )
                        self.assertEqual(completed.returncode, 0, completed.stderr)
                        self.assertEqual(completed.stdout, "")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_eigenvalue_direct_script_without_pythonpath(self):
        env = dict(os.environ, MPLBACKEND="Agg")
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(ROOT / "examples" / "eigenvalue.py")],
                cwd=directory, env=env, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Saved eigenvalue visualizations:", completed.stdout)
            output = Path(directory) / "figs" / "eigenvalue_visual_output"
            self.assertTrue((output / "cantilever_frequency_errors.csv").is_file())
            self.assert_image(output / "cantilever_mode_shape_comparison.png")
            self.assert_image(output / "cantilever_mode_1_standing_wave_animation.gif", animated=True)

    def assert_image(self, path, animated=False):
        self.assertTrue(Path(path).is_file(), str(path))
        with Image.open(path) as image:
            self.assertGreater(image.width, 100)
            self.assertGreater(np.asarray(image.convert("RGB")).std(), 1)
            if animated:
                self.assertGreater(image.n_frames, 1)

    def test_framework_plots_and_animations(self):
        solver = FrameworkEigenvalueAnalysis(num_modes=3)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            _, paths = solver.save_static_point_load_visualizations(output_dir=directory)
            for path in paths.values():
                self.assert_image(path)
            result = solver.solve()
            for path in solver.plot_first_modes(result, num_modes=2, output_dir=directory):
                self.assert_image(path)
            self.assert_image(solver.animate_mode(result, output_path=directory / "mode.gif", periods=1, frames_per_period=4), animated=True)
            response = solver.solve_newmark_release_from_point_load(total_time=0.001, time_step=0.0002)
            for name, path in solver.save_newmark_visualizations(response, output_dir=directory, frame_step=1).items():
                self.assert_image(path, animated=name == "animation")

    def test_beam_plots_and_animations(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            modal = EigenvalueBeamAnalysis(num_elements=8, num_modes=3)
            result = modal.solve()
            modal.plot_mode_shape_comparison(result, num_modes=2, save_path=directory / "comparison.png", show=False)
            modal.plot_frequency_errors(result, num_modes=2, save_path=directory / "errors.png", show=False)
            modal.plot_first_eigenmodes(result, num_modes=2, save_path=directory / "modes.png", show=False)
            modal.save_frequency_error_table(result, num_modes=2, save_path=directory / "errors.csv")
            self.assertTrue((directory / "errors.csv").is_file())
            for filename in ("comparison.png", "errors.png", "modes.png"):
                self.assert_image(directory / filename)
            paths = modal.save_standing_wave_visualizations(result, output_dir=directory, periods=1, frames_per_period=4)
            for name, path in paths.items():
                self.assert_image(path, animated=name == "animation")
            dynamic = NewmarkBeamAnalysis(load_case="point_load_end", P=1000, total_time=0.002, time_step=0.0005)
            for name, path in create_all_visualizations(dynamic, output_dir=directory).items():
                self.assert_image(path, animated=name == "animation")
            with contextlib.redirect_stdout(io.StringIO()):
                dynamic.plot_tip_response()
                modal.plot_mode_shapes(result, num_modes=2)
                modal.plot_tip_response(modal.free_vibration_response(total_time=0.001))

    def test_extended_matrix_plot_delegation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            analysis = BeamFEM(load_case="uniform", q0=10.0)
            analysis.solve_static_extended()
            analysis.plot_piecewise_polynomial()
        self.assertEqual(len(plt.gca().lines), 2)


if __name__ == "__main__":
    unittest.main()
