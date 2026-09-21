"""Numerical snapshots used to check the file-layout refactor."""

import argparse
import contextlib
import dataclasses
import importlib
import io
import json
from pathlib import Path
import sys
import subprocess
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def snapshot(legacy=False, reference=None):
    if reference:
        for name in ("my_plot_fun", "constraints", "beam_loads", "static_beam", "extended_matrix", "newmark", "eigenvalue", "framework"):
            source = subprocess.check_output(
                ["git", "show", f"{reference}:code/{name}.py"], cwd=ROOT, text=True, encoding="utf-8",
            )
            module = types.ModuleType(name)
            module.__file__ = str(ROOT / "code" / (name + ".py"))
            sys.modules[name] = module
            exec(compile(source, module.__file__, "exec"), module.__dict__)
        legacy = True
    sys.path.insert(0, str(ROOT / "code" if legacy else ROOT))
    prefix = "" if legacy else "beam_fem."
    modules = {
        name: importlib.import_module(prefix + name)
        for name in ("static_beam", "extended_matrix", "newmark", "eigenvalue", "framework", "beam_loads")
    }
    values = {}

    def record(name, value):
        if dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                record(name + "." + field.name, getattr(value, field.name))
        else:
            values[name] = np.asarray(value).tolist()

    static = modules["static_beam"]
    params = static.BeamParameters(num_elements=4)
    for name, matrix in zip(("stiffness", "mass"), static.FiniteElementMatrices(params).assemble_global_matrices()):
        record(name, matrix)
    cases = [("uniform", {"q0": 10.0}), ("point_load_end", {"P": 1000.0}), ("end_moment", {"M": 1000.0})]
    for bc in ("cantilever", "simply_supported"):
        for case, kwargs in cases:
            tag = bc + "." + case
            beam = static.BeamAnalysis(beam_type=bc, load_case=case, **kwargs)
            beam.solve()
            record(tag + ".static", beam.full_dof)
            extended = modules["extended_matrix"].BeamFEM(load_case=case, **kwargs)
            u, multipliers = extended.solve_static_extended(bc)
            record(tag + ".extended", u)
            record(tag + ".multipliers", multipliers)
            dynamic = modules["newmark"].NewmarkBeamAnalysis(
                beam_type=bc, load_case=case, total_time=0.002, time_step=0.0005,
                damping_mass=0.01, damping_stiffness=1e-6,
                load_time_function=lambda t: 1.0 + t, **kwargs,
            )
            record(tag + ".newmark", dynamic.solve())
        modal = modules["eigenvalue"].EigenvalueBeamAnalysis(beam_type=bc, num_elements=8, num_modes=4)
        record(bc + ".modes", modal.solve())
        record(bc + ".free", modal.free_vibration_response(total_time=0.002, time_step=0.0005))
        record(bc + ".standing", modal.standing_wave_response(frames_per_period=4, periods=1))
        record(bc + ".analytic", modal.analytic_frequencies())
    for fixed_rotation in (False, True):
        frame = modules["framework"].FrameworkEigenvalueAnalysis(constrain_p4_rotation=fixed_rotation)
        tag = "framework." + str(fixed_rotation)
        record(tag + ".static", frame.solve_static_point_load())
        record(tag + ".modes", frame.solve())
        record(tag + ".free", frame.free_vibration_response(total_time=0.002, time_step=0.0005))
        record(tag + ".release", frame.solve_newmark_release_from_point_load(total_time=0.002, time_step=0.0005))
        record(tag + ".forced", frame.solve_newmark_point_load(
            total_time=0.002, time_step=0.0005, damping_mass=0.01,
            damping_stiffness=1e-6, load_time_function=lambda t: 1.0 + t,
        ))
    loads = modules["beam_loads"]
    record("interior_point_load", loads.point_load_vector(params.node_positions, 123.0, 3.2))
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--legacy", action="store_true", help="Use the compatibility imports")
    mode.add_argument("--reference", help="Read the original flat modules from a Git revision")
    mode.add_argument("--compare-ref", help="Compare the package with an original Git revision")
    args = parser.parse_args()
    if args.compare_ref:
        original = subprocess.check_output([sys.executable, __file__, "--reference", args.compare_ref], text=True)
        current = subprocess.check_output([sys.executable, __file__], text=True)
        before, after = json.loads(original), json.loads(current)
        assert before.keys() == after.keys()
        for name in before:
            np.testing.assert_array_equal(before[name], after[name], err_msg=name)
        print(f"All {len(before)} numerical arrays/scalars exactly match {args.compare_ref}.")
        sys.exit(0)
    with contextlib.redirect_stdout(io.StringIO()):
        values = snapshot(legacy=args.legacy, reference=args.reference)
    print(json.dumps(values))
