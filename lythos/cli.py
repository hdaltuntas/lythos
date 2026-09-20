"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="lythos",
        description="2D finite element analysis for geotechnical engineering.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="analyse a model file and write a report")
    run.add_argument("model", help="path to a .json model file")
    run.add_argument("-o", "--out", default="out", help="output directory")
    run.add_argument("--mesh-size", type=float, default=None, help="target element size [m]")
    run.add_argument("--tolerance", type=float, default=3e-3, help="convergence tolerance")
    run.add_argument("--quiet", action="store_true")

    mesh = sub.add_parser("mesh", help="mesh a model and report its statistics")
    mesh.add_argument("model")
    mesh.add_argument("-o", "--out", default=None, help="write a mesh image here")
    mesh.add_argument("--mesh-size", type=float, default=None)

    gui = sub.add_parser("gui", help="start the interactive interface in a browser")
    gui.add_argument("--port", type=int, default=8777)
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--no-browser", action="store_true")

    examples = sub.add_parser("examples", help="write the built-in example models")
    examples.add_argument("-o", "--out", default="examples/models")

    args = parser.parse_args(argv)

    if args.command == "run":
        from .core.serialize import load_model
        from .report import run_and_report
        model = load_model(args.model)
        summary = run_and_report(model, out_dir=args.out, mesh_size=args.mesh_size,
                                 tolerance=args.tolerance, verbose=not args.quiet)
        worst = [s.get("factor_of_safety") for s in summary["stages"]
                 if s.get("factor_of_safety") is not None]
        if worst and not args.quiet:
            print(f"lowest factor of safety: {min(worst):.3f}")
        return 0 if all(s["converged"] for s in summary["stages"]) else 1

    if args.command == "mesh":
        from .core.serialize import load_model
        model = load_model(args.model)
        problem = model.build(mesh_size=args.mesh_size)
        areas = problem.continuum.volumes()
        print(f"elements: {problem.continuum.n_elements}")
        print(f"nodes:    {problem.mesh.n_nodes}")
        print(f"dofs:     {problem.dofs.n_dof}")
        print(f"area:     {areas.sum():.2f} m2   "
              f"(smallest element {areas.min():.4f}, largest {areas.max():.4f})")
        for structure in problem.structures:
            print(f"structure {structure.name!r}: {structure.beam.n_elements} beam elements, "
                  f"{len(structure.interfaces)} interface set(s)")
        if args.out:
            from .viz import plots
            plots.save(plots.plot_mesh(problem), args.out)
            print(f"mesh image written to {args.out}")
        return 0

    if args.command == "gui":
        from .gui.server import serve
        serve(host=args.host, port=args.port, open_browser=not args.no_browser)
        return 0

    if args.command == "examples":
        from .examples import write_examples
        paths = write_examples(args.out)
        for path in paths:
            print(path)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
