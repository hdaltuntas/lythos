"""A complete Lythos analysis, written out step by step for Thonny.

Press the green Run button.  It builds a cut slope, finds its factor of
safety, writes the figures and an HTML report next to this file, and opens
the report in a browser.

Everything the command line does is available here as ordinary Python, which
is the point: change the numbers below and run it again.

    pip install lythosfea
"""

import os
import webbrowser

try:
    from lythos.core.materials import MohrCoulomb
    from lythos.core.model import Model, SoilLayer, Stage, WaterTable
    from lythos.core.solver import Solver
    from lythos.report import run_and_report
    from lythos.viz import plots
except ImportError:
    raise SystemExit(
        "Lythos is not installed in the interpreter Thonny is using.\n"
        "In Thonny: Tools > Manage packages... > search 'lythosfea' > Install")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "lythos_out")


# ---------------------------------------------------------------- the ground
# Everything is in kN, m and kPa.  Compression is negative in the stress
# output; c' and phi' are effective strength, psi is dilatancy.
soil = MohrCoulomb(
    name="silty clay",
    E=1.0e5,        # Young's modulus [kPa]
    nu=0.3,         # Poisson's ratio
    c=10.0,         # effective cohesion c' [kPa]
    phi=20.0,       # effective friction angle phi' [degrees]
    psi=0.0,        # dilatancy angle [degrees]
    gamma=20.0,     # unit weight above the water table [kN/m3]
    gamma_sat=21.0,  # saturated unit weight [kN/m3]
)

# --------------------------------------------------------------- the geometry
# A 10 m high cut at 1 vertical to 2 horizontal, on a rigid base at toe level.
# Points go anticlockwise: (x, y) in metres.
outline = [(5, 0), (40, 0), (40, 10), (25, 10)]

model = Model(
    name="cut slope",
    layers=[SoilLayer("silty clay", outline, soil, mesh_size=2.0)],
    water=WaterTable([(5, 2.0), (40, 2.0)]),   # phreatic surface, or leave it out
    stages=[
        Stage("1 - self weight", kind="initial", increments=10),
        Stage("2 - factor of safety", kind="ssr", srf_min=0.8, srf_max=2.5),
    ],
    mesh_size=2.0,
    initial_stress="gravity",   # "k0" for level ground, "gravity" for a slope
)


def main() -> None:
    problems = model.validate()
    if problems:
        print("The model is not ready:")
        for problem in problems:
            print("   ", problem)
        return

    print(f"Analysing {model.name!r}. The safety analysis takes a minute or so.\n")

    # run_and_report meshes, solves every stage, draws the figures and writes
    # a single self-contained HTML report.
    summary = run_and_report(model, out_dir=OUT, verbose=True)

    print("\n--- results " + "-" * 50)
    for stage in summary["stages"]:
        line = (f"{stage['stage']:<26} "
                f"{'converged' if stage['converged'] else 'DID NOT CONVERGE':<18}"
                f"max displacement {stage['max_displacement_mm']:7.2f} mm")
        if stage.get("factor_of_safety") is not None:
            line += f"   factor of safety {stage['factor_of_safety']:.3f}"
        print(line)

    report = os.path.join(OUT, "report.html")
    print(f"\nFigures and report written to {OUT}")
    try:
        webbrowser.open("file://" + report)
    except Exception:
        print(f"Open {report} to see them.")


def without_the_report() -> None:
    """The same analysis, driven directly, if you want the numbers only."""
    problem = model.build()
    print(f"{problem.continuum.n_elements} elements, {problem.dofs.n_dof} unknowns")

    solver = Solver(problem, tolerance=2.0e-3)
    results = solver.run()
    print("factor of safety:", results[-1].srf)

    # any figure can be saved on its own
    plots.save(plots.plot_field(problem, results[-1], "deviatoric_strain"),
               os.path.join(OUT, "slip_surface.png"))


if __name__ == "__main__":
    main()
