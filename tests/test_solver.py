"""End-to-end checks against closed-form and independent results."""

import numpy as np
import pytest

from lythos.core.materials import LinearElastic, MohrCoulomb
from lythos.core.model import Model, SoilLayer, Stage, WaterTable
from lythos.core.solver import Solver


def _column(E=1.0e4, nu=0.0, gamma=20.0, height=10.0, size=1.5):
    soil = LinearElastic(name="soil", E=E, nu=nu, gamma=gamma, K0=0.0)
    outline = [(0, 0), (10, 0), (10, height), (0, height)]
    return Model(
        name="column",
        layers=[SoilLayer("soil", outline, soil, mesh_size=size)],
        stages=[Stage("gravity", kind="initial", increments=4)],
        initial_stress="gravity",
    )


def test_gravity_gives_the_exact_geostatic_stress_and_settlement():
    """Self weight on a laterally confined elastic column has a closed form."""
    E, gamma, H = 1.0e4, 20.0, 10.0
    model = _column(E=E, gamma=gamma, height=H)
    problem = model.build()
    solver = Solver(problem, tolerance=1e-8)
    result = solver.run_stage(model.resolved_stages()[0])
    assert result.converged

    gauss = problem.continuum.gauss_xy.reshape(-1, 2)
    stress = solver._state.stress
    for depth in (1.0, 5.0, 9.0):
        i = int(np.argmin(np.abs(gauss[:, 1] - (H - depth))))
        actual_depth = H - gauss[i, 1]
        assert stress[i, 1] == pytest.approx(-gamma * actual_depth, rel=1e-6)
        assert abs(stress[i, 0]) < 1e-6 * gamma * H     # nu = 0 -> no lateral stress

    u = solver._u[: 2 * problem.mesh.n_nodes].reshape(-1, 2)
    top = int(np.argmax(problem.mesh.nodes[:, 1]))
    assert u[top, 1] == pytest.approx(-gamma * H * H / (2 * E), rel=1e-6)


def test_buoyancy_reduces_the_effective_stress_by_the_water_pressure():
    gamma, gamma_sat, H = 18.0, 21.0, 10.0
    soil = LinearElastic(name="soil", E=5.0e4, nu=0.0, gamma=gamma,
                         gamma_sat=gamma_sat, K0=0.0)
    outline = [(0, 0), (10, 0), (10, H), (0, H)]
    model = Model(
        name="submerged column",
        layers=[SoilLayer("soil", outline, soil, mesh_size=1.5)],
        water=WaterTable([(0, 5.0), (10, 5.0)]),
        stages=[Stage("gravity", kind="initial", increments=4)],
        initial_stress="gravity",
    )
    problem = model.build()
    solver = Solver(problem, tolerance=1e-8)
    solver.run_stage(model.resolved_stages()[0])
    gauss = problem.continuum.gauss_xy.reshape(-1, 2)
    stress = solver._state.stress
    i = int(np.argmin(np.abs(gauss[:, 1] - 1.0)))
    y = gauss[i, 1]
    expected = -(gamma * (H - 5.0) + (gamma_sat - model.gamma_water) * (5.0 - y))
    # The phreatic surface is resolved at the Gauss points, so an element
    # straddling it carries a slightly wrong weight; that error is second
    # order and shrinks with the mesh, hence the loose tolerance.
    assert stress[i, 1] == pytest.approx(expected, rel=3e-3)


def test_earth_pressure_at_rest_follows_k0():
    k0 = 0.55
    soil = MohrCoulomb(name="soil", E=3.0e4, nu=0.3, c=50.0, phi=25.0, gamma=20.0, K0=k0)
    outline = [(0, 0), (20, 0), (20, 12), (0, 12)]
    model = Model(
        name="k0 column",
        layers=[SoilLayer("soil", outline, soil, mesh_size=2.0)],
        stages=[Stage("initial", kind="initial", increments=4)],
        initial_stress="k0",
    )
    problem = model.build()
    solver = Solver(problem, tolerance=1e-4)
    solver.run_stage(model.resolved_stages()[0])
    gauss = problem.continuum.gauss_xy.reshape(-1, 2)
    stress = solver._state.stress
    inner = (gauss[:, 0] > 4) & (gauss[:, 0] < 16) & (gauss[:, 1] > 2) & (gauss[:, 1] < 10)
    ratio = stress[inner, 0] / stress[inner, 1]
    assert np.allclose(ratio, k0, atol=0.03)


def _slope_model(mesh_size=2.5):
    """The shipped slope example at a chosen element size.

    Using the example itself keeps the numbers in the validation notes and the
    numbers this test checks from drifting apart.
    """
    from lythos.examples import slope

    model = slope()
    model.mesh_size = mesh_size
    for layer in model.layers:
        layer.mesh_size = mesh_size
    return model


@pytest.mark.slow
def test_slope_factor_of_safety_matches_limit_equilibrium():
    """2:1 slope, c' = 10 kPa, phi' = 20 deg, gamma = 20 kN/m3.

    An independent Bishop simplified search over circular surfaces, with the
    same rigid base at the toe, gives 1.377.  Strength reduction finds the
    critical surface itself rather than the best circle, so it should land
    close to that and a little below it.
    """
    model = _slope_model()
    problem = model.build()
    solver = Solver(problem, tolerance=2e-3)
    results = solver.run()
    assert results[0].converged
    fos = results[-1].srf
    assert 1.30 < fos < 1.45, fos
    # displacements must grow towards failure, which is what makes the search valid
    curve = results[-1].srf_curve
    assert curve[-1][1] > 5 * max(curve[0][1], 1e-6)


@pytest.mark.slow
def test_factor_of_safety_converges_downwards_as_the_mesh_is_refined():
    """Refinement must reduce the factor of safety towards a limit.

    A coarse mesh cannot resolve the shear band, so it makes the slope look
    stronger than it is.  What matters is that refining moves the answer the
    right way and that successive refinements change it by less and less - a
    factor of safety that wandered with the mesh would not be usable.
    """
    values = [Solver(_slope_model(size).build(), tolerance=2e-3).run()[-1].srf
              for size in (3.0, 2.5, 2.0)]
    assert values[0] >= values[1] >= values[2], values
    first_step = values[0] - values[1]
    second_step = values[1] - values[2]
    assert second_step <= first_step + 0.01, values
    assert abs(values[1] - values[2]) / values[2] < 0.04, values


def test_deactivating_a_layer_removes_its_weight():
    """Excavating a layer must unload the ground beneath it."""
    soil = LinearElastic(name="soil", E=2.0e4, nu=0.25, gamma=20.0, K0=0.5)
    lower = [(0, 0), (20, 0), (20, 8), (0, 8)]
    upper = [(0, 8), (20, 8), (20, 12), (0, 12)]
    model = Model(
        name="excavation",
        layers=[SoilLayer("lower", lower, soil, mesh_size=2.0),
                SoilLayer("upper", upper, soil, mesh_size=2.0)],
        stages=[
            Stage("initial", kind="initial", active_layers=["lower", "upper"], increments=4),
            Stage("excavate", kind="plastic", active_layers=["lower"], increments=6),
        ],
        initial_stress="gravity",
    )
    problem = model.build()
    solver = Solver(problem, tolerance=1e-6)
    results = solver.run()
    assert all(r.converged for r in results)

    gauss = problem.continuum.gauss_xy.reshape(-1, 2)
    stress = solver._state.stress
    keep = np.repeat(results[-1].active_elements, problem.continuum.n_gauss)
    i = int(np.argmin(np.where(keep, np.abs(gauss[:, 1] - 1.0), 1e9)))
    assert stress[i, 1] == pytest.approx(-20.0 * (8.0 - gauss[i, 1]), rel=1e-4)
    # and the ground must rebound upwards
    u = results[-1].displacement[: 2 * problem.mesh.n_nodes].reshape(-1, 2)
    surface = np.abs(problem.mesh.nodes[:, 1] - 8.0) < 1e-6
    assert u[surface, 1].mean() > 0
