"""Element formulations: patch tests and closed-form structural checks."""


import numpy as np
import pytest

from lythos.core.assembly import DofManager, SystemAssembler, solve_constrained
from lythos.core.elements import BeamElements, BeamSection, ContinuumElements
from lythos.core.materials import elastic_matrix
from lythos.core.mesh import build_mesh
from lythos.core.mesher import MeshGenerator, PSLG


def _unit_square_mesh(size=0.4):
    p = PSLG()
    outline = [(0, 0), (1, 0), (1, 1), (0, 1)]
    p.add_polyline(outline, marker=1, closed=True)
    p.polygons.append((0, outline, 0.5 * size * size))
    return build_mesh(MeshGenerator(p, min_angle=25.0).run())


def test_element_stiffness_is_symmetric_and_has_no_energy_in_rigid_motion():
    mesh = _unit_square_mesh()
    ce = ContinuumElements(mesh.nodes, mesh.elements)
    D = elastic_matrix(2.0e4, 0.25)
    Ke = ce.stiffness(np.broadcast_to(D, (ce.n_points, 4, 4)).copy())
    assert np.abs(Ke - np.transpose(Ke, (0, 2, 1))).max() < 1e-8 * np.abs(Ke).max()
    for motion in (np.tile([1.0, 0.0], 6), np.tile([0.0, 1.0], 6)):
        assert np.abs(Ke @ motion).max() < 1e-8 * np.abs(Ke).max()


def test_patch_test_reproduces_a_linear_displacement_field_exactly():
    """Prescribing a linear field on the boundary must give constant stress.

    This is the standard patch test: an element that fails it cannot converge
    to the right answer however fine the mesh.
    """
    mesh = _unit_square_mesh(0.3)
    ce = ContinuumElements(mesh.nodes, mesh.elements)
    E, nu = 3.0e4, 0.2
    D = elastic_matrix(E, nu)
    dofs = DofManager(mesh.n_nodes)

    # u = a x + b y, v = c x + d y  ->  uniform strain everywhere
    a, b, c, d = 1e-3, -4e-4, 7e-4, 2e-3
    exact = np.zeros(2 * mesh.n_nodes)
    exact[0::2] = a * mesh.nodes[:, 0] + b * mesh.nodes[:, 1]
    exact[1::2] = c * mesh.nodes[:, 0] + d * mesh.nodes[:, 1]

    asm = SystemAssembler(dofs.n_dof)
    asm.add(ce.dofs(), ce.stiffness(np.broadcast_to(D, (ce.n_points, 4, 4)).copy()))
    K = asm.matrix()

    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    tol = 1e-9
    on_edge = (np.abs(x) < tol) | (np.abs(x - 1) < tol) | (np.abs(y) < tol) | (np.abs(y - 1) < tol)
    fixed = np.sort(np.concatenate([2 * np.nonzero(on_edge)[0], 2 * np.nonzero(on_edge)[0] + 1]))
    u = solve_constrained(K, np.zeros(dofs.n_dof), fixed, exact[fixed])

    assert np.abs(u - exact).max() < 1e-12
    strains = ce.strains(u)
    assert np.allclose(strains[:, 0], a, atol=1e-12)
    assert np.allclose(strains[:, 1], d, atol=1e-12)
    assert np.allclose(strains[:, 3], b + c, atol=1e-12)
    stress = strains @ D.T
    assert np.allclose(stress, stress[0], atol=1e-8)


def _cantilever(length=5.0, n_elements=4, EI=1.0e4, EA=1.0e7, GA=1.0e9, load=10.0):
    """Tip-loaded cantilever built from quadratic beam elements."""
    n_nodes = 2 * n_elements + 1
    xs = np.linspace(0.0, length, n_nodes)
    nodes = np.column_stack([xs, np.zeros(n_nodes)])
    chain = list(range(n_nodes))
    beam = BeamElements(nodes, [chain], BeamSection(EA=EA, EI=EI, GA=GA))
    dofs = DofManager(n_nodes)
    for n in chain:
        dofs.add_rotation(n)
    dof_map = np.array([dofs.beam_dofs(el) for el in beam.elements])

    u = np.zeros(dofs.n_dof)
    Ke, _Fe = beam.stiffness_and_force(u, dof_map)
    asm = SystemAssembler(dofs.n_dof)
    asm.add(dof_map, Ke)
    f = np.zeros(dofs.n_dof)
    f[2 * (n_nodes - 1) + 1] = -load
    fixed = np.array([0, 1, dofs.rotation_of[0]])
    u = solve_constrained(asm.matrix(), f, fixed)
    return beam, dof_map, u, dofs, n_nodes


def test_cantilever_tip_deflection_matches_beam_theory():
    L, EI, GA, P = 5.0, 1.0e4, 1.0e9, 10.0
    _beam, _map, u, _dofs, n_nodes = _cantilever(length=L, EI=EI, GA=GA, load=P)
    tip = u[2 * (n_nodes - 1) + 1]
    expected = -(P * L ** 3 / (3 * EI) + P * L / GA)
    assert tip == pytest.approx(expected, rel=2e-3)


def test_cantilever_root_moment_matches_statics():
    L, P = 5.0, 10.0
    beam, dof_map, u, _dofs, _n = _cantilever(length=L, load=P)
    forces = beam.section_forces(u, dof_map)
    # extrapolate the moment back to the root from the first two stations
    s = forces[:2, 0]
    m = forces[:2, 4]
    root = m[0] + (m[1] - m[0]) * (0.0 - s[0]) / (s[1] - s[0])
    assert abs(abs(root) - P * L) / (P * L) < 0.02
    assert abs(forces[:, 3]).max() == pytest.approx(P, rel=0.02)


def test_thin_beam_does_not_lock():
    """A very slender beam must still reach its theoretical deflection.

    Full integration of the shear term would make this element far too stiff;
    the selective reduced integration is what keeps the answer right.
    """
    L, P = 10.0, 1.0
    E, b, h = 2.1e8, 1.0, 0.01                 # h/L = 1/1000
    EI = E * b * h ** 3 / 12.0
    GA = (E / 2.4) * (5.0 / 6.0) * b * h
    _beam, _map, u, _dofs, n_nodes = _cantilever(length=L, n_elements=6, EI=EI,
                                                 EA=E * b * h, GA=GA, load=P)
    tip = u[2 * (n_nodes - 1) + 1]
    slender = -P * L ** 3 / (3 * EI)
    assert tip / slender > 0.98
