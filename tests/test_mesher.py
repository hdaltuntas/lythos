"""Mesh generation: does it cover the domain, honour the geometry and stay valid."""

import math

import numpy as np
import pytest

from lythos.core.mesh import build_mesh, order_chain
from lythos.core.mesher import MeshGenerator, PSLG, polygon_area


def _mesh(pslg, min_angle=25.0, max_area=None):
    gen = MeshGenerator(pslg, min_angle=min_angle, max_area=max_area)
    T = gen.run()
    assert T.check_consistency() == [], T.check_consistency()[:4]
    return T


def _min_angle(mesh):
    angles = []
    for tri in mesh.corner_triangles():
        p = mesh.nodes[tri]
        L = [math.dist(p[k], p[(k + 1) % 3]) for k in range(3)]
        for k in range(3):
            a, b, c = L[k], L[(k + 1) % 3], L[(k + 2) % 3]
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, (b * b + c * c - a * a) / (2 * b * c))))))
    return min(angles)


def test_square_area_and_quality():
    p = PSLG()
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    p.add_polyline(square, marker=1, closed=True)
    p.polygons.append((0, square, 0.01))
    mesh = build_mesh(_mesh(p))
    assert mesh.element_areas().sum() == pytest.approx(1.0, rel=1e-12)
    assert _min_angle(mesh) > 24.0
    assert mesh.n_elements > 50


def test_hole_is_excluded():
    p = PSLG()
    outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
    hole = [(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)]
    p.add_polyline(outer, marker=1, closed=True)
    p.add_polyline(hole, marker=2, closed=True)
    p.polygons.append((0, outer, 0.05))
    p.holes.append((2.0, 2.0))
    mesh = build_mesh(_mesh(p))
    assert mesh.element_areas().sum() == pytest.approx(15.0, rel=1e-12)


def test_layers_keep_their_own_areas():
    p = PSLG()
    lower = [(0, 0), (40, 0), (40, 8), (0, 8)]
    upper = [(0, 8), (40, 8), (40, 10), (25, 10), (15, 20), (0, 20)]
    p.add_polyline(lower, marker=1, closed=True)
    p.add_polyline(upper, marker=1, closed=True)
    p.polygons += [(0, lower, 6.0), (1, upper, 2.0)]
    mesh = build_mesh(_mesh(p))
    areas = mesh.element_areas()
    assert areas[mesh.element_tags == 0].sum() == pytest.approx(320.0, rel=1e-12)
    assert areas[mesh.element_tags == 1].sum() == pytest.approx(abs(polygon_area(upper)), rel=1e-12)


def test_internal_line_does_not_leak_between_regions():
    """A wall cutting a layer in two must not merge the two halves."""
    p = PSLG()
    left = [(0, 0), (10, 0), (10, 10), (0, 10)]
    right = [(10, 0), (20, 0), (20, 10), (10, 10)]
    p.add_polyline(left, marker=1, closed=True)
    p.add_polyline(right, marker=1, closed=True)
    p.add_polyline([(10, 10), (10, 4)], marker=99)
    p.polygons += [(0, left, 2.0), (1, right, 2.0)]
    mesh = build_mesh(_mesh(p))
    areas = mesh.element_areas()
    assert areas[mesh.element_tags == 0].sum() == pytest.approx(100.0, rel=1e-12)
    assert areas[mesh.element_tags == 1].sum() == pytest.approx(100.0, rel=1e-12)
    chains = order_chain(mesh.segments[99])
    assert len(chains) == 1
    ends = sorted([mesh.nodes[chains[0][0]][1], mesh.nodes[chains[0][-1]][1]])
    assert ends == pytest.approx([4.0, 10.0])


def test_crossing_lines_are_planarised():
    """Lines a user draws across one another must become a valid arrangement."""
    p = PSLG()
    outline = [(0, 0), (10, 0), (10, 10), (0, 10)]
    p.add_polyline(outline, marker=1, closed=True)
    p.add_polyline([(-1, 5), (11, 5)], marker=2)       # runs past both edges
    p.add_polyline([(5, -1), (5, 11)], marker=3)       # crosses it in the middle
    p.polygons.append((0, outline, 1.0))
    mesh = build_mesh(_mesh(p))
    assert mesh.element_areas().sum() == pytest.approx(100.0, rel=1e-12)
    # the crossing point must exist as a node
    assert mesh.node_at(5.0, 5.0, tol=1e-9) >= 0


def test_every_element_is_counter_clockwise_and_positive():
    p = PSLG()
    outline = [(0, 0), (30, 0), (30, 6), (18, 6), (10, 14), (0, 14)]
    p.add_polyline(outline, marker=1, closed=True)
    p.polygons.append((0, outline, 2.0))
    mesh = build_mesh(_mesh(p))
    p0 = mesh.nodes[mesh.elements[:, 0]]
    p1 = mesh.nodes[mesh.elements[:, 1]]
    p2 = mesh.nodes[mesh.elements[:, 2]]
    cross = ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
             - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    assert np.all(cross > 0)
    # midside nodes really sit at the edge midpoints
    for a, b, m in ((0, 1, 3), (1, 2, 4), (2, 0, 5)):
        mid = 0.5 * (mesh.nodes[mesh.elements[:, a]] + mesh.nodes[mesh.elements[:, b]])
        assert np.allclose(mid, mesh.nodes[mesh.elements[:, m]])


def test_smaller_target_size_gives_more_elements():
    counts = []
    for size in (4.0, 2.0, 1.0):
        p = PSLG()
        outline = [(0, 0), (20, 0), (20, 10), (0, 10)]
        p.add_polyline(outline, marker=1, closed=True)
        p.polygons.append((0, outline, 0.5 * size * size))
        counts.append(build_mesh(_mesh(p)).n_elements)
    assert counts[0] < counts[1] < counts[2]


def test_a_sharp_input_corner_does_not_produce_degenerate_elements():
    """Two lines meeting at a sharp angle must not send refinement runaway.

    Ruppert refinement splits the two segments at such a corner against each
    other without end unless the splitting is bounded, leaving elements many
    orders of magnitude smaller than the rest and a stiffness matrix to match.
    """
    p = PSLG()
    # a wedge: the two long edges meet at about 18 degrees
    outline = [(0, 0), (60, 0), (40, 6.5)]
    p.add_polyline(outline, marker=1, closed=True)
    p.polygons.append((0, outline, 2.0))
    mesh = build_mesh(_mesh(p))
    areas = mesh.element_areas()
    assert areas.sum() == pytest.approx(abs(polygon_area(outline)), rel=1e-9)
    assert areas.min() > 1e-5 * areas.max(), (areas.min(), areas.max())
    assert areas.min() > 0


def test_a_zero_width_lobe_is_discarded():
    """A polygon that doubles back on itself must not leave sliver elements."""
    p = PSLG()
    # the run from (5, 0) back to (0, 0) is collinear with the base: no area
    outline = [(0, 0), (40, 0), (40, 10), (25, 10), (5, 0)]
    p.add_polyline(outline, marker=1, closed=True)
    p.polygons.append((0, outline, 2.0))
    mesh = build_mesh(_mesh(p))
    areas = mesh.element_areas()
    assert areas.sum() == pytest.approx(250.0, rel=1e-9)
    assert areas.min() > 1e-6 * areas.max()
