"""Reading geometry out of DXF drawings."""

import math

import pytest

from lythos.core.topology import find_faces, join_chains
from lythos.dxf import ImportRules, build_model, import_dxf, parse_dxf, read_dxf


# ------------------------------------------------------------------- fixtures
def tag(code, value):
    return f"{code}\n{value}\n"


def line(layer, a, b):
    return (tag(0, "LINE") + tag(8, layer) + tag(10, a[0]) + tag(20, a[1])
            + tag(11, b[0]) + tag(21, b[1]))


def lwpolyline(layer, points, closed=False, bulges=None):
    out = (tag(0, "LWPOLYLINE") + tag(8, layer) + tag(90, len(points))
           + tag(70, 1 if closed else 0))
    for i, p in enumerate(points):
        out += tag(10, p[0]) + tag(20, p[1])
        if bulges and i < len(bulges) and bulges[i]:
            out += tag(42, bulges[i])
    return out


def drawing(body, insunits=6):
    return (tag(0, "SECTION") + tag(2, "HEADER") + tag(9, "$INSUNITS")
            + tag(70, insunits) + tag(0, "ENDSEC")
            + tag(0, "SECTION") + tag(2, "ENTITIES") + body
            + tag(0, "ENDSEC") + tag(0, "EOF"))


# --------------------------------------------------------------------- reading
def test_reads_lines_and_polylines():
    text = drawing(line("A", (0, 0), (10, 0))
                   + lwpolyline("B", [(0, 0), (5, 5), (10, 0)], closed=True))
    doc = parse_dxf(text)
    assert [e.kind for e in doc.entities] == ["line", "polyline"]
    assert doc.entities[0].points == [(0.0, 0.0), (10.0, 0.0)]
    assert doc.entities[1].closed
    assert doc.entities[1].layer == "B"


def test_reads_units_from_the_header():
    assert parse_dxf(drawing(line("A", (0, 0), (1, 0)), insunits=4)).units == 0.001
    assert parse_dxf(drawing(line("A", (0, 0), (1, 0)), insunits=6)).units == 1.0
    assert parse_dxf(drawing(line("A", (0, 0), (1, 0)), insunits=2)).units == pytest.approx(0.3048)


def test_arcs_and_circles_become_polylines():
    text = drawing(tag(0, "ARC") + tag(8, "A") + tag(10, 0.0) + tag(20, 0.0)
                   + tag(40, 5.0) + tag(50, 0.0) + tag(51, 90.0)
                   + tag(0, "CIRCLE") + tag(8, "A") + tag(10, 0.0) + tag(20, 0.0)
                   + tag(40, 3.0))
    doc = parse_dxf(text)
    arc, circle = doc.entities
    assert arc.kind == "arc" and len(arc.points) > 4
    assert all(abs(math.hypot(*p) - 5.0) < 1e-9 for p in arc.points)
    assert arc.points[0] == pytest.approx((5.0, 0.0))
    assert arc.points[-1] == pytest.approx((0.0, 5.0), abs=1e-9)
    assert circle.closed and all(abs(math.hypot(*p) - 3.0) < 1e-9 for p in circle.points)


def test_a_bulged_segment_becomes_an_arc():
    """A bulge of 1 is a half circle; the polyline must bow out to match."""
    text = drawing(lwpolyline("A", [(0, 0), (10, 0)], bulges=[1.0]))
    points = parse_dxf(text).entities[0].points
    assert len(points) > 5
    # every point should sit on the circle of radius 5 centred at (5, 0)
    for p in points:
        assert math.dist(p, (5.0, 0.0)) == pytest.approx(5.0, abs=1e-6)


def test_a_malformed_file_does_not_crash():
    assert parse_dxf("").entities == []
    assert parse_dxf("not a dxf at all").entities == []


def test_binary_dxf_is_reported_clearly(tmp_path):
    path = tmp_path / "binary.dxf"
    path.write_bytes(b"AutoCAD Binary DXF\r\n\x1a\x00" + b"\x00" * 64)
    with pytest.raises(ValueError, match="binary DXF"):
        read_dxf(str(path))


# ------------------------------------------------------------------- mapping
@pytest.mark.parametrize("layer,expected", [
    ("SOIL-CLAY", "soil"), ("zemin_kum", "soil"), ("Stiff Clay", "soil"),
    ("WALL", "structure"), ("PERDE-SECANT", "structure"), ("pile row", "structure"),
    ("WATER-TABLE", "water"), ("SU-SEVIYESI", "water"), ("gwl", "water"),
    ("ANCHOR-ROW-1", "anchor"), ("ankraj", "anchor"), ("STRUT", "anchor"),
    ("SURCHARGE", "load"), ("yuk", "load"),
    ("TEXT", "ignore"), ("DIM-LEVELS", "ignore"), ("Defpoints", "ignore"),
    ("XYZ", None),
])
def test_layer_names_map_to_roles(layer, expected):
    assert ImportRules().role(layer) == expected


def test_separators_do_not_matter():
    rules = ImportRules()
    for name in ("SU SEVIYESI", "SU-SEVIYESI", "su_seviyesi", "SuSeviyesi"):
        assert rules.role(name) == "water", name


def test_ignored_layers_are_left_out():
    text = drawing(lwpolyline("SOIL", [(0, 0), (10, 0), (10, 5), (0, 5)], closed=True)
                   + lwpolyline("TEXT-NOTES", [(0, 8), (10, 8)])
                   + line("DIM", (0, -2), (10, -2)))
    model, report = build_model(parse_dxf(text))
    assert report.roles["TEXT-NOTES"] == "ignored"
    assert len(model.layers) == 1
    assert not model.structures


# ------------------------------------------------------------------ importing
def test_an_outline_cut_by_a_line_becomes_two_layers():
    """The usual CAD drawing: one boundary plus the lines dividing it."""
    text = drawing(
        line("SOIL-OUTLINE", (0, 0), (40, 0)) + line("SOIL-OUTLINE", (40, 0), (40, 12))
        + line("SOIL-OUTLINE", (40, 12), (25, 12)) + line("SOIL-OUTLINE", (25, 12), (10, 0))
        + line("SOIL-OUTLINE", (10, 0), (0, 0))
        + lwpolyline("SOIL-CLAY", [(0, 5), (40, 5)]))
    model, report = build_model(parse_dxf(text))
    assert len(model.layers) == 2
    assert sum(layer.area() for layer in model.layers) == pytest.approx(270.0, rel=1e-9)
    assert model.validate() == []
    assert any("names are taken from" in w for w in report.warnings)


def test_closed_outlines_keep_their_own_names():
    """Each stratum drawn closed on its own layer names its region exactly."""
    text = drawing(
        lwpolyline("ZEMIN-KUM", [(0, 0), (40, 0), (40, 5), (16.25, 5), (10, 0)], closed=True)
        + lwpolyline("ZEMIN-KIL", [(0, 5), (40, 5), (40, 12), (25, 12), (16.25, 5)],
                     closed=True))
    model, report = build_model(parse_dxf(text))
    assert sorted(layer.name for layer in model.layers) == ["ZEMIN-KIL", "ZEMIN-KUM"]
    assert not any("names are taken from" in w for w in report.warnings)
    assert not any("cut into smaller regions" in w for w in report.warnings)


def test_overlapping_outlines_are_reported():
    text = drawing(
        lwpolyline("A", [(0, 0), (10, 0), (10, 10), (0, 10)], closed=True)
        + lwpolyline("B", [(5, 5), (15, 5), (15, 15), (5, 15)], closed=True))
    _model, report = build_model(parse_dxf(text))
    assert any("cut into smaller regions" in w for w in report.warnings)


def test_walls_water_loads_and_anchors_are_picked_up():
    text = drawing(
        lwpolyline("SOIL", [(0, 0), (40, 0), (40, 12), (0, 12)], closed=True)
        + lwpolyline("WALL", [(22, 12), (22, 2)])
        + lwpolyline("WATER", [(0, 4), (40, 4)])
        + lwpolyline("SURCHARGE", [(28, 12), (36, 12)])
        + line("ANCHOR-1", (22, 11), (14, 7)))
    model, _report = build_model(parse_dxf(text))
    assert [s.name for s in model.structures] == ["WALL"]
    assert model.structures[0].path == [(22.0, 12.0), (22.0, 2.0)]
    assert model.water.points == [(0.0, 4.0), (40.0, 4.0)]
    assert [load.name for load in model.line_loads] == ["SURCHARGE"]
    assert [a.name for a in model.anchors] == ["ANCHOR-1"]
    assert model.anchors[0].start == (22.0, 11.0)


def test_drawing_units_are_converted_to_metres():
    text = drawing(lwpolyline("SOIL", [(0, 0), (40000, 0), (40000, 5000), (0, 5000)],
                              closed=True), insunits=4)     # millimetres
    model, report = build_model(parse_dxf(text))
    assert report.scale == pytest.approx(0.001)
    assert model.layers[0].area() == pytest.approx(200.0, rel=1e-9)


def test_survey_coordinates_are_brought_to_the_origin():
    """CAD sections often sit on survey grids far from the origin."""
    big = 452000.0
    text = drawing(lwpolyline("SOIL", [(big, 0), (big + 40, 0), (big + 40, 12), (big, 12)],
                              closed=True))
    model, report = build_model(parse_dxf(text))
    x0, _y0, x1, _y1 = model.bounds()
    assert x0 == pytest.approx(0.0, abs=1e-6)
    assert x1 == pytest.approx(40.0, abs=1e-6)
    assert report.offset[0] == pytest.approx(big)

    kept, _ = build_model(parse_dxf(text), ImportRules(shift_to_origin=False))
    assert kept.bounds()[0] == pytest.approx(big)


def test_an_imported_model_can_be_meshed():
    text = drawing(
        lwpolyline("SOIL", [(0, 0), (40, 0), (40, 12), (25, 12), (10, 0)], closed=True)
        + lwpolyline("WALL", [(22, 12), (22, 2)]))
    model, _report = build_model(parse_dxf(text))
    model.mesh_size = 3.0
    problem = model.build()
    assert problem.continuum.n_elements > 20
    assert problem.structures and problem.structures[0].beam.n_elements > 0
    assert problem.mesh.element_areas().sum() == pytest.approx(
        sum(layer.area() for layer in model.layers), rel=1e-9)


def test_import_dxf_reads_from_a_file(tmp_path):
    path = tmp_path / "section.dxf"
    path.write_text(drawing(lwpolyline("SOIL", [(0, 0), (10, 0), (10, 5), (0, 5)],
                                       closed=True)))
    model, report = import_dxf(str(path), name="from file")
    assert model.name == "from file"
    assert len(model.layers) == 1
    assert "section.dxf" in report.describe()


def test_a_drawing_with_no_soil_says_so():
    model, report = build_model(parse_dxf(drawing(lwpolyline("WALL", [(0, 0), (0, 5)]))))
    assert not model.layers
    assert any("no soil" in w or "no layer matched" in w for w in report.warnings)


# ------------------------------------------------------------------- topology
def test_find_faces_splits_a_divided_square():
    points = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 4), (10, 4)]
    segments = [(0, 1), (1, 5), (5, 2), (2, 3), (3, 4), (4, 0), (4, 5)]
    faces = find_faces(points, segments)
    assert len(faces) == 2
    from lythos.core.topology import _signed_area
    areas = sorted(_signed_area(points, f) for f in faces)
    assert areas == pytest.approx([40.0, 60.0])


def test_find_faces_ignores_dangling_lines():
    points = [(0, 0), (10, 0), (10, 10), (0, 10), (20, 5)]
    segments = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4)]
    faces = find_faces(points, segments)
    assert len(faces) == 1
    assert len(faces[0]) == 4


def test_join_chains_closes_a_boundary_drawn_as_separate_lines():
    pieces = [[(0, 0), (10, 0)], [(10, 10), (10, 0)], [(10, 10), (0, 10)], [(0, 10), (0, 0)]]
    chains, closed = join_chains(pieces)
    assert len(chains) == 1 and closed == [True]
    assert len(chains[0]) == 4
