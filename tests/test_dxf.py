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
    assert any("more than one region ended up named" in w for w in report.warnings)


def test_closed_outlines_keep_their_own_names():
    """Each stratum drawn closed on its own layer names its region exactly."""
    text = drawing(
        lwpolyline("ZEMIN-KUM", [(0, 0), (40, 0), (40, 5), (16.25, 5), (10, 0)], closed=True)
        + lwpolyline("ZEMIN-KIL", [(0, 5), (40, 5), (40, 12), (25, 12), (16.25, 5)],
                     closed=True))
    model, report = build_model(parse_dxf(text))
    assert sorted(layer.name for layer in model.layers) == ["ZEMIN-KIL", "ZEMIN-KUM"]
    assert not any("more than one region ended up named" in w for w in report.warnings)
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


# ------------------------------------------------------------------- staging
@pytest.mark.parametrize("layer,step", [
    ("KAZI-2", 2), ("EXC-1", 1), ("LIFT 3", 3), ("PERDE-1", 1),
    ("ANKRAJ-10", 10), ("WALL", None), ("SOIL-CLAY", None), ("EXC-100", None),
])
def test_step_numbers_are_read_from_layer_names(layer, step):
    assert ImportRules().step(layer) == step


def _staged_drawing():
    """A wall, two anchors and three lifts, each carrying its step number."""
    W, R = 20, 60
    return drawing(
        lwpolyline("ZEMIN-KUM", [(0, 0), (R, 0), (R, 8), (0, 8)], closed=True)
        + lwpolyline("ZEMIN-KIL", [(0, 8), (R, 8), (R, 22), (0, 22)], closed=True)
        + lwpolyline("ZEMIN-DOLGU", [(0, 22), (R, 22), (R, 30), (0, 30)], closed=True)
        + lwpolyline("KAZI-2", [(W, 27), (R, 27), (R, 30), (W, 30)], closed=True)
        + lwpolyline("KAZI-4", [(W, 24), (R, 24), (R, 27), (W, 27)], closed=True)
        + lwpolyline("KAZI-6", [(W, 22), (R, 22), (R, 24), (W, 24)], closed=True)
        + lwpolyline("PERDE-1", [(W, 30), (W, 14)])
        + line("ANKRAJ-3", (W, 28.5), (W - 12, 21))
        + line("ANKRAJ-5", (W, 25.5), (W - 11, 18.5))
        + lwpolyline("SURSARJ-1", [(2, 30), (16, 30)]))


def test_a_staged_drawing_becomes_a_construction_sequence():
    model, report = build_model(parse_dxf(_staged_drawing()))
    kinds = [stage.kind for stage in model.stages]
    assert kinds[0] == "initial" and kinds[-1] == "ssr"
    assert [n for n, _what in report.steps] == [1, 2, 3, 4, 5, 6]

    stages = model.resolved_stages()
    # the wall is built and the surcharge applied first, before any digging
    assert stages[1].active_structures == ["PERDE-1"]
    assert stages[1].active_loads == ["SURSARJ-1"]
    assert "KAZI-2" in stages[1].active_layers

    # each lift leaves in its own step, and stays gone
    gone_by = {}
    for stage in stages:
        for lift in ("KAZI-2", "KAZI-4", "KAZI-6"):
            if lift not in stage.active_layers and lift not in gone_by:
                gone_by[lift] = stage.name
    assert set(gone_by) == {"KAZI-2", "KAZI-4", "KAZI-6"}
    assert "excavate KAZI-2" in gone_by["KAZI-2"]

    # anchors are stressed in order and stay stressed
    stressed = [tuple(stage.active_anchors) for stage in stages]
    assert ("ANKRAJ-3",) in stressed
    assert ("ANKRAJ-3", "ANKRAJ-5") == stressed[-1]

    # nothing that was dug out comes back
    for earlier, later in zip(stages[1:-1], stages[2:]):
        assert set(later.active_layers) <= set(earlier.active_layers)


def test_excavated_regions_are_still_meshed():
    """A lift has to exist in the mesh before it can be taken away."""
    model, _report = build_model(parse_dxf(_staged_drawing()))
    model.mesh_size = 4.0
    for layer in model.layers:
        layer.mesh_size = 4.0
    problem = model.build()
    total = sum(layer.area() for layer in model.layers)
    assert problem.mesh.element_areas().sum() == pytest.approx(total, rel=1e-9)
    assert total == pytest.approx(60 * 30, rel=1e-9)


def test_excavation_levels_drawn_as_lines_also_work():
    """The other convention: dig levels as lines, not as closed regions."""
    text = drawing(
        lwpolyline("ZEMIN", [(0, 0), (40, 0), (40, 20), (0, 20)], closed=True)
        + lwpolyline("PERDE-1", [(20, 20), (20, 10)])
        + lwpolyline("KAZI-2", [(20, 16), (40, 16)])
        + lwpolyline("KAZI-3", [(20, 12), (40, 12)]))
    model, report = build_model(parse_dxf(text))
    assert [n for n, _w in report.steps] == [1, 2, 3]

    stages = model.resolved_stages()
    counts = [len(stage.active_layers) for stage in stages]
    # one region leaves at step 2 and another at step 3
    assert counts[1] > counts[2] > counts[3]
    areas = {layer.name: layer.area() for layer in model.layers}
    removed = set(stages[1].active_layers) - set(stages[3].active_layers)
    assert sum(areas[name] for name in removed) == pytest.approx(
        20 * 4 + 20 * 4, rel=1e-6)          # the two 20 x 4 m lifts


def test_an_unstaged_drawing_keeps_the_simple_sequence():
    text = drawing(lwpolyline("SOIL", [(0, 0), (10, 0), (10, 5), (0, 5)], closed=True)
                   + lwpolyline("WALL", [(5, 5), (5, 1)]))
    model, report = build_model(parse_dxf(text))
    assert report.steps == []
    assert [s.kind for s in model.stages] == ["initial", "plastic", "ssr"]
    assert model.stages[1].active_structures == ["WALL"]


def test_the_safety_stage_can_be_turned_off():
    model, _report = build_model(parse_dxf(_staged_drawing()),
                                 ImportRules(add_safety_stage=False))
    assert "ssr" not in [stage.kind for stage in model.stages]


def test_step_numbers_can_be_ignored():
    model, report = build_model(parse_dxf(_staged_drawing()),
                                ImportRules(read_steps=False))
    assert report.steps == []
    # every region stays present: nothing was read as an excavation
    for stage in model.resolved_stages():
        if stage.kind == "plastic":
            assert "KAZI-2" in stage.active_layers
