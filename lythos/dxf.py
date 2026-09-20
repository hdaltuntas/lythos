"""Reading geometry out of DXF drawings.

Sections are drawn in CAD, not typed in as coordinates, so a model usually
starts life as a DXF.  This module reads the entity types that carry the
geometry of a section - lines, polylines, arcs and circles - and turns the
layers of the drawing into the layers, walls and water table of a model.

The reader is self-contained and handles ASCII DXF, which is what every CAD
program can export and what nearly every exchanged drawing is.  If `ezdxf` is
installed it is used instead, which adds binary DXF, splines and block
references; nothing here requires it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

#: $INSUNITS values that matter, as metres per drawing unit
INSUNITS_TO_METRES = {
    0: 1.0,        # unitless: assume metres
    1: 0.0254,     # inches
    2: 0.3048,     # feet
    4: 0.001,      # millimetres
    5: 0.01,       # centimetres
    6: 1.0,        # metres
    8: 1.0e-6,     # microns
    9: 0.001,      # millimetres (alias)
    14: 0.1,       # decimetres
    15: 10.0,      # decametres
    16: 100.0,     # hectometres
    21: 1609.344,  # miles
}


@dataclass
class DxfEntity:
    """One piece of drawn geometry, reduced to a polyline."""

    layer: str
    points: list[tuple[float, float]]
    closed: bool = False
    kind: str = "line"

    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.points[:-1], self.points[1:]))


@dataclass
class DxfDrawing:
    """Everything read from a drawing."""

    entities: list[DxfEntity] = field(default_factory=list)
    units: float = 1.0                 # metres per drawing unit
    source: str = ""

    def layers(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entity in self.entities:
            counts[entity.layer] = counts.get(entity.layer, 0) + 1
        return dict(sorted(counts.items()))

    def bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for e in self.entities for p in e.points]
        ys = [p[1] for e in self.entities for p in e.points]
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))


# --------------------------------------------------------------------- reading
def read_dxf(path: str, arc_tolerance: float = 0.02) -> DxfDrawing:
    """Read a DXF file into polylines.

    ``arc_tolerance`` is the greatest distance a chord may fall short of the
    true arc, as a fraction of the radius; it sets how finely arcs are broken
    into straight segments.
    """
    try:
        return _read_with_ezdxf(path, arc_tolerance)
    except ImportError:
        pass
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    if "\x00" in text[:2048]:
        raise ValueError(
            f"{path} looks like a binary DXF. Re-export it as ASCII DXF, "
            "or install ezdxf (pip install ezdxf) to read it directly.")
    return parse_dxf(text, arc_tolerance=arc_tolerance, source=path)


def _read_with_ezdxf(path: str, arc_tolerance: float) -> DxfDrawing:
    import ezdxf                                   # noqa: PLC0415 - optional

    doc = ezdxf.readfile(path)
    scale = INSUNITS_TO_METRES.get(int(doc.header.get("$INSUNITS", 0) or 0), 1.0)
    drawing = DxfDrawing(units=scale, source=path)
    for entity in doc.modelspace():
        drawing.entities.extend(_from_ezdxf(entity, arc_tolerance))
    return drawing


def _from_ezdxf(entity, arc_tolerance: float) -> list[DxfEntity]:
    kind = entity.dxftype()
    layer = str(entity.dxf.layer)
    if kind == "LINE":
        return [DxfEntity(layer, [(entity.dxf.start.x, entity.dxf.start.y),
                                  (entity.dxf.end.x, entity.dxf.end.y)], kind="line")]
    if kind in ("LWPOLYLINE", "POLYLINE"):
        points = [(p[0], p[1]) for p in entity.get_points("xy")] \
            if kind == "LWPOLYLINE" else [(v.dxf.location.x, v.dxf.location.y)
                                          for v in entity.vertices]
        return [DxfEntity(layer, points, closed=bool(entity.closed), kind="polyline")]
    if kind == "ARC":
        return [DxfEntity(layer, _arc_points(
            (entity.dxf.center.x, entity.dxf.center.y), entity.dxf.radius,
            entity.dxf.start_angle, entity.dxf.end_angle, arc_tolerance), kind="arc")]
    if kind == "CIRCLE":
        return [DxfEntity(layer, _arc_points(
            (entity.dxf.center.x, entity.dxf.center.y), entity.dxf.radius,
            0.0, 360.0, arc_tolerance), closed=True, kind="circle")]
    if kind in ("SPLINE", "ELLIPSE"):
        try:
            points = [(p[0], p[1]) for p in entity.flattening(arc_tolerance)]
        except (AttributeError, TypeError):
            return []
        return [DxfEntity(layer, points, kind=kind.lower())]
    return []


def parse_dxf(text: str, arc_tolerance: float = 0.02, source: str = "") -> DxfDrawing:
    """Parse ASCII DXF text.  Exposed separately so it can be tested directly."""
    tags = _tags(text)
    scale = _insunits(tags)
    drawing = DxfDrawing(units=scale, source=source)

    index = _entities_start(tags)
    pending: list[tuple[int, str]] = []
    name = ""
    polyline: DxfEntity | None = None

    def flush():
        nonlocal name, pending, polyline
        if not name:
            return
        if name == "VERTEX" and polyline is not None:
            x = _first(pending, 10)
            y = _first(pending, 20)
            if x is not None and y is not None:
                polyline.points.append((x, y))
        elif name == "SEQEND":
            if polyline is not None and len(polyline.points) >= 2:
                drawing.entities.append(polyline)
            polyline = None
        elif name == "POLYLINE":
            flags = int(_first(pending, 70) or 0)
            polyline = DxfEntity(_layer(pending), [], closed=bool(flags & 1),
                                 kind="polyline")
        else:
            drawing.entities.extend(_entity(name, pending, arc_tolerance))
        pending = []
        name = ""

    while index < len(tags):
        code, value = tags[index]
        if code == 0:
            flush()
            if value in ("ENDSEC", "EOF"):
                break
            name = value
        else:
            pending.append((code, value))
        index += 1
    flush()
    return drawing


def _tags(text: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    tags: list[tuple[int, str]] = []
    for i in range(0, len(lines) - 1, 2):
        raw = lines[i].strip()
        if not raw or not re.fullmatch(r"-?\d+", raw):
            # A stray line would shift every pair that follows, so resynchronise
            # by scanning forward to the next line that is a group code.
            shifted = _resynchronise(lines, i)
            if shifted is None:
                break
            return tags + _tags("\n".join(lines[shifted:]))
        tags.append((int(raw), lines[i + 1].strip()))
    return tags


def _resynchronise(lines: list[str], start: int) -> int | None:
    for j in range(start, min(start + 8, len(lines) - 1)):
        if re.fullmatch(r"-?\d+", lines[j].strip()):
            return j
    return None


def _entities_start(tags: list[tuple[int, str]]) -> int:
    for i, (code, value) in enumerate(tags):
        if code == 2 and value == "ENTITIES":
            return i + 1
    return 0


def _insunits(tags: list[tuple[int, str]]) -> float:
    for i, (code, value) in enumerate(tags):
        if code == 9 and value == "$INSUNITS":
            for code2, value2 in tags[i + 1:i + 4]:
                if code2 == 70:
                    try:
                        return INSUNITS_TO_METRES.get(int(value2), 1.0)
                    except ValueError:
                        return 1.0
    return 1.0


def _first(pending, code) -> float | None:
    for c, v in pending:
        if c == code:
            try:
                return float(v)
            except ValueError:
                return None
    return None


def _layer(pending) -> str:
    for c, v in pending:
        if c == 8:
            return v
    return "0"


def _entity(name: str, pending, arc_tolerance: float) -> list[DxfEntity]:
    layer = _layer(pending)
    if name == "LINE":
        x1, y1 = _first(pending, 10), _first(pending, 20)
        x2, y2 = _first(pending, 11), _first(pending, 21)
        if None in (x1, y1, x2, y2):
            return []
        return [DxfEntity(layer, [(x1, y1), (x2, y2)], kind="line")]

    if name == "LWPOLYLINE":
        points: list[tuple[float, float]] = []
        bulges: list[float] = []
        x = None
        for code, value in pending:
            if code == 10:
                if x is not None:
                    points.append((x, 0.0))
                x = float(value)
                bulges.append(0.0)
            elif code == 20 and x is not None:
                points.append((x, float(value)))
                x = None
            elif code == 42 and bulges:
                bulges[-1] = float(value)
        flags = int(_first(pending, 70) or 0)
        closed = bool(flags & 1)
        if len(points) < 2:
            return []
        return [DxfEntity(layer, _apply_bulges(points, bulges, closed, arc_tolerance),
                          closed=closed, kind="polyline")]

    if name == "ARC":
        cx, cy = _first(pending, 10), _first(pending, 20)
        r = _first(pending, 40)
        a0, a1 = _first(pending, 50), _first(pending, 51)
        if None in (cx, cy, r) or r <= 0:
            return []
        return [DxfEntity(layer, _arc_points((cx, cy), r, a0 or 0.0, a1 or 360.0,
                                             arc_tolerance), kind="arc")]

    if name == "CIRCLE":
        cx, cy = _first(pending, 10), _first(pending, 20)
        r = _first(pending, 40)
        if None in (cx, cy, r) or r <= 0:
            return []
        return [DxfEntity(layer, _arc_points((cx, cy), r, 0.0, 360.0, arc_tolerance),
                          closed=True, kind="circle")]

    if name == "POINT":
        x, y = _first(pending, 10), _first(pending, 20)
        if None in (x, y):
            return []
        return [DxfEntity(layer, [(x, y)], kind="point")]

    return []


def _arc_points(centre, radius, start_deg, end_deg, tolerance):
    """Break an arc into chords no further than ``tolerance * radius`` from it."""
    sweep = (end_deg - start_deg) % 360.0
    if sweep <= 1e-9:
        sweep = 360.0
    step = 2.0 * math.acos(max(-1.0, min(1.0, 1.0 - max(tolerance, 1e-4))))
    count = max(2, int(math.ceil(math.radians(sweep) / max(step, 1e-3))))
    return [(centre[0] + radius * math.cos(math.radians(start_deg) + i * math.radians(sweep) / count),
             centre[1] + radius * math.sin(math.radians(start_deg) + i * math.radians(sweep) / count))
            for i in range(count + 1)]


def _apply_bulges(points, bulges, closed, tolerance):
    """Replace bulged polyline segments with the arcs they stand for."""
    if not any(abs(b) > 1e-12 for b in bulges):
        return points
    out: list[tuple[float, float]] = []
    count = len(points)
    last = count if closed else count - 1
    for i in range(last):
        a = points[i]
        b = points[(i + 1) % count]
        out.append(a)
        bulge = bulges[i] if i < len(bulges) else 0.0
        if abs(bulge) <= 1e-12:
            continue
        chord = math.dist(a, b)
        if chord <= 1e-12:
            continue
        # bulge is tan(theta / 4) for an included angle theta
        theta = 4.0 * math.atan(bulge)
        radius = chord / (2.0 * math.sin(abs(theta) / 2.0))
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        height = radius * math.cos(abs(theta) / 2.0)
        nx, ny = -(b[1] - a[1]) / chord, (b[0] - a[0]) / chord
        sign = 1.0 if theta > 0 else -1.0
        centre = (mid[0] - sign * nx * height, mid[1] - sign * ny * height)
        start = math.degrees(math.atan2(a[1] - centre[1], a[0] - centre[0]))
        end = start + math.degrees(theta)
        arc = _arc_points(centre, radius, min(start, end), max(start, end), tolerance)
        if theta < 0:
            arc.reverse()
        out.extend(arc[1:-1])
    if not closed:
        out.append(points[-1])
    return out


# ------------------------------------------------------------------- importing
#: default mapping from DXF layer name to what the geometry becomes.  Matching
#: is case-insensitive and on substrings, so "SOIL-CLAY-2" and "clay" both hit
#: the soil rule.  English and Turkish names are both recognised because both
#: turn up in the same drawings.
DEFAULT_PATTERNS = {
    "soil": ("soil", "zemin", "strata", "stratum", "clay", "kil", "sand", "kum",
             "silt", "gravel", "cakil", "çakıl", "rock", "kaya", "fill", "dolgu",
             "layer", "tabaka", "ground"),
    # note: matching is done on the layer name with separators stripped, so
    # "SU-SEVIYESI" and "su seviyesi" both reduce to "suseviyesi"
    "structure": ("wall", "perde", "pile", "kazik", "kazık", "sheet", "palplanş",
                  "palplans", "diaphragm", "lining", "kaplama", "secant", "fore"),
    "water": ("water", "phreatic", "gwl", "ysz", "suseviye", "watertable",
              "yeraltisu", "yeraltısu"),
    "load": ("load", "surcharge", "yuk", "yük", "sürşarj", "sursarj"),
    "anchor": ("anchor", "ankraj", "strut", "prop", "tie", "destek", "payanda"),
    # An excavation region or level.  Its geometry is soil like any other; what
    # marks it out is that it is taken away, at the construction step given by
    # the number in the layer name.
    "excavation": ("exc", "kazi", "kazı", "lift", "dig", "kademe", "hafriyat",
                   "remove", "kaldir", "kaldır"),
    "ignore": ("text", "yazi", "yazı", "dim", "olcu", "ölçü", "hatch", "tarama",
               "title", "antet", "grid", "aks", "defpoints", "frame", "cerceve"),
}


@dataclass
class ImportRules:
    """How a drawing's layers map onto a model."""

    patterns: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in DEFAULT_PATTERNS.items()})
    #: metres per drawing unit; None reads it from the file's $INSUNITS
    scale: float | None = None
    #: move the geometry so its lower left corner sits at the origin.  CAD
    #: drawings are often on survey coordinates in the hundreds of thousands,
    #: where the mesher's geometric tests lose precision.
    shift_to_origin: bool = True
    #: endpoints closer than this are treated as the same point
    snap: float = 1.0e-3
    #: ignore faces smaller than this, which are artefacts of drawing slop
    min_area: float = 1.0e-4
    #: what an unrecognised layer becomes: soil if closed, a structure if not
    fallback_by_shape: bool = True
    #: read the trailing number in a layer name as the construction step at
    #: which that thing happens: "KAZI-2" is excavated at step 2, "PERDE-1" is
    #: built at step 1, "ANKRAJ-3" is stressed at step 3
    read_steps: bool = True
    #: add a strength reduction stage at the end of the sequence
    add_safety_stage: bool = True

    def step(self, layer: str) -> int | None:
        """The construction step a layer name carries, if any."""
        if not self.read_steps:
            return None
        match = re.search(r"(\d+)\s*$", layer.strip())
        if not match:
            return None
        value = int(match.group(1))
        return value if 1 <= value <= 99 else None

    def role(self, layer: str) -> str | None:
        """The role of a DXF layer.

        Returns ``"ignore"`` for layers the rules say to skip and ``None`` for
        ones no rule mentions, which are two different things: the first is a
        decision, the second leaves the fallback to decide.
        """
        # Drawing layers separate words with hyphens, underscores or spaces
        # in no consistent way, so flatten them all before matching.
        name = re.sub(r"[\s_\-.]+", "", layer.strip().lower())
        for role in ("ignore", "water", "anchor", "load", "excavation",
                     "structure", "soil"):
            for pattern in self.patterns.get(role, ()):
                if pattern in name:
                    return role
        return None


@dataclass
class ImportReport:
    """What the import made of the drawing."""

    source: str = ""
    scale: float = 1.0
    offset: tuple[float, float] = (0.0, 0.0)
    roles: dict[str, str] = field(default_factory=dict)      # dxf layer -> role
    counts: dict[str, int] = field(default_factory=dict)     # role -> entities
    soil_layers: list[tuple[str, float]] = field(default_factory=list)
    #: what happens at each construction step, in order
    steps: list[tuple[int, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"Imported {self.source or 'drawing'}",
                 f"  scale      {self.scale:g} m per drawing unit"]
        if self.offset != (0.0, 0.0):
            lines.append(f"  moved by   ({-self.offset[0]:.3f}, {-self.offset[1]:.3f}) m "
                         "to bring the geometry to the origin")
        for layer, role in sorted(self.roles.items()):
            lines.append(f"  layer {layer!r} -> {role}")
        for name, area in self.soil_layers:
            lines.append(f"  soil region {name!r}: {area:.2f} m2")
        for number, what in self.steps:
            lines.append(f"  step {number}: {what}")
        for warning in self.warnings:
            lines.append(f"  warning: {warning}")
        return "\n".join(lines)


def import_dxf(path: str, rules: ImportRules | None = None, name: str | None = None):
    """Read a DXF file and build a model from it.

    Returns ``(model, report)``.  The model has materials and stages at their
    defaults: the drawing fixes the geometry, and the properties are yours to
    set afterwards, in the interface or in the model file.
    """
    drawing = read_dxf(path)
    return build_model(drawing, rules=rules, name=name)


def build_model(drawing: DxfDrawing, rules: ImportRules | None = None,
                name: str | None = None):
    from .core.materials import MohrCoulomb
    from .core.mesher import PSLG, polygon_area
    from .core.model import Anchor, LineLoad, Model, SoilLayer, Structure, WaterTable
    from .core.elements import AnchorProperties
    from .core.pile import PileSection
    from .core.topology import find_faces, join_chains

    rules = rules or ImportRules()
    scale = rules.scale if rules.scale is not None else drawing.units
    report = ImportReport(source=drawing.source, scale=scale)

    grouped: dict[str, list[DxfEntity]] = {}
    unmatched: set[str] = set()
    for entity in drawing.entities:
        role = rules.role(entity.layer)
        if role is None:
            unmatched.add(entity.layer)
            if not rules.fallback_by_shape:
                continue
            # Nothing named it, so let the shape decide: a closed outline is a
            # region of soil, an open run of line is a structure.
            role = "soil" if entity.closed else "structure"
        if role == "ignore":
            report.roles[entity.layer] = "ignored"
            continue
        report.roles[entity.layer] = role
        grouped.setdefault(role, []).append(entity)
    if unmatched and rules.fallback_by_shape:
        report.warnings.append(
            "no rule named " + ", ".join(sorted(repr(u) for u in unmatched))
            + "; closed outlines there were read as soil and open lines as "
              "structures. Rename the drawing layers, or pass your own rules, "
              "if that is not what they are.")
    report.counts = {role: len(items) for role, items in grouped.items()}

    offset = (0.0, 0.0)
    if rules.shift_to_origin and drawing.entities:
        x0, y0, _x1, _y1 = drawing.bounds()
        offset = (x0 * scale, y0 * scale)
    report.offset = offset

    def place(points):
        return [(p[0] * scale - offset[0], p[1] * scale - offset[1]) for p in points]

    # ------------------------------------------------------------ soil regions
    soil_layers: list[SoilLayer] = []
    #: excavation geometry is soil like any other - it has to be meshed before
    #: it can be taken away - so the two are gathered together, with the step
    #: each excavation region belongs to kept alongside
    soil_entities = grouped.get("soil", []) + grouped.get("excavation", [])
    excavated_at: dict[str, int] = {}      # soil layer name -> step removed
    if soil_entities:
        pslg = PSLG()
        drawn: list[tuple[str, list[tuple[float, float]]]] = []
        for entity in soil_entities:
            pts = place(entity.points)
            pslg.add_polyline(pts, marker=1, closed=entity.closed)
            drawn.append((entity.layer, pts + ([pts[0]] if entity.closed else [])))
        # Walls bound regions too.  Excavation levels are usually drawn only
        # across the dig, from the wall outwards, so without the wall in the
        # arrangement they enclose nothing and the lifts never appear.  The
        # wall contributes edges but no name: regions are still named after
        # the soil that drew them.
        for entity in grouped.get("structure", []):
            wall = place(entity.points)
            if len(wall) >= 2:
                pslg.add_polyline(wall, marker=2, closed=entity.closed)
        # Planarising splits and renumbers everything, so which drawing layer a
        # segment came from has to be recovered from where it lies, not from
        # the indices it had beforehand.
        pslg.planarize(tol=rules.snap)
        owners = _segment_owners(pslg, drawn, rules.snap)

        faces = find_faces(pslg.points, pslg.segments, min_area=rules.min_area)
        # A stratum drawn as its own closed outline names its region exactly;
        # one bounded by shared lines can only be named by whichever layer drew
        # most of its boundary, which is a guess.
        closed_outlines = [(e.layer, place(e.points)) for e in soil_entities if e.closed]
        # Excavation levels drawn as open lines rather than closed regions: the
        # region they cut off from below is what comes out at that step.
        levels = [(rules.step(e.layer), place(e.points))
                  for e in grouped.get("excavation", [])
                  if not e.closed and rules.step(e.layer)]
        guessed = False
        used_names: dict[str, int] = {}
        for face in faces:
            outline = [pslg.points[i] for i in face]
            area = abs(polygon_area(outline))
            base = _matching_outline(outline, closed_outlines, rules.snap)
            if base is None:
                base = _face_name(face, owners) or "soil"
                guessed = True
            used_names[base] = used_names.get(base, 0) + 1
            label = base if used_names[base] == 1 else f"{base} {used_names[base]}"
            soil_layers.append(SoilLayer(
                name=label, polygon=outline,
                material=MohrCoulomb(name=label, color=_colour(len(soil_layers)))))
            report.soil_layers.append((label, area))

            step = rules.step(base) if rules.role(base) == "excavation" else None
            if step is None:
                step = _level_step(outline, levels, rules.snap)
            if step is not None:
                excavated_at[label] = step
        # A stratum cut up by the excavation regions drawn over it is the
        # intended arrangement, not a mistake, so only report the cuts that
        # something else made.
        excavation_outlines = [place(e.points) for e in grouped.get("excavation", [])
                               if e.closed and len(e.points) >= 3]
        structure_paths = [place(e.points) for e in grouped.get("structure", [])
                           if len(e.points) >= 2]
        subdivided = [layer for layer, outline in closed_outlines
                      if len(outline) >= 3
                      and _matching_face(outline, faces, pslg.points, rules.snap) is None
                      and not _cut_by_excavation(outline, excavation_outlines,
                                                 structure_paths)]
        if subdivided:
            report.warnings.append(
                "these closed outlines were cut into smaller regions, which "
                "happens when they overlap one another or a line crosses them: "
                + ", ".join(sorted(set(repr(s) for s in subdivided)))
                + ". Check the drawing if that was not intended.")
        ambiguous = sorted(base for base, count in used_names.items() if count > 1)
        if guessed and ambiguous:
            report.warnings.append(
                "more than one region ended up named after "
                + ", ".join(repr(a) for a in ambiguous)
                + ", because they are bounded by lines shared with their "
                  "neighbours rather than by their own outline. Rename them "
                  "once imported, or draw each region as its own closed polyline.")
        if not faces:
            report.warnings.append(
                "no closed region was found in the soil layers; check that the "
                "boundaries meet, or close the polylines in the drawing")
    else:
        report.warnings.append("no layer matched the soil rules, so the model has "
                               "no soil; rename the layers or pass your own rules")

    # -------------------------------------------------------------- structures
    structures: list[Structure] = []
    for entity in grouped.get("structure", []):
        path = place(entity.points)
        if len(path) < 2:
            continue
        label = _unique(entity.layer or "wall", [s.name for s in structures])
        structures.append(Structure(name=label, path=path, section=PileSection(name=label)))

    # ------------------------------------------------------------------ water
    water_points: list[tuple[float, float]] = []
    for entity in grouped.get("water", []):
        water_points.extend(place(entity.points))
    water = WaterTable(sorted(water_points)) if len(water_points) >= 2 else WaterTable()

    # ------------------------------------------------------------------ loads
    loads: list[LineLoad] = []
    for entity in grouped.get("load", []):
        path = place(entity.points)
        if len(path) >= 2:
            loads.append(LineLoad(name=_unique(entity.layer or "load",
                                               [l.name for l in loads]),
                                  path=path, qy=0.0))
    if loads:
        report.warnings.append("line loads were imported with zero magnitude; "
                               "set q before running")

    # ---------------------------------------------------------------- anchors
    anchors: list[Anchor] = []
    for entity in grouped.get("anchor", []):
        path = place(entity.points)
        if len(path) < 2:
            continue
        anchors.append(Anchor(name=_unique(entity.layer or "anchor",
                                           [a.name for a in anchors]),
                              start=path[0], end=path[-1],
                              properties=AnchorProperties()))

    # ----------------------------------------------------------------- joining
    open_soil = [e for e in soil_entities if not e.closed]
    if open_soil and not soil_layers:
        chains, closed_flags = join_chains([place(e.points) for e in open_soil],
                                           tolerance=rules.snap)
        for chain, is_closed in zip(chains, closed_flags):
            if is_closed and len(chain) >= 3:
                label = _unique("soil", [s.name for s in soil_layers])
                soil_layers.append(SoilLayer(name=label, polygon=chain,
                                             material=MohrCoulomb(name=label)))

    names = [layer.name for layer in soil_layers]
    stages, step_notes = _stages(soil_layers, excavated_at, structures, anchors, loads,
                                 rules)
    report.steps = step_notes
    if excavated_at:
        removed = sum(layer.area() for layer in soil_layers
                      if layer.name in excavated_at)
        report.warnings.append(
            f"{len(excavated_at)} region(s), {removed:.1f} m2 in all, are "
            "excavated by the end of the sequence. Check the stage list before "
            "running: the drawing fixes what comes out and when, nothing else.")

    model = Model(
        name=name or "imported model",
        layers=soil_layers,
        structures=structures,
        anchors=anchors,
        line_loads=loads,
        water=water,
        stages=stages,
    )
    issues = model.validate()
    report.warnings.extend(issues)
    return model, report


def _cut_by_excavation(outline, excavation_outlines, structures=()) -> bool:
    """Whether something meant to divide this outline is drawn across it.

    An excavation region drawn over a stratum, or a wall running through it,
    subdivides it by design; only the cuts nothing accounts for are worth
    reporting.
    """
    from .core.mesher import point_in_polygon

    for other in excavation_outlines:
        _area, centre = _polygon_area_and_centroid(other)
        if point_in_polygon(centre[0], centre[1], outline):
            return True
    for path in structures:
        if any(point_in_polygon(p[0], p[1], outline) for p in path):
            return True
    return False


def _level_step(face, levels, tol: float) -> int | None:
    """The step at which a region bounded below by an excavation level goes.

    Digging works downwards, so the material taken out at a step is what sits
    directly above that step's level.  A region is assigned to the level that
    forms its floor: the highest one that runs below it and touches its
    boundary.
    """
    if not levels:
        return None
    _area, centre = _polygon_area_and_centroid(face)
    best: tuple[float, int] | None = None
    for step, path in levels:
        if not any(_on_path(_midpoint(face[i], face[(i + 1) % len(face)]), path, tol)
                   for i in range(len(face))):
            continue
        elevation = _elevation_at(path, centre[0])
        if elevation is None or elevation > centre[1]:
            continue                      # the level is above: not this floor
        if best is None or elevation > best[0]:
            best = (elevation, step)
    return None if best is None else best[1]


def _midpoint(a, b):
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def _elevation_at(path, x: float) -> float | None:
    """Height of a polyline at a given x, or None where it does not reach."""
    best = None
    for a, b in zip(path[:-1], path[1:]):
        lo, hi = min(a[0], b[0]), max(a[0], b[0])
        if x < lo - 1e-9 or x > hi + 1e-9:
            continue
        if abs(b[0] - a[0]) < 1e-12:
            y = max(a[1], b[1])
        else:
            t = (x - a[0]) / (b[0] - a[0])
            y = a[1] + t * (b[1] - a[1])
        best = y if best is None else max(best, y)
    return best


def _matching_face(outline, faces, points, tol: float):
    """The face an input outline traces exactly, if it traces one."""
    return _matching_outline(outline, [(i, [points[k] for k in face])
                                       for i, face in enumerate(faces)], tol)


def _matching_outline(face, closed_outlines, tol: float):
    """The label of a closed outline that traces this face, if one does.

    Matching is by area and centroid, both computed as polygon integrals so
    that they do not change when planarising adds vertices along an edge -
    which it always does where a neighbouring stratum meets the boundary.
    """
    area, centre = _polygon_area_and_centroid(face)
    if area <= 0.0:
        return None
    reach = max(tol, 1e-9) + 1e-4 * math.sqrt(area)
    for label, outline in closed_outlines:
        if len(outline) < 3:
            continue
        other_area, other_centre = _polygon_area_and_centroid(outline)
        if other_area <= 0.0 or abs(other_area - area) > 1e-6 * max(area, other_area):
            continue
        if math.dist(centre, other_centre) <= reach:
            return label
    return None


def _polygon_area_and_centroid(poly):
    """Unsigned area and centroid of a polygon, as integrals over its interior."""
    twice_area = 0.0
    cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        twice_area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(twice_area) < 1e-300:
        return 0.0, (0.0, 0.0)
    return abs(twice_area) / 2.0, (cx / (3.0 * twice_area), cy / (3.0 * twice_area))


def _segment_owners(pslg, drawn, tol: float) -> dict[tuple[int, int], str]:
    """Which drawing layer each planarised segment came from."""
    owners: dict[tuple[int, int], str] = {}
    reach = max(tol, 1e-9)
    for (i, j) in pslg.segments:
        ax, ay = pslg.points[i]
        bx, by = pslg.points[j]
        mid = (0.5 * (ax + bx), 0.5 * (ay + by))
        for layer, path in drawn:
            if _on_path(mid, path, reach):
                owners[(min(i, j), max(i, j))] = layer
                break
    return owners


def _on_path(point, path, tol: float) -> bool:
    px, py = point
    for a, b in zip(path[:-1], path[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        if length2 <= 0.0:
            continue
        t = ((px - a[0]) * dx + (py - a[1]) * dy) / length2
        if t < -1e-9 or t > 1.0 + 1e-9:
            continue
        t = min(1.0, max(0.0, t))
        if math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy)) <= tol:
            return True
    return False


def _stages(soil_layers, excavated_at, structures, anchors, loads, rules):
    """Build the construction sequence the drawing describes.

    A step number in a layer name says when that thing happens: soil on
    "KAZI-2" is dug out at step 2, a wall on "PERDE-1" is built at step 1, an
    anchor on "ANKRAJ-3" is stressed at step 3.  Anything unnumbered is there
    from the first step.
    """
    from .core.model import Stage

    numbered_structures = {s.name: rules.step(s.name) for s in structures}
    numbered_anchors = {a.name: rules.step(a.name) for a in anchors}
    numbered_loads = {load.name: rules.step(load.name) for load in loads}

    steps = sorted({n for n in list(excavated_at.values())
                    + list(numbered_structures.values())
                    + list(numbered_anchors.values())
                    + list(numbered_loads.values()) if n})
    all_names = [layer.name for layer in soil_layers]

    def upto(mapping, n):
        return [k for k, v in mapping.items() if v is None or v <= n]

    notes: list[tuple[int, str]] = []
    stages = [Stage("1 - initial stresses", kind="initial", active_layers=all_names,
                    active_structures=[], active_anchors=[], active_loads=[])]
    if not steps:
        stages.append(Stage("2 - construction", kind="plastic",
                            active_structures=[s.name for s in structures],
                            active_anchors=[a.name for a in anchors],
                            active_loads=[load.name for load in loads]))
    else:
        for n in steps:
            gone = {k for k, v in excavated_at.items() if v <= n}
            built = upto(numbered_structures, n)
            stressed = upto(numbered_anchors, n)
            applied = upto(numbered_loads, n)
            what = _describe_step(n, excavated_at, numbered_structures,
                                  numbered_anchors, numbered_loads)
            notes.append((n, what))
            stages.append(Stage(
                f"{len(stages) + 1} - {what}", kind="plastic",
                active_layers=[name for name in all_names if name not in gone],
                active_structures=built, active_anchors=stressed,
                active_loads=applied,
                reset_displacements=(len(stages) == 1)))
    if rules.add_safety_stage:
        stages.append(Stage(f"{len(stages) + 1} - factor of safety", kind="ssr",
                            srf_min=0.8, srf_max=3.0))
    return stages, notes


def _describe_step(n, excavated_at, structures, anchors, loads) -> str:
    parts = []
    built = sorted(k for k, v in structures.items() if v == n)
    if built:
        parts.append("build " + ", ".join(built))
    stressed = sorted(k for k, v in anchors.items() if v == n)
    if stressed:
        parts.append("stress " + ", ".join(stressed))
    applied = sorted(k for k, v in loads.items() if v == n)
    if applied:
        parts.append("apply " + ", ".join(applied))
    dug = sorted(k for k, v in excavated_at.items() if v == n)
    if dug:
        parts.append("excavate " + ", ".join(dug))
    return "; ".join(parts) if parts else f"step {n}"


def _face_name(face, owners) -> str | None:
    """Name a region after whichever drawing layer drew most of its boundary."""
    tally: dict[str, int] = {}
    for i in range(len(face)):
        key = (min(face[i], face[(i + 1) % len(face)]),
               max(face[i], face[(i + 1) % len(face)]))
        layer = owners.get(key)
        if layer:
            tally[layer] = tally.get(layer, 0) + 1
    if not tally:
        return None
    return max(tally, key=tally.get)


def _unique(base: str, taken: list[str]) -> str:
    if base not in taken:
        return base
    i = 2
    while f"{base} {i}" in taken:
        i += 1
    return f"{base} {i}"


def _colour(index: int) -> str:
    palette = ("#c8b273", "#a8907a", "#9aa7a0", "#d2b48c", "#b9a878",
               "#8fa2ad", "#c2a68c", "#7d8a83")
    return palette[index % len(palette)]
