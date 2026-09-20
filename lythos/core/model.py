"""Problem definition: geometry, materials, structures, water and stages.

A :class:`Model` is what the user builds in the interface or in a script.  It
holds only geometry and properties; :meth:`Model.build` turns it into a meshed
:class:`~lythos.core.problem.FEProblem` ready for the solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .elements import AnchorProperties, InterfaceProperties
from .materials import Material, MohrCoulomb
from .mesher import polygon_area
from .pile import PileSection, WallSection

GAMMA_WATER = 9.81


@dataclass
class SoilLayer:
    """A closed region of soil or rock with one material."""

    name: str
    polygon: list[tuple[float, float]]
    material: Material = field(default_factory=MohrCoulomb)
    mesh_size: float | None = None      # target element size [m]

    def area(self) -> float:
        return abs(polygon_area(self.polygon))

    def normalised(self) -> list[tuple[float, float]]:
        """Outline in counter-clockwise order."""
        return list(self.polygon) if polygon_area(self.polygon) > 0 else list(reversed(self.polygon))


@dataclass
class Structure:
    """A wall, pile row, lining or other beam-like member along a polyline."""

    name: str
    path: list[tuple[float, float]]
    section: PileSection | WallSection = field(default_factory=WallSection)
    interface: InterfaceProperties | None = None
    #: strength reduction factor applied to the soil at the interface
    r_inter: float = 0.67

    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.path[:-1], self.path[1:]))


@dataclass
class Anchor:
    """A strut, prop or ground anchor connecting two points."""

    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    properties: AnchorProperties = field(default_factory=AnchorProperties)


@dataclass
class LineLoad:
    """Distributed load along a line, in kN/m of length."""

    name: str
    path: list[tuple[float, float]]
    qx: float = 0.0
    qy: float = 0.0        # negative acts downwards


@dataclass
class PointLoad:
    name: str
    x: float
    y: float
    fx: float = 0.0
    fy: float = 0.0


@dataclass
class WaterTable:
    """Phreatic surface as a polyline; pore pressure is hydrostatic below it."""

    points: list[tuple[float, float]] = field(default_factory=list)

    def head(self, x):
        """Phreatic elevation at ``x`` (extrapolated flat beyond the ends)."""
        import numpy as np
        if not self.points:
            return np.full_like(np.asarray(x, float), -1.0e30)
        pts = sorted(self.points)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return np.interp(np.asarray(x, float), xs, ys)


@dataclass
class Stage:
    """One construction step.

    Anything left as ``None`` is inherited from the previous stage, which keeps
    a long excavation sequence readable.
    """

    name: str
    kind: str = "plastic"                     # initial | plastic | ssr
    active_layers: list[str] | None = None    # names of layers that exist
    active_structures: list[str] | None = None
    active_anchors: list[str] | None = None
    active_loads: list[str] | None = None
    water: WaterTable | None = None
    reset_displacements: bool = False
    increments: int = 10
    #: SSR search bracket, used when ``kind == "ssr"``
    srf_min: float = 0.4
    srf_max: float = 4.0
    notes: str = ""


@dataclass
class BoundaryConditions:
    """Standard fixities, or explicit dof constraints."""

    fix_bottom: bool = True          # both directions on the lowest edge
    fix_sides: bool = True           # horizontal restraint on the vertical edges
    fixed_nodes: list[tuple[float, float, bool, bool]] = field(default_factory=list)


@dataclass
class Model:
    """A complete 2D geotechnical finite element model."""

    name: str = "model"
    layers: list[SoilLayer] = field(default_factory=list)
    structures: list[Structure] = field(default_factory=list)
    anchors: list[Anchor] = field(default_factory=list)
    line_loads: list[LineLoad] = field(default_factory=list)
    point_loads: list[PointLoad] = field(default_factory=list)
    water: WaterTable = field(default_factory=WaterTable)
    stages: list[Stage] = field(default_factory=list)
    boundary: BoundaryConditions = field(default_factory=BoundaryConditions)
    mesh_size: float | None = None
    min_angle: float = 25.0
    initial_stress: str = "k0"        # k0 | gravity
    gamma_water: float = GAMMA_WATER

    # ------------------------------------------------------------------ lookup
    def layer(self, name: str) -> SoilLayer:
        for lay in self.layers:
            if lay.name == name:
                return lay
        raise KeyError(f"no layer named {name!r}")

    def structure(self, name: str) -> Structure:
        for s in self.structures:
            if s.name == name:
                return s
        raise KeyError(f"no structure named {name!r}")

    def bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for lay in self.layers for p in lay.polygon]
        ys = [p[1] for lay in self.layers for p in lay.polygon]
        if not xs:
            raise ValueError("the model has no soil layers")
        return min(xs), min(ys), max(xs), max(ys)

    def default_mesh_size(self) -> float:
        x0, y0, x1, y1 = self.bounds()
        return max(x1 - x0, y1 - y0) / 25.0

    def resolved_stages(self) -> list[Stage]:
        """Stages with inherited fields filled in."""
        out: list[Stage] = []
        current = {
            "active_layers": [lay.name for lay in self.layers],
            "active_structures": [],
            "active_anchors": [],
            "active_loads": [],
            "water": self.water,
        }
        for st in self.stages:
            for key in ("active_layers", "active_structures", "active_anchors",
                        "active_loads", "water"):
                value = getattr(st, key)
                if value is None:
                    setattr(st, key, current[key])
                else:
                    current[key] = value
            out.append(st)
        return out

    def validate(self) -> list[str]:
        """Human-readable problems with the model (empty when it is sound)."""
        issues: list[str] = []
        if not self.layers:
            issues.append("the model has no soil layers")
        for lay in self.layers:
            if len(lay.polygon) < 3:
                issues.append(f"layer {lay.name!r} needs at least three points")
            elif lay.area() <= 0:
                issues.append(f"layer {lay.name!r} has zero area")
        names = [lay.name for lay in self.layers]
        if len(set(names)) != len(names):
            issues.append("layer names must be unique")
        for s in self.structures:
            if len(s.path) < 2:
                issues.append(f"structure {s.name!r} needs at least two points")
        if not self.stages:
            issues.append("the model has no calculation stages")
        elif self.stages[0].kind != "initial":
            issues.append("the first stage should be the initial stress stage")
        known = set(names)
        for st in self.stages:
            for lay_name in (st.active_layers or []):
                if lay_name not in known:
                    issues.append(f"stage {st.name!r} refers to unknown layer {lay_name!r}")
        return issues

    # --------------------------------------------------------------- building
    def build(self, mesh_size: float | None = None):
        from .problem import build_problem
        return build_problem(self, mesh_size=mesh_size)
