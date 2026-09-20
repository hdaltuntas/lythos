"""Turning a :class:`~lythos.core.model.Model` into a solvable FE problem."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .assembly import DofManager
from .elements import (AnchorElements, AnchorProperties, BeamElements, ContinuumElements,
                       InterfaceElements, InterfaceProperties, LINE_GAUSS_3, line3_shape)
from .materials import MaterialState, Material, MohrCoulomb
from .mesh import Mesh, build_mesh, order_chain
from .mesher import MeshGenerator, PSLG
from .model import GAMMA_WATER, Model, WaterTable

STRUCTURE_MARKER = 1000
LOAD_MARKER = 5000


@dataclass
class StructureFE:
    """Meshed form of a wall or pile row."""

    name: str
    beam: BeamElements
    dof_map: np.ndarray
    chain: list[int]
    interfaces: list[InterfaceElements] = field(default_factory=list)
    interface_dofs: list[np.ndarray] = field(default_factory=list)

    def arc_length(self, nodes: np.ndarray) -> np.ndarray:
        xy = nodes[self.chain]
        d = np.r_[0.0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))]
        return d


@dataclass
class FEProblem:
    """Mesh, elements, materials and boundary conditions for one model."""

    model: Model
    mesh: Mesh
    continuum: ContinuumElements
    dofs: DofManager
    layer_of_element: np.ndarray
    materials: list[Material]
    structures: list[StructureFE] = field(default_factory=list)
    anchors: AnchorElements | None = None
    anchor_names: list[str] = field(default_factory=list)
    fixed_dofs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    state: MaterialState | None = None
    displacement: np.ndarray | None = None

    # ------------------------------------------------------------------ groups
    def material_groups(self):
        """(material, element indices, Gauss point indices) for each layer."""
        ngp = self.continuum.n_gauss
        for i, mat in enumerate(self.materials):
            elems = np.nonzero(self.layer_of_element == i)[0]
            if len(elems) == 0:
                continue
            gp = (elems[:, None] * ngp + np.arange(ngp)[None, :]).ravel()
            yield mat, elems, gp

    def active_elements(self, layer_names) -> np.ndarray:
        names = {lay.name for lay in self.model.layers if lay.name in set(layer_names)}
        keep = np.array([self.model.layers[i].name in names for i in self.layer_of_element])
        return keep

    # ------------------------------------------------------------- water & body
    def pore_pressure(self, xy: np.ndarray, water: WaterTable) -> np.ndarray:
        """Hydrostatic pore pressure (positive in compression) at points."""
        if not water or not water.points:
            return np.zeros(len(xy))
        head = np.asarray(water.head(xy[:, 0]), float)
        return np.maximum(head - xy[:, 1], 0.0) * self.model.gamma_water

    def body_force(self, active: np.ndarray, water: WaterTable) -> np.ndarray:
        """Effective-stress body force on the active soil skeleton.

        Working in effective stress, equilibrium reads div(sigma') + b - grad(p)
        = 0.  Below a horizontal phreatic surface that is simply the buoyant
        unit weight; where the surface is inclined the horizontal part of
        grad(p) is the seepage force, and it is included here.
        """
        ce = self.continuum
        gw = self.model.gamma_water
        gauss = ce.gauss_xy.reshape(-1, 2)
        submerged = np.zeros(len(gauss), bool)
        dhdx = np.zeros(len(gauss))
        if water and water.points:
            head = np.asarray(water.head(gauss[:, 0]), float)
            submerged = gauss[:, 1] < head
            step = max(1e-4, 1e-3 * (self.mesh.bounds()[2] - self.mesh.bounds()[0]))
            dhdx = (np.asarray(water.head(gauss[:, 0] + step), float)
                    - np.asarray(water.head(gauss[:, 0] - step), float)) / (2.0 * step)

        gamma = np.zeros(len(gauss))
        for i, mat in enumerate(self.materials):
            mask = np.repeat(self.layer_of_element == i, ce.n_gauss)
            wet = mask & submerged
            dry = mask & ~submerged
            gamma[dry] = mat.unit_weight(False)
            gamma[wet] = mat.unit_weight(True) - gw

        fx = np.where(submerged, -gw * dhdx, 0.0)
        fy = -gamma

        f = np.zeros((ce.n_elements, 12))
        fx = fx.reshape(ce.n_elements, ce.n_gauss)
        fy = fy.reshape(ce.n_elements, ce.n_gauss)
        for g in range(ce.n_gauss):
            w = ce.detJw[:, g]
            f[:, 0::2] += np.outer(fx[:, g] * w, ce.N[g])
            f[:, 1::2] += np.outer(fy[:, g] * w, ce.N[g])
        f[~active] = 0.0
        return f

    # ---------------------------------------------------------------- stresses
    def initial_k0_stress(self, active: np.ndarray, water: WaterTable) -> np.ndarray:
        """Vertical integration of self weight, with horizontal stress K0 sigma_v."""
        ce = self.continuum
        gauss = ce.gauss_xy.reshape(-1, 2)
        active_gp = np.repeat(active, ce.n_gauss)
        sv = np.zeros(len(gauss))
        k0 = np.ones(len(gauss))

        active_layers = [self.model.layers[i] for i in range(len(self.materials))
                         if np.any(active & (self.layer_of_element == i))]
        for p_index, (x, y) in enumerate(gauss):
            if not active_gp[p_index]:
                continue
            total = 0.0
            for lay in active_layers:
                mat = lay.material
                for (y0, y1) in _vertical_spans(lay.polygon, x):
                    lo, hi = max(y0, y), y1
                    if hi <= lo:
                        continue
                    hw = float(water.head(np.array([x]))[0]) if (water and water.points) else -1e30
                    wet = max(0.0, min(hi, hw) - lo)
                    dry = (hi - lo) - wet
                    total += dry * mat.unit_weight(False)
                    total += wet * (mat.unit_weight(True) - self.model.gamma_water)
            sv[p_index] = total

        for i, mat in enumerate(self.materials):
            mask = np.repeat(self.layer_of_element == i, ce.n_gauss)
            k0[mask] = mat.k0()

        stress = np.zeros((len(gauss), 4))
        stress[:, 1] = -sv                    # tension positive
        stress[:, 0] = -k0 * sv
        stress[:, 2] = -k0 * sv
        return stress

    # ------------------------------------------------------------------- loads
    def line_load_vector(self, names) -> np.ndarray:
        f = np.zeros(self.dofs.n_dof)
        chosen = {n for n in (names or [])}
        for i, load in enumerate(self.model.line_loads):
            if load.name not in chosen:
                continue
            edges = self.mesh.segments.get(LOAD_MARKER + i, [])
            for (a, m, b) in edges:
                xy = self.mesh.nodes[[a, m, b]]
                for xi, w in LINE_GAUSS_3:
                    N, dN = line3_shape(xi)
                    jac = float(np.hypot(*(dN @ xy)))
                    for k, node in enumerate((a, m, b)):
                        f[2 * node] += w * jac * N[k] * load.qx
                        f[2 * node + 1] += w * jac * N[k] * load.qy
        for load in self.model.point_loads:
            node = self.mesh.nearest_node(load.x, load.y)
            f[2 * node] += load.fx
            f[2 * node + 1] += load.fy
        return f

    def structure_weight(self) -> dict[str, np.ndarray]:
        out = {}
        for s in self.structures:
            out[s.name] = s.beam.self_weight(s.dof_map)
        return out


def _vertical_spans(polygon, x: float) -> list[tuple[float, float]]:
    """Intervals of the vertical line at ``x`` that lie inside ``polygon``."""
    crossings = []
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if (x0 <= x < x1) or (x1 <= x < x0):
            t = (x - x0) / (x1 - x0)
            crossings.append(y0 + t * (y1 - y0))
    crossings.sort()
    return [(crossings[i], crossings[i + 1]) for i in range(0, len(crossings) - 1, 2)]


def _side_of_edge(nodes, a, b, c) -> float:
    """Sign of the cross product (b - a) x (c - a)."""
    ax, ay = nodes[a]
    bx, by = nodes[b]
    cx, cy = nodes[c]
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def build_problem(model: Model, mesh_size: float | None = None) -> FEProblem:
    """Mesh the model and assemble all element groups."""
    issues = model.validate()
    fatal = [m for m in issues if "no soil layers" in m or "zero area" in m]
    if fatal:
        raise ValueError("; ".join(fatal))

    target = mesh_size or model.mesh_size or model.default_mesh_size()
    pslg = PSLG()
    for i, lay in enumerate(model.layers):
        outline = lay.normalised()
        pslg.add_polyline(outline, marker=1 + i, closed=True)
        size = lay.mesh_size or target
        pslg.polygons.append((i, outline, 0.5 * size * size))
    for i, s in enumerate(model.structures):
        pslg.add_polyline(s.path, marker=STRUCTURE_MARKER + i)
    for i, load in enumerate(model.line_loads):
        pslg.add_polyline(load.path, marker=LOAD_MARKER + i)

    gen = MeshGenerator(pslg, min_angle=model.min_angle, max_area=0.5 * target * target)
    T = gen.run()
    mesh = build_mesh(T)

    # Node splitting has to happen before the element operators are built, so
    # every structure that needs an interface is processed first.
    split: dict[int, tuple[list[int], list[int], list[int] | None]] = {}
    for i, st in enumerate(model.structures):
        edges = mesh.segments.get(STRUCTURE_MARKER + i, [])
        if not edges:
            continue
        chains = order_chain(edges)
        if not chains:
            continue
        chain = [int(v) for v in max(chains, key=len)]
        if st.interface is None:
            split[i] = (chain, chain, None)
        else:
            chain_w, chain_b = _split_structure_nodes(mesh, chain)
            split[i] = (chain, chain_w, chain_b)

    continuum = ContinuumElements(mesh.nodes, mesh.elements)
    dofs = DofManager(mesh.n_nodes)
    materials = [lay.material for lay in model.layers]
    layer_of_element = mesh.element_tags.copy()

    problem = FEProblem(model=model, mesh=mesh, continuum=continuum, dofs=dofs,
                        layer_of_element=layer_of_element, materials=materials)

    for i, st in enumerate(model.structures):
        if i not in split:
            continue
        chain_a, chain_w, chain_b = split[i]
        for n in chain_w:
            dofs.add_rotation(int(n))
        beam = BeamElements(mesh.nodes, [list(chain_w)], st.section.section())
        dof_map = (np.array([dofs.beam_dofs(el) for el in beam.elements], dtype=np.int64)
                   if beam.n_elements else np.zeros((0, 9), np.int64))
        sfe = StructureFE(name=st.name, beam=beam, dof_map=dof_map, chain=list(chain_w))
        if st.interface is not None and chain_b is not None:
            props = _interface_properties(st, model, problem, chain_a)
            for pairs in (_pair_chain(chain_a, chain_w), _pair_chain(chain_w, chain_b)):
                if pairs:
                    ie = InterfaceElements(mesh.nodes, pairs, props)
                    sfe.interfaces.append(ie)
                    sfe.interface_dofs.append(ie.dofs())
        problem.structures.append(sfe)

    # ----------------------------------------------------------------- anchors
    if model.anchors:
        pairs = []
        props = []
        for a in model.anchors:
            i0 = mesh.nearest_node(*a.start)
            i1 = mesh.nearest_node(*a.end)
            if i0 == i1:
                continue
            pairs.append((i0, i1))
            props.append(a.properties)
        if pairs:
            problem.anchors = AnchorElements(mesh.nodes, pairs, props)
            problem.anchor_names = [a.name for a in model.anchors]

    problem.fixed_dofs = _boundary_dofs(model, mesh, dofs)
    problem.state = MaterialState.zeros(continuum.n_points)
    problem.displacement = np.zeros(dofs.n_dof)
    return problem


def _pair_chain(chain_a, chain_b):
    """Quadratic interface element node pairs between two parallel chains."""
    if chain_b is None or len(chain_a) != len(chain_b):
        return []
    pairs = []
    for i in range(0, len(chain_a) - 2, 2):
        pairs.append((tuple(int(v) for v in chain_a[i:i + 3]),
                      tuple(int(v) for v in chain_b[i:i + 3])))
    return pairs


def _interface_properties(structure, model, problem, chain) -> InterfaceProperties:
    """Interface strength as R_inter times the strength of the adjacent soil."""
    props = structure.interface
    if props is None:
        return InterfaceProperties()
    if props.c or props.phi:
        return props
    soil = _adjacent_material(problem, chain)
    c = getattr(soil, "c", 0.0) * structure.r_inter
    phi = math.degrees(math.atan(structure.r_inter
                                 * math.tan(math.radians(getattr(soil, "phi", 25.0)))))
    # A stiff interface: soft enough to avoid ill-conditioning, stiff enough
    # that elastic slip stays small compared with the soil deformation.
    E = getattr(soil, "E", 3.0e4)
    kn = props.kn if props.kn else 100.0 * E
    ks = props.ks if props.ks else 100.0 * E
    return InterfaceProperties(kn=kn, ks=ks, c=c, phi=phi, tensile=props.tensile)


def _adjacent_material(problem, chain):
    """Material of the soil layer that most of the chain runs through."""
    counts: dict[int, int] = {}
    chain_set = set(int(v) for v in chain)
    for e, el in enumerate(problem.mesh.elements):
        if chain_set.intersection(int(v) for v in el):
            tag = int(problem.layer_of_element[e])
            counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return problem.materials[0]
    return problem.materials[max(counts, key=counts.get)]


def _split_structure_nodes(mesh: Mesh, chain: list[int]):
    """Duplicate the nodes along a structure so it can slip against the soil.

    Three sets of nodes end up on the line: the soil on the negative side keeps
    the originals, the structure gets its own copy, and the soil on the
    positive side gets a second copy.  Elements that only touch the line at a
    single node - around the toe of a wall, for instance - keep the original
    node, so the soil stays continuous where it wraps round the end.
    """
    chain_set = set(int(v) for v in chain)
    nodes = mesh.nodes
    n0 = len(nodes)

    wall_nodes = {int(v): n0 + i for i, v in enumerate(chain)}
    n1 = n0 + len(chain)
    side_nodes = {int(v): n1 + i for i, v in enumerate(chain)}

    extra = np.vstack([nodes[[int(v) for v in chain]], nodes[[int(v) for v in chain]]])
    mesh.nodes = np.vstack([nodes, extra])

    # orientation reference: the chain direction
    ref = nodes[int(chain[-1])] - nodes[int(chain[0])]
    for e, el in enumerate(mesh.elements):
        on_line = [int(v) for v in el if int(v) in chain_set]
        if len(on_line) < 2:
            continue
        centroid = nodes[el[:3]].mean(axis=0)
        anchor = nodes[on_line[0]]
        cross = ref[0] * (centroid[1] - anchor[1]) - ref[1] * (centroid[0] - anchor[0])
        if cross > 0:                       # positive side -> duplicated nodes
            for k, v in enumerate(el):
                if int(v) in side_nodes:
                    mesh.elements[e, k] = side_nodes[int(v)]

    chain_w = [wall_nodes[int(v)] for v in chain]
    chain_b = [side_nodes[int(v)] for v in chain]
    return chain_w, chain_b


def _boundary_dofs(model: Model, mesh: Mesh, dofs: DofManager) -> np.ndarray:
    """Standard fixities: rollers on the sides, pinned along the base."""
    x0, y0, x1, y1 = mesh.bounds()
    tol = 1e-6 * max(x1 - x0, y1 - y0, 1.0)
    fixed: set[int] = set()
    bc = model.boundary
    if bc.fix_bottom:
        for n in np.nonzero(np.abs(mesh.nodes[:, 1] - y0) < tol)[0]:
            fixed.add(2 * int(n))
            fixed.add(2 * int(n) + 1)
    if bc.fix_sides:
        side = (np.abs(mesh.nodes[:, 0] - x0) < tol) | (np.abs(mesh.nodes[:, 0] - x1) < tol)
        for n in np.nonzero(side)[0]:
            fixed.add(2 * int(n))
    for (x, y, fx, fy) in bc.fixed_nodes:
        n = mesh.nearest_node(x, y)
        if fx:
            fixed.add(2 * n)
        if fy:
            fixed.add(2 * n + 1)
    return np.array(sorted(fixed), dtype=np.int64)
