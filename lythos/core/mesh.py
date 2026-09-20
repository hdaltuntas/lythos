"""Finite element mesh built from a refined triangulation.

Continuum elements are 6-node quadratic triangles.  Quadratic triangles are
used rather than 3-node triangles because constant-strain triangles lock badly
once a Mohr-Coulomb soil starts flowing plastically at constant volume, and the
collapse loads they predict are far too high - which matters directly for a
strength reduction factor of safety.

Node ordering follows the usual convention::

        2
        | \\
        5   4
        |     \\
        0 - 3 - 1

corner nodes 0-1-2 counter-clockwise, then the midside nodes of edges 0-1,
1-2 and 2-0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesher import EXTERIOR, MeshGenerator, PSLG, Triangulation

#: local corner pairs for each element edge, in node order
EDGE_NODES = ((0, 1, 3), (1, 2, 4), (2, 0, 5))


@dataclass
class Mesh:
    """Nodes, elements and the named lines that structural elements live on."""

    nodes: np.ndarray                      # (nn, 2) coordinates
    elements: np.ndarray                   # (ne, 6) connectivity
    element_tags: np.ndarray               # (ne,) region tag of each element
    #: marker -> ordered list of (start, mid, end) node triples along a line
    segments: dict[int, list[tuple[int, int, int]]] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    def corner_triangles(self) -> np.ndarray:
        return self.elements[:, :3]

    def element_areas(self) -> np.ndarray:
        p = self.nodes[self.elements[:, :3]]
        return 0.5 * np.abs(
            (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
            - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
        )

    def centroids(self) -> np.ndarray:
        return self.nodes[self.elements[:, :3]].mean(axis=1)

    def bounds(self) -> tuple[float, float, float, float]:
        return (float(self.nodes[:, 0].min()), float(self.nodes[:, 1].min()),
                float(self.nodes[:, 0].max()), float(self.nodes[:, 1].max()))

    # ------------------------------------------------------------------ lookup
    def node_at(self, x: float, y: float, tol: float = 1e-6) -> int:
        d = np.hypot(self.nodes[:, 0] - x, self.nodes[:, 1] - y)
        i = int(np.argmin(d))
        return i if d[i] <= tol else -1

    def nearest_node(self, x: float, y: float) -> int:
        return int(np.argmin(np.hypot(self.nodes[:, 0] - x, self.nodes[:, 1] - y)))

    def nodes_on_line(self, p0, p1, tol: float = 1e-6) -> np.ndarray:
        """Indices of nodes lying on the closed segment p0-p1."""
        a = np.asarray(p0, float)
        b = np.asarray(p1, float)
        d = b - a
        length2 = float(d @ d)
        if length2 == 0.0:
            return np.array([self.node_at(a[0], a[1], tol)], dtype=int)
        rel = self.nodes - a
        t = (rel @ d) / length2
        perp = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]) / np.sqrt(length2)
        m = (t >= -tol) & (t <= 1.0 + tol) & (perp <= tol)
        idx = np.nonzero(m)[0]
        return idx[np.argsort(t[idx])]

    def boundary_edges(self) -> list[tuple[int, int, int, int]]:
        """Edges used by a single element, as (elem, start, mid, end)."""
        seen: dict[frozenset[int], tuple[int, tuple[int, int, int]]] = {}
        for e, el in enumerate(self.elements):
            for (i, j, m) in EDGE_NODES:
                key = frozenset((int(el[i]), int(el[j])))
                if key in seen:
                    seen[key] = None  # shared -> interior
                else:
                    seen[key] = (e, (int(el[i]), int(el[m]), int(el[j])))
        return [(e, *nodes) for v in seen.values() if v for e, nodes in [v]]

    def element_nodes_xy(self, e: int) -> np.ndarray:
        return self.nodes[self.elements[e]]


def build_mesh(T: Triangulation, segment_markers: dict[tuple[int, int], int] | None = None) -> Mesh:
    """Convert a refined triangulation into a quadratic triangle mesh."""
    tris = T.interior_triangles()
    if not tris:
        raise ValueError("triangulation contains no interior elements; check the region seed points")

    used: dict[int, int] = {}
    coords: list[tuple[float, float]] = []
    for t in tris:
        for v in T.tris[t]:
            if v not in used:
                used[v] = len(coords)
                coords.append(T.verts[v])

    corner_conn = np.array([[used[v] for v in T.tris[t]] for t in tris], dtype=np.int64)
    tags = np.array([T.tags[t] for t in tris], dtype=np.int64)

    # midside nodes, created once per shared edge
    mids: dict[tuple[int, int], int] = {}
    for tri in corner_conn:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = int(tri[a]), int(tri[b])
            key = (i, j) if i < j else (j, i)
            if key not in mids:
                mids[key] = len(coords)
                coords.append((0.5 * (coords[i][0] + coords[j][0]),
                               0.5 * (coords[i][1] + coords[j][1])))

    conn = np.empty((len(corner_conn), 6), dtype=np.int64)
    conn[:, :3] = corner_conn
    for e, tri in enumerate(corner_conn):
        for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
            i, j = int(tri[a]), int(tri[b])
            conn[e, 3 + k] = mids[(i, j) if i < j else (j, i)]

    nodes = np.array(coords, dtype=float)

    # named lines: map constrained triangulation edges onto quadratic mesh edges
    markers = segment_markers if segment_markers is not None else T.subsegs
    segments: dict[int, list[tuple[int, int, int]]] = {}
    for (vi, vj), marker in markers.items():
        if not marker or vi not in used or vj not in used:
            continue
        i, j = used[vi], used[vj]
        key = (i, j) if i < j else (j, i)
        mid = mids.get(key)
        if mid is None:
            continue
        segments.setdefault(int(marker), []).append((i, mid, j))

    return Mesh(nodes=nodes, elements=conn, element_tags=tags, segments=segments)


def mesh_from_pslg(pslg: PSLG, min_angle: float = 25.0, max_area: float | None = None) -> Mesh:
    gen = MeshGenerator(pslg, min_angle=min_angle, max_area=max_area)
    T = gen.run()
    return build_mesh(T)


def order_chain(edges: list[tuple[int, int, int]]) -> list[list[int]]:
    """Order a bag of quadratic edges into continuous node chains.

    Structural members (walls, piles, geogrids) are meshed as a set of element
    edges with a shared marker; beam theory needs them in order along the
    member, with a consistent local axis.
    """
    adj: dict[int, list[tuple[int, int]]] = {}
    for (a, m, b) in edges:
        adj.setdefault(a, []).append((m, b))
        adj.setdefault(b, []).append((m, a))

    unused = {(min(a, b), max(a, b)): (a, m, b) for (a, m, b) in edges}
    chains: list[list[int]] = []
    while unused:
        ends = [n for n, links in adj.items() if len(links) == 1]
        start = None
        for n in ends:
            for (m, other) in adj[n]:
                if (min(n, other), max(n, other)) in unused:
                    start = n
                    break
            if start is not None:
                break
        if start is None:
            start = next(iter(unused.values()))[0]
        chain = [start]
        node = start
        while True:
            nxt = None
            for (m, other) in adj.get(node, ()):
                key = (min(node, other), max(node, other))
                if key in unused:
                    nxt = (m, other, key)
                    break
            if nxt is None:
                break
            m, other, key = nxt
            del unused[key]
            chain += [m, other]
            node = other
        if len(chain) >= 3:
            chains.append(chain)
    return chains
