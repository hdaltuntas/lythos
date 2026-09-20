"""Unstructured triangular mesh generation for planar geotechnical sections.

The generator is a Delaunay refinement mesher in the spirit of Ruppert's
algorithm:

1. an incremental Bowyer-Watson Delaunay triangulation of the input vertices,
2. recovery of the input segments by splitting them until each one appears as a
   union of triangulation edges (a conforming Delaunay triangulation),
3. flood fill from region seed points to label soil layers and discard holes and
   the region outside the domain,
4. refinement, driven by a minimum-angle bound and per-region target element
   sizes, that splits encroached segments in preference to inserting
   circumcentres.

The result is a graded mesh that honours every layer boundary, excavation line
and structural element, which is what a staged geotechnical analysis needs: the
same mesh has to remain valid as elements are switched on and off.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .predicates import incircle, orient2d

EXTERIOR = -1


@dataclass
class PSLG:
    """Planar straight line graph: the input to the mesher."""

    points: list[tuple[float, float]] = field(default_factory=list)
    segments: list[tuple[int, int]] = field(default_factory=list)
    segment_markers: list[int] = field(default_factory=list)
    #: (x, y, tag, max_area) seed points, one inside each material region
    regions: list[tuple[float, float, int, float]] = field(default_factory=list)
    #: (tag, outline, max_area) material regions labelled by containment; more
    #: reliable than seed points, because an internal wall or excavation line
    #: may cut one soil layer into several disconnected pieces
    polygons: list[tuple[int, list[tuple[float, float]], float]] = field(default_factory=list)
    holes: list[tuple[float, float]] = field(default_factory=list)

    def add_point(self, x: float, y: float, tol: float = 1e-9) -> int:
        for i, (px, py) in enumerate(self.points):
            if abs(px - x) <= tol and abs(py - y) <= tol:
                return i
        self.points.append((float(x), float(y)))
        return len(self.points) - 1

    def add_segment(self, i: int, j: int, marker: int = 0) -> None:
        if i == j:
            return
        key = (min(i, j), max(i, j))
        for k, (a, b) in enumerate(self.segments):
            if (min(a, b), max(a, b)) == key:
                if marker:
                    self.segment_markers[k] = marker
                return
        self.segments.append((i, j))
        self.segment_markers.append(marker)

    def planarize(self, tol: float = 1e-9) -> None:
        """Turn the input into a valid planar arrangement.

        Users draw layer boundaries, excavation lines and walls as independent
        polylines that cross one another and end part-way along other lines.
        A PSLG may not contain such crossings, so every intersection point is
        promoted to a vertex and the segments through it are split.
        """
        scale = self._scale()
        tol = max(tol, 1e-12 * scale)

        # 1. every proper crossing becomes a vertex
        extra: list[tuple[float, float]] = []
        for i in range(len(self.segments)):
            a1, b1 = (self.points[k] for k in self.segments[i])
            for j in range(i + 1, len(self.segments)):
                if set(self.segments[i]) & set(self.segments[j]):
                    continue
                a2, b2 = (self.points[k] for k in self.segments[j])
                pt = _segment_intersection(a1, b1, a2, b2, tol)
                if pt is not None:
                    extra.append(pt)
        for (x, y) in extra:
            self.add_point(x, y, tol)

        # 2. every vertex lying inside a segment splits it (T-junctions)
        segments, markers = [], []
        for (i, j), marker in zip(self.segments, self.segment_markers):
            a, b = self.points[i], self.points[j]
            length = math.dist(a, b)
            if length <= tol:
                continue
            cuts = [(0.0, i), (1.0, j)]
            for k, q in enumerate(self.points):
                if k == i or k == j:
                    continue
                t = _param_on_segment(a, b, q, tol)
                if t is not None and tol / length < t < 1.0 - tol / length:
                    cuts.append((t, k))
            cuts.sort()
            for (_, p0), (_, p1) in zip(cuts[:-1], cuts[1:]):
                if p0 != p1:
                    segments.append((p0, p1))
                    markers.append(marker)

        self.segments, self.segment_markers = [], []
        for (i, j), marker in zip(segments, markers):
            self.add_segment(i, j, marker)

    def _scale(self) -> float:
        if not self.points:
            return 1.0
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    def add_polyline(self, pts, marker: int = 0, closed: bool = False) -> list[int]:
        idx = [self.add_point(x, y) for x, y in pts]
        for a, b in zip(idx[:-1], idx[1:]):
            self.add_segment(a, b, marker)
        if closed and len(idx) > 2:
            self.add_segment(idx[-1], idx[0], marker)
        return idx


def point_in_polygon(x: float, y: float, poly) -> bool:
    """Crossing-number test for a point against a closed polygon outline."""
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xc:
                inside = not inside
    return inside


def polygon_area(poly) -> float:
    """Signed area of a polygon (positive when counter-clockwise)."""
    total = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def _param_on_segment(a, b, p, tol):
    """Parameter of ``p`` along ab when it lies on the segment, else None."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 <= 0.0:
        return None
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2
    if t < 0.0 or t > 1.0:
        return None
    cx, cy = a[0] + t * dx, a[1] + t * dy
    if math.hypot(p[0] - cx, p[1] - cy) > tol:
        return None
    return t


def _segment_intersection(a1, b1, a2, b2, tol):
    """Intersection of two open segments, or None when they do not cross."""
    d1x, d1y = b1[0] - a1[0], b1[1] - a1[1]
    d2x, d2y = b2[0] - a2[0], b2[1] - a2[1]
    den = d1x * d2y - d1y * d2x
    if abs(den) < 1e-300:
        return None
    t = ((a2[0] - a1[0]) * d2y - (a2[1] - a1[1]) * d2x) / den
    u = ((a2[0] - a1[0]) * d1y - (a2[1] - a1[1]) * d1x) / den
    if not (0.0 <= t <= 1.0 and 0.0 <= u <= 1.0):
        return None
    return (a1[0] + t * d1x, a1[1] + t * d1y)


class _SegmentGrid:
    """Uniform bucket grid over subsegment diametral circles.

    Used to answer "which subsegments does this candidate point encroach upon?"
    without scanning every subsegment on every insertion.
    """

    def __init__(self, bbox, ncells: int = 48):
        xmin, ymin, xmax, ymax = bbox
        span = max(xmax - xmin, ymax - ymin, 1e-12)
        self.h = span / max(ncells, 1)
        self.x0, self.y0 = xmin - self.h, ymin - self.h
        self.cells: dict[tuple[int, int], set[int]] = {}
        self.owned: dict[int, list[tuple[int, int]]] = {}

    def _key(self, x, y):
        if not (math.isfinite(x) and math.isfinite(y)):
            return (0, 0)
        return (int((x - self.x0) // self.h), int((y - self.y0) // self.h))

    def add(self, sid: int, a, b) -> None:
        cx, cy = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        r = 0.5 * math.hypot(b[0] - a[0], b[1] - a[1])
        i0, j0 = self._key(cx - r, cy - r)
        i1, j1 = self._key(cx + r, cy + r)
        keys = []
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self.cells.setdefault((i, j), set()).add(sid)
                keys.append((i, j))
        self.owned[sid] = keys

    def remove(self, sid: int) -> None:
        for key in self.owned.pop(sid, ()):
            bucket = self.cells.get(key)
            if bucket is not None:
                bucket.discard(sid)

    def candidates(self, x, y):
        return self.cells.get(self._key(x, y), ())


class Triangulation:
    """Incremental Delaunay triangulation with segment constraints."""

    def __init__(self, points, bbox=None):
        self.verts: list[tuple[float, float]] = [tuple(map(float, p)) for p in points]
        self.tris: list[list[int] | None] = []
        self.nbrs: list[list[int] | None] = []  # nbrs[t][k] shares the edge opposite vertex k
        self.tags: list[int] = []
        self.subsegs: dict[tuple[int, int], int] = {}  # edge key -> marker
        self._last = 0
        self._rng = random.Random(20240101)  # fixed seed: meshes are reproducible

        xs = [p[0] for p in self.verts]
        ys = [p[1] for p in self.verts]
        if bbox is None:
            bbox = (min(xs), min(ys), max(xs), max(ys))
        self.bbox = bbox
        cx, cy = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1.0)
        big = 1000.0 * scale
        self._super = len(self.verts)
        self.verts += [(cx - big, cy - big), (cx + big, cy - big), (cx, cy + big)]
        self.super_ids = frozenset((self._super, self._super + 1, self._super + 2))
        self._add_tri(self._super, self._super + 1, self._super + 2)
        self.nbrs[0] = [-1, -1, -1]
        for i in range(self._super):
            self.insert_vertex(i)

    # ---------------------------------------------------------------- topology
    def _add_tri(self, a, b, c) -> int:
        self.tris.append([a, b, c])
        self.nbrs.append([-1, -1, -1])
        self.tags.append(0)
        return len(self.tris) - 1

    def _set_nbr(self, t, k, other):
        if t >= 0:
            self.nbrs[t][k] = other

    @staticmethod
    def _key(i, j):
        return (i, j) if i < j else (j, i)

    def is_subseg(self, i, j) -> bool:
        return self._key(i, j) in self.subsegs

    def alive(self, t) -> bool:
        return self.tris[t] is not None

    def is_super(self, v: int) -> bool:
        """True for the three vertices of the bounding super-triangle."""
        return v in self.super_ids

    def real_vertices(self):
        return [v for v in range(len(self.verts)) if v not in self.super_ids]

    def edge_of(self, t, k):
        """Vertices of the edge opposite local vertex ``k``."""
        tri = self.tris[t]
        return tri[(k + 1) % 3], tri[(k + 2) % 3]

    # ---------------------------------------------------------------- location
    def locate(self, p) -> int:
        """Stochastic visibility walk to the triangle containing ``p``.

        The random choice of which edge to test first is what makes the walk
        provably terminate; a fixed order can cycle between triangles.
        """
        t = self._last
        if t >= len(self.tris) or self.tris[t] is None:
            t = next(i for i in range(len(self.tris)) if self.tris[i] is not None)
        previous = -1
        for _ in range(16 * len(self.tris) + 64):
            moved = False
            offset = self._rng.randrange(3)
            for kk in range(3):
                k = (kk + offset) % 3
                n = self.nbrs[t][k]
                if n < 0 or n == previous or self.tris[n] is None:
                    continue
                a, b = self.edge_of(t, k)
                pa, pb = self.verts[a], self.verts[b]
                if orient2d(pa[0], pa[1], pb[0], pb[1], p[0], p[1]) < 0.0:
                    previous = t
                    t = n
                    moved = True
                    break
            if not moved:
                # Either p is inside t, or the only way forward is the triangle
                # we came from, in which case p sits on their shared edge.
                if self._contains(t, p):
                    self._last = t
                    return t
                if previous >= 0 and self.tris[previous] is not None and self._contains(previous, p):
                    self._last = previous
                    return previous
                for k in range(3):
                    n = self.nbrs[t][k]
                    if n < 0 or self.tris[n] is None:
                        continue
                    a, b = self.edge_of(t, k)
                    pa, pb = self.verts[a], self.verts[b]
                    if orient2d(pa[0], pa[1], pb[0], pb[1], p[0], p[1]) < 0.0:
                        previous = t
                        t = n
                        moved = True
                        break
                if not moved:
                    self._last = t
                    return t
        self._last = t
        return t

    def _contains(self, t, p) -> bool:
        for k in range(3):
            a, b = self.edge_of(t, k)
            pa, pb = self.verts[a], self.verts[b]
            if orient2d(pa[0], pa[1], pb[0], pb[1], p[0], p[1]) < 0.0:
                return False
        return True

    def check_consistency(self) -> list[str]:
        """Return a list of topology violations (empty when the mesh is sound)."""
        errors = []
        for t, tri in enumerate(self.tris):
            if tri is None:
                continue
            q = [self.verts[v] for v in tri]
            if orient2d(q[0][0], q[0][1], q[1][0], q[1][1], q[2][0], q[2][1]) <= 0.0:
                errors.append(f"triangle {t} is not counter-clockwise")
            for k in range(3):
                n = self.nbrs[t][k]
                if n < 0:
                    continue
                if self.tris[n] is None:
                    errors.append(f"triangle {t} points at deleted neighbour {n}")
                    continue
                if self.find_edge(n, *self.edge_of(t, k)) < 0:
                    errors.append(f"triangles {t}/{n} do not share edge {self.edge_of(t, k)}")
                elif t not in self.nbrs[n]:
                    errors.append(f"neighbour link {t}->{n} is not reciprocated")
        return errors

    # --------------------------------------------------------------- insertion
    def insert_point(self, p, tag_from: int | None = None) -> int:
        self.verts.append((float(p[0]), float(p[1])))
        return self.insert_vertex(len(self.verts) - 1, tag_from=tag_from)

    def insert_vertex(self, vi: int, tag_from: int | None = None) -> int:
        """Bowyer-Watson insertion of an interior point.

        The cavity is never grown across a subsegment, so constrained edges
        survive and every cavity triangle belongs to a single region.
        """
        p = self.verts[vi]
        start = self.locate(p)
        tag = self.tags[start] if tag_from is None else tag_from

        cavity = {start}
        stack = [start]
        while stack:
            t = stack.pop()
            for k in range(3):
                n = self.nbrs[t][k]
                if n < 0 or n in cavity:
                    continue
                a, b = self.edge_of(t, k)
                if self.is_subseg(a, b):
                    continue
                q = [self.verts[v] for v in self.tris[n]]
                if incircle(q[0][0], q[0][1], q[1][0], q[1][1], q[2][0], q[2][1], p[0], p[1]) > 0.0:
                    cavity.add(n)
                    stack.append(n)

        border = []
        for t in cavity:
            for k in range(3):
                n = self.nbrs[t][k]
                if n not in cavity:
                    a, b = self.edge_of(t, k)
                    border.append((a, b, n))

        for t in cavity:
            self.tris[t] = None
        free = list(cavity)

        made: dict[tuple[int, int], int] = {}
        for a, b, outside in border:
            if free:
                t = free.pop()
                self.tris[t] = [vi, a, b]
                self.nbrs[t] = [-1, -1, -1]
            else:
                t = self._add_tri(vi, a, b)
            self.tags[t] = tag
            self.nbrs[t][0] = outside
            self._backlink(outside, a, b, t)
            made[(a, b)] = t

        for (a, b), t in made.items():
            # Local vertices are (vi, a, b), so neighbour 1 is across edge
            # (b, vi) and neighbour 2 across edge (vi, a).
            self.nbrs[t][1] = self._neighbour_with_edge(made, b, vi, t)
            self.nbrs[t][2] = self._neighbour_with_edge(made, vi, a, t)

        self._last = next(iter(made.values())) if made else start
        return vi

    def _neighbour_with_edge(self, made, i, j, exclude):
        key = self._key(i, j)
        for t in made.values():
            if t == exclude:
                continue
            for k in range(3):
                if self._key(*self.edge_of(t, k)) == key:
                    return t
        return -1

    def _backlink(self, other, a, b, t) -> None:
        """Point ``other``'s neighbour slot for edge ab back at ``t``."""
        if other < 0 or self.tris[other] is None:
            return
        key = self._key(a, b)
        for k in range(3):
            if self._key(*self.edge_of(other, k)) == key:
                self.nbrs[other][k] = t
                return

    def find_edge(self, t, i, j) -> int:
        key = self._key(i, j)
        for k in range(3):
            if self._key(*self.edge_of(t, k)) == key:
                return k
        return -1

    def triangle_with_edge(self, i, j) -> tuple[int, int]:
        """Find a triangle owning edge (i, j), searching outward from a guess."""
        mid = (0.5 * (self.verts[i][0] + self.verts[j][0]),
               0.5 * (self.verts[i][1] + self.verts[j][1]))
        start = self.locate(mid)
        seen = {start}
        queue = [start]
        while queue:
            t = queue.pop(0)
            k = self.find_edge(t, i, j)
            if k >= 0:
                return t, k
            for n in self.nbrs[t]:
                if n >= 0 and n not in seen and self.tris[n] is not None:
                    seen.add(n)
                    queue.append(n)
            if len(seen) > 64:
                break
        return -1, -1

    def insert_on_edge(self, i, j, p) -> int:
        """Insert a point lying on edge (i, j) by splitting both adjacent
        triangles and restoring the Delaunay property with edge flips.

        Splitting this way (rather than by a Bowyer-Watson cavity) keeps the
        two sides of a constrained edge separate, so region labels on either
        side of a layer boundary are preserved.
        """
        t, k = self.triangle_with_edge(i, j)
        if t < 0:
            return self.insert_point(p)
        self.verts.append((float(p[0]), float(p[1])))
        m = len(self.verts) - 1

        tri = self.tris[t]
        a, b, c = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]  # edge (b, c) == (i, j)
        n_ca = self.nbrs[t][(k + 1) % 3]
        n_ab = self.nbrs[t][(k + 2) % 3]
        n = self.nbrs[t][k]

        self.tris[t] = [a, b, m]
        self.nbrs[t] = [-1, -1, n_ab]
        t2 = self._add_tri(a, m, c)
        self.tags[t2] = self.tags[t]
        self.nbrs[t2] = [-1, n_ca, t]
        self.nbrs[t][1] = t2
        self._backlink(n_ab, a, b, t)
        self._backlink(n_ca, c, a, t2)

        flips = [(t, 2), (t2, 1)]
        if n >= 0 and self.tris[n] is not None:
            kn = self.find_edge(n, b, c)
            ntri = self.tris[n]
            d = ntri[kn]
            # nbrs[n][x] lies across the edge opposite vertex ntri[x]; the
            # neighbour opposite c is the one sharing edge (b, d).
            if ntri[(kn + 1) % 3] == c:
                n_bd, n_dc = self.nbrs[n][(kn + 1) % 3], self.nbrs[n][(kn + 2) % 3]
            else:
                n_bd, n_dc = self.nbrs[n][(kn + 2) % 3], self.nbrs[n][(kn + 1) % 3]
            self.tris[n] = [d, c, m]
            self.nbrs[n] = [t2, -1, n_dc]
            n2 = self._add_tri(d, m, b)
            self.tags[n2] = self.tags[n]
            self.nbrs[n2] = [t, n_bd, n]
            self.nbrs[n][1] = n2
            self.nbrs[t][0] = n2
            self.nbrs[t2][0] = n
            self._backlink(n_dc, d, c, n)
            self._backlink(n_bd, b, d, n2)
            flips += [(n, 2), (n2, 1)]

        self._restore_delaunay(flips, m)
        self._last = t
        return m

    def _restore_delaunay(self, stack, m) -> None:
        guard = 0
        while stack and guard < 10_000:
            guard += 1
            t, k = stack.pop()
            if self.tris[t] is None or self.tris[t][k] != m:
                continue
            n = self.nbrs[t][k]
            if n < 0 or self.tris[n] is None:
                continue
            a, b = self.edge_of(t, k)
            if self.is_subseg(a, b):
                continue
            kn = self.find_edge(n, a, b)
            if kn < 0:
                continue
            d = self.tris[n][kn]
            q = [self.verts[v] for v in self.tris[t]]
            if incircle(q[0][0], q[0][1], q[1][0], q[1][1], q[2][0], q[2][1],
                        self.verts[d][0], self.verts[d][1]) <= 0.0:
                continue
            t_new, n_new = self._flip(t, k)
            stack.append((t_new, 0))
            stack.append((n_new, 0))

    def _flip(self, t, k) -> tuple[int, int]:
        """Flip the edge opposite local vertex ``k`` of triangle ``t``."""
        n = self.nbrs[t][k]
        tri = self.tris[t]
        a, b, c = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
        n_ca = self.nbrs[t][(k + 1) % 3]
        n_ab = self.nbrs[t][(k + 2) % 3]
        kn = self.find_edge(n, b, c)
        ntri = self.tris[n]
        d = ntri[kn]
        if ntri[(kn + 1) % 3] == c:
            n_bd, n_dc = self.nbrs[n][(kn + 1) % 3], self.nbrs[n][(kn + 2) % 3]
        else:
            n_bd, n_dc = self.nbrs[n][(kn + 2) % 3], self.nbrs[n][(kn + 1) % 3]

        self.tris[t] = [a, b, d]
        self.nbrs[t] = [n_bd, n, n_ab]
        self.tris[n] = [a, d, c]
        self.nbrs[n] = [n_dc, n_ca, t]
        self.tags[n] = self.tags[t]
        self._backlink(n_bd, b, d, t)
        self._backlink(n_ab, a, b, t)
        self._backlink(n_dc, d, c, n)
        self._backlink(n_ca, c, a, n)
        return t, n

    # ------------------------------------------------------------------ output
    def interior_triangles(self):
        """Real elements of the domain.

        Slivers of negligible area are dropped.  They appear where two
        boundary segments meet almost collinearly - a polygon with a
        zero-width lobe, say, which is easy to draw by accident - and an
        element of 1e-14 m2 in a domain of hundreds contributes nothing but a
        ruinous condition number.
        """
        span = max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1], 1e-12)
        floor = 1e-12 * span * span
        out = []
        for t, tri in enumerate(self.tris):
            if tri is None or self.tags[t] == EXTERIOR:
                continue
            if any(self.is_super(v) for v in tri):
                continue
            if _tri_area(*[self.verts[v] for v in tri]) <= floor:
                continue
            out.append(t)
        return out


def _encroached(pa, pb, p) -> bool:
    """True when ``p`` lies strictly inside the diametral circle of ab."""
    return (pa[0] - p[0]) * (pb[0] - p[0]) + (pa[1] - p[1]) * (pb[1] - p[1]) < -1e-14


def _circumcenter(a, b, c):
    """Circumcentre of a triangle, computed in coordinates local to ``a``."""
    bx, by = b[0] - a[0], b[1] - a[1]
    cx, cy = c[0] - a[0], c[1] - a[1]
    d = 2.0 * (bx * cy - by * cx)
    scale = max(abs(bx), abs(by), abs(cx), abs(cy), 1e-300)
    if abs(d) < 1e-12 * scale * scale:
        return None
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (cy * b2 - by * c2) / d
    uy = (bx * c2 - cx * b2) / d
    if not (math.isfinite(ux) and math.isfinite(uy)):
        return None
    return (a[0] + ux, a[1] + uy)


def _tri_area(a, b, c) -> float:
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


class MeshGenerator:
    """Delaunay refinement mesher."""

    def __init__(self, pslg: PSLG, min_angle: float = 25.0, max_area: float | None = None,
                 max_points: int = 200_000, min_length_ratio: float = 0.05):
        self.pslg = pslg
        self.min_angle = min_angle
        self.max_area = max_area
        self.max_points = max_points
        self.quality_bound = 1.0 / (2.0 * math.sin(math.radians(min_angle)))
        #: shortest subsegment the refinement may create, as a fraction of the
        #: target element size.  Where two boundary lines meet at a sharp
        #: angle - the toe of a slope, the shoulder of an embankment -
        #: refinement otherwise splits the two segments against each other
        #: without end, leaving elements millions of times smaller than the
        #: rest and a stiffness matrix to match.
        self.min_length_ratio = min_length_ratio
        self.min_length = 0.0

    def run(self) -> Triangulation:
        p = self.pslg
        if len(p.points) < 3:
            raise ValueError("a PSLG needs at least three points")
        p.planarize()
        T = Triangulation(p.points)
        for (i, j), marker in zip(p.segments, p.segment_markers):
            T.subsegs[T._key(i, j)] = marker

        xs = [q[0] for q in p.points]
        ys = [q[1] for q in p.points]
        self.grid = _SegmentGrid((min(xs), min(ys), max(xs), max(ys)))
        self._seg_ids: dict[tuple[int, int], int] = {}
        self._next_sid = 0
        for key in T.subsegs:
            self._index_segment(T, key)

        self._set_minimum_length()
        self._recover_segments(T)
        self._label_regions(T)
        self._refine(T)
        return T

    def _set_minimum_length(self) -> None:
        """Smallest subsegment the mesher may create."""
        areas = [a for (_tag, _outline, a) in self.pslg.polygons if a and a > 0]
        areas += [a for (_x, _y, _tag, a) in self.pslg.regions if a and a > 0]
        if self.max_area and self.max_area > 0:
            areas.append(self.max_area)
        if areas:
            target = math.sqrt(2.0 * min(areas))
        else:
            xs = [p[0] for p in self.pslg.points]
            ys = [p[1] for p in self.pslg.points]
            target = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) / 40.0
        self.min_length = self.min_length_ratio * target

    # --------------------------------------------------------------- segments
    def _index_segment(self, T, key):
        sid = self._next_sid
        self._next_sid += 1
        self._seg_ids[key] = sid
        self._sid_key = getattr(self, "_sid_key", {})
        self._sid_key[sid] = key
        self.grid.add(sid, T.verts[key[0]], T.verts[key[1]])

    def _drop_segment(self, key):
        sid = self._seg_ids.pop(key, None)
        if sid is not None:
            self.grid.remove(sid)
            self._sid_key.pop(sid, None)

    def _split_segment(self, T, key, limit: bool = True):
        """Split a subsegment at its midpoint and insert the new vertex.

        Returns the new vertex, or None when the subsegment is already at the
        shortest length refinement is allowed to create.
        """
        i, j = key
        if limit and self.min_length > 0.0:
            if math.dist(T.verts[i], T.verts[j]) < self.min_length:
                return None
        marker = T.subsegs.pop(key)
        self._drop_segment(key)
        a, b = T.verts[i], T.verts[j]
        mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
        if T.triangle_with_edge(i, j)[0] >= 0:
            vi = T.insert_on_edge(i, j, mid)
        else:
            # The segment is still crossed by other edges; a plain insertion
            # drives the triangulation towards conformity.
            vi = T.insert_point(mid)
        for k in (T._key(i, vi), T._key(vi, j)):
            T.subsegs[k] = marker
            self._index_segment(T, k)
        return vi

    def _encroached_by(self, T, p, skip_keys=()):
        hits = []
        for sid in list(self.grid.candidates(p[0], p[1])):
            key = self._sid_key.get(sid)
            if key is None or key in skip_keys:
                continue
            if _encroached(T.verts[key[0]], T.verts[key[1]], p):
                hits.append(key)
        return hits

    def _recover_segments(self, T) -> None:
        """Split subsegments until each is an edge and none is encroached."""
        for _ in range(64):
            edges = set()
            for t, tri in enumerate(T.tris):
                if tri is None:
                    continue
                for k in range(3):
                    edges.add(T._key(*T.edge_of(t, k)))
            todo = [k for k in T.subsegs if k not in edges]
            for key in todo:
                if key in T.subsegs:
                    self._split_segment(T, key, limit=False)
            pending = []
            for key in list(T.subsegs):
                a, b = T.verts[key[0]], T.verts[key[1]]
                for vi, v in enumerate(T.verts):
                    if vi in key or T.is_super(vi):
                        continue
                    if _encroached(a, b, v):
                        pending.append(key)
                        break
            split_any = False
            for key in pending:
                if key in T.subsegs and self._split_segment(T, key) is not None:
                    split_any = True
            if not todo and not split_any:
                # Either nothing was encroached, or what was is already as
                # short as the input geometry allows.
                return

    # ---------------------------------------------------------------- regions
    def _flood(self, T, start: int, tag: int) -> None:
        stack = [start]
        seen = {start}
        T.tags[start] = tag
        while stack:
            t = stack.pop()
            for k in range(3):
                n = T.nbrs[t][k]
                if n < 0 or n in seen or T.tris[n] is None:
                    continue
                a, b = T.edge_of(t, k)
                if T.is_subseg(a, b):
                    continue
                seen.add(n)
                T.tags[n] = tag
                stack.append(n)

    def _label_regions(self, T) -> None:
        """Assign a material region to every triangle.

        Region outlines are tested by containment of the triangle centroid,
        which stays correct when an internal line (a diaphragm wall, an
        excavation stage boundary) cuts one soil layer into several pieces.
        Seed points are supported as a fallback for callers that supply them.
        """
        for t, tri in enumerate(T.tris):
            if tri is not None:
                T.tags[t] = EXTERIOR
        self._area_limit: dict[int, float] = {}

        for (tag, _outline, area) in self.pslg.polygons:
            if area and area > 0:
                self._area_limit[int(tag)] = float(area)
        if self.pslg.polygons:
            for t, tri in enumerate(T.tris):
                if tri is None or any(T.is_super(v) for v in tri):
                    continue
                pts = [T.verts[v] for v in tri]
                cx = (pts[0][0] + pts[1][0] + pts[2][0]) / 3.0
                cy = (pts[0][1] + pts[1][1] + pts[2][1]) / 3.0
                for (tag, outline, _a) in reversed(self.pslg.polygons):
                    if point_in_polygon(cx, cy, outline):
                        T.tags[t] = int(tag)
                        break

        for (x, y, tag, area) in self.pslg.regions:
            t = T.locate((x, y))
            if T.tris[t] is None:
                continue
            self._flood(T, t, int(tag))
            if area and area > 0:
                self._area_limit[int(tag)] = float(area)

        for (x, y) in self.pslg.holes:
            t = T.locate((x, y))
            if T.tris[t] is not None:
                self._flood(T, t, EXTERIOR)

    # -------------------------------------------------------------- refinement
    def _target_area(self, T, t) -> float:
        lim = self._area_limit.get(T.tags[t], self.max_area or float("inf"))
        return lim if lim and lim > 0 else float("inf")

    def _is_bad(self, T, t):
        tri = T.tris[t]
        if tri is None or T.tags[t] == EXTERIOR or any(T.is_super(v) for v in tri):
            return False
        pts = [T.verts[v] for v in tri]
        area = _tri_area(*pts)
        if area <= 1e-18:
            return False
        lengths = [math.dist(pts[k], pts[(k + 1) % 3]) for k in range(3)]
        shortest = min(lengths)
        circumradius = (lengths[0] * lengths[1] * lengths[2]) / (4.0 * area)
        if area > self._target_area(T, t):
            return True
        if circumradius / shortest > self.quality_bound:
            # Do not chase a small angle that the input geometry itself imposes.
            k = lengths.index(shortest)
            a, b = tri[k], tri[(k + 1) % 3]
            apex = tri[(k + 2) % 3]
            if T.is_subseg(a, apex) and T.is_subseg(b, apex):
                v0 = (pts[k][0] - pts[(k + 2) % 3][0], pts[k][1] - pts[(k + 2) % 3][1])
                v1 = (pts[(k + 1) % 3][0] - pts[(k + 2) % 3][0],
                      pts[(k + 1) % 3][1] - pts[(k + 2) % 3][1])
                dot = v0[0] * v1[0] + v0[1] * v1[1]
                n0 = math.hypot(*v0) * math.hypot(*v1)
                if n0 > 0 and dot / n0 > math.cos(math.radians(60.0)):
                    return False
            return True
        return False

    def _refine(self, T) -> None:
        queue = [t for t in range(len(T.tris)) if self._is_bad(T, t)]
        guard = 0
        while queue and len(T.verts) < self.max_points:
            guard += 1
            if guard > 40 * self.max_points:
                break
            t = queue.pop()
            if T.tris[t] is None or not self._is_bad(T, t):
                continue
            tri = T.tris[t]
            cc = _circumcenter(*[T.verts[v] for v in tri])
            if cc is None:
                continue
            keys = self._encroached_by(T, cc)
            if keys:
                split_any = False
                for key in keys:
                    if key in T.subsegs and self._split_segment(T, key) is not None:
                        split_any = True
                if not split_any:
                    # Every encroached segment is already as short as it is
                    # allowed to get: this triangle is as good as the input
                    # geometry permits.
                    continue
                queue.append(t)
                queue.extend(range(max(0, len(T.tris) - 12), len(T.tris)))
                continue
            host = T.locate(cc)
            if T.tris[host] is None or T.tags[host] == EXTERIOR:
                continue
            before = len(T.tris)
            T.insert_point(cc, tag_from=T.tags[host])
            queue.extend(t2 for t2 in range(before - 6, len(T.tris)) if t2 >= 0)
