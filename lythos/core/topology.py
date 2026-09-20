"""Finding the regions enclosed by a set of drawn lines.

A section drawn in CAD is usually an outer boundary plus the lines dividing it
into strata, not one closed polygon per layer.  Turning that into a model means
recovering the faces of the planar subdivision those lines make.
"""

from __future__ import annotations

import math


def find_faces(points, segments, min_area: float = 0.0) -> list[list[int]]:
    """Minimal closed regions enclosed by ``segments``.

    ``points`` are vertex coordinates and ``segments`` index pairs that must
    already form a planar arrangement - no crossings except at shared
    endpoints, which :meth:`lythos.core.mesher.PSLG.planarize` arranges.

    Each face comes back as a list of vertex indices in counter-clockwise
    order.  The unbounded face outside everything is not included.
    """
    neighbours: dict[int, list[int]] = {}
    for i, j in segments:
        if i == j:
            continue
        neighbours.setdefault(i, []).append(j)
        neighbours.setdefault(j, []).append(i)

    # Around each vertex, order the edges by direction.  Walking a face means
    # repeatedly taking the next edge clockwise from the one just used.
    order: dict[int, list[int]] = {}
    position: dict[tuple[int, int], int] = {}
    for v, links in neighbours.items():
        unique = sorted(set(links),
                        key=lambda w: math.atan2(points[w][1] - points[v][1],
                                                 points[w][0] - points[v][0]))
        order[v] = unique
        for k, w in enumerate(unique):
            position[(v, w)] = k

    faces: list[list[int]] = []
    visited: set[tuple[int, int]] = set()
    for i, j in segments:
        for u, v in ((i, j), (j, i)):
            if (u, v) in visited or v not in order:
                continue
            face = _walk(u, v, order, position, visited)
            if face is None or len(face) < 3:
                continue
            area = _signed_area(points, face)
            if area > max(min_area, 0.0):
                faces.append(face)
    return faces


def _walk(start_u, start_v, order, position, visited):
    """Follow half-edges clockwise around one face."""
    face = [start_u]
    u, v = start_u, start_v
    for _ in range(4 * len(position) + 8):
        if (u, v) in visited:
            return None
        visited.add((u, v))
        face.append(v)
        links = order.get(v)
        if not links:
            return None
        k = position.get((v, u))
        if k is None:
            return None
        nxt = links[(k - 1) % len(links)]
        u, v = v, nxt
        if (u, v) == (start_u, start_v):
            face.pop()
            return face
    return None


def _signed_area(points, face) -> float:
    total = 0.0
    n = len(face)
    for i in range(n):
        x0, y0 = points[face[i]]
        x1, y1 = points[face[(i + 1) % n]]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def join_chains(polylines, tolerance: float = 1e-6):
    """Join polylines that meet end to end into longer runs.

    Boundaries are often drawn as a handful of separate lines; a layer outline
    only becomes usable once they are strung together.  Returns
    ``(chains, closed_flags)``.
    """
    pieces = [list(p) for p in polylines if len(p) >= 2]
    used = [False] * len(pieces)
    chains: list[list[tuple[float, float]]] = []
    closed: list[bool] = []

    def near(a, b):
        return math.dist(a, b) <= tolerance

    for index in range(len(pieces)):
        if used[index]:
            continue
        used[index] = True
        chain = list(pieces[index])
        extended = True
        while extended:
            extended = False
            for other in range(len(pieces)):
                if used[other]:
                    continue
                candidate = pieces[other]
                if near(chain[-1], candidate[0]):
                    chain.extend(candidate[1:])
                elif near(chain[-1], candidate[-1]):
                    chain.extend(reversed(candidate[:-1]))
                elif near(chain[0], candidate[-1]):
                    chain = candidate[:-1] + chain
                elif near(chain[0], candidate[0]):
                    chain = list(reversed(candidate[1:])) + chain
                else:
                    continue
                used[other] = True
                extended = True
        is_closed = len(chain) > 3 and near(chain[0], chain[-1])
        if is_closed:
            chain = chain[:-1]
        chains.append(chain)
        closed.append(is_closed)
    return chains, closed
