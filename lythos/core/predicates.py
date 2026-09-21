"""Robust 2D geometric predicates.

Floating point determinants are evaluated with a static error filter; when the
filter cannot certify the sign the determinant is recomputed exactly with
``fractions.Fraction`` (binary floats convert to rationals exactly, so the
fallback is an exact arithmetic evaluation, not merely a more precise one).

Correct signs matter here: the Delaunay kernel in :mod:`lythos.core.mesher`
walks and flips triangles based on these tests, and a single inconsistent sign
can produce a non-planar or inverted triangulation.
"""

from __future__ import annotations

from fractions import Fraction

# Error bounds derived from the standard forward error analysis of the 2x2 and
# 3x3 determinant expansions (Shewchuk, "Adaptive Precision Floating-Point
# Arithmetic and Fast Robust Geometric Predicates", 1997).
_EPS = 2.220446049250313e-16
_ORIENT_BOUND = (3.0 + 16.0 * _EPS) * _EPS
_INCIRCLE_BOUND = (10.0 + 96.0 * _EPS) * _EPS


def orient2d(ax, ay, bx, by, cx, cy) -> float:
    """Return > 0 if a->b->c turns counter-clockwise, < 0 if clockwise, 0 if collinear."""
    detleft = (ax - cx) * (by - cy)
    detright = (ay - cy) * (bx - cx)
    det = detleft - detright

    if detleft > 0.0:
        if detright <= 0.0:
            return det
        summ = detleft + detright
    elif detleft < 0.0:
        if detright >= 0.0:
            return det
        summ = -detleft - detright
    else:
        return det

    if abs(det) >= _ORIENT_BOUND * summ:
        return det
    return _orient2d_exact(ax, ay, bx, by, cx, cy)


def _orient2d_exact(ax, ay, bx, by, cx, cy) -> float:
    fa = Fraction
    det = (fa(ax) - fa(cx)) * (fa(by) - fa(cy)) - (fa(ay) - fa(cy)) * (fa(bx) - fa(cx))
    if det == 0:
        return 0.0
    return 1.0 if det > 0 else -1.0


def incircle(ax, ay, bx, by, cx, cy, dx, dy) -> float:
    """Return > 0 if d lies inside the circle through the CCW triangle a, b, c."""
    adx, ady = ax - dx, ay - dy
    bdx, bdy = bx - dx, by - dy
    cdx, cdy = cx - dx, cy - dy

    bdxcdy = bdx * cdy
    cdxbdy = cdx * bdy
    alift = adx * adx + ady * ady

    cdxady = cdx * ady
    adxcdy = adx * cdy
    blift = bdx * bdx + bdy * bdy

    adxbdy = adx * bdy
    bdxady = bdx * ady
    clift = cdx * cdx + cdy * cdy

    det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady)

    permanent = (
        (abs(bdxcdy) + abs(cdxbdy)) * alift
        + (abs(cdxady) + abs(adxcdy)) * blift
        + (abs(adxbdy) + abs(bdxady)) * clift
    )
    if abs(det) > _INCIRCLE_BOUND * permanent:
        return det
    return _incircle_exact(ax, ay, bx, by, cx, cy, dx, dy)


def _incircle_exact(ax, ay, bx, by, cx, cy, dx, dy) -> float:
    fa = Fraction
    adx, ady = fa(ax) - fa(dx), fa(ay) - fa(dy)
    bdx, bdy = fa(bx) - fa(dx), fa(by) - fa(dy)
    cdx, cdy = fa(cx) - fa(dx), fa(cy) - fa(dy)
    det = (
        (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
        + (bdx * bdx + bdy * bdy) * (cdx * ady - adx * cdy)
        + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady)
    )
    if det == 0:
        return 0.0
    return 1.0 if det > 0 else -1.0


def in_circumcircle_of(tri_pts, p) -> bool:
    """``incircle`` for a triangle of unknown orientation."""
    (ax, ay), (bx, by), (cx, cy) = tri_pts
    if orient2d(ax, ay, bx, by, cx, cy) < 0.0:
        bx, by, cx, cy = cx, cy, bx, by
    return incircle(ax, ay, bx, by, cx, cy, p[0], p[1]) > 0.0
