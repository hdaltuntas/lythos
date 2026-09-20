"""Constitutive models for soil, rock and structural concrete.

Stresses use the **tension-positive** continuum convention throughout, and are
stored as the four plane-strain components ``[sxx, syy, szz, sxy]``.  The
out-of-plane component is carried explicitly because plastic flow makes the
out-of-plane elastic strain non-zero even though the total strain there is
zero, and because every yield criterion here depends on it.

The Mohr-Coulomb update is an exact return mapping in principal stress space
(Clausen, Damkilde & Andersen, 2006): because the criterion is linear in the
principal stresses, the return to a face, to an edge, or to the apex can be
written in closed form rather than iterated.  All updates are vectorised over
Gauss points, which is what makes a strength reduction analysis - dozens of
full non-linear solutions - finish in seconds rather than hours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

DRAINED = "drained"
UNDRAINED = "undrained"


def elastic_matrix(E: float, nu: float) -> np.ndarray:
    """4x4 plane-strain elasticity matrix for [xx, yy, zz, xy]."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((4, 4))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[3, 3] = mu
    return D


def principal_stresses(sig: np.ndarray):
    """In-plane principal stresses and the rotation angle of the principal axes.

    Returns ``(s_a, s_b, szz, cos2t, sin2t)`` where ``s_a >= s_b`` are the
    in-plane principal values.
    """
    sxx, syy, szz, sxy = sig[:, 0], sig[:, 1], sig[:, 2], sig[:, 3]
    mean = 0.5 * (sxx + syy)
    half = 0.5 * (sxx - syy)
    radius = np.sqrt(half * half + sxy * sxy)
    safe = np.where(radius > 1e-300, radius, 1.0)
    cos2t = np.where(radius > 1e-300, half / safe, 1.0)
    sin2t = np.where(radius > 1e-300, sxy / safe, 0.0)
    return mean + radius, mean - radius, szz, cos2t, sin2t


def from_principal(s_a, s_b, szz, cos2t, sin2t) -> np.ndarray:
    """Rebuild Cartesian stresses from in-plane principal values."""
    mean = 0.5 * (s_a + s_b)
    half = 0.5 * (s_a - s_b)
    out = np.empty((len(s_a), 4))
    out[:, 0] = mean + half * cos2t
    out[:, 1] = mean - half * cos2t
    out[:, 2] = szz
    out[:, 3] = half * sin2t
    return out


@dataclass
class MaterialState:
    """Per-Gauss-point history."""

    stress: np.ndarray            # (n, 4) current stress
    plastic_strain: np.ndarray    # (n, 4) accumulated plastic strain
    eps_p_eq: np.ndarray          # (n,) equivalent plastic strain
    yielding: np.ndarray          # (n,) bool, plastic at the last update

    @classmethod
    def zeros(cls, n: int) -> "MaterialState":
        return cls(np.zeros((n, 4)), np.zeros((n, 4)), np.zeros(n), np.zeros(n, bool))

    def copy(self) -> "MaterialState":
        return MaterialState(self.stress.copy(), self.plastic_strain.copy(),
                             self.eps_p_eq.copy(), self.yielding.copy())


@dataclass
class Material:
    """Base soil/rock material."""

    name: str = "material"
    E: float = 3.0e4           # Young's modulus [kPa]
    nu: float = 0.3            # Poisson's ratio
    gamma: float = 18.0        # bulk unit weight above the water table [kN/m3]
    gamma_sat: float | None = None   # saturated unit weight [kN/m3]
    K0: float | None = None    # coefficient of earth pressure at rest
    drainage: str = DRAINED
    color: str = "#c8b273"

    # stress dependent stiffness: E = E_ref * ((c cot(phi) + s3) / p_ref)^m
    m_stiffness: float = 0.0
    p_ref: float = 100.0

    def unit_weight(self, saturated: bool = False) -> float:
        if saturated and self.gamma_sat:
            return self.gamma_sat
        return self.gamma

    def k0(self) -> float:
        return 1.0 - math.sin(math.radians(getattr(self, "phi", 30.0))) if self.K0 is None else self.K0

    def elastic(self) -> np.ndarray:
        return elastic_matrix(self.E, self.nu)

    def reduced(self, srf: float) -> "Material":
        """Material with strength divided by ``srf`` (used by SSR)."""
        return self

    # ------------------------------------------------------------------ update
    def update(self, state: MaterialState, dstrain: np.ndarray):
        """Advance the stress from ``state`` by ``dstrain``.

        Returns ``(stress, tangent, new_state)`` with ``tangent`` of shape
        ``(n, 4, 4)``.
        """
        D = self.elastic()
        stress = state.stress + dstrain @ D.T
        n = len(stress)
        tangent = np.broadcast_to(D, (n, 4, 4)).copy()
        new = MaterialState(stress, state.plastic_strain.copy(), state.eps_p_eq.copy(),
                            np.zeros(n, bool))
        return stress, tangent, new


@dataclass
class LinearElastic(Material):
    """Isotropic linear elastic soil or rock."""

    phi: float = 0.0


@dataclass
class MohrCoulomb(Material):
    """Elastic-perfectly plastic Mohr-Coulomb with a tension cut-off.

    ``psi < phi`` gives non-associated flow, which is the realistic choice for
    soils; the resulting non-symmetric tangent is symmetrised, and global
    convergence is maintained by the line search in the solver.
    """

    c: float = 5.0             # effective cohesion [kPa]
    phi: float = 30.0          # effective friction angle [deg]
    psi: float = 0.0           # dilatancy angle [deg]
    #: fraction of the elastic stiffness retained where the material has no
    #: strength left (the apex of the cone, or a fully tensile point); a small
    #: residual keeps the global stiffness matrix invertible where a crack has
    #: opened, without measurably changing the answer
    residual_stiffness: float = 1.0e-3
    #: tensile strength [kPa]; 0.0 means no tension is carried at all, and
    #: None disables the cut-off so the material may reach the Mohr-Coulomb
    #: apex at c cot(phi) - the cut-off is on by default because real soil
    #: does not sustain the tension the apex implies
    tension_cutoff: float | None = 0.0

    def reduced(self, srf: float) -> "MohrCoulomb":
        phi_r = math.degrees(math.atan(math.tan(math.radians(self.phi)) / srf))
        psi_r = math.degrees(math.atan(math.tan(math.radians(self.psi)) / srf))
        cut = None if self.tension_cutoff is None else self.tension_cutoff / srf
        return replace(self, c=self.c / srf, phi=phi_r, psi=min(psi_r, phi_r),
                       tension_cutoff=cut)

    # ------------------------------------------------------------------ update
    def update(self, state: MaterialState, dstrain: np.ndarray):
        D = self.elastic()
        Dinv = np.linalg.inv(D)
        trial = state.stress + dstrain @ D.T
        n = len(trial)

        s_a, s_b, szz, cos2t, sin2t = principal_stresses(trial)
        pr = np.stack([s_a, s_b, szz], axis=1)
        order = np.argsort(-pr, axis=1)              # descending: s1 >= s2 >= s3
        rank = np.argsort(order, axis=1)             # component -> its rank
        rows = np.arange(n)
        sorted_pr = pr[rows[:, None], order]

        returned, D_sorted, plastic = self._return_map(sorted_pr)

        # undo the sort for both the stresses and the tangent
        final_pr = np.empty_like(returned)
        final_pr[rows[:, None], order] = returned
        D_pr = D_sorted[rows[:, None, None], rank[:, :, None], rank[:, None, :]]

        stress = from_principal(final_pr[:, 0], final_pr[:, 1], final_pr[:, 2], cos2t, sin2t)
        tangent = self._cartesian_tangent(D_pr, pr, final_pr, cos2t, sin2t)

        dplastic = dstrain - (stress - state.stress) @ Dinv.T
        dev = dplastic - dplastic[:, :3].sum(axis=1, keepdims=True) / 3.0 * np.array([1.0, 1.0, 1.0, 0.0])
        deq = np.sqrt(np.maximum(
            2.0 / 3.0 * (dev[:, 0] ** 2 + dev[:, 1] ** 2 + dev[:, 2] ** 2 + 0.5 * dev[:, 3] ** 2), 0.0))
        new = MaterialState(stress, state.plastic_strain + dplastic,
                            state.eps_p_eq + deq * plastic, plastic)
        return stress, tangent, new

    # ---------------------------------------------------------- principal space
    def _params(self):
        sphi = math.sin(math.radians(self.phi))
        spsi = math.sin(math.radians(self.psi))
        cphi = math.cos(math.radians(self.phi))
        return sphi, spsi, 2.0 * self.c * cphi

    def _elastic_principal(self) -> np.ndarray:
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        mu = self.E / (2.0 * (1.0 + self.nu))
        return lam * np.ones((3, 3)) + 2.0 * mu * np.eye(3)

    def tension_limit(self) -> float:
        """Largest admissible major principal stress (tension positive)."""
        apex = self.c / math.tan(math.radians(self.phi)) if self.phi > 1e-9 else math.inf
        if self.tension_cutoff is None:
            return apex
        return min(self.tension_cutoff, apex)

    @staticmethod
    def _tension_return(s: np.ndarray, limit: float, De: np.ndarray):
        """Associated return onto the tension cut-off planes.

        The criterion is one plane per principal stress, ``s_i <= limit``.
        Because the stresses arrive sorted, the set of violated planes is
        always a prefix, so trying the prefixes in turn finds the active set
        without a general search.  Returning along ``De`` rather than simply
        truncating the stress is what makes the accompanying tangent
        consistent, and that is what the global Newton iteration needs in
        order to converge inside a cracked zone.
        """
        out = s.copy()
        tang = np.broadcast_to(De, (len(s), 3, 3)).copy()
        remaining = np.ones(len(s), bool)
        for m in (1, 2, 3):
            if not remaining.any():
                break
            M = np.linalg.inv(De[:m, :m])
            cols = De[:, :m]
            rows = np.nonzero(remaining)[0]
            lam = (s[rows][:, :m] - limit) @ M.T
            cand = s[rows] - lam @ cols.T
            ok = np.all(lam >= -1e-12, axis=1)
            if m < 3:
                ok &= np.all(cand[:, m:] <= limit + 1e-9, axis=1)
            idx = rows[ok]
            out[idx] = cand[ok]
            tang[idx] = De - cols @ M @ De[:m, :]
            remaining[idx] = False
        if remaining.any():          # degenerate: fall back to truncation
            out[remaining] = np.minimum(s[remaining], limit)
            tang[remaining] = 1e-6 * De
        return out, tang

    def _criteria(self):
        """Yield surfaces as (normal, flow direction, offset) in sorted principal space.

        The Mohr-Coulomb pyramid contributes its main face plus the two faces
        that bound the sorted sector; intersecting the main face with f(s2, s3)
        gives the edge s1 = s2 and with f(s1, s2) the edge s2 = s3, which is
        why the pairs below look swapped relative to the violated ordering.
        The tension cut-off contributes one plane per principal stress.
        """
        sphi, spsi, k = self._params()
        st = self.tension_limit()
        eye = np.eye(3)
        return {
            "main": (np.array([1.0 + sphi, 0.0, -1.0 + sphi]),
                     np.array([1.0 + spsi, 0.0, -1.0 + spsi]), k),
            "s12": (np.array([1.0 + sphi, -1.0 + sphi, 0.0]),
                    np.array([1.0 + spsi, -1.0 + spsi, 0.0]), k),
            "s23": (np.array([0.0, 1.0 + sphi, -1.0 + sphi]),
                    np.array([0.0, 1.0 + spsi, -1.0 + spsi]), k),
            "t0": (eye[0], eye[0], st),
            "t1": (eye[1], eye[1], st),
            "t2": (eye[2], eye[2], st),
        }

    #: candidate active sets, tried in order of increasing size.  The first one
    #: that returns a stress satisfying both criteria with non-negative plastic
    #: multipliers is the correct region for that point.
    ACTIVE_SETS = (
        ("main",),
        ("t0",),
        ("main", "s23"),
        ("main", "s12"),
        ("main", "t0"),
        ("t0", "t1"),
        ("main", "s23", "t0"),
        ("main", "s12", "t0"),
        ("main", "t0", "t1"),
        ("t0", "t1", "t2"),
    )

    def _return_map(self, s: np.ndarray):
        """Map sorted trial principal stresses back onto the yield surface.

        ``s`` holds s1 >= s2 >= s3 (tension positive).  Both criteria are
        linear in the principal stresses, so for any given set of active
        surfaces the return is the solution of a small linear system, and its
        consistent tangent follows in closed form.  Which surfaces are active
        is found by trying the candidate sets in turn and keeping the first
        that satisfies the loading-unloading conditions - a search rather than
        a chain of geometric tests, which is what makes the corners between
        the shear criterion and the tension cut-off come out right.
        """
        crit = self._criteria()
        a_main, _b_main, k = crit["main"]
        De = self._elastic_principal()
        n = len(s)
        st_max = self.tension_limit()
        scale = max(abs(k), self.E * 1e-6, 1.0)
        tol = 1e-10 * scale

        plastic = (s @ a_main - k > tol) | (s[:, 0] > st_max + tol)
        out = s.copy()
        tangents = np.broadcast_to(De, (n, 3, 3)).copy()
        if not plastic.any():
            return out, tangents, plastic

        idx = np.nonzero(plastic)[0]
        trial = s[idx]
        res = trial.copy()
        tang = np.broadcast_to(De, (len(idx), 3, 3)).copy()
        unsolved = np.ones(len(idx), bool)

        for names in self.ACTIVE_SETS:
            if not unsolved.any():
                break
            if st_max == math.inf and any(name.startswith("t") for name in names):
                continue
            A = np.column_stack([crit[nm][0] for nm in names])
            B = np.column_stack([crit[nm][1] for nm in names])
            offsets = np.array([crit[nm][2] for nm in names])
            M = A.T @ De @ B
            if abs(np.linalg.det(M)) < 1e-12 * scale ** len(names):
                continue
            Minv = np.linalg.inv(M)
            rows = np.nonzero(unsolved)[0]
            lam = (trial[rows] @ A - offsets) @ Minv.T
            cand = trial[rows] - lam @ (De @ B).T
            ok = (np.all(lam >= -1e-9 * scale, axis=1)
                  & (cand @ a_main - k <= 1e-7 * scale)
                  & (cand[:, 0] <= st_max + 1e-7 * scale)
                  & (cand[:, 0] >= cand[:, 1] - 1e-7 * scale)
                  & (cand[:, 1] >= cand[:, 2] - 1e-7 * scale))
            take = rows[ok]
            res[take] = cand[ok]
            tang[take] = De - (De @ B) @ Minv @ (A.T @ De)
            unsolved[take] = False

        if unsolved.any():
            # The tip of the cone: no combination of planes applies, so the
            # only admissible point is the apex itself (or the cut-off, when
            # that sits lower).  It carries no strength, hence the token
            # stiffness that keeps the global system solvable.
            sub = np.nonzero(unsolved)[0]
            apex = (self.c / math.tan(math.radians(self.phi))
                    if self.phi > 1e-9 else math.inf)
            limit = min(apex, st_max)
            if not math.isfinite(limit):
                limit = float(np.min(trial[sub]))
            res[sub] = limit
            tang[sub] = self.residual_stiffness * De

        out[idx] = res
        tangents[idx] = tang
        return out, tangents, plastic

    # ------------------------------------------------------------- tangent map
    def _cartesian_tangent(self, D_pr, trial_pr, final_pr, cos2t, sin2t) -> np.ndarray:
        """Rotate a principal-space tangent into global plane-strain axes.

        Besides the principal stiffness, the rotation of the principal
        directions contributes the shear term (sa - sb) / (2 (ea - eb)), which
        collapses to the shear modulus for an elastic step.  Leaving it out
        would cost quadratic convergence wherever the material is plastic.
        """
        n = len(trial_pr)
        mu = self.E / (2.0 * (1.0 + self.nu))

        d_trial = trial_pr[:, 0] - trial_pr[:, 1]
        safe = np.where(np.abs(d_trial) > 1e-12, d_trial, 1.0)
        shear = np.where(np.abs(d_trial) > 1e-12,
                         mu * (final_pr[:, 0] - final_pr[:, 1]) / safe, mu)

        Dp = np.zeros((n, 4, 4))
        Dp[:, :3, :3] = D_pr
        Dp[:, 3, 3] = np.clip(shear, self.residual_stiffness * mu, mu)

        c = np.sqrt(np.clip(0.5 * (1.0 + cos2t), 0.0, 1.0))
        s = np.where(c > 1e-12, sin2t / (2.0 * np.where(c > 1e-12, c, 1.0)), 1.0)
        cc, ss, cs = c * c, s * s, c * s

        T = np.zeros((n, 4, 4))
        T[:, 0, 0] = cc; T[:, 0, 1] = ss; T[:, 0, 3] = cs
        T[:, 1, 0] = ss; T[:, 1, 1] = cc; T[:, 1, 3] = -cs
        T[:, 2, 2] = 1.0
        T[:, 3, 0] = -2.0 * cs; T[:, 3, 1] = 2.0 * cs; T[:, 3, 3] = cc - ss
        return np.matmul(T.transpose(0, 2, 1), np.matmul(Dp, T))


@dataclass
class Concrete(LinearElastic):
    """Linear elastic concrete, with stiffness derived from strength class.

    ``fck`` is the characteristic cylinder strength in MPa; the secant modulus
    follows EN 1992-1-1: ``Ecm = 22000 (fcm/10)^0.3`` MPa with
    ``fcm = fck + 8`` MPa.
    """

    fck: float = 30.0
    gamma: float = 24.0

    def __post_init__(self):
        if self.E <= 0 or self.E == 3.0e4:
            self.E = concrete_modulus(self.fck)


def concrete_modulus(fck_mpa: float) -> float:
    """Secant modulus of elasticity of concrete in kPa, from EN 1992-1-1."""
    fcm = fck_mpa + 8.0
    return 22000.0 * (fcm / 10.0) ** 0.3 * 1.0e3


def undrained_from_drained(mat: MohrCoulomb, su: float) -> MohrCoulomb:
    """Total-stress (phi = 0) equivalent of a drained material."""
    return replace(mat, c=su, phi=0.0, psi=0.0, drainage=UNDRAINED,
                   name=f"{mat.name} (undrained)")
