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

    def _return_map(self, s: np.ndarray):
        """Map sorted trial principal stresses back onto the yield surface.

        ``s`` holds s1 >= s2 >= s3 (tension positive).  The Mohr-Coulomb
        pyramid is linear, so each return - to the main face, to one of the two
        edges bounding the sorted sector, or to the apex - is a closed-form
        expression rather than a local Newton iteration.
        """
        sphi, spsi, k = self._params()
        De = self._elastic_principal()
        n = len(s)
        st_max = self.tension_limit()

        a_main = np.array([1.0 + sphi, 0.0, -1.0 + sphi])
        b_main = np.array([1.0 + spsi, 0.0, -1.0 + spsi])
        a_12 = np.array([1.0 + sphi, -1.0 + sphi, 0.0])
        b_12 = np.array([1.0 + spsi, -1.0 + spsi, 0.0])
        a_23 = np.array([0.0, 1.0 + sphi, -1.0 + sphi])
        b_23 = np.array([0.0, 1.0 + spsi, -1.0 + spsi])

        tol = 1e-10 * max(k, self.E * 1e-6, 1.0)
        f_mc = s @ a_main - k
        plastic = (f_mc > tol) | (s[:, 0] > st_max + tol)

        out = s.copy()
        tangents = np.broadcast_to(De, (n, 3, 3)).copy()
        if not plastic.any():
            return out, tangents, plastic

        idx = np.nonzero(plastic)[0]
        trial = s[idx]
        res = trial.copy()
        tang = np.broadcast_to(De, (len(idx), 3, 3)).copy()

        # Two passes let a state that violates both the shear criterion and the
        # tension cut-off settle onto their common corner.
        for _pass in range(2):
            active = res @ a_main - k > tol
            if active.any():
                sub = np.nonzero(active)[0]
                start = res[sub]

                denom = float(a_main @ De @ b_main)
                lam = (start @ a_main - k) / denom
                cand = start - lam[:, None] * (De @ b_main)
                local = cand.copy()
                local_t = np.broadcast_to(De - np.outer(De @ b_main, De @ a_main) / denom,
                                          (len(sub), 3, 3)).copy()

                # A face return that leaves the sorted sector means the point
                # belongs to an edge of the pyramid.  Intersecting the main
                # face with f(s2, s3) gives the edge s1 = s2, and with
                # f(s1, s2) the edge s2 = s3 - so the criteria pair up the
                # opposite way round from the violated inequality.
                for mask, a2, b2 in ((cand[:, 0] < cand[:, 1] - 1e-12, a_23, b_23),
                                     (cand[:, 1] < cand[:, 2] - 1e-12, a_12, b_12)):
                    if not mask.any():
                        continue
                    A = np.column_stack([a_main, a2])
                    B = np.column_stack([b_main, b2])
                    Minv = np.linalg.inv(A.T @ De @ B)
                    fv = np.column_stack([start[mask] @ a_main - k, start[mask] @ a2 - k])
                    local[mask] = start[mask] - (fv @ Minv.T) @ (De @ B).T
                    local_t[mask] = De - (De @ B) @ Minv @ (A.T @ De)

                if sphi > 1e-9:
                    apex = self.c / math.tan(math.radians(self.phi))
                    beyond = local[:, 2] > apex + 1e-9
                    if beyond.any():
                        local[beyond] = apex
                        local_t[beyond] = self.residual_stiffness * De

                res[sub] = local
                tang[sub] = local_t

            over = res[:, 0] > st_max + tol
            if not over.any():
                break
            # Return along the principal axes onto the tension cut-off plane.
            # Only the clipped components lose their stiffness; the others stay
            # elastic, which is what keeps a cracked zone from going rigid-body.
            clipped = res[over] > st_max
            res[over] = np.minimum(res[over], st_max)
            keep = (~clipped).astype(float)
            proj = keep[:, :, None] * keep[:, None, :]
            tang[over] = proj * De + self.residual_stiffness * De

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
