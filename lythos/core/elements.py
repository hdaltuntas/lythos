"""Element formulations.

* :class:`ContinuumElements` - 6-node quadratic triangles in plane strain,
  vectorised over the whole mesh.
* :class:`BeamElements` - 3-node quadratic Timoshenko beams sharing their
  nodes with a continuum element edge, used for retaining walls, piles and
  tunnel linings.  Selective reduced integration of the shear term avoids
  shear locking in thin members.
* :class:`InterfaceElements` - zero-thickness quadratic interfaces with a
  Mohr-Coulomb slip criterion, for soil-structure contact.
* :class:`AnchorElements` - two-node elasto-plastic bars for struts, anchors
  and props, with optional pre-stress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 3-point Gauss rule for triangles, exact to degree 2
TRI_GAUSS = (
    ((2.0 / 3.0, 1.0 / 6.0), 1.0 / 3.0),
    ((1.0 / 6.0, 2.0 / 3.0), 1.0 / 3.0),
    ((1.0 / 6.0, 1.0 / 6.0), 1.0 / 3.0),
)

#: Gauss rules on [-1, 1]
LINE_GAUSS_3 = ((-np.sqrt(0.6), 5.0 / 9.0), (0.0, 8.0 / 9.0), (np.sqrt(0.6), 5.0 / 9.0))
LINE_GAUSS_2 = ((-1.0 / np.sqrt(3.0), 1.0), (1.0 / np.sqrt(3.0), 1.0))


def tri6_shape(xi: float, eta: float):
    """Shape functions and natural derivatives of the 6-node triangle."""
    lam = 1.0 - xi - eta
    N = np.array([
        lam * (2.0 * lam - 1.0),
        xi * (2.0 * xi - 1.0),
        eta * (2.0 * eta - 1.0),
        4.0 * lam * xi,
        4.0 * xi * eta,
        4.0 * eta * lam,
    ])
    dN = np.array([
        [1.0 - 4.0 * lam, 4.0 * xi - 1.0, 0.0, 4.0 * (lam - xi), 4.0 * eta, -4.0 * eta],
        [1.0 - 4.0 * lam, 0.0, 4.0 * eta - 1.0, -4.0 * xi, 4.0 * xi, 4.0 * (lam - eta)],
    ])
    return N, dN


def line3_shape(xi: float):
    """Quadratic line shape functions ordered (start, mid, end)."""
    N = np.array([0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)])
    dN = np.array([xi - 0.5, -2.0 * xi, xi + 0.5])
    return N, dN


class ContinuumElements:
    """All 6-node triangles of a mesh, with pre-computed strain operators."""

    n_gauss = 3
    dof_per_node = 2

    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        self.nodes = nodes
        self.elements = elements
        ne = len(elements)
        self.B = np.zeros((ne, self.n_gauss, 4, 12))
        self.detJw = np.zeros((ne, self.n_gauss))
        self.N = np.zeros((self.n_gauss, 6))
        self.gauss_xy = np.zeros((ne, self.n_gauss, 2))

        xy = nodes[elements]                                   # (ne, 6, 2)
        for g, ((xi, eta), w) in enumerate(TRI_GAUSS):
            N, dN = tri6_shape(xi, eta)
            self.N[g] = N
            J = np.einsum("kn,enj->ekj", dN, xy)               # (ne, 2, 2)
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            if np.any(det <= 0):
                bad = int(np.argmin(det))
                raise ValueError(f"element {bad} has a non-positive Jacobian ({det[bad]:.3e})")
            Jinv = np.empty_like(J)
            Jinv[:, 0, 0] = J[:, 1, 1] / det
            Jinv[:, 1, 1] = J[:, 0, 0] / det
            Jinv[:, 0, 1] = -J[:, 0, 1] / det
            Jinv[:, 1, 0] = -J[:, 1, 0] / det
            dNxy = np.einsum("eij,jn->ein", Jinv, dN)          # (ne, 2, 6)
            self.B[:, g, 0, 0::2] = dNxy[:, 0, :]
            self.B[:, g, 1, 1::2] = dNxy[:, 1, :]
            self.B[:, g, 3, 0::2] = dNxy[:, 1, :]
            self.B[:, g, 3, 1::2] = dNxy[:, 0, :]
            self.detJw[:, g] = det * 0.5 * w                   # 0.5 = area of the reference triangle
            self.gauss_xy[:, g] = np.einsum("n,enj->ej", N, xy)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def n_points(self) -> int:
        return self.n_elements * self.n_gauss

    def dofs(self) -> np.ndarray:
        """Global dof indices of every element, shape (ne, 12)."""
        d = np.empty((len(self.elements), 12), dtype=np.int64)
        d[:, 0::2] = 2 * self.elements
        d[:, 1::2] = 2 * self.elements + 1
        return d

    def strains(self, u: np.ndarray) -> np.ndarray:
        """Strain at every Gauss point, shape (ne * ngp, 4)."""
        ue = u[self.dofs()]                                    # (ne, 12)
        eps = np.matmul(self.B, ue[:, None, :, None])[..., 0]
        return eps.reshape(-1, 4)

    def internal_forces(self, stress: np.ndarray) -> np.ndarray:
        """Element internal force vectors, shape (ne, 12)."""
        sig = stress.reshape(self.n_elements, self.n_gauss, 4) * self.detJw[:, :, None]
        return np.matmul(self.B.transpose(0, 1, 3, 2), sig[..., None])[..., 0].sum(axis=1)

    def stiffness(self, tangent: np.ndarray) -> np.ndarray:
        """Element stiffness matrices, shape (ne, 12, 12)."""
        D = tangent.reshape(self.n_elements, self.n_gauss, 4, 4)
        # Two batched matrix products rather than one four-operand einsum:
        # these map onto BLAS and are several times faster.
        DB = np.matmul(D, self.B)                              # (ne, ngp, 4, 12)
        K = np.matmul(self.B.transpose(0, 1, 3, 2), DB)        # (ne, ngp, 12, 12)
        return (K * self.detJw[:, :, None, None]).sum(axis=1)

    def body_force(self, gamma_per_element: np.ndarray) -> np.ndarray:
        """Consistent nodal forces from a vertical body force (downwards)."""
        f = np.zeros((self.n_elements, 12))
        for g in range(self.n_gauss):
            w = self.detJw[:, g]
            f[:, 1::2] -= np.outer(gamma_per_element * w, self.N[g])
        return f

    def volumes(self) -> np.ndarray:
        return self.detJw.sum(axis=1)


@dataclass
class BeamSection:
    """Cross-section properties of a plate/beam element, per metre of wall."""

    EA: float           # axial rigidity [kN/m]
    EI: float           # bending rigidity [kN m2/m]
    GA: float           # shear rigidity [kN/m]
    weight: float = 0.0     # self weight [kN/m/m]
    Mp: float = 0.0         # plastic moment capacity [kNm/m], 0 = unlimited
    Np: float = 0.0         # axial capacity [kN/m], 0 = unlimited


class BeamElements:
    """3-node Timoshenko beams; dofs are (ux, uy, rotation) per node."""

    dof_per_node = 3

    def __init__(self, nodes: np.ndarray, chains: list[list[int]], section: BeamSection):
        self.nodes = nodes
        self.section = section
        conn = []
        for chain in chains:
            for i in range(0, len(chain) - 2, 2):
                conn.append(chain[i:i + 3])
        self.elements = np.array(conn, dtype=np.int64) if conn else np.zeros((0, 3), np.int64)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    def _geometry(self, e: int, xi: float):
        xy = self.nodes[self.elements[e]]
        N, dN = line3_shape(xi)
        dxy = dN @ xy
        jac = float(np.hypot(dxy[0], dxy[1]))
        t = dxy / jac                       # unit tangent
        return N, dN, jac, t

    def stiffness_and_force(self, u: np.ndarray, dof_map: np.ndarray):
        """Assemble beam stiffness and internal force.

        ``dof_map`` gives the global dof of (node, component) for the beam's
        own rotational dofs plus the shared translational dofs.
        """
        ne = self.n_elements
        Ke = np.zeros((ne, 9, 9))
        Fe = np.zeros((ne, 9))
        sec = self.section
        for e in range(ne):
            ue = u[dof_map[e]]
            k = np.zeros((9, 9))
            fi = np.zeros(9)
            # bending and axial: full 3-point integration
            for xi, w in LINE_GAUSS_3:
                N, dN, jac, t = self._geometry(e, xi)
                Bax = np.zeros(9)
                Bbend = np.zeros(9)
                for a in range(3):
                    Bax[3 * a + 0] = dN[a] / jac * t[0]
                    Bax[3 * a + 1] = dN[a] / jac * t[1]
                    Bbend[3 * a + 2] = dN[a] / jac
                k += w * jac * (sec.EA * np.outer(Bax, Bax) + sec.EI * np.outer(Bbend, Bbend))
                fi += w * jac * (sec.EA * (Bax @ ue) * Bax + sec.EI * (Bbend @ ue) * Bbend)
            # transverse shear: reduced 2-point integration prevents locking
            for xi, w in LINE_GAUSS_2:
                N, dN, jac, t = self._geometry(e, xi)
                n = np.array([-t[1], t[0]])
                Bs = np.zeros(9)
                for a in range(3):
                    Bs[3 * a + 0] = dN[a] / jac * n[0]
                    Bs[3 * a + 1] = dN[a] / jac * n[1]
                    Bs[3 * a + 2] = -N[a]
                k += w * jac * sec.GA * np.outer(Bs, Bs)
                fi += w * jac * sec.GA * (Bs @ ue) * Bs
            Ke[e] = k
            Fe[e] = fi
        return Ke, Fe

    def section_forces(self, u: np.ndarray, dof_map: np.ndarray):
        """Axial force, shear force and bending moment at the element nodes.

        Returns ``(s, N, V, M)`` where ``s`` is the distance along the member.
        """
        sec = self.section
        out = []
        for e in range(self.n_elements):
            ue = u[dof_map[e]]
            for xi in (-1.0, 0.0, 1.0):
                N, dN, jac, t = self._geometry(e, xi)
                n = np.array([-t[1], t[0]])
                eps = sum(dN[a] / jac * (t[0] * ue[3 * a] + t[1] * ue[3 * a + 1]) for a in range(3))
                kap = sum(dN[a] / jac * ue[3 * a + 2] for a in range(3))
                gam = (sum(dN[a] / jac * (n[0] * ue[3 * a] + n[1] * ue[3 * a + 1]) for a in range(3))
                       - sum(N[a] * ue[3 * a + 2] for a in range(3)))
                xy = line3_shape(xi)[0] @ self.nodes[self.elements[e]]
                out.append((xy[0], xy[1], sec.EA * eps, sec.GA * gam, sec.EI * kap))
        return np.array(out) if out else np.zeros((0, 5))

    def self_weight(self, dof_map: np.ndarray) -> np.ndarray:
        f = np.zeros((self.n_elements, 9))
        for e in range(self.n_elements):
            for xi, w in LINE_GAUSS_3:
                N, dN, jac, t = self._geometry(e, xi)
                for a in range(3):
                    f[e, 3 * a + 1] -= w * jac * self.section.weight * N[a]
        return f


@dataclass
class InterfaceProperties:
    """Mohr-Coulomb contact between a structure and the soil."""

    kn: float = 1.0e6      # normal stiffness [kN/m3]
    ks: float = 1.0e5      # shear stiffness [kN/m3]
    c: float = 0.0         # adhesion [kPa]
    phi: float = 20.0      # interface friction angle [deg]
    tensile: float = 0.0   # tensile capacity [kPa]


class InterfaceElements:
    """Zero-thickness quadratic interface elements (three node pairs)."""

    def __init__(self, nodes: np.ndarray, pairs: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
                 props: InterfaceProperties):
        self.nodes = nodes
        self.props = props
        self.pairs = pairs
        self.state = np.zeros((len(pairs), 2, 2))   # per element, per Gauss point: (tn, ts)

    @property
    def n_elements(self) -> int:
        return len(self.pairs)

    def dofs(self) -> np.ndarray:
        d = np.zeros((self.n_elements, 12), dtype=np.int64)
        for e, (a, b) in enumerate(self.pairs):
            for i in range(3):
                d[e, 2 * i] = 2 * a[i]
                d[e, 2 * i + 1] = 2 * a[i] + 1
                d[e, 6 + 2 * i] = 2 * b[i]
                d[e, 6 + 2 * i + 1] = 2 * b[i] + 1
        return d

    def stiffness_and_force(self, u: np.ndarray):
        ne = self.n_elements
        Ke = np.zeros((ne, 12, 12))
        Fe = np.zeros((ne, 12))
        dofs = self.dofs()
        p = self.props
        tan_phi = np.tan(np.radians(p.phi))
        for e in range(ne):
            a, b = self.pairs[e]
            xy = self.nodes[list(a)]
            ue = u[dofs[e]]
            k = np.zeros((12, 12))
            fi = np.zeros(12)
            for g, (xi, w) in enumerate(LINE_GAUSS_2):
                N, dN = line3_shape(xi)
                dxy = dN @ xy
                jac = float(np.hypot(dxy[0], dxy[1]))
                t = dxy / jac
                n = np.array([-t[1], t[0]])
                # relative displacement operator (side B minus side A)
                Bn = np.zeros(12)
                Bs = np.zeros(12)
                for i in range(3):
                    Bn[2 * i], Bn[2 * i + 1] = -N[i] * n[0], -N[i] * n[1]
                    Bn[6 + 2 * i], Bn[6 + 2 * i + 1] = N[i] * n[0], N[i] * n[1]
                    Bs[2 * i], Bs[2 * i + 1] = -N[i] * t[0], -N[i] * t[1]
                    Bs[6 + 2 * i], Bs[6 + 2 * i + 1] = N[i] * t[0], N[i] * t[1]
                dn = float(Bn @ ue)
                ds = float(Bs @ ue)
                tn = p.kn * dn
                ts = p.ks * ds
                kn_eff, ks_eff = p.kn, p.ks
                if tn > p.tensile:                     # gap opens
                    tn, kn_eff, ks_eff = p.tensile, 1e-6 * p.kn, 1e-6 * p.ks
                    ts = 0.0
                tmax = p.c - min(tn, 0.0) * tan_phi if tn < 0 else p.c
                if abs(ts) > tmax:                     # slip
                    ts = np.sign(ts) * tmax
                    ks_eff = 1e-4 * p.ks
                k += w * jac * (kn_eff * np.outer(Bn, Bn) + ks_eff * np.outer(Bs, Bs))
                fi += w * jac * (tn * Bn + ts * Bs)
                self.state[e, g] = (tn, ts)
            Ke[e] = k
            Fe[e] = fi
        return Ke, Fe


@dataclass
class AnchorProperties:
    """Node-to-node anchor, strut or prop."""

    EA: float = 1.0e5        # axial rigidity [kN]
    spacing: float = 1.0     # out-of-plane spacing [m]
    prestress: float = 0.0   # applied pre-stress force [kN]
    Fmax: float = 0.0        # yield force [kN], 0 = unlimited
    compression: bool = True  # False for a tension-only ground anchor


class AnchorElements:
    """Two-node bars connecting a structure to an anchorage point."""

    def __init__(self, nodes: np.ndarray, pairs: list[tuple[int, int]],
                 props: list[AnchorProperties]):
        self.nodes = nodes
        self.pairs = pairs
        self.props = props
        self.force = np.zeros(len(pairs))

    @property
    def n_elements(self) -> int:
        return len(self.pairs)

    def stiffness_and_force(self, u: np.ndarray):
        ne = self.n_elements
        Ke = np.zeros((ne, 4, 4))
        Fe = np.zeros((ne, 4))
        for e, (i, j) in enumerate(self.pairs):
            p = self.props[e]
            d = self.nodes[j] - self.nodes[i]
            L = float(np.hypot(*d))
            t = d / L
            B = np.array([-t[0], -t[1], t[0], t[1]]) / L
            ue = np.array([u[2 * i], u[2 * i + 1], u[2 * j], u[2 * j + 1]])
            k_ax = p.EA / max(p.spacing, 1e-9)       # rigidity per metre of wall
            direction = np.array([-t[0], -t[1], t[0], t[1]])
            force = k_ax * float(B @ ue) + p.prestress
            stiff = k_ax / L
            if p.Fmax and abs(force) > p.Fmax:       # yielded: cap the force
                force = float(np.sign(force)) * p.Fmax
                stiff *= 1e-4
            if not p.compression and force < 0.0:    # ground anchor went slack
                force, stiff = 0.0, 1e-6 * k_ax / L
            self.force[e] = force
            Ke[e] = stiff * np.outer(direction, direction)
            Fe[e] = force * direction
        return Ke, Fe

    def dofs(self) -> np.ndarray:
        d = np.zeros((self.n_elements, 4), dtype=np.int64)
        for e, (i, j) in enumerate(self.pairs):
            d[e] = (2 * i, 2 * i + 1, 2 * j, 2 * j + 1)
        return d
