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
        """Axial force, shear force and bending moment along the member.

        Sampled at the reduced integration points rather than at the nodes.
        The shear term is integrated at two points to avoid shear locking, so
        that is where the shear strain is accurate; reading it at the nodes
        instead produces a diagram that zigzags from element to element and
        hides the real distribution.
        """
        sec = self.section
        out = []
        stations = [xi for xi, _w in LINE_GAUSS_2]
        for e in range(self.n_elements):
            ue = u[dof_map[e]]
            for xi in stations:
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
    """Mohr-Coulomb contact between a structure and the soil.

    Leaving a value as ``None`` means "derive it from the adjacent soil":
    strength from the structure's ``r_inter``, stiffness from the soil moduli
    and a virtual interface thickness.  That is what an engineer expects from
    an interface they have not tuned by hand, and it keeps the contact stiff
    enough to be realistic without wrecking the conditioning of the system.
    """

    kn: float | None = None      # normal stiffness [kN/m3]
    ks: float | None = None      # shear stiffness [kN/m3]
    c: float | None = None       # adhesion [kPa]
    phi: float | None = None     # interface friction angle [deg]
    tensile: float = 0.0         # tensile capacity [kPa]
    #: interface thickness as a fraction of the local element size
    virtual_thickness: float = 0.1


class InterfaceElements:
    """Zero-thickness quadratic interface elements (three node pairs).

    The contact is elasto-plastic and incremental: tractions are carried
    forward from the last converged state and updated by the change in
    relative displacement, so unloading after slip recovers elastically
    instead of retracing the loading curve.  A total-displacement form is much
    simpler, but it makes the tangent inconsistent as soon as any point on the
    wall unloads, and the global iteration then stalls short of equilibrium.
    """

    def __init__(self, nodes: np.ndarray, pairs, props: InterfaceProperties):
        self.nodes = nodes
        self.props = props
        self.pairs = pairs
        n = len(pairs)
        # per element, per Gauss point: [gap, slip, normal traction, shear traction]
        self.committed = np.zeros((n, 2, 4))
        self.trial = np.zeros((n, 2, 4))
        #: contact state per Gauss point: 0 stuck, 1 sliding +, 2 sliding -, 3 open
        self.mode = np.zeros((n, 2), dtype=np.int8)
        #: when frozen, the contact states are held fixed.  Newton cannot
        #: converge while points keep switching between sticking and sliding
        #: from one iteration to the next, so the set is frozen once the
        #: iteration stops making progress and released at the next increment.
        self.frozen = False

    @property
    def n_elements(self) -> int:
        return len(self.pairs)

    def commit(self) -> None:
        self.committed = self.trial.copy()

    def snapshot(self):
        return (self.committed.copy(), self.mode.copy())

    def restore(self, state) -> None:
        committed, mode = state if isinstance(state, tuple) else (state, self.mode)
        self.committed = committed.copy()
        self.trial = committed.copy()
        self.mode = np.asarray(mode).copy()
        self.frozen = False

    def tractions(self):
        """Normal and shear traction at each Gauss point, and its position."""
        out = []
        for e, (a, _b) in enumerate(self.pairs):
            xy = self.nodes[list(a)]
            for g, (xi, _w) in enumerate(LINE_GAUSS_2):
                point = line3_shape(xi)[0] @ xy
                out.append((point[0], point[1], self.committed[e, g, 2],
                            self.committed[e, g, 3]))
        return np.array(out) if out else np.zeros((0, 4))

    def dofs(self) -> np.ndarray:
        d = np.zeros((self.n_elements, 12), dtype=np.int64)
        for e, (a, b) in enumerate(self.pairs):
            for i in range(3):
                d[e, 2 * i] = 2 * a[i]
                d[e, 2 * i + 1] = 2 * a[i] + 1
                d[e, 6 + 2 * i] = 2 * b[i]
                d[e, 6 + 2 * i + 1] = 2 * b[i] + 1
        return d

    def _operators(self, e: int, xi: float):
        """Relative-displacement operators and the Jacobian at one Gauss point."""
        xy = self.nodes[list(self.pairs[e][0])]
        N, dN = line3_shape(xi)
        dxy = dN @ xy
        jac = float(np.hypot(dxy[0], dxy[1]))
        t = dxy / jac
        n = np.array([-t[1], t[0]])
        Bn = np.zeros(12)
        Bs = np.zeros(12)
        for i in range(3):
            Bn[2 * i], Bn[2 * i + 1] = -N[i] * n[0], -N[i] * n[1]
            Bn[6 + 2 * i], Bn[6 + 2 * i + 1] = N[i] * n[0], N[i] * n[1]
            Bs[2 * i], Bs[2 * i + 1] = -N[i] * t[0], -N[i] * t[1]
            Bs[6 + 2 * i], Bs[6 + 2 * i + 1] = N[i] * t[0], N[i] * t[1]
        return Bn, Bs, jac

    def stiffness_and_force(self, u: np.ndarray, rigid: bool = False):
        """Interface forces and stiffness.

        ``rigid`` ties the two sides together instead of letting them slip.
        That is what an interface does before its structure is installed: the
        node pairs exist in the mesh from the first stage, and without the tie
        the soil would be split along the future wall line from the start.
        """
        ne = self.n_elements
        Ke = np.zeros((ne, 12, 12))
        Fe = np.zeros((ne, 12))
        dofs = self.dofs()
        p = self.props
        kn = p.kn or 1.0e6
        ks = p.ks or 1.0e5
        cohesion = p.c or 0.0
        tan_phi = np.tan(np.radians(p.phi or 0.0))
        tie = 100.0 * max(kn, ks)
        residual = 1.0e-3

        for e in range(ne):
            ue = u[dofs[e]]
            k = np.zeros((12, 12))
            fi = np.zeros(12)
            for g, (xi, w) in enumerate(LINE_GAUSS_2):
                Bn, Bs, jac = self._operators(e, xi)
                dn = float(Bn @ ue)
                ds = float(Bs @ ue)
                coupling = 0.0
                if rigid:
                    tn, ts = tie * dn, tie * ds
                    kn_eff = ks_eff = tie
                else:
                    dn_c, ds_c, tn_c, ts_c = self.committed[e, g]
                    tn = tn_c + kn * (dn - dn_c)
                    ts = ts_c + ks * (ds - ds_c)
                    kn_eff, ks_eff = kn, ks
                    if self.frozen:
                        mode = int(self.mode[e, g])
                    elif tn > p.tensile:
                        mode = 3
                    else:
                        limit = cohesion - tn * tan_phi
                        mode = 0 if abs(ts) <= limit else (1 if ts > 0 else 2)
                    self.mode[e, g] = mode

                    if mode == 3:                      # the gap has opened
                        tn, ts = p.tensile, 0.0
                        kn_eff, ks_eff = residual * kn, residual * ks
                    elif mode != 0:                    # sliding
                        sign = 1.0 if mode == 1 else -1.0
                        tmax = cohesion - tn * tan_phi  # tn <= 0 in contact
                        ts = sign * max(tmax, 0.0)
                        ks_eff = residual * ks
                        # While sliding the shear traction is set by the
                        # normal traction, so the tangent needs the coupling
                        # term d(tau)/d(sigma_n).  It is far larger than the
                        # residual shear stiffness, and leaving it out stalls
                        # the global iteration however small the load step.
                        coupling = -sign * tan_phi * kn
                self.trial[e, g] = (dn, ds, tn, ts)
                k += w * jac * (kn_eff * np.outer(Bn, Bn) + ks_eff * np.outer(Bs, Bs))
                if coupling:
                    k += w * jac * coupling * np.outer(Bs, Bn)
                fi += w * jac * (tn * Bn + ts * Bs)
            Ke[e] = k
            Fe[e] = fi
        return Ke, Fe


@dataclass
class AnchorProperties:
    """A strut, prop or ground anchor."""

    EA: float = 1.0e5        # axial rigidity [kN]
    spacing: float = 1.0     # out-of-plane spacing [m]
    prestress: float = 0.0   # force locked in when the anchor is stressed [kN]
    Fmax: float = 0.0        # capacity [kN], 0 = unlimited
    compression: bool = True  # False for a tension-only ground anchor
    #: length of the fixed (grout) length at the far end [m].  The anchor load
    #: is shared over the soil along it; tying a ground anchor to one node
    #: instead lets the anchorage be dragged through the mesh and the
    #: pre-stress simply bleeds away.
    grout_length: float = 4.0


class AnchorElements:
    """Bars connecting a structure to an anchorage in the soil.

    The near end attaches to a single node on the wall.  The far end is a
    weighted set of nodes spanning the grout length, which spreads the
    anchorage force into the soil the way a grouted bond length does.
    """

    def __init__(self, nodes: np.ndarray, anchors, props: list[AnchorProperties]):
        self.nodes = nodes
        self.props = props
        #: (near node, far node indices, far weights) for each anchor
        self.anchors = anchors
        #: force per anchor [kN] (not per metre of wall)
        self.force = np.zeros(len(anchors))
        #: strain locked in when the anchor was stressed, one per anchor
        self.reference_strain = np.zeros(len(anchors))

    @property
    def n_elements(self) -> int:
        return len(self.anchors)

    @property
    def pairs(self):
        """Near and representative far node, for drawing."""
        return [(near, far[int(np.argmax(w))]) for (near, far, w) in self.anchors]

    def geometry(self, e: int):
        """``(dofs, direction, length)`` of one anchor.

        ``direction`` is the operator whose product with the nodal
        displacements gives the elongation of the anchor.
        """
        near, far, weights = self.anchors[e]
        d = (weights @ self.nodes[far]) - self.nodes[near]
        L = float(np.hypot(*d))
        if L < 1e-9:
            return None, None, 0.0
        t = d / L
        dofs = np.array([2 * near, 2 * near + 1]
                        + [v for n in far for v in (2 * n, 2 * n + 1)], dtype=np.int64)
        direction = np.concatenate([[-t[0], -t[1]], np.outer(weights, t).ravel()])
        return dofs, direction, L

    def prestress_load(self, e: int) -> tuple[np.ndarray, np.ndarray] | None:
        """External force pair equivalent to jacking the anchor to its lock-off load.

        While an anchor is being stressed its force is what the jack sets, not
        what the surrounding ground decides, so during that stage it acts as a
        prescribed force rather than as an elastic bar.
        """
        p = self.props[e]
        if not p.prestress:
            return None
        dofs, direction, _L = self.geometry(e)
        if dofs is None:
            return None
        return dofs, -(p.prestress / max(p.spacing, 1e-9)) * direction

    def set_reference(self, e: int, u: np.ndarray) -> None:
        """Lock in the current length as the anchor's unstressed reference."""
        dofs, direction, L = self.geometry(e)
        if dofs is None:
            return
        self.reference_strain[e] = float(direction @ u[dofs]) / L

    def contributions(self, u: np.ndarray, skip: set[int] | None = None):
        """Yield ``(index, dofs, Ke, Fe)`` for each elastic anchor."""
        for e, _spec in enumerate(self.anchors):
            if skip and e in skip:
                continue
            p = self.props[e]
            dofs, direction, L = self.geometry(e)
            if dofs is None:
                continue
            k_ax = p.EA / max(p.spacing, 1e-9)              # per metre of wall
            locked = p.prestress / max(p.spacing, 1e-9)
            strain = float(direction @ u[dofs]) / L - self.reference_strain[e]
            force = k_ax * strain + locked                  # [kN per metre]
            stiff = k_ax / L
            capacity = p.Fmax / max(p.spacing, 1e-9) if p.Fmax else 0.0
            if capacity and abs(force) > capacity:          # capacity reached
                force = float(np.sign(force)) * capacity
                stiff *= 1e-4
            if not p.compression and force < 0.0:           # gone slack
                force, stiff = 0.0, 1e-6 * k_ax / L
            self.force[e] = force * max(p.spacing, 1e-9)    # report per anchor
            yield e, dofs, stiff * np.outer(direction, direction), force * direction
