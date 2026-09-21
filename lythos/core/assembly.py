"""Degree-of-freedom bookkeeping and global system assembly."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class DofManager:
    """Maps nodes and structural rotations onto global equation numbers.

    Continuum nodes own two translational dofs.  Beam nodes additionally own a
    rotation, allocated after all translations so that the soil-only part of
    the system keeps a contiguous numbering.
    """

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.n_translation = 2 * n_nodes
        self.rotation_of: dict[int, int] = {}

    def add_rotation(self, node: int) -> int:
        if node not in self.rotation_of:
            self.rotation_of[node] = self.n_translation + len(self.rotation_of)
        return self.rotation_of[node]

    @property
    def n_dof(self) -> int:
        return self.n_translation + len(self.rotation_of)

    def ux(self, node: int) -> int:
        return 2 * node

    def uy(self, node: int) -> int:
        return 2 * node + 1

    def beam_dofs(self, element_nodes) -> np.ndarray:
        out = []
        for n in element_nodes:
            out += [2 * n, 2 * n + 1, self.rotation_of[n]]
        return np.array(out, dtype=np.int64)


class SystemAssembler:
    """Accumulates element contributions into a sparse global system."""

    def __init__(self, n_dof: int):
        self.n_dof = n_dof
        self._rows: list[np.ndarray] = []
        self._cols: list[np.ndarray] = []
        self._vals: list[np.ndarray] = []
        self.force = np.zeros(n_dof)

    def add(self, dofs: np.ndarray, Ke: np.ndarray, Fe: np.ndarray | None = None,
            active: np.ndarray | None = None) -> None:
        """Scatter element matrices ``Ke`` (n, m, m) at ``dofs`` (n, m)."""
        if len(dofs) == 0:
            return
        if active is not None:
            dofs, Ke = dofs[active], Ke[active]
            Fe = None if Fe is None else Fe[active]
            if len(dofs) == 0:
                return
        m = dofs.shape[1]
        self._rows.append(np.repeat(dofs, m, axis=1).ravel())
        self._cols.append(np.tile(dofs, (1, m)).ravel())
        self._vals.append(Ke.reshape(len(dofs), -1).ravel())
        if Fe is not None:
            np.add.at(self.force, dofs.ravel(), Fe.ravel())

    def add_vector(self, dofs: np.ndarray, Fe: np.ndarray,
                   active: np.ndarray | None = None) -> None:
        if len(dofs) == 0:
            return
        if active is not None:
            dofs, Fe = dofs[active], Fe[active]
            if len(dofs) == 0:
                return
        np.add.at(self.force, dofs.ravel(), Fe.ravel())

    def matrix(self, symmetrise: bool = False) -> sp.csr_matrix:
        if not self._vals:
            return sp.csr_matrix((self.n_dof, self.n_dof))
        rows = np.concatenate(self._rows)
        cols = np.concatenate(self._cols)
        vals = np.concatenate(self._vals)
        K = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_dof, self.n_dof)).tocsr()
        if symmetrise:
            K = 0.5 * (K + K.T)
        # The tangent is genuinely unsymmetric - non-associated plastic flow
        # and frictional sliding both make it so - and the sparse LU solver
        # handles that directly.  Symmetrising it costs Newton its convergence
        # rate exactly where the soil is failing.
        return K


def solve_constrained(K: sp.csr_matrix, f: np.ndarray, fixed: np.ndarray,
                      prescribed: np.ndarray | None = None) -> np.ndarray:
    """Solve ``K u = f`` with ``fixed`` dofs held at ``prescribed`` values."""
    n = K.shape[0]
    free = np.ones(n, dtype=bool)
    free[fixed] = False
    u = np.zeros(n)
    if prescribed is not None and len(fixed):
        u[fixed] = prescribed
        f = f - K @ u
    Kff = K[free][:, free].tocsc()
    rhs = f[free]
    if Kff.shape[0] == 0:
        return u
    try:
        u[free] = spla.spsolve(Kff, rhs)
    except (RuntimeError, spla.MatrixRankWarning):
        u[free] = spla.lsqr(Kff, rhs)[0]
    if not np.all(np.isfinite(u)):
        # Singular system: regularise slightly rather than returning NaNs.
        diag = Kff.diagonal()
        scale = float(np.mean(np.abs(diag[diag != 0]))) if np.any(diag != 0) else 1.0
        u[free] = spla.spsolve(Kff + 1e-10 * scale * sp.eye(Kff.shape[0], format="csc"), rhs)
    return u
