"""Non-linear solver: staged construction, plastic analysis and strength reduction."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .assembly import SystemAssembler, solve_constrained
from .materials import MaterialState
from .model import Stage
from .problem import FEProblem


@dataclass
class IterationLog:
    increment: int
    iteration: int
    residual: float
    displacement_norm: float


@dataclass
class StageResult:
    """Everything the post-processor needs from one construction stage."""

    name: str
    kind: str
    converged: bool
    #: displacement since the last reset, which is what the contours show
    displacement: np.ndarray
    state: MaterialState
    active_elements: np.ndarray
    active_structures: list[str] = field(default_factory=list)
    active_anchors: list[str] = field(default_factory=list)
    srf: float | None = None
    srf_curve: list[tuple[float, float]] = field(default_factory=list)
    iterations: list[IterationLog] = field(default_factory=list)
    message: str = ""
    seconds: float = 0.0
    #: total displacement from the undeformed mesh.  Structural section forces
    #: are measured from a member's own installation reference, not from the
    #: display reset, so they need this rather than ``displacement``.
    total_displacement: np.ndarray | None = None
    plastic_fraction: float = 0.0
    max_displacement: float = 0.0
    anchor_forces: dict[str, float] = field(default_factory=dict)


class Solver:
    """Drives a model through its construction stages."""

    def __init__(self, problem: FEProblem, tolerance: float = 1.0e-3,
                 max_iterations: int = 60, verbose: bool = False):
        self.p = problem
        self.tol = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.results: list[StageResult] = []
        self._u = np.zeros(problem.dofs.n_dof)
        self._state = MaterialState.zeros(problem.continuum.n_points)
        self._u_offset = np.zeros(problem.dofs.n_dof)   # displacements reset by a stage
        self._installed: set[str] = set()               # structures already built
        self._progress = None                           # reported per stage and per trial
        self._stressed: set[str] = set()                # anchors already jacked
        x0, y0, x1, y1 = problem.mesh.bounds()
        self._size = max(x1 - x0, y1 - y0, 1.0)          # model size, for step limits
        self._has_contacts = any(st.interfaces for st in problem.structures)

    # ------------------------------------------------------------------ public
    def run(self, progress=None) -> list[StageResult]:
        stages = self.p.model.resolved_stages()
        self._progress = progress
        for i, stage in enumerate(stages):
            if progress:
                progress(i, len(stages), stage.name)
            result = self.run_stage(stage)
            self.results.append(result)
        return self.results

    def run_stage(self, stage: Stage) -> StageResult:
        t0 = time.time()
        active = self.p.active_elements(stage.active_layers or [])
        if stage.kind == "initial":
            result = self._initial_stage(stage, active)
        elif stage.kind == "ssr":
            result = self._ssr_stage(stage, active)
        else:
            result = self._plastic_stage(stage, active)
        result.seconds = time.time() - t0
        result.plastic_fraction = float(np.mean(self._state.yielding[np.repeat(active, 3)])) \
            if active.any() else 0.0
        disp = (self._u - self._u_offset)[: 2 * self.p.mesh.n_nodes].reshape(-1, 2)
        result.max_displacement = float(np.max(np.hypot(disp[:, 0], disp[:, 1]))) if len(disp) else 0.0
        if self.p.anchors is not None:
            result.anchor_forces = {name: float(f) for name, f
                                    in zip(self.p.anchor_names, self.p.anchors.force)}
        return result

    # ----------------------------------------------------------------- stages
    def _initial_stage(self, stage: Stage, active: np.ndarray) -> StageResult:
        """Initial stresses, either by the K0 procedure or by gravity loading."""
        water = stage.water or self.p.model.water
        if self.p.model.initial_stress == "k0":
            stress = self.p.initial_k0_stress(active, water)
            self._state = MaterialState.zeros(self.p.continuum.n_points)
            self._state.stress[:] = stress
            self._u[:] = 0.0
            self._u_offset[:] = 0.0
            # Let the solver iron out any out-of-balance the K0 field leaves
            # behind (it is exact only for horizontal ground).
            res = self._newton(stage, active, increments=max(1, stage.increments // 2),
                               label="initial")
            res.kind = "initial"
        else:
            self._state = MaterialState.zeros(self.p.continuum.n_points)
            res = self._newton(stage, active, increments=stage.increments, label="initial")
            res.kind = "initial"
        self._u_offset = self._u.copy()      # initial stage leaves no displacement
        res.displacement = np.zeros_like(self._u)
        return res

    def _plastic_stage(self, stage: Stage, active: np.ndarray) -> StageResult:
        if stage.reset_displacements:
            self._u_offset = self._u.copy()
        return self._newton(stage, active, increments=stage.increments, label=stage.name)

    def _ssr_stage(self, stage: Stage, active: np.ndarray) -> StageResult:
        """Shear strength reduction: find the largest factor that still equilibrates.

        Cohesion and tan(phi) are divided by a trial factor and the whole
        non-linear problem is re-solved.  The factor of safety is the largest
        factor for which an equilibrium state still exists; past it the
        deforming zone links up into a mechanism, displacements run away, and
        the Newton iteration stops converging.

        Each trial starts from the previous converged trial rather than from
        the original state.  That both halves the work and makes the
        displacement-versus-factor curve meaningful, because it traces one
        continuous loading path instead of a set of unrelated solutions.
        """
        base_u = self._u.copy()
        base_state = self._state.copy()
        base_structures = self._snapshot_structures()
        base_materials = list(self.p.materials)
        reference_offset = self._u_offset.copy()

        curve: list[tuple[float, float]] = []
        trials: list[tuple[float, bool, float, float, int]] = []
        successful_iterations: list[int] = []
        anchor_u, anchor_state = base_u.copy(), base_state.copy()
        anchor_structures = base_structures
        best_u, best_state = base_u.copy(), base_state.copy()

        def attempt(srf: float, warm_u, warm_state, warm_structures):
            self.p.materials = [m.reduced(srf) for m in base_materials]
            self._u = warm_u.copy()
            self._state = warm_state.copy()
            self._restore_structures(warm_structures)
            started = time.time()
            # A trial that has already taken many times the work of the
            # successful ones is at failure; spending the rest of the
            # sub-stepping ladder on it only confirms that slowly.
            budget = max(200, 8 * max(successful_iterations, default=25))
            res = self._newton(stage, active, increments=max(stage.increments, 6),
                               label=f"SRF {srf:.3f}", budget=budget)
            if res.converged:
                successful_iterations.append(len(res.iterations))
            disp = (self._u - reference_offset)[: 2 * self.p.mesh.n_nodes].reshape(-1, 2)
            dmax = float(np.max(np.hypot(disp[:, 0], disp[:, 1]))) if len(disp) else 0.0
            curve.append((srf, dmax))
            trials.append((srf, res.converged, dmax, time.time() - started,
                           len(res.iterations)))
            if self.verbose:
                print(f"      trial SRF {srf:.3f}: "
                      f"{'equilibrium found' if res.converged else 'no equilibrium'}, "
                      f"{1000 * dmax:.1f} mm, {len(res.iterations)} iterations, "
                      f"{time.time() - started:.1f} s", flush=True)
            if self._progress:
                self._progress(-1, 0, f"{stage.name}: trial factor {srf:.2f}")
            return res.converged, dmax

        # 1. march upwards from the lower bound until equilibrium is lost
        srf = max(stage.srf_min, 0.5)
        step = 0.2
        last_ok = None
        hi = stage.srf_max
        for _ in range(30):
            ok, _dmax = attempt(srf, anchor_u, anchor_state, anchor_structures)
            if ok:
                last_ok = srf
                anchor_u, anchor_state = self._u.copy(), self._state.copy()
                anchor_structures = self._snapshot_structures()
                best_u, best_state = anchor_u, anchor_state
                if srf >= stage.srf_max:
                    hi = stage.srf_max
                    break
                # A trial that went through easily says the slope is still far
                # from failing, so take a longer stride towards it.
                step = min(1.5 * step, 0.5)
                srf = min(srf + step, stage.srf_max)
            else:
                hi = srf
                break
        else:
            hi = srf

        # 2. bisect between the last stable and the first unstable factor
        if last_ok is None:
            fos = stage.srf_min
            message = ("no equilibrium even at the lowest trial factor: "
                       "the slope is unstable as modelled")
        else:
            lo = last_ok
            for _ in range(8):
                if hi - lo <= 0.01:
                    break
                mid = 0.5 * (lo + hi)
                ok, _d = attempt(mid, anchor_u, anchor_state, anchor_structures)
                if ok:
                    lo = mid
                    anchor_u, anchor_state = self._u.copy(), self._state.copy()
                    anchor_structures = self._snapshot_structures()
                    best_u, best_state = anchor_u, anchor_state
                else:
                    hi = mid
            fos = lo
            message = f"factor of safety {fos:.3f}, bracketed within {hi - lo:.3f}"

        self._u, self._state = best_u, best_state
        self._restore_structures(anchor_structures)
        self.p.materials = base_materials
        result = StageResult(name=stage.name, kind="ssr", converged=last_ok is not None,
                             displacement=self._u - reference_offset, state=self._state,
                             active_elements=active, srf=fos,
                             srf_curve=sorted(curve), message=message,
                             total_displacement=self._u.copy())
        result.active_structures = list(stage.active_structures or [])
        result.active_anchors = list(stage.active_anchors or [])
        return result

    # ------------------------------------------------------- structural state
    def _snapshot_structures(self):
        """Committed interface tractions, so a trial can be rolled back."""
        return [[ie.snapshot() for ie in st.interfaces] for st in self.p.structures]

    def _restore_structures(self, saved) -> None:
        for st, states in zip(self.p.structures, saved):
            for ie, state in zip(st.interfaces, states):
                ie.restore(state)

    def _commit_structures(self) -> None:
        for st in self.p.structures:
            for ie in st.interfaces:
                ie.commit()

    def _freeze_contacts(self, frozen: bool) -> None:
        """Hold the stick/slip/open state of every interface point fixed.

        A contact point that keeps switching state from one Newton iteration
        to the next leaves a pair of equal and opposite residuals that no step
        size can remove.  Freezing the set lets the iteration converge on it;
        the next increment starts from a fresh set.
        """
        for st in self.p.structures:
            for ie in st.interfaces:
                ie.frozen = frozen

    # ----------------------------------------------------------------- newton
    def _newton(self, stage: Stage, active: np.ndarray, increments: int,
                label: str, budget: int | None = None) -> StageResult:
        p = self.p
        water = stage.water or p.model.water
        fixed = p.fixed_dofs
        logs: list[IterationLog] = []

        f_body = p.body_force(active, water)
        f_ext = np.zeros(p.dofs.n_dof)
        np.add.at(f_ext, p.continuum.dofs()[active].ravel(), f_body[active].ravel())
        f_ext += p.line_load_vector(stage.active_loads)
        active_structs = [s for s in p.structures if s.name in set(stage.active_structures or [])]
        # Anchors being stressed in this stage act as a prescribed jack force.
        stressing = set()
        if p.anchors is not None:
            for e, name in enumerate(p.anchor_names):
                if name in set(stage.active_anchors or []) and name not in self._stressed:
                    stressing.add(e)
                    load = p.anchors.prestress_load(e)
                    if load is not None:
                        f_ext[load[0]] += load[1]
        stage_stressing = stressing
        for s in active_structs:
            if s.beam.n_elements:
                w = s.beam.self_weight(s.dof_map)
                np.add.at(f_ext, s.dof_map.ravel(), w.ravel())

        self._stressing_now = stage_stressing
        u_committed = self._u.copy()
        # A member is built into ground that has already deformed, so it starts
        # free of force: its reference displacement is the field at the moment
        # of installation, and any interface tie tractions from before are
        # cleared rather than carried into the contact.
        for st in active_structs:
            if st.name not in self._installed:
                st.u_reference = u_committed.copy()
                for ie in st.interfaces:
                    ie.restore(np.zeros_like(ie.committed))
                self._installed.add(st.name)
        state_committed = self._state.copy()
        structures_committed = self._snapshot_structures()
        f_int0, _, _ = self._internal(u_committed, u_committed, state_committed, active,
                                      active_structs, stage, tangent=False)

        # Adaptive load stepping: an increment that will not converge is
        # retried at half the size rather than abandoning the stage.  This is
        # what lets a strength reduction run approach collapse closely enough
        # for the factor of safety to be meaningful.
        message = ""
        lam = 0.0
        dlam = 1.0 / max(increments, 1)
        min_dlam = dlam / 64.0
        # Once a step size has failed there is no point growing back to it and
        # walking into the same wall again, which is what made a doomed
        # strength reduction trial cost ten times a successful one.
        dlam_ceiling = 1.0 / max(increments, 1)
        cuts = 0
        scale_ref = max(np.linalg.norm(f_ext), 1e-8)

        while lam < 1.0 - 1e-10:
            if budget is not None and len(logs) > budget:
                message = (f"gave up after {len(logs)} iterations at "
                           f"{lam * 100:.0f}% of the stage load")
                break
            trial = min(1.0, lam + dlam)
            target = f_int0 + trial * (f_ext - f_int0)
            u_try = u_committed.copy()
            ok = False
            stalled = 0
            self._freeze_contacts(False)
            for it in range(self.max_iterations):
                f_int, K, state = self._internal(u_try, u_committed, state_committed, active,
                                                 active_structs, stage, tangent=True)
                r = target - f_int
                r[fixed] = 0.0
                rn = float(np.linalg.norm(r)) / scale_ref
                logs.append(IterationLog(len(logs), it, rn, float(np.linalg.norm(u_try))))
                if rn < self.tol:
                    ok = True
                    break
                du = solve_constrained(K, r, fixed)
                if not np.all(np.isfinite(du)):
                    message = "singular stiffness matrix"
                    break
                du = self._limit_step(du)
                u_try, improved = self._line_search(u_try, du, u_committed, state_committed,
                                                    target, active, active_structs, stage,
                                                    fixed, float(np.linalg.norm(r)))
                if not improved:
                    stalled += 1
                    if stalled == 1 and self._has_contacts and not self._contacts_frozen():
                        # First try holding the contact states: the stall is
                        # usually points switching between sticking and sliding.
                        self._freeze_contacts(True)
                    elif stalled >= 3:
                        message = "the Newton step stopped reducing the imbalance"
                        break
                else:
                    stalled = 0
                if not np.all(np.isfinite(u_try)) or np.linalg.norm(u_try) > 1.0e6 * self._size:
                    message = "displacements ran away"
                    break

            if ok:
                _f, _K, state = self._internal(u_try, u_committed, state_committed, active,
                                               active_structs, stage, tangent=False)
                self._commit_structures()
                structures_committed = self._snapshot_structures()
                u_committed = u_try
                state_committed = state
                lam = trial
                if it < 6 and dlam < dlam_ceiling:
                    dlam = min(2.0 * dlam, dlam_ceiling)
            else:
                self._restore_structures(structures_committed)
                dlam_ceiling = min(dlam_ceiling, 0.5 * dlam)
                dlam *= 0.5
                cuts += 1
                if dlam < min_dlam or cuts > 20:
                    if not message:
                        message = f"no equilibrium beyond {lam * 100:.0f}% of the stage load"
                    break

        converged_all = lam >= 1.0 - 1e-10
        self._freeze_contacts(False)
        self._restore_structures(structures_committed)
        self._u = u_committed
        self._state = state_committed
        # Once jacked, the anchor keeps the force it was stressed to and from
        # here on responds elastically about that length.
        if p.anchors is not None:
            for e in stage_stressing:
                p.anchors.set_reference(e, u_committed)
                self._stressed.add(p.anchor_names[e])
        self._stressing_now = set()
        if p.anchors is not None:
            list(p.anchors.contributions(u_committed))

        return StageResult(name=label, kind=stage.kind, converged=converged_all,
                           displacement=self._u - self._u_offset, state=self._state,
                           active_elements=active,
                           active_structures=[s.name for s in active_structs],
                           active_anchors=list(stage.active_anchors or []),
                           iterations=logs, message=message,
                           total_displacement=self._u.copy())

    def _contacts_frozen(self) -> bool:
        return any(ie.frozen for st in self.p.structures for ie in st.interfaces)

    def _limit_step(self, du: np.ndarray) -> np.ndarray:
        """Cap a Newton step so one bad tangent cannot throw the solution away."""
        cap = 0.05 * self._size
        peak = float(np.max(np.abs(du))) if du.size else 0.0
        if peak > cap > 0.0:
            return du * (cap / peak)
        return du

    def _line_search(self, u, du, u_committed, state_committed, target, active,
                     active_structs, stage, fixed, r0_norm):
        """Backtracking line search, needed because non-associated flow makes
        the symmetrised tangent only an approximation of the true Jacobian.

        The full Newton step is tried first and kept whenever it reduces the
        residual, so the extra stress updates are only paid for where the step
        actually misbehaves.
        """
        def residual(alpha):
            f_int, _, _ = self._internal(u + alpha * du, u_committed, state_committed,
                                         active, active_structs, stage, tangent=False)
            r = target - f_int
            r[fixed] = 0.0
            return float(np.linalg.norm(r))

        full = residual(1.0)
        if full <= r0_norm or r0_norm == 0.0:
            return u + du, True
        best_alpha, best = 1.0, full
        for alpha in (0.5, 0.25, 0.1, 0.03):
            value = residual(alpha)
            if value < best:
                best_alpha, best = alpha, value
            if value < r0_norm:
                return u + alpha * du, True
        # Even the shortest step makes the imbalance worse: the tangent is not
        # describing this state, and pressing on only diverges.  Report the
        # failure so the increment is cut instead.
        return u + best_alpha * du, best < 1.5 * r0_norm

    # ------------------------------------------------------------- assembly
    def _internal(self, u, u_committed, state_committed, active, active_structs,
                  stage, tangent: bool = True):
        """Return ``(internal force, tangent or None, updated material state)``."""
        p = self.p
        ce = p.continuum
        n_dof = p.dofs.n_dof
        asm = SystemAssembler(n_dof)

        dstrain = ce.strains(u - u_committed)
        stress = np.empty_like(state_committed.stress)
        tangents = np.zeros((ce.n_points, 4, 4))
        new_state = state_committed.copy()

        for mat, _elems, gp in p.material_groups():
            sub = MaterialState(state_committed.stress[gp], state_committed.plastic_strain[gp],
                                state_committed.eps_p_eq[gp], state_committed.yielding[gp])
            s, t, ns = mat.update(sub, dstrain[gp])
            stress[gp] = s
            tangents[gp] = t
            new_state.stress[gp] = ns.stress
            new_state.plastic_strain[gp] = ns.plastic_strain
            new_state.eps_p_eq[gp] = ns.eps_p_eq
            new_state.yielding[gp] = ns.yielding

        f_elem = ce.internal_forces(stress)
        dofs = ce.dofs()
        asm.add_vector(dofs, f_elem, active=active)
        if tangent:
            Ke = ce.stiffness(tangents)
            asm.add(dofs, Ke, active=active)

        active_names = {s.name for s in active_structs}
        for s in p.structures:
            live = s.name in active_names
            if live and s.beam.n_elements:
                u_beam = u if s.u_reference is None else u - s.u_reference
                Kb, Fb = s.beam.stiffness_and_force(u_beam, s.dof_map)
                asm.add_vector(s.dof_map, Fb)
                if tangent:
                    asm.add(s.dof_map, Kb)
            # Interfaces are always assembled: as a Mohr-Coulomb contact once
            # the structure is installed, and as a rigid tie before that.  An
            # interface whose soil has been excavated away is left out
            # entirely, so it cannot hold the wall against nothing.
            for k, (ie, idofs) in enumerate(zip(s.interfaces, s.interface_dofs)):
                support = s.interface_support[k] if k < len(s.interface_support) else None
                if support is None:
                    mask = None
                else:
                    mask = np.array([bool(sup) and bool(active[sup].any()) for sup in support])
                    if not mask.any():
                        continue
                u_int = u if (not live or s.u_reference is None) else u - s.u_reference
                Ki, Fi = ie.stiffness_and_force(u_int, rigid=not live)
                asm.add_vector(idofs, Fi, active=mask)
                if tangent:
                    asm.add(idofs, Ki, active=mask)

        if p.anchors is not None and stage.active_anchors:
            wanted = set(stage.active_anchors)
            skip = getattr(self, "_stressing_now", set())
            for e, dofs_a, Ka, Fa in p.anchors.contributions(u, skip=skip):
                if p.anchor_names[e] not in wanted:
                    continue
                asm.add_vector(dofs_a[None, :], Fa[None, :])
                if tangent:
                    asm.add(dofs_a[None, :], Ka[None, :, :])

        # Elements that are switched off still own dofs; pin them so the
        # system stays non-singular without affecting the active soil.
        if tangent:
            K = _pin_orphan_dofs(asm.matrix(), self._live_dofs(active, active_structs))
            return asm.force, K, new_state
        return asm.force, None, new_state

    def _live_dofs(self, active, active_structs) -> np.ndarray:
        p = self.p
        live = np.zeros(p.dofs.n_dof, bool)
        nodes = np.unique(p.mesh.elements[active]) if active.any() else np.zeros(0, int)
        live[2 * nodes] = True
        live[2 * nodes + 1] = True
        for s in p.structures:
            if s.name in {a.name for a in active_structs}:
                live[s.dof_map.ravel()] = True
            for k, d in enumerate(s.interface_dofs):
                support = s.interface_support[k] if k < len(s.interface_support) else None
                if support is None:
                    live[d.ravel()] = True
                    continue
                for e, sup in enumerate(support):
                    if sup and active[sup].any():
                        live[d[e]] = True
        return live


def _pin_orphan_dofs(K, live: np.ndarray):
    """Give inactive dofs a unit diagonal so the factorisation stays regular."""
    import scipy.sparse as sp
    row_weight = np.asarray(abs(K).sum(axis=1)).ravel()
    reference = float(np.mean(row_weight[row_weight > 0])) if np.any(row_weight > 0) else 1.0
    orphan = row_weight <= 1e-12 * reference
    if not orphan.any():
        return K
    # An orphan dof has an all-zero row, so a unit diagonal regularises the
    # factorisation without touching the active part of the system.
    return (K + sp.diags(np.where(orphan, reference, 0.0))).tocsr()
