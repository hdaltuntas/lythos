"""Post-processing: nodal fields, invariants and structural section forces."""

from __future__ import annotations

import numpy as np

from .materials import principal_stresses
from .problem import FEProblem
from .solver import StageResult


def gauss_to_nodes(problem: FEProblem, values: np.ndarray) -> np.ndarray:
    """Average Gauss point values onto the mesh nodes.

    The three Gauss points of a quadratic triangle are spread by simple
    area-weighted averaging onto all six of its nodes.  This is smoothing, not
    a stress recovery scheme, so peak values at a singularity are damped - it
    is contour plotting, not a design check.
    """
    ce = problem.continuum
    vals = np.asarray(values, float).reshape(ce.n_elements, ce.n_gauss)
    weights = ce.detJw
    elem_value = (vals * weights).sum(axis=1) / weights.sum(axis=1)

    total = np.zeros(problem.mesh.n_nodes)
    count = np.zeros(problem.mesh.n_nodes)
    area = ce.volumes()
    for k in range(6):
        np.add.at(total, problem.mesh.elements[:, k], elem_value * area)
        np.add.at(count, problem.mesh.elements[:, k], area)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def element_values(problem: FEProblem, values: np.ndarray) -> np.ndarray:
    ce = problem.continuum
    vals = np.asarray(values, float).reshape(ce.n_elements, ce.n_gauss)
    return (vals * ce.detJw).sum(axis=1) / ce.detJw.sum(axis=1)


def invariants(stress: np.ndarray) -> dict[str, np.ndarray]:
    """Common stress measures, all tension-positive except where noted."""
    s_a, s_b, szz, _, _ = principal_stresses(stress)
    pr = np.sort(np.stack([s_a, s_b, szz], axis=1), axis=1)[:, ::-1]
    s1, s2, s3 = pr[:, 0], pr[:, 1], pr[:, 2]
    mean = (s1 + s2 + s3) / 3.0
    q = np.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2))
    return {
        "sigma_1": s1,
        "sigma_2": s2,
        "sigma_3": s3,
        "sigma_xx": stress[:, 0],
        "sigma_yy": stress[:, 1],
        "sigma_zz": stress[:, 2],
        "tau_xy": stress[:, 3],
        "mean_stress": mean,
        "p_compression": -mean,          # positive in compression, as used in soil mechanics
        "deviatoric_stress": q,
        "max_shear": 0.5 * (s1 - s3),
    }


def strain_measures(problem: FEProblem, u: np.ndarray) -> dict[str, np.ndarray]:
    """Strain invariants at the Gauss points for a displacement field."""
    eps = problem.continuum.strains(u)
    exx, eyy, ezz, gxy = eps[:, 0], eps[:, 1], eps[:, 2], eps[:, 3]
    vol = exx + eyy + ezz
    ex, ey, ez = exx - vol / 3.0, eyy - vol / 3.0, ezz - vol / 3.0
    dev = np.sqrt(2.0 / 3.0 * (ex ** 2 + ey ** 2 + ez ** 2 + 0.5 * gxy ** 2))
    radius = np.sqrt(0.25 * (exx - eyy) ** 2 + 0.25 * gxy ** 2)
    return {
        "eps_xx": exx, "eps_yy": eyy, "gamma_xy": gxy,
        "volumetric_strain": vol,
        "deviatoric_strain": dev,
        "max_shear_strain": 2.0 * radius,
    }


def displacement_field(problem: FEProblem, result: StageResult) -> dict[str, np.ndarray]:
    u = result.displacement[: 2 * problem.mesh.n_nodes].reshape(-1, 2)
    return {"ux": u[:, 0], "uy": u[:, 1],
            "u_total": np.hypot(u[:, 0], u[:, 1])}


def structure_forces(problem: FEProblem, result: StageResult, name: str) -> dict:
    """Axial force, shear and bending moment along one structural member."""
    for s in problem.structures:
        if s.name != name:
            continue
        if not s.beam.n_elements:
            break
        # Section forces follow the displacement since installation.
        u = result.displacement if s.u_reference is None else result.displacement
        raw = s.beam.section_forces(u, s.dof_map)
        if len(raw) == 0:
            break
        # order the sample points along the member and drop duplicates at joins
        start = problem.mesh.nodes[s.chain[0]]
        dist = np.hypot(raw[:, 0] - start[0], raw[:, 1] - start[1])
        order = np.argsort(dist)
        raw, dist = raw[order], dist[order]
        keep = np.r_[True, np.diff(dist) > 1e-9]
        raw, dist = raw[keep], dist[keep]
        section = _section_of(problem, name)
        out = {
            "name": name,
            "s": dist,
            "x": raw[:, 0],
            "y": raw[:, 1],
            "N": raw[:, 2],
            "V": raw[:, 3],
            "M": raw[:, 4],
        }
        out["N_max"] = float(np.max(np.abs(raw[:, 2]))) if len(raw) else 0.0
        out["V_max"] = float(np.max(np.abs(raw[:, 3]))) if len(raw) else 0.0
        out["M_max"] = float(np.max(np.abs(raw[:, 4]))) if len(raw) else 0.0
        if section is not None and getattr(section, "Mp", 0.0):
            out["M_capacity"] = section.Mp
            out["utilisation"] = out["M_max"] / section.Mp if section.Mp else None
        return out
    return {"name": name, "s": np.zeros(0), "x": np.zeros(0), "y": np.zeros(0),
            "N": np.zeros(0), "V": np.zeros(0), "M": np.zeros(0),
            "N_max": 0.0, "V_max": 0.0, "M_max": 0.0}


def _section_of(problem: FEProblem, name: str):
    for st in problem.model.structures:
        if st.name == name:
            return st.section.section()
    return None


FIELDS = {
    "u_total": "total displacement |u| [m]",
    "ux": "horizontal displacement [m]",
    "uy": "vertical displacement [m]",
    "deviatoric_strain": "deviatoric strain [-]",
    "max_shear_strain": "maximum shear strain [-]",
    "volumetric_strain": "volumetric strain [-]",
    "sigma_1": "major principal effective stress [kPa]",
    "sigma_3": "minor principal effective stress [kPa]",
    "sigma_xx": "horizontal effective stress [kPa]",
    "sigma_yy": "vertical effective stress [kPa]",
    "tau_xy": "shear stress [kPa]",
    "p_compression": "mean effective stress p' [kPa]",
    "deviatoric_stress": "deviatoric stress q [kPa]",
    "max_shear": "maximum shear stress [kPa]",
    "pore_pressure": "pore water pressure [kPa]",
}


def field(problem: FEProblem, result: StageResult, name: str) -> tuple[np.ndarray, str]:
    """Return a nodal field by name, together with its label."""
    disp = displacement_field(problem, result)
    if name in disp:
        return disp[name], FIELDS.get(name, name)
    strains = strain_measures(problem, result.displacement)
    if name in strains:
        return gauss_to_nodes(problem, strains[name]), FIELDS.get(name, name)
    inv = invariants(result.state.stress)
    if name in inv:
        return gauss_to_nodes(problem, inv[name]), FIELDS.get(name, name)
    if name == "pore_pressure":
        stage_water = problem.model.water
        return problem.pore_pressure(problem.mesh.nodes, stage_water), FIELDS[name]
    raise KeyError(f"unknown result field {name!r}; choose from {sorted(FIELDS)}")
