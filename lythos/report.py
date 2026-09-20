"""Run a model and produce a complete set of results.

Writes an HTML report with the figures embedded, so a whole analysis is one
self-contained file that can be filed, mailed or printed.
"""

from __future__ import annotations

import base64
import html
import json
import os
from datetime import datetime

import numpy as np

from .core import results as R
from .core.model import Model
from .core.problem import FEProblem
from .core.serialize import model_to_dict
from .core.solver import Solver, StageResult
from .viz import plots

#: fields plotted for every stage
DEFAULT_FIELDS = ("u_total", "deviatoric_strain", "p_compression")


def run_and_report(model: Model, out_dir: str = "out", fields=DEFAULT_FIELDS,
                   mesh_size: float | None = None, tolerance: float = 3.0e-3,
                   verbose: bool = True) -> dict:
    """Mesh, solve and write a full report.  Returns the summary dictionary."""
    os.makedirs(out_dir, exist_ok=True)
    issues = model.validate()
    if verbose and issues:
        for item in issues:
            print(f"  warning: {item}")

    problem = model.build(mesh_size=mesh_size)
    if verbose:
        print(f"mesh: {problem.continuum.n_elements} elements, "
              f"{problem.mesh.n_nodes} nodes, {problem.dofs.n_dof} degrees of freedom")

    solver = Solver(problem, tolerance=tolerance)
    def progress(i, n, name):
        if verbose:
            print(f"  stage {i + 1}/{n}: {name}", flush=True)
    results = solver.run(progress=progress)

    summary = summarise(problem, results)
    figures = write_figures(problem, results, out_dir, fields)
    write_html(model, problem, results, summary, figures, out_dir)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as fh:
        json.dump(model_to_dict(model), fh, indent=2, default=float)

    if verbose:
        print(f"report written to {os.path.join(out_dir, 'report.html')}")
    return summary


def summarise(problem: FEProblem, results: list[StageResult]) -> dict:
    """Numbers an engineer checks first, for every stage."""
    stages = []
    for res in results:
        entry = {
            "stage": res.name,
            "type": res.kind,
            "converged": bool(res.converged),
            "message": res.message,
            "seconds": round(res.seconds, 2),
            "max_displacement_mm": round(1000.0 * res.max_displacement, 2),
            "plastic_point_fraction": round(float(res.plastic_fraction), 3),
        }
        if res.srf is not None:
            entry["factor_of_safety"] = round(res.srf, 3)
            entry["srf_curve"] = [[round(a, 3), round(1000 * b, 3)] for a, b in res.srf_curve]
        structures = {}
        for s in problem.structures:
            if s.name not in res.active_structures:
                continue
            forces = R.structure_forces(problem, res, s.name)
            item = {
                "axial_force_max_kN_per_m": round(forces["N_max"], 1),
                "shear_force_max_kN_per_m": round(forces["V_max"], 1),
                "bending_moment_max_kNm_per_m": round(forces["M_max"], 1),
            }
            if forces.get("M_capacity"):
                item["moment_capacity_kNm_per_m"] = round(forces["M_capacity"], 1)
                item["moment_utilisation"] = round(forces["M_max"] / forces["M_capacity"], 3)
            xy = problem.mesh.nodes[s.chain]
            u = res.displacement[: 2 * problem.mesh.n_nodes].reshape(-1, 2)[s.chain]
            item["max_deflection_mm"] = round(1000.0 * float(np.max(np.abs(u[:, 0]))), 2)
            item["head_deflection_mm"] = round(
                1000.0 * float(u[int(np.argmax(xy[:, 1])), 0]), 2)
            structures[s.name] = item
        if structures:
            entry["structures"] = structures
        if res.anchor_forces:
            entry["anchor_forces_kN"] = {k: round(v, 1) for k, v in res.anchor_forces.items()
                                         if k in res.active_anchors}
        stages.append(entry)

    return {
        "model": problem.model.name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "elements": int(problem.continuum.n_elements),
        "nodes": int(problem.mesh.n_nodes),
        "degrees_of_freedom": int(problem.dofs.n_dof),
        "stages": stages,
    }


def write_figures(problem: FEProblem, results: list[StageResult], out_dir: str,
                  fields=DEFAULT_FIELDS) -> list[tuple[str, str]]:
    """Render every figure and return ``(caption, path)`` pairs."""
    figures: list[tuple[str, str]] = []

    def save(fig, name, caption):
        path = os.path.join(out_dir, name)
        plots.save(fig, path)
        figures.append((caption, path))

    save(plots.plot_mesh(problem), "mesh.png", "Finite element mesh and boundary conditions")

    for i, res in enumerate(results):
        tag = f"stage{i + 1}"
        if res.kind == "initial":
            continue
        for fname in fields:
            try:
                fig = plots.plot_field(problem, res, fname,
                                       deformed=(fname == "u_total"))
            except (KeyError, ValueError):
                continue
            save(fig, f"{tag}_{fname}.png", f"{res.name}: {R.FIELDS.get(fname, fname)}")
        save(plots.plot_plastic_points(problem, res), f"{tag}_plastic.png",
             f"{res.name}: plastic points")
        if res.srf is not None:
            save(plots.plot_ssr_curve(res), f"{tag}_ssr.png",
                 f"{res.name}: strength reduction curve")
        for s in problem.structures:
            if s.name in res.active_structures:
                save(plots.plot_structure_forces(problem, res, s.name),
                     f"{tag}_{_slug(s.name)}.png", f"{res.name}: {s.name} section forces")
    return figures


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def _embed(path: str) -> str:
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")


def write_html(model: Model, problem: FEProblem, results, summary: dict,
               figures, out_dir: str) -> str:
    """Write a self-contained HTML report."""
    rows = []
    for entry in summary["stages"]:
        fos = entry.get("factor_of_safety")
        rows.append(
            "<tr>"
            f"<td>{html.escape(entry['stage'])}</td>"
            f"<td>{entry['type']}</td>"
            f"<td class='{'ok' if entry['converged'] else 'bad'}'>"
            f"{'converged' if entry['converged'] else 'not converged'}</td>"
            f"<td class='num'>{entry['max_displacement_mm']:.1f}</td>"
            f"<td class='num'>{100 * entry['plastic_point_fraction']:.0f}%</td>"
            f"<td class='num'>{'' if fos is None else f'{fos:.3f}'}</td>"
            f"<td class='num'>{entry['seconds']:.1f}</td>"
            "</tr>"
        )

    structure_rows = []
    for entry in summary["stages"]:
        for name, item in (entry.get("structures") or {}).items():
            structure_rows.append(
                "<tr>"
                f"<td>{html.escape(entry['stage'])}</td><td>{html.escape(name)}</td>"
                f"<td class='num'>{item['axial_force_max_kN_per_m']:.0f}</td>"
                f"<td class='num'>{item['shear_force_max_kN_per_m']:.0f}</td>"
                f"<td class='num'>{item['bending_moment_max_kNm_per_m']:.0f}</td>"
                f"<td class='num'>{item.get('moment_utilisation', '')}</td>"
                f"<td class='num'>{item['max_deflection_mm']:.1f}</td>"
                "</tr>"
            )

    anchor_rows = []
    for entry in summary["stages"]:
        for name, value in (entry.get("anchor_forces_kN") or {}).items():
            anchor_rows.append(f"<tr><td>{html.escape(entry['stage'])}</td>"
                               f"<td>{html.escape(name)}</td>"
                               f"<td class='num'>{value:.0f}</td></tr>")

    material_rows = []
    for lay in model.layers:
        mat = lay.material
        material_rows.append(
            "<tr>"
            f"<td>{html.escape(lay.name)}</td>"
            f"<td>{type(mat).__name__}</td>"
            f"<td class='num'>{mat.E:,.0f}</td><td class='num'>{mat.nu:.2f}</td>"
            f"<td class='num'>{getattr(mat, 'c', 0):.1f}</td>"
            f"<td class='num'>{getattr(mat, 'phi', 0):.1f}</td>"
            f"<td class='num'>{getattr(mat, 'psi', 0):.1f}</td>"
            f"<td class='num'>{mat.gamma:.1f}</td>"
            "</tr>"
        )

    structure_props = []
    for s in model.structures:
        described = s.section.describe()
        structure_props.append(
            f"<tr><td>{html.escape(s.name)}</td>"
            f"<td>{html.escape(str(described.get('name', '')))}</td>"
            f"<td class='num'>{described.get('EA_kN_per_m', 0):,.0f}</td>"
            f"<td class='num'>{described.get('EI_kNm2_per_m', 0):,.0f}</td>"
            f"<td class='num'>{described.get('Mp_kNm_per_m', 0):,.0f}</td></tr>"
        )

    images = "\n".join(
        f"<figure><img src='{_embed(path)}' alt='{html.escape(caption)}'>"
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        for caption, path in figures
    )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(model.name)} - Lythos analysis report</title>
<style>
 :root {{ --ink:#1b1f24; --muted:#5b6672; --line:#d8dee6; --accent:#1565c0; --bg:#ffffff; }}
 body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
        margin:0 auto; max-width:1100px; padding:32px 16px; color:var(--ink); background:var(--bg); }}
 h1 {{ font-size:1.7rem; margin-bottom:4px; }}
 h2 {{ font-size:1.15rem; margin-top:36px; border-bottom:2px solid var(--line); padding-bottom:6px; }}
 .sub {{ color:var(--muted); margin-top:0; }}
 table {{ border-collapse:collapse; width:100%; margin:12px 0 24px; font-size:0.9rem; }}
 th,td {{ border-bottom:1px solid var(--line); padding:7px 10px; text-align:left; }}
 th {{ background:#f4f6f9; font-weight:600; }}
 td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .ok {{ color:#2e7d32; }} .bad {{ color:#c62828; font-weight:600; }}
 figure {{ margin:24px 0; }}
 img {{ max-width:100%; border:1px solid var(--line); border-radius:6px; }}
 figcaption {{ color:var(--muted); font-size:0.85rem; margin-top:6px; }}
 .cards {{ display:flex; gap:16px; flex-wrap:wrap; margin:16px 0; }}
 .card {{ border:1px solid var(--line); border-radius:8px; padding:12px 16px; min-width:150px; }}
 .card b {{ display:block; font-size:1.5rem; color:var(--accent); }}
 .card span {{ color:var(--muted); font-size:0.8rem; }}
 @media print {{ figure {{ break-inside:avoid; }} }}
</style></head><body>
<h1>{html.escape(model.name)}</h1>
<p class="sub">Lythos 2D geotechnical finite element analysis &middot; {summary['generated']}</p>
<div class="cards">
  <div class="card"><b>{summary['elements']}</b><span>elements</span></div>
  <div class="card"><b>{summary['nodes']}</b><span>nodes</span></div>
  <div class="card"><b>{summary['degrees_of_freedom']}</b><span>degrees of freedom</span></div>
  {_fos_card(summary)}
</div>

<h2>Stages</h2>
<table><thead><tr><th>Stage</th><th>Type</th><th>Result</th><th>Max displacement [mm]</th>
<th>Plastic points</th><th>Factor of safety</th><th>Time [s]</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>

<h2>Soil properties</h2>
<table><thead><tr><th>Layer</th><th>Model</th><th>E [kPa]</th><th>&nu;</th><th>c&prime; [kPa]</th>
<th>&phi;&prime; [&deg;]</th><th>&psi; [&deg;]</th><th>&gamma; [kN/m&sup3;]</th></tr></thead>
<tbody>{''.join(material_rows)}</tbody></table>

{_section('Structural sections', '<table><thead><tr><th>Structure</th><th>Section</th>'
          '<th>EA [kN/m]</th><th>EI [kNm&sup2;/m]</th><th>M<sub>p</sub> [kNm/m]</th></tr></thead>'
          '<tbody>' + ''.join(structure_props) + '</tbody></table>', structure_props)}

{_section('Structural forces', '<table><thead><tr><th>Stage</th><th>Structure</th>'
          '<th>N<sub>max</sub> [kN/m]</th><th>V<sub>max</sub> [kN/m]</th>'
          '<th>M<sub>max</sub> [kNm/m]</th><th>M utilisation</th>'
          '<th>Max deflection [mm]</th></tr></thead><tbody>'
          + ''.join(structure_rows) + '</tbody></table>', structure_rows)}

{_section('Anchor forces', '<table><thead><tr><th>Stage</th><th>Anchor</th>'
          '<th>Force [kN]</th></tr></thead><tbody>' + ''.join(anchor_rows)
          + '</tbody></table>', anchor_rows)}

<h2>Figures</h2>
{images}

<h2>Notes</h2>
<ul>
 <li>Stresses are effective stresses; pore pressure is hydrostatic below the phreatic
     surface and the horizontal gradient of the pressure is carried as a seepage force.</li>
 <li>Compression is negative in the stress output and positive in p&prime;.</li>
 <li>Structural forces are per metre run of wall; divide by the spacing given in the
     section table to get the force in one pile.</li>
 <li>The factor of safety comes from strength reduction: c&prime; and tan&phi;&prime; are divided
     by a trial factor until no equilibrium state exists.</li>
</ul>
</body></html>
"""
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


def _section(title: str, body: str, items) -> str:
    return f"<h2>{title}</h2>{body}" if items else ""


def _fos_card(summary: dict) -> str:
    values = [s.get("factor_of_safety") for s in summary["stages"]
              if s.get("factor_of_safety") is not None]
    if not values:
        return ""
    return (f"<div class='card'><b>{min(values):.2f}</b>"
            "<span>lowest factor of safety</span></div>")
