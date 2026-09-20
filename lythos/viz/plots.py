"""Graphical output: contours, deformed meshes and structural force diagrams.

Everything here renders through Matplotlib's Agg backend so that the same code
serves scripts, the command line and the web interface without needing a
display.
"""

from __future__ import annotations

import io
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation

from ..core import results as R
from ..core.problem import FEProblem
from ..core.solver import StageResult

#: a perceptually even blue-to-red ramp that also reads in greyscale
GEO_CMAP = LinearSegmentedColormap.from_list("lythos", [
    "#2b3a67", "#3d6ea8", "#63b0c4", "#a8d5b5", "#f2e394",
    "#f2a65a", "#e2653f", "#b02e26",
])

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "font.size": 9,
    "axes.titlesize": 10,
}


def _triangulation(problem: FEProblem) -> Triangulation:
    """Split each quadratic triangle into four linear ones for smooth plots."""
    el = problem.mesh.elements
    sub = np.vstack([
        el[:, [0, 3, 5]],
        el[:, [3, 1, 4]],
        el[:, [5, 4, 2]],
        el[:, [3, 4, 5]],
    ])
    nodes = problem.mesh.nodes
    return Triangulation(nodes[:, 0], nodes[:, 1], sub)


def _active_mask(problem: FEProblem, result: StageResult) -> np.ndarray:
    """Sub-triangle mask hiding elements that are switched off in this stage."""
    active = result.active_elements
    return np.tile(~active, 4)


def _setup_axes(ax, problem, title=None):
    x0, y0, x1, y1 = problem.mesh.bounds()
    pad = 0.03 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.15, linewidth=0.5)


def plot_mesh(problem: FEProblem, ax=None, show_layers: bool = True,
              show_structures: bool = True, title: str = "Finite element mesh"):
    """The mesh, coloured by soil layer, with structures and supports marked."""
    with plt.rc_context(PLOT_STYLE):
        if ax is None:
            _fig, ax = plt.subplots(figsize=(9, 5))
        nodes = problem.mesh.nodes
        tri = _triangulation(problem)
        if show_layers:
            colors = [getattr(lay.material, "color", "#c8b273") for lay in problem.model.layers]
            for i, lay in enumerate(problem.model.layers):
                mask = np.tile(problem.layer_of_element != i, 4)
                t = Triangulation(tri.x, tri.y, tri.triangles)
                t.set_mask(mask)
                ax.tripcolor(t, facecolors=np.zeros(len(tri.triangles)),
                             cmap=LinearSegmentedColormap.from_list(
                                 f"l{i}", [colors[i % len(colors)]] * 2),
                             edgecolors="none")
        ax.triplot(tri, color="#37474f", linewidth=0.25, alpha=0.7)
        if show_structures:
            _draw_structures(ax, problem)
        _draw_supports(ax, problem)
        _draw_water(ax, problem, problem.model.water)
        _setup_axes(ax, problem, title)
        _layer_legend(ax, problem)
        return ax.figure


def _draw_structures(ax, problem: FEProblem, linewidth=2.4):
    for s in problem.structures:
        xy = problem.mesh.nodes[s.chain]
        ax.plot(xy[:, 0], xy[:, 1], color="#1a237e", linewidth=linewidth,
                solid_capstyle="round", zorder=6, label="_nolegend_")
    if problem.anchors is not None:
        for (near, far, weights) in problem.anchors.anchors:
            p0 = problem.mesh.nodes[near]
            p1 = weights @ problem.mesh.nodes[far]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#b71c1c",
                    linewidth=1.6, linestyle="--", zorder=6)
            grout = problem.mesh.nodes[far]
            ax.plot(grout[:, 0], grout[:, 1], linestyle="none", marker="o",
                    color="#b71c1c", markersize=2.5, alpha=0.7, zorder=6)


def _draw_supports(ax, problem: FEProblem):
    nodes = problem.mesh.nodes
    fixed = problem.fixed_dofs
    both, hor = [], []
    fixed_set = set(int(d) for d in fixed)
    for n in range(problem.mesh.n_nodes):
        fx, fy = 2 * n in fixed_set, 2 * n + 1 in fixed_set
        if fx and fy:
            both.append(n)
        elif fx:
            hor.append(n)
    if both:
        ax.plot(nodes[both, 0], nodes[both, 1], marker="^", linestyle="none",
                color="#455a64", markersize=3.5, zorder=5)
    if hor:
        ax.plot(nodes[hor, 0], nodes[hor, 1], marker=">", linestyle="none",
                color="#90a4ae", markersize=3, zorder=5)


def _draw_water(ax, problem: FEProblem, water):
    if not water or not water.points:
        return
    pts = sorted(water.points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color="#0277bd", linewidth=1.4, zorder=7)
    ax.plot(xs[:1], ys[:1], marker="v", color="#0277bd", markersize=7, zorder=7)


def _layer_legend(ax, problem: FEProblem):
    handles = []
    for lay in problem.model.layers:
        color = getattr(lay.material, "color", "#c8b273")
        handles.append(plt.Line2D([], [], marker="s", linestyle="none", markersize=8,
                                  markerfacecolor=color, markeredgecolor="#555555",
                                  label=lay.name))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)


def plot_field(problem: FEProblem, result: StageResult, field: str = "u_total",
               ax=None, deformed: bool = False, scale: float | None = None,
               levels: int = 18, title: str | None = None, show_mesh: bool = False):
    """Filled contours of a result field over the active part of the mesh."""
    values, label = R.field(problem, result, field)
    with plt.rc_context(PLOT_STYLE):
        if ax is None:
            _fig, ax = plt.subplots(figsize=(9.5, 5.2))
        nodes = problem.mesh.nodes.copy()
        if deformed:
            u = result.displacement[: 2 * problem.mesh.n_nodes].reshape(-1, 2)
            if scale is None:
                scale = _auto_scale(problem, u)
            nodes = nodes + scale * u
        tri = Triangulation(nodes[:, 0], nodes[:, 1], _triangulation(problem).triangles)
        tri.set_mask(_active_mask(problem, result))

        finite = values[np.isfinite(values)]
        if len(finite) == 0 or np.allclose(finite, finite[0]):
            contour = ax.tripcolor(tri, values, cmap=GEO_CMAP, shading="gouraud")
        else:
            contour = ax.tricontourf(tri, values, levels=levels, cmap=GEO_CMAP)
        if show_mesh:
            ax.triplot(tri, color="#37474f", linewidth=0.2, alpha=0.4)
        cbar = ax.figure.colorbar(contour, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(label)
        _draw_structures(ax, problem)
        _draw_water(ax, problem, problem.model.water)
        head = title or f"{result.name} - {label}"
        if deformed:
            head += f"  (deformation x{scale:.0f})"
        _setup_axes(ax, problem, head)
        return ax.figure


def _auto_scale(problem: FEProblem, u: np.ndarray) -> float:
    x0, y0, x1, y1 = problem.mesh.bounds()
    peak = float(np.max(np.hypot(u[:, 0], u[:, 1]))) if len(u) else 0.0
    if peak <= 0:
        return 1.0
    target = 0.06 * max(x1 - x0, y1 - y0)
    return max(1.0, round(target / peak))


def plot_plastic_points(problem: FEProblem, result: StageResult, ax=None,
                        title: str | None = None):
    """Where the soil is at failure - the shape of the developing mechanism."""
    with plt.rc_context(PLOT_STYLE):
        if ax is None:
            _fig, ax = plt.subplots(figsize=(9, 5))
        tri = _triangulation(problem)
        tri.set_mask(_active_mask(problem, result))
        ax.triplot(tri, color="#cfd8dc", linewidth=0.3)
        gauss = problem.continuum.gauss_xy.reshape(-1, 2)
        yielding = result.state.yielding
        active_gp = np.repeat(result.active_elements, problem.continuum.n_gauss)
        show = yielding & active_gp
        if show.any():
            ax.plot(gauss[show, 0], gauss[show, 1], linestyle="none", marker="s",
                    markersize=2.2, color="#c62828", label="at failure")
        rest = ~yielding & active_gp
        if rest.any():
            ax.plot(gauss[rest, 0], gauss[rest, 1], linestyle="none", marker=".",
                    markersize=1.0, color="#b0bec5", label="elastic")
        _draw_structures(ax, problem)
        ax.legend(loc="upper right", fontsize=8)
        fraction = 100.0 * show.sum() / max(active_gp.sum(), 1)
        _setup_axes(ax, problem,
                    title or f"{result.name} - plastic points ({fraction:.0f}% at failure)")
        return ax.figure


def plot_deformed_mesh(problem: FEProblem, result: StageResult, scale: float | None = None,
                       ax=None, title: str | None = None):
    with plt.rc_context(PLOT_STYLE):
        if ax is None:
            _fig, ax = plt.subplots(figsize=(9, 5))
        u = result.displacement[: 2 * problem.mesh.n_nodes].reshape(-1, 2)
        if scale is None:
            scale = _auto_scale(problem, u)
        base = _triangulation(problem)
        base.set_mask(_active_mask(problem, result))
        ax.triplot(base, color="#cfd8dc", linewidth=0.3)
        moved = problem.mesh.nodes + scale * u
        tri = Triangulation(moved[:, 0], moved[:, 1], base.triangles)
        tri.set_mask(_active_mask(problem, result))
        ax.triplot(tri, color="#37474f", linewidth=0.35)
        _setup_axes(ax, problem, title or f"{result.name} - deformed mesh (x{scale:.0f})")
        return ax.figure


def plot_displacement_vectors(problem: FEProblem, result: StageResult, ax=None,
                              every: int = 3, title: str | None = None):
    with plt.rc_context(PLOT_STYLE):
        if ax is None:
            _fig, ax = plt.subplots(figsize=(9, 5))
        tri = _triangulation(problem)
        tri.set_mask(_active_mask(problem, result))
        ax.triplot(tri, color="#eceff1", linewidth=0.3)
        u = result.displacement[: 2 * problem.mesh.n_nodes].reshape(-1, 2)
        nodes = problem.mesh.nodes
        pick = slice(None, None, max(1, every))
        mag = np.hypot(u[:, 0], u[:, 1])
        ax.quiver(nodes[pick, 0], nodes[pick, 1], u[pick, 0], u[pick, 1], mag[pick],
                  cmap=GEO_CMAP, angles="xy", width=0.0025)
        _draw_structures(ax, problem)
        _setup_axes(ax, problem, title or f"{result.name} - displacement directions")
        return ax.figure


def plot_structure_forces(problem: FEProblem, result: StageResult, name: str,
                          fig=None):
    """Axial force, shear and bending moment along a wall or pile row."""
    data = R.structure_forces(problem, result, name)
    with plt.rc_context(PLOT_STYLE):
        if fig is None:
            fig, axes = plt.subplots(1, 3, figsize=(10, 5.5), sharey=True)
        else:
            axes = fig.subplots(1, 3, sharey=True)
        if len(data["s"]) == 0:
            for ax in axes:
                ax.text(0.5, 0.5, "no results", ha="center", va="center")
            return fig

        depth = data["y"]
        series = (("N", "axial force [kN/m]", "#1565c0"),
                  ("V", "shear force [kN/m]", "#2e7d32"),
                  ("M", "bending moment [kNm/m]", "#c62828"))
        for ax, (key, label, color) in zip(axes, series):
            ax.plot(data[key], depth, color=color, linewidth=1.8)
            ax.fill_betweenx(depth, 0, data[key], color=color, alpha=0.18)
            ax.axvline(0, color="#90a4ae", linewidth=0.8)
            ax.set_xlabel(label)
            ax.grid(True, alpha=0.2)
            peak = np.argmax(np.abs(data[key]))
            ax.annotate(f"{data[key][peak]:.0f}", (data[key][peak], depth[peak]),
                        textcoords="offset points", xytext=(6, 0), fontsize=8, color=color)
        if data.get("M_capacity"):
            axes[2].axvline(data["M_capacity"], color="#c62828", linestyle=":", linewidth=1)
            axes[2].axvline(-data["M_capacity"], color="#c62828", linestyle=":", linewidth=1)
        axes[0].set_ylabel("elevation y [m]")
        fig.suptitle(f"{name} - section forces ({result.name})")
        fig.tight_layout()
        return fig


def plot_ssr_curve(result: StageResult, ax=None):
    """Displacement against strength reduction factor: the failure knee."""
    with plt.rc_context(PLOT_STYLE):
        if ax is None:
            _fig, ax = plt.subplots(figsize=(6, 4))
        if result.srf_curve:
            srf = [a for a, _b in result.srf_curve]
            disp = [1000.0 * b for _a, b in result.srf_curve]
            ax.plot(srf, disp, marker="o", markersize=4, color="#1565c0")
        if result.srf:
            ax.axvline(result.srf, color="#c62828", linestyle="--", linewidth=1.2)
            ax.annotate(f"FoS = {result.srf:.2f}", (result.srf, ax.get_ylim()[1]),
                        textcoords="offset points", xytext=(6, -14),
                        color="#c62828", fontsize=10, fontweight="bold")
        ax.set_xlabel("strength reduction factor")
        ax.set_ylabel("maximum displacement [mm]")
        ax.set_title("Strength reduction")
        ax.grid(True, alpha=0.25)
        return ax.figure


def plot_model(model, ax=None, title: str = "Model geometry"):
    """Draw the input geometry before any mesh exists."""
    with plt.rc_context(PLOT_STYLE):
        if ax is None:
            _fig, ax = plt.subplots(figsize=(9, 5))
        for lay in model.layers:
            poly = np.array(lay.normalised() + [lay.normalised()[0]])
            color = getattr(lay.material, "color", "#c8b273")
            ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.85, zorder=1)
            ax.plot(poly[:, 0], poly[:, 1], color="#5d4037", linewidth=1.0, zorder=2)
        for s in model.structures:
            path = np.array(s.path)
            ax.plot(path[:, 0], path[:, 1], color="#1a237e", linewidth=2.6, zorder=4)
        for a in model.anchors:
            ax.plot([a.start[0], a.end[0]], [a.start[1], a.end[1]],
                    color="#b71c1c", linewidth=1.6, linestyle="--", zorder=4)
        for load in model.line_loads:
            path = np.array(load.path)
            ax.plot(path[:, 0], path[:, 1], color="#ef6c00", linewidth=3.0, zorder=4)
        if model.water.points:
            pts = np.array(sorted(model.water.points))
            ax.plot(pts[:, 0], pts[:, 1], color="#0277bd", linewidth=1.4, zorder=5)
        x0, y0, x1, y1 = model.bounds()
        pad = 0.05 * max(x1 - x0, y1 - y0)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title)
        ax.grid(True, alpha=0.15)
        return ax.figure


def figure_to_png(fig, dpi: int = 110) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def save(fig, path: str, dpi: int = 130) -> str:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
