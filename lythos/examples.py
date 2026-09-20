"""Worked example models.

These double as the starting points offered by the interface and as
integration tests, so every one of them is a model that runs.
"""

from __future__ import annotations

import os

from .core.elements import AnchorProperties, InterfaceProperties
from .core.materials import MohrCoulomb
from .core.model import (Anchor, LineLoad, Model, SoilLayer, Stage, Structure,
                         WaterTable)
from .core.pile import PileSection

WALL_X = 20.0
RIGHT = 60.0
from .core.serialize import save_model


def slope() -> Model:
    """A 2:1 cut slope in a single soil, analysed for its factor of safety.

    The geometry and parameters are the classic strength-reduction benchmark:
    10 m high, 1 vertical to 2 horizontal, c' = 10 kPa, phi' = 20 degrees,
    unit weight 20 kN/m3.  Limit equilibrium puts the factor of safety at
    about 1.38, and the strength reduction analysis should land just below it.
    """
    soil = MohrCoulomb(name="silty clay", E=1.0e5, nu=0.3, c=10.0, phi=20.0, psi=0.0,
                       gamma=20.0, gamma_sat=21.0, color="#b9a878")
    outline = [(5, 0), (40, 0), (40, 10), (25, 10)]
    return Model(
        name="cut slope factor of safety",
        layers=[SoilLayer("silty clay", outline, soil, mesh_size=2.0)],
        stages=[
            Stage("1 - self weight", kind="initial", increments=10),
            Stage("2 - factor of safety", kind="ssr", increments=10,
                  srf_min=0.8, srf_max=2.5),
        ],
        mesh_size=2.0,
        initial_stress="gravity",
    )


def embankment() -> Model:
    """A road embankment raised in three lifts over soft clay.

    Staged filling on a soft foundation is the case where the construction
    sequence matters most.  There is no consolidation here, so every lift is
    carried by the undrained strength the clay already has - the conservative
    reading, and the one that decides whether the next lift can go on.

    Six metres of fill at 20 kN/m3 puts about 120 kPa on the clay against an
    undrained bearing capacity of roughly (2 + pi) su = 165 kPa, so the
    embankment stands at full height with something in hand.  Drop su much
    below 25 kPa and it will not: the analysis then reports that no
    equilibrium exists for the last lift, which is the right answer.
    """
    soft = MohrCoulomb(name="soft clay", E=8.0e3, nu=0.35, c=32.0, phi=0.0, psi=0.0,
                       gamma=16.5, gamma_sat=17.0, K0=0.9, color="#9aa7a0")
    stiff = MohrCoulomb(name="stiff clay", E=3.0e4, nu=0.3, c=75.0, phi=0.0, psi=0.0,
                        gamma=19.0, gamma_sat=19.5, K0=0.8, color="#7d8a83")
    fill = MohrCoulomb(name="granular fill", E=3.0e4, nu=0.3, c=1.0, phi=34.0, psi=4.0,
                       gamma=20.0, gamma_sat=21.0, color="#d2b48c")

    layers = [
        SoilLayer("stiff clay", [(0, 0), (80, 0), (80, 6), (0, 6)], stiff, mesh_size=4.0),
        SoilLayer("soft clay", [(0, 6), (80, 6), (80, 14), (0, 14)], soft, mesh_size=2.0),
        SoilLayer("lift 1", [(0, 14), (46, 14), (40, 16), (0, 16)], fill, mesh_size=1.5),
        SoilLayer("lift 2", [(0, 16), (40, 16), (34, 18), (0, 18)], fill, mesh_size=1.5),
        SoilLayer("lift 3", [(0, 18), (34, 18), (28, 20), (0, 20)], fill, mesh_size=1.5),
    ]
    names = [lay.name for lay in layers]
    ground = names[:2]
    return Model(
        name="staged embankment on soft clay",
        layers=layers,
        water=WaterTable([(0, 13.0), (80, 13.0)]),
        stages=[
            Stage("1 - initial stresses", kind="initial", active_layers=ground, increments=6),
            Stage("2 - first lift", kind="plastic", active_layers=ground + ["lift 1"],
                  increments=10, reset_displacements=True),
            Stage("3 - second lift", kind="plastic",
                  active_layers=ground + ["lift 1", "lift 2"], increments=10),
            Stage("4 - third lift", kind="plastic", active_layers=names, increments=10),
            Stage("5 - factor of safety", kind="ssr", increments=10, srf_min=0.8, srf_max=3.0),
        ],
        mesh_size=3.0,
        initial_stress="k0",
    )


def excavation() -> Model:
    sand = MohrCoulomb(name="dense sand", E=6.0e4, nu=0.3, c=1.0, phi=36.0, psi=6.0,
                       gamma=19.0, gamma_sat=21.0, color="#e0c068")
    clay = MohrCoulomb(name="stiff clay", E=2.5e4, nu=0.33, c=15.0, phi=24.0, psi=0.0,
                       gamma=19.5, gamma_sat=20.0, color="#a8907a")
    fill = MohrCoulomb(name="sand fill", E=3.0e4, nu=0.3, c=2.0, phi=32.0, psi=2.0,
                       gamma=18.0, gamma_sat=20.0, color="#d8c9a0")

    layers = [
        SoilLayer("dense sand", [(0, 0), (RIGHT, 0), (RIGHT, 8), (0, 8)], sand, mesh_size=3.5),
        SoilLayer("stiff clay", [(0, 8), (RIGHT, 8), (RIGHT, 22), (0, 22)], clay, mesh_size=2.5),
        SoilLayer("retained fill", [(0, 22), (WALL_X, 22), (WALL_X, 30), (0, 30)], fill,
                  mesh_size=1.5),
        SoilLayer("lift 1", [(WALL_X, 27), (RIGHT, 27), (RIGHT, 30), (WALL_X, 30)], fill,
                  mesh_size=1.5),
        SoilLayer("lift 2", [(WALL_X, 24), (RIGHT, 24), (RIGHT, 27), (WALL_X, 27)], fill,
                  mesh_size=1.5),
        SoilLayer("lift 3", [(WALL_X, 22), (RIGHT, 22), (RIGHT, 24), (WALL_X, 24)], fill,
                  mesh_size=1.5),
    ]

    wall = Structure(
        name="secant pile wall",
        path=[(WALL_X, 30.0), (WALL_X, 14.0)],
        section=PileSection(name="D1000 @ 1.2 m", diameter=1.0, spacing=1.2, fck=32.0,
                            rho_s=0.012, stiffness_factor=0.7),
        interface=InterfaceProperties(),
        r_inter=0.67,
    )

    anchors = [
        Anchor("anchor row 1", (WALL_X, 28.5), (WALL_X - 12.0, 21.0),
               AnchorProperties(EA=2.4e5, spacing=2.5, prestress=350.0, Fmax=900.0,
                                compression=False)),
        Anchor("anchor row 2", (WALL_X, 25.5), (WALL_X - 11.0, 18.5),
               AnchorProperties(EA=2.4e5, spacing=2.5, prestress=450.0, Fmax=900.0,
                                compression=False)),
    ]

    surcharge = LineLoad("site surcharge", [(2.0, 30.0), (16.0, 30.0)], qy=-20.0)

    everything = [lay.name for lay in layers]
    stages = [
        Stage("1 - initial stresses", kind="initial", active_layers=everything,
              active_structures=[], active_anchors=[], active_loads=[], increments=6),
        Stage("2 - install wall", kind="plastic", active_structures=["secant pile wall"],
              active_loads=["site surcharge"], increments=6, reset_displacements=True),
        Stage("3 - excavate to 27.0 m", kind="plastic",
              active_layers=[n for n in everything if n not in {"lift 1"}], increments=8),
        Stage("4 - stress anchor row 1, excavate to 24.0 m", kind="plastic",
              active_layers=[n for n in everything if n not in {"lift 1", "lift 2"}],
              active_anchors=["anchor row 1"], increments=10),
        Stage("5 - stress anchor row 2, excavate to 22.0 m", kind="plastic",
              active_layers=[n for n in everything if n not in {"lift 1", "lift 2", "lift 3"}],
              active_anchors=["anchor row 1", "anchor row 2"], increments=10),
        Stage("6 - factor of safety", kind="ssr", increments=8, srf_min=0.8, srf_max=3.0),
    ]

    return Model(
        name="braced deep excavation",
        layers=layers,
        structures=[wall],
        anchors=anchors,
        line_loads=[surcharge],
        water=WaterTable([(0.0, 18.0), (RIGHT, 18.0)]),
        stages=stages,
        mesh_size=2.5,
        initial_stress="k0",
    )


def pile_wall() -> Model:
    """A cantilever contiguous pile wall retaining a 4 m cut.

    The simplest case in which pile bending governs: no props, no anchors, so
    the whole retained load turns into moment in the piles.
    """
    sand = MohrCoulomb(name="medium dense sand", E=3.5e4, nu=0.3, c=1.0, phi=33.0, psi=3.0,
                       gamma=18.5, gamma_sat=20.5, color="#dcc48e")
    clay = MohrCoulomb(name="firm clay", E=2.0e4, nu=0.32, c=25.0, phi=22.0, psi=0.0,
                       gamma=19.0, gamma_sat=19.5, color="#a89880")
    wall_x = 15.0
    layers = [
        SoilLayer("firm clay", [(0, 0), (40, 0), (40, 6), (0, 6)], clay, mesh_size=2.5),
        SoilLayer("sand (retained)", [(0, 6), (wall_x, 6), (wall_x, 14), (0, 14)], sand,
                  mesh_size=1.2),
        SoilLayer("sand (in front)", [(wall_x, 6), (40, 6), (40, 10), (wall_x, 10)], sand,
                  mesh_size=1.2),
        SoilLayer("excavation", [(wall_x, 10), (40, 10), (40, 14), (wall_x, 14)], sand,
                  mesh_size=1.2),
    ]
    names = [lay.name for lay in layers]
    wall = Structure(
        name="contiguous pile wall",
        path=[(wall_x, 14.0), (wall_x, 3.0)],
        section=PileSection(name="D750 @ 0.9 m", diameter=0.75, spacing=0.9, fck=30.0,
                            rho_s=0.01, stiffness_factor=0.7),
        interface=InterfaceProperties(),
        r_inter=0.7,
    )
    return Model(
        name="cantilever pile wall",
        layers=layers,
        structures=[wall],
        line_loads=[LineLoad("surcharge", [(2.0, 14.0), (12.0, 14.0)], qy=-15.0)],
        water=WaterTable([(0, 5.0), (40, 5.0)]),
        stages=[
            Stage("1 - initial stresses", kind="initial", active_layers=names,
                  active_structures=[], active_loads=[], increments=6),
            Stage("2 - install wall", kind="plastic",
                  active_structures=["contiguous pile wall"],
                  active_loads=["surcharge"], increments=6, reset_displacements=True),
            Stage("3 - excavate to 10.0 m", kind="plastic",
                  active_layers=[n for n in names if n != "excavation"], increments=10),
            Stage("4 - factor of safety", kind="ssr", increments=8, srf_min=0.8, srf_max=3.0),
        ],
        mesh_size=2.0,
        initial_stress="k0",
    )


#: every built-in example, by key
EXAMPLES = {
    "slope": slope,
    "embankment": embankment,
    "pile_wall": pile_wall,
    "excavation": excavation,
}

DESCRIPTIONS = {
    "slope": "Cut slope, factor of safety by strength reduction",
    "embankment": "Road embankment raised in three lifts over soft clay",
    "pile_wall": "Cantilever contiguous pile wall retaining a 4 m cut",
    "excavation": "Braced deep excavation with a secant pile wall and ground anchors",
}


def write_examples(out_dir: str = "examples/models") -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for key, factory in EXAMPLES.items():
        path = os.path.join(out_dir, f"{key}.json")
        save_model(factory(), path)
        paths.append(path)
    return paths
