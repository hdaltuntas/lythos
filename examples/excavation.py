"""Braced deep excavation behind a secant pile wall.

Three soil layers, a 16 m secant pile wall of 1.0 m piles at 1.2 m centres,
two rows of pre-stressed ground anchors and a three-lift excavation to 8 m
depth.  The analysis reports wall deflection, bending moment and axial force
in the piles, anchor forces, and the factor of safety of the final excavation.
"""

from lythos.core.elements import AnchorProperties, InterfaceProperties
from lythos.core.materials import MohrCoulomb
from lythos.core.model import (Anchor, LineLoad, Model, SoilLayer, Stage, Structure,
                               WaterTable)
from lythos.core.pile import PileSection

WALL_X = 20.0
RIGHT = 60.0


def build() -> Model:
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


if __name__ == "__main__":
    import sys
    from lythos.report import run_and_report
    run_and_report(build(), out_dir=sys.argv[1] if len(sys.argv) > 1 else "out/excavation")
