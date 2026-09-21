"""Cross-section properties for piles, walls and other structural members.

Engineers specify a pile by diameter, concrete grade and spacing, not by EA and
EI.  These helpers do that conversion once, so the same numbers appear in the
input echo, in the analysis and in the reported section forces.

A plane-strain analysis models a row of piles as an equivalent continuous
plate, so all rigidities are divided by the out-of-plane spacing and carry
units "per metre run of wall".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .elements import BeamSection
from .materials import concrete_modulus

#: shear area factor for a solid circular section
KAPPA_CIRCLE = 0.9
#: shear area factor for a rectangular section
KAPPA_RECTANGLE = 5.0 / 6.0


@dataclass
class PileSection:
    """A bored or driven pile, analysed as a row at a given spacing."""

    name: str = "pile"
    diameter: float = 0.8        # [m]
    spacing: float = 2.0         # centre-to-centre, out of plane [m]
    fck: float = 30.0            # characteristic cylinder strength [MPa]
    E: float | None = None       # override the concrete modulus [kPa]
    nu: float = 0.2
    gamma: float = 25.0          # reinforced concrete unit weight [kN/m3]
    #: longitudinal reinforcement ratio, used for the plastic moment estimate
    rho_s: float = 0.01
    fyk: float = 500.0           # reinforcement yield strength [MPa]
    stiffness_factor: float = 1.0  # e.g. 0.7 for cracked-section bending

    # ------------------------------------------------------------------ shape
    @property
    def area(self) -> float:
        return math.pi * self.diameter ** 2 / 4.0

    @property
    def inertia(self) -> float:
        return math.pi * self.diameter ** 4 / 64.0

    @property
    def modulus(self) -> float:
        return self.E if self.E else concrete_modulus(self.fck)

    # ------------------------------------------------------------- rigidities
    def section(self) -> BeamSection:
        """Equivalent plate rigidities per metre run of wall."""
        s = max(self.spacing, 1e-6)
        E = self.modulus
        G = E / (2.0 * (1.0 + self.nu))
        return BeamSection(
            EA=E * self.area / s,
            EI=self.stiffness_factor * E * self.inertia / s,
            GA=G * KAPPA_CIRCLE * self.area / s,
            weight=self.gamma * self.area / s,
            Mp=self.plastic_moment() / s,
            Np=0.0,
        )

    def plastic_moment(self) -> float:
        """Approximate ultimate moment of a circular reinforced section [kNm].

        The steel is taken as a thin ring at 0.4 D lever arm, which is the
        usual first-order estimate for a bored pile cage; supply a measured
        capacity instead where one is available.
        """
        as_total = self.rho_s * self.area                  # [m2]
        fyd = self.fyk * 1.0e3                             # MPa -> kPa
        return 0.4 * self.diameter * as_total * fyd * 0.8

    def describe(self) -> dict:
        sec = self.section()
        return {
            "name": self.name,
            "diameter_m": self.diameter,
            "spacing_m": self.spacing,
            "fck_MPa": self.fck,
            "E_kPa": self.modulus,
            "area_m2": self.area,
            "inertia_m4": self.inertia,
            "EA_kN_per_m": sec.EA,
            "EI_kNm2_per_m": sec.EI,
            "weight_kN_per_m2": sec.weight,
            "Mp_kNm_per_m": sec.Mp,
        }


@dataclass
class WallSection:
    """A continuous wall: diaphragm wall, sheet pile or shotcrete lining."""

    name: str = "wall"
    thickness: float = 0.6       # [m]
    fck: float = 30.0
    E: float | None = None
    nu: float = 0.2
    gamma: float = 25.0
    stiffness_factor: float = 1.0
    #: for sheet piles, give EA/EI directly instead of a thickness
    EA_override: float | None = None
    EI_override: float | None = None
    Mp: float = 0.0

    @property
    def modulus(self) -> float:
        return self.E if self.E else concrete_modulus(self.fck)

    def section(self) -> BeamSection:
        E = self.modulus
        G = E / (2.0 * (1.0 + self.nu))
        t = self.thickness
        EA = self.EA_override if self.EA_override else E * t
        EI = self.EI_override if self.EI_override else self.stiffness_factor * E * t ** 3 / 12.0
        return BeamSection(EA=EA, EI=EI, GA=G * KAPPA_RECTANGLE * t,
                           weight=self.gamma * t, Mp=self.Mp)

    def describe(self) -> dict:
        sec = self.section()
        return {"name": self.name, "thickness_m": self.thickness, "fck_MPa": self.fck,
                "E_kPa": self.modulus, "EA_kN_per_m": sec.EA, "EI_kNm2_per_m": sec.EI,
                "weight_kN_per_m2": sec.weight}


def equivalent_plate_from_piles(diameter: float, spacing: float, fck: float = 30.0) -> BeamSection:
    """Shorthand for the rigidities of a contiguous or secant pile wall."""
    return PileSection(diameter=diameter, spacing=spacing, fck=fck).section()
