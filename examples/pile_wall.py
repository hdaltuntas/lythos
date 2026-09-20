"""Cantilever contiguous pile wall retaining a 4 m cut.

Reports the bending moment and shear in the piles, and the factor of safety.

    python examples/pile_wall.py out/pile_wall
"""

import sys

from lythos.examples import pile_wall
from lythos.report import run_and_report

if __name__ == "__main__":
    summary = run_and_report(pile_wall(),
                             out_dir=sys.argv[1] if len(sys.argv) > 1 else "out/pile_wall")
    for stage in summary["stages"]:
        for name, item in (stage.get("structures") or {}).items():
            print(f"{stage['stage']:38s} {name}: "
                  f"M = {item['bending_moment_max_kNm_per_m']:.0f} kNm/m, "
                  f"deflection = {item['max_deflection_mm']:.1f} mm")
