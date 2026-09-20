"""Factor of safety of a 2:1 cut slope by strength reduction.

    python examples/slope.py out/slope
"""

import sys

from lythos.examples import slope
from lythos.report import run_and_report

if __name__ == "__main__":
    summary = run_and_report(slope(), out_dir=sys.argv[1] if len(sys.argv) > 1 else "out/slope")
    fos = summary["stages"][-1]["factor_of_safety"]
    print(f"\nfactor of safety: {fos:.3f}")
