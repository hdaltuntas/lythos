"""Braced deep excavation behind a secant pile wall.

Runs the worked example from :mod:`lythos.examples` and writes a full report.

    python examples/excavation.py out/excavation
"""

import sys

from lythos.examples import excavation
from lythos.report import run_and_report

#: kept so that existing scripts calling ``build()`` still work
build = excavation

if __name__ == "__main__":
    run_and_report(excavation(),
                   out_dir=sys.argv[1] if len(sys.argv) > 1 else "out/excavation")
