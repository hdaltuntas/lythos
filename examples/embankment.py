"""Road embankment raised in three lifts over soft clay.

    python examples/embankment.py out/embankment
"""

import sys

from lythos.examples import embankment
from lythos.report import run_and_report

if __name__ == "__main__":
    run_and_report(embankment(),
                   out_dir=sys.argv[1] if len(sys.argv) > 1 else "out/embankment")
