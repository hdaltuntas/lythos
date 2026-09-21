"""Sample drawings shipped with the package.

These are here rather than in the repository's examples folder so that they
are still there after ``pip install``, when the repository is not.
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))

#: a braced excavation: three strata, three lifts, a wall, two anchors and a
#: surcharge, with the construction step in every layer name
BRACED_EXCAVATION = os.path.join(HERE, "braced_excavation.dxf")


def sample_drawing(name: str = "braced_excavation") -> str:
    """Path to a sample DXF drawing."""
    path = os.path.join(HERE, f"{name}.dxf")
    if not os.path.isfile(path):
        available = sorted(f[:-4] for f in os.listdir(HERE) if f.endswith(".dxf"))
        raise FileNotFoundError(f"no sample drawing {name!r}; there is {available}")
    return path
