# Changelog

What changed in each released version.  The version numbers are the ones on
[PyPI](https://pypi.org/project/lythosfea/); each has a matching tag and a
[release](https://github.com/hdaltuntas/lythos/releases) here, with the built
files attached.

## 0.1.0 - 21 September 2026

First release.  2D plane-strain finite element analysis for geotechnical
engineering: slopes, embankments and deep excavations.

- Factors of safety by shear strength reduction
- Mohr-Coulomb with non-associated flow and a tension cut-off, returned
  exactly in principal stress space
- 6-node quadratic triangles with Delaunay refinement meshing
- Timoshenko beams, interfaces and anchors for walls, piles and props
- Staged construction, with the sequence readable straight from a DXF drawing
- A browser interface, a command line and a Python API

Verified against an independently written Bishop search: the cut slope
converges to 1.381 under mesh refinement, against 1.377.
[`docs/validation.md`](docs/validation.md) records where the agreement holds
and where it does not - coarse meshes are unconservative, undrained analysis
at phi = 0 suffers volumetric locking, and the embankment example's 1.61 is
not independently validated.

[On PyPI](https://pypi.org/project/lythosfea/0.1.0/) ·
[release](https://github.com/hdaltuntas/lythos/releases/tag/0.1.0)
