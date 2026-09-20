"""Reading and writing models as JSON.

The interface, the command line and the test suite all exchange models in this
format, so it is the one definition of what a Lythos model file contains.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass

from .elements import AnchorProperties, InterfaceProperties
from .materials import Concrete, LinearElastic, Material, MohrCoulomb
from .model import (Anchor, BoundaryConditions, LineLoad, Model, PointLoad, SoilLayer,
                    Stage, Structure, WaterTable)
from .pile import PileSection, WallSection

MATERIAL_TYPES = {
    "MohrCoulomb": MohrCoulomb,
    "LinearElastic": LinearElastic,
    "Concrete": Concrete,
}
SECTION_TYPES = {"PileSection": PileSection, "WallSection": WallSection}


def _clean(obj):
    if is_dataclass(obj):
        return {f.name: _clean(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    return obj


def _build(cls, data: dict):
    """Instantiate a dataclass, ignoring keys it does not define."""
    if data is None:
        return None
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        data = {k: v for k, v in data.items() if k in allowed}
    return cls(**data)


def material_to_dict(mat: Material) -> dict:
    out = _clean(mat)
    out["type"] = type(mat).__name__
    return out


def material_from_dict(data: dict) -> Material:
    kind = data.get("type", "MohrCoulomb")
    cls = MATERIAL_TYPES.get(kind, MohrCoulomb)
    return _build(cls, {k: v for k, v in data.items() if k != "type"})


def section_to_dict(section) -> dict:
    out = _clean(section)
    out["type"] = type(section).__name__
    return out


def section_from_dict(data: dict):
    cls = SECTION_TYPES.get(data.get("type", "WallSection"), WallSection)
    return _build(cls, {k: v for k, v in data.items() if k != "type"})


def model_to_dict(model: Model) -> dict:
    return {
        "format": "lythos-model",
        "version": 1,
        "name": model.name,
        "mesh_size": model.mesh_size,
        "min_angle": model.min_angle,
        "initial_stress": model.initial_stress,
        "gamma_water": model.gamma_water,
        "layers": [
            {"name": lay.name,
             "polygon": [list(p) for p in lay.polygon],
             "mesh_size": lay.mesh_size,
             "material": material_to_dict(lay.material)}
            for lay in model.layers
        ],
        "structures": [
            {"name": s.name,
             "path": [list(p) for p in s.path],
             "r_inter": s.r_inter,
             "section": section_to_dict(s.section),
             "interface": _clean(s.interface) if s.interface else None}
            for s in model.structures
        ],
        "anchors": [
            {"name": a.name, "start": list(a.start), "end": list(a.end),
             "properties": _clean(a.properties)}
            for a in model.anchors
        ],
        "line_loads": [_clean(load) for load in model.line_loads],
        "point_loads": [_clean(load) for load in model.point_loads],
        "water": {"points": [list(p) for p in model.water.points]},
        "boundary": _clean(model.boundary),
        "stages": [_clean(st) for st in model.stages],
    }


def model_from_dict(data: dict) -> Model:
    layers = [
        SoilLayer(name=lay["name"],
                  polygon=[tuple(p) for p in lay["polygon"]],
                  material=material_from_dict(lay.get("material", {})),
                  mesh_size=lay.get("mesh_size"))
        for lay in data.get("layers", [])
    ]
    structures = []
    for s in data.get("structures", []):
        interface = _build(InterfaceProperties, s["interface"]) if s.get("interface") else None
        structures.append(Structure(
            name=s["name"],
            path=[tuple(p) for p in s["path"]],
            section=section_from_dict(s.get("section", {})),
            interface=interface,
            r_inter=s.get("r_inter", 0.67),
        ))
    anchors = [
        Anchor(name=a["name"], start=tuple(a["start"]), end=tuple(a["end"]),
               properties=_build(AnchorProperties, a.get("properties", {})) or AnchorProperties())
        for a in data.get("anchors", [])
    ]
    stages = []
    for st in data.get("stages", []):
        st = dict(st)
        water = st.pop("water", None)
        stage = _build(Stage, st)
        if isinstance(water, dict):
            stage.water = WaterTable([tuple(p) for p in water.get("points", [])])
        stages.append(stage)
    model = Model(
        name=data.get("name", "model"),
        layers=layers,
        structures=structures,
        anchors=anchors,
        line_loads=[_build(LineLoad, dict(d, path=[tuple(p) for p in d["path"]]))
                    for d in data.get("line_loads", [])],
        point_loads=[_build(PointLoad, d) for d in data.get("point_loads", [])],
        water=WaterTable([tuple(p) for p in data.get("water", {}).get("points", [])]),
        stages=stages,
        boundary=_build(BoundaryConditions, data.get("boundary", {})) or BoundaryConditions(),
        mesh_size=data.get("mesh_size"),
        min_angle=data.get("min_angle", 25.0),
        initial_stress=data.get("initial_stress", "k0"),
        gamma_water=data.get("gamma_water", 9.81),
    )
    return model


def save_model(model: Model, path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model_to_dict(model), fh, indent=2)
    return path


def load_model(path: str) -> Model:
    with open(path, encoding="utf-8") as fh:
        return model_from_dict(json.load(fh))
