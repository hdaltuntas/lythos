"""Model definition, serialisation and section properties."""

import json
import math

import pytest

from lythos.core.elements import AnchorProperties, InterfaceProperties
from lythos.core.materials import MohrCoulomb, concrete_modulus
from lythos.core.model import (Anchor, LineLoad, Model, SoilLayer, Stage, Structure,
                               WaterTable)
from lythos.core.pile import PileSection, WallSection
from lythos.core.serialize import model_from_dict, model_to_dict
from lythos.examples import EXAMPLES


def _demo_model():
    return Model(
        name="demo",
        layers=[SoilLayer("clay", [(0, 0), (10, 0), (10, 5), (0, 5)],
                          MohrCoulomb(name="clay", c=5, phi=25))],
        structures=[Structure("wall", [(5, 5), (5, 1)],
                              PileSection(diameter=0.9, spacing=1.2),
                              InterfaceProperties())],
        anchors=[Anchor("a1", (5, 4), (9, 4.5), AnchorProperties(EA=2e5, prestress=100))],
        line_loads=[LineLoad("q", [(6, 5), (9, 5)], qy=-20)],
        water=WaterTable([(0, 2), (10, 2)]),
        stages=[Stage("initial", kind="initial"),
                Stage("dig", kind="plastic", active_structures=["wall"])],
    )


def test_model_round_trips_through_json():
    model = _demo_model()
    once = model_to_dict(model)
    twice = model_to_dict(model_from_dict(json.loads(json.dumps(once))))
    assert once == twice


def test_validation_reports_real_problems():
    empty = Model(name="empty")
    assert any("no soil layers" in m for m in empty.validate())

    duplicate = _demo_model()
    duplicate.layers.append(duplicate.layers[0])
    assert any("unique" in m for m in duplicate.validate())

    unknown = _demo_model()
    unknown.stages[1].active_layers = ["not a layer"]
    assert any("unknown layer" in m for m in unknown.validate())

    assert _demo_model().validate() == []


def test_stages_inherit_what_they_do_not_change():
    model = _demo_model()
    model.stages.append(Stage("deeper", kind="plastic"))
    stages = model.resolved_stages()
    assert stages[-1].active_structures == ["wall"]
    assert stages[0].active_layers == ["clay"]


def test_pile_section_properties():
    pile = PileSection(diameter=0.8, spacing=2.0, fck=30.0)
    assert pile.area == pytest.approx(math.pi * 0.16, rel=1e-12)
    assert pile.inertia == pytest.approx(math.pi * 0.8 ** 4 / 64, rel=1e-12)
    section = pile.section()
    E = concrete_modulus(30.0)
    assert section.EA == pytest.approx(E * pile.area / 2.0, rel=1e-12)
    assert section.EI == pytest.approx(E * pile.inertia / 2.0, rel=1e-12)
    # halving the spacing doubles the rigidity per metre of wall
    closer = PileSection(diameter=0.8, spacing=1.0, fck=30.0).section()
    assert closer.EI == pytest.approx(2 * section.EI, rel=1e-12)


def test_wall_section_overrides_win():
    wall = WallSection(thickness=0.5, EA_override=1.0e6, EI_override=2.0e4)
    section = wall.section()
    assert section.EA == 1.0e6
    assert section.EI == 2.0e4


def test_cracked_section_factor_reduces_only_bending():
    full = PileSection(diameter=1.0, spacing=1.0).section()
    cracked = PileSection(diameter=1.0, spacing=1.0, stiffness_factor=0.7).section()
    assert cracked.EI == pytest.approx(0.7 * full.EI, rel=1e-12)
    assert cracked.EA == pytest.approx(full.EA, rel=1e-12)


@pytest.mark.parametrize("key", sorted(EXAMPLES))
def test_examples_are_valid_and_can_be_meshed(key):
    model = EXAMPLES[key]()
    assert model.validate() == []
    problem = model.build()
    assert problem.continuum.n_elements > 50
    assert problem.mesh.element_areas().min() > 0
    total = sum(layer.area() for layer in model.layers)
    assert problem.mesh.element_areas().sum() == pytest.approx(total, rel=1e-9)


@pytest.mark.parametrize("key", sorted(EXAMPLES))
def test_examples_round_trip(key):
    model = EXAMPLES[key]()
    assert model_to_dict(model) == model_to_dict(model_from_dict(model_to_dict(model)))
