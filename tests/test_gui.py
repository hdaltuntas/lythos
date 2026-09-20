"""The interface's backend: meshing, running and rendering through the session API."""

import json
import time

import pytest

from lythos.core.serialize import model_to_dict
from lythos.examples import EXAMPLES
from lythos.gui.server import Session


def _pile_wall_quick():
    """The pile wall example, coarsened and without the safety stage."""
    model = EXAMPLES["pile_wall"]()
    model.mesh_size = 3.0
    for layer in model.layers:
        layer.mesh_size = 3.0
    model.stages = [s for s in model.stages if s.kind != "ssr"]
    return model


def test_mesh_endpoint_reports_the_mesh():
    session = Session()
    reply = session.mesh(model_to_dict(_pile_wall_quick()))
    assert reply["ok"]
    assert reply["elements"] > 50
    assert reply["dofs"] > 2 * reply["nodes"] - 1
    assert reply["structures"][0]["elements"] > 0
    assert reply["structures"][0]["interfaces"] == 2


def test_mesh_endpoint_refuses_an_empty_model():
    session = Session()
    reply = session.mesh({"layers": [], "stages": []})
    assert reply["ok"] is False
    assert any("no soil layers" in m for m in reply["issues"])


def test_mesh_plot_is_a_png():
    session = Session()
    session.mesh(model_to_dict(_pile_wall_quick()))
    data = session.plot({"kind": ["mesh"]})
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 5000


def test_plot_before_meshing_is_an_error():
    session = Session()
    with pytest.raises(ValueError):
        session.plot({"kind": ["mesh"]})


@pytest.mark.slow
def test_full_run_through_the_session_produces_every_plot():
    session = Session()
    assert session.start(model_to_dict(_pile_wall_quick()))["ok"]
    for _ in range(600):
        state = session.state()
        if state["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    state = session.state()
    assert state["status"] == "done", state.get("error")
    assert all(s["converged"] for s in state["stages"])

    last = len(state["stages"]) - 1
    for kind, extra in (("field", {"field": ["u_total"], "deformed": ["1"]}),
                        ("plastic", {}), ("deformed", {}), ("vectors", {}),
                        ("forces", {"name": ["contiguous pile wall"]})):
        query = {"kind": [kind], "stage": [str(last)], **extra}
        assert session.plot(query)[:4] == b"\x89PNG"

    summary = state["summary"]
    assert summary["elements"] > 0
    forces = [s for s in summary["stages"] if s.get("structures")]
    assert forces, "the wall should report section forces once installed"
    wall = forces[-1]["structures"]["contiguous pile wall"]
    assert wall["bending_moment_max_kNm_per_m"] > 0
    assert wall["moment_utilisation"] < 1.0


def test_examples_endpoint_lists_every_example():
    keys = {e["key"] for e in json.loads(json.dumps(
        {"examples": [{"key": k} for k in EXAMPLES]}))["examples"]}
    assert keys == set(EXAMPLES)


def test_static_assets_exist():
    import os
    from lythos.gui.server import STATIC
    for name in ("index.html", "app.js", "style.css"):
        assert os.path.isfile(os.path.join(STATIC, name))
    page = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    for element in ("btnRun", "btnMesh", "canvas", "exampleSelect", "stageList"):
        assert f'id="{element}"' in page


def test_launcher_runs_without_installation():
    """main.py must work from a clone with nothing installed."""
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    launcher = os.path.join(root, "main.py")
    assert os.path.isfile(launcher)

    # An environment with no PYTHONPATH and a working directory elsewhere:
    # the launcher has to find the package by itself.
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, launcher, "--help"],
                            capture_output=True, text=True, cwd="/", env=env, timeout=120)
    assert result.returncode == 0, result.stderr
    for command in ("run", "mesh", "gui", "examples"):
        assert command in result.stdout


def test_launcher_defaults_to_the_interface():
    """Bare arguments belong to the interface, not to a missing command."""
    import importlib.util
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("lythos_launcher",
                                                  os.path.join(root, "main.py"))
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    assert launcher.DEFAULT_COMMAND == "gui"
    assert launcher.check_dependencies() == []

    import lythos.cli                      # main.py imports this lazily

    seen = {}
    argv = sys.argv
    real_main = lythos.cli.main

    def fake_cli(args):
        seen["args"] = args
        return 0

    lythos.cli.main = fake_cli
    try:
        for given, expected_first in (([], "gui"),
                                      (["--port", "9000"], "gui"),
                                      (["run", "m.json"], "run"),
                                      (["mesh", "m.json"], "mesh")):
            sys.argv = ["main.py"] + given
            launcher.main()
            assert seen["args"][0] == expected_first, (given, seen["args"])
    finally:
        sys.argv = argv
        lythos.cli.main = real_main
