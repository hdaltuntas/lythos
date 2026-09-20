"""A local web interface for building and running models.

The interface runs as a small HTTP server on the machine it was started on and
is driven from a browser.  That choice keeps the program usable over a remote
session or inside a container, where a desktop toolkit would need a display it
does not have, and it costs no dependencies beyond the standard library.

Nothing is exposed beyond the loopback address unless the host is changed
deliberately, and the server runs one analysis at a time in a worker thread so
the interface stays responsive while a strength reduction analysis grinds
through its trials.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ..core import results as R
from ..core.serialize import model_from_dict, model_to_dict
from ..core.solver import Solver
from ..examples import DESCRIPTIONS, EXAMPLES
from ..report import summarise, write_figures, write_html
from ..viz import plots

STATIC = os.path.join(os.path.dirname(__file__), "static")

#: a small mark for the browser tab: ground line over a slope
_FAVICON = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    b'<rect width="32" height="32" rx="6" fill="#1565c0"/>'
    b'<path d="M4 25h24L20 9 12 25z" fill="#f2e394"/>'
    b'<path d="M4 25h24" stroke="#ffffff" stroke-width="2"/></svg>'
)


class Session:
    """The one analysis this server knows about."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.problem = None
        self.results = []
        self.status = "idle"
        self.message = ""
        self.progress = (0, 0, "")
        self.summary = None
        self.error = ""

    # ------------------------------------------------------------------ meshing
    def mesh(self, data: dict) -> dict:
        model = model_from_dict(data)
        issues = model.validate()
        blocking = [m for m in issues if "no soil layers" in m or "zero area" in m]
        if blocking:
            return {"ok": False, "issues": issues}
        problem = model.build()
        with self.lock:
            self.model, self.problem, self.results = model, problem, []
            self.summary = None
        areas = problem.continuum.volumes()
        return {
            "ok": True,
            "issues": issues,
            "elements": int(problem.continuum.n_elements),
            "nodes": int(problem.mesh.n_nodes),
            "dofs": int(problem.dofs.n_dof),
            "area": float(areas.sum()),
            "structures": [
                {"name": s.name, "elements": int(s.beam.n_elements),
                 "interfaces": len(s.interfaces)}
                for s in problem.structures
            ],
        }

    # ------------------------------------------------------------------ running
    def start(self, data: dict) -> dict:
        with self.lock:
            if self.status == "running":
                return {"ok": False, "message": "an analysis is already running"}
            self.status = "running"
            self.error = ""
            self.results = []
            self.summary = None
        thread = threading.Thread(target=self._run, args=(data,), daemon=True)
        thread.start()
        return {"ok": True}

    def _run(self, data: dict) -> None:
        try:
            model = model_from_dict(data)
            problem = model.build()
            with self.lock:
                self.model, self.problem = model, problem
            solver = Solver(problem, tolerance=float(data.get("tolerance", 3e-3)))

            def progress(i, n, name):
                with self.lock:
                    self.progress = (i, n, name)

            results = solver.run(progress=progress)
            with self.lock:
                self.results = results
                self.summary = summarise(problem, results)
                self.status = "done"
                self.progress = (len(results), len(results), "finished")
        except Exception as exc:                       # surfaced in the interface
            with self.lock:
                self.status = "error"
                self.error = f"{type(exc).__name__}: {exc}"
                self.message = traceback.format_exc(limit=4)

    def state(self) -> dict:
        with self.lock:
            i, n, name = self.progress
            return {
                "status": self.status,
                "stage_index": i,
                "stage_count": n,
                "stage_name": name,
                "error": self.error,
                "detail": self.message if self.status == "error" else "",
                "summary": self.summary,
                "stages": [
                    {"name": r.name, "kind": r.kind, "converged": bool(r.converged),
                     "srf": r.srf, "message": r.message}
                    for r in self.results
                ],
                "structures": [s.name for s in (self.problem.structures if self.problem else [])],
                "fields": sorted(R.FIELDS),
            }

    # ------------------------------------------------------------------- images
    def plot(self, query: dict) -> bytes:
        kind = query.get("kind", ["mesh"])[0]
        with self.lock:
            problem, results = self.problem, list(self.results)
        if problem is None:
            raise ValueError("no mesh yet; press Mesh or Run first")
        if kind == "mesh":
            return plots.figure_to_png(plots.plot_mesh(problem))

        index = int(query.get("stage", ["0"])[0])
        if not results:
            raise ValueError("no results yet")
        index = max(0, min(index, len(results) - 1))
        result = results[index]
        if kind == "field":
            field = query.get("field", ["u_total"])[0]
            deformed = query.get("deformed", ["0"])[0] == "1"
            return plots.figure_to_png(
                plots.plot_field(problem, result, field, deformed=deformed))
        if kind == "plastic":
            return plots.figure_to_png(plots.plot_plastic_points(problem, result))
        if kind == "deformed":
            return plots.figure_to_png(plots.plot_deformed_mesh(problem, result))
        if kind == "vectors":
            return plots.figure_to_png(plots.plot_displacement_vectors(problem, result))
        if kind == "ssr":
            return plots.figure_to_png(plots.plot_ssr_curve(result))
        if kind == "forces":
            name = query.get("name", [""])[0]
            return plots.figure_to_png(plots.plot_structure_forces(problem, result, name))
        raise ValueError(f"unknown plot {kind!r}")

    def report(self, out_dir: str) -> str:
        with self.lock:
            problem, results, model = self.problem, list(self.results), self.model
        if not results:
            raise ValueError("run the analysis first")
        os.makedirs(out_dir, exist_ok=True)
        summary = summarise(problem, results)
        figures = write_figures(problem, results, out_dir)
        return write_html(model, problem, results, summary, figures, out_dir)


SESSION = Session()


class Handler(BaseHTTPRequestHandler):
    server_version = "Lythos"

    def log_message(self, fmt, *args):      # keep the console for analysis output
        pass

    # ---------------------------------------------------------------- utilities
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, code: int = 200) -> None:
        self._send(code, json.dumps(payload, default=float).encode("utf-8"),
                   "application/json")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # --------------------------------------------------------------------- GET
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        try:
            if route in ("/", "/index.html"):
                return self._static("index.html")
            if route.startswith("/static/"):
                return self._static(os.path.basename(route))
            if route == "/favicon.ico":
                return self._send(200, _FAVICON, "image/svg+xml")
            if route == "/api/state":
                return self._json(SESSION.state())
            if route == "/api/examples":
                return self._json({
                    "examples": [{"key": k, "title": DESCRIPTIONS.get(k, k)}
                                 for k in EXAMPLES]
                })
            if route == "/api/example":
                key = query.get("key", ["slope"])[0]
                if key not in EXAMPLES:
                    return self._json({"error": "unknown example"}, 404)
                return self._json(model_to_dict(EXAMPLES[key]()))
            if route == "/api/plot":
                return self._send(200, SESSION.plot(query), "image/png")
            self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

    def _static(self, name: str) -> None:
        path = os.path.join(STATIC, name)
        if not os.path.isfile(path):
            return self._json({"error": "not found"}, 404)
        kinds = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
                 ".css": "text/css", ".svg": "image/svg+xml"}
        ext = os.path.splitext(name)[1]
        with open(path, "rb") as fh:
            self._send(200, fh.read(), kinds.get(ext, "application/octet-stream"))

    # -------------------------------------------------------------------- POST
    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            if route == "/api/mesh":
                return self._json(SESSION.mesh(self._body()))
            if route == "/api/run":
                return self._json(SESSION.start(self._body()))
            if route == "/api/report":
                data = self._body()
                path = SESSION.report(data.get("out", "out/report"))
                return self._json({"ok": True, "path": os.path.abspath(path)})
            if route == "/api/save":
                data = self._body()
                path = data.get("path", "model.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(data.get("model", {}), fh, indent=2)
                return self._json({"ok": True, "path": os.path.abspath(path)})
            self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)


def serve(host: str = "127.0.0.1", port: int = 8777, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Lythos is running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
