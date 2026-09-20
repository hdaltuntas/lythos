"use strict";
/* Lythos interactive interface.
 *
 * The model lives here as the same JSON the solver reads, so what is drawn,
 * what is saved and what is analysed are one description with no translation
 * layer to drift out of step.
 */

/* ------------------------------------------------------------------ defaults */
const SOIL_COLORS = ["#c8b273", "#a8907a", "#9aa7a0", "#d2b48c", "#b9a878",
                     "#8fa2ad", "#c2a68c", "#7d8a83"];

function defaultMaterial(index) {
  return {
    type: "MohrCoulomb", name: "soil " + (index + 1),
    E: 30000, nu: 0.3, gamma: 18, gamma_sat: 20, K0: null,
    c: 5, phi: 30, psi: 0, tension_cutoff: 0,
    color: SOIL_COLORS[index % SOIL_COLORS.length],
    drainage: "drained", m_stiffness: 0, p_ref: 100, residual_stiffness: 0.001
  };
}

function emptyModel() {
  return {
    format: "lythos-model", version: 1, name: "new model",
    mesh_size: null, min_angle: 25, initial_stress: "k0", gamma_water: 9.81,
    layers: [], structures: [], anchors: [], line_loads: [], point_loads: [],
    water: { points: [] }, boundary: { fix_bottom: true, fix_sides: true, fixed_nodes: [] },
    stages: []
  };
}

/* --------------------------------------------------------------------- state */
const S = {
  model: emptyModel(),
  tool: "select",
  draft: [],                 // points of the shape being drawn
  selection: null,           // {kind, index}
  view: { scale: 12, ox: 60, oy: 40 },
  drag: null,
  hoverPoint: null,
  pane: "props",
  stage: 0,
  results: null,
  polling: null,
  resultView: { stage: 0, kind: "field", field: "u_total", deformed: true, structure: "" }
};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

/* ------------------------------------------------------------- view helpers */
function toScreen(p) {
  return [p[0] * S.view.scale + S.view.ox, canvas.clientHeight - (p[1] * S.view.scale + S.view.oy)];
}
function toWorld(x, y) {
  return [(x - S.view.ox) / S.view.scale, (canvas.clientHeight - y - S.view.oy) / S.view.scale];
}
function snap(p) {
  const step = parseFloat(document.getElementById("snapStep").value) || 0;
  if (step <= 0) return p;
  return [Math.round(p[0] / step) * step, Math.round(p[1] / step) * step];
}

function modelBounds() {
  const pts = [];
  S.model.layers.forEach(l => pts.push(...l.polygon));
  S.model.structures.forEach(s => pts.push(...s.path));
  S.model.line_loads.forEach(l => pts.push(...l.path));
  S.model.anchors.forEach(a => pts.push(a.start, a.end));
  pts.push(...(S.model.water.points || []));
  if (!pts.length) return null;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function fitView() {
  const b = modelBounds();
  if (!b) { S.view = { scale: 12, ox: 60, oy: 40 }; return; }
  const w = Math.max(b[2] - b[0], 1), h = Math.max(b[3] - b[1], 1);
  const scale = Math.min((canvas.clientWidth - 80) / w, (canvas.clientHeight - 80) / h);
  S.view.scale = scale;
  S.view.ox = 40 - b[0] * scale + (canvas.clientWidth - 80 - w * scale) / 2;
  S.view.oy = 40 - b[1] * scale + (canvas.clientHeight - 80 - h * scale) / 2;
}

/* ------------------------------------------------------------------ drawing */
function resize() {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * ratio;
  canvas.height = canvas.clientHeight * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}

function niceStep(targetPx) {
  const raw = targetPx / S.view.scale;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (raw <= m * mag) return m * mag;
  return 10 * mag;
}

function drawGrid() {
  const step = niceStep(70);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const [x0, y1] = toWorld(0, 0), [x1, y0] = toWorld(w, h);
  ctx.lineWidth = 1;
  ctx.font = "10px -apple-system, sans-serif";
  for (let x = Math.floor(x0 / step) * step; x <= x1; x += step) {
    const sx = toScreen([x, 0])[0];
    ctx.strokeStyle = Math.abs(x) < 1e-9 ? "#c3ccd8" : "#eaeef4";
    ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, h); ctx.stroke();
    ctx.fillStyle = "#9aa5b1";
    ctx.fillText(x.toFixed(step < 1 ? 1 : 0), sx + 3, h - 4);
  }
  for (let y = Math.floor(y0 / step) * step; y <= y1; y += step) {
    const sy = toScreen([0, y])[1];
    ctx.strokeStyle = Math.abs(y) < 1e-9 ? "#c3ccd8" : "#eaeef4";
    ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(w, sy); ctx.stroke();
    ctx.fillStyle = "#9aa5b1";
    ctx.fillText(y.toFixed(step < 1 ? 1 : 0), 3, sy - 3);
  }
}

function pathFrom(points, close) {
  ctx.beginPath();
  points.forEach((p, i) => {
    const s = toScreen(p);
    if (i === 0) ctx.moveTo(s[0], s[1]); else ctx.lineTo(s[0], s[1]);
  });
  if (close) ctx.closePath();
}

function isSelected(kind, index) {
  return S.selection && S.selection.kind === kind && S.selection.index === index;
}

function draw() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  drawGrid();

  S.model.layers.forEach((layer, i) => {
    if (layer.polygon.length < 2) return;
    pathFrom(layer.polygon, true);
    ctx.fillStyle = layer.material.color || "#c8b273";
    ctx.globalAlpha = 0.88; ctx.fill(); ctx.globalAlpha = 1;
    ctx.strokeStyle = isSelected("layer", i) ? "#1565c0" : "#7a6a52";
    ctx.lineWidth = isSelected("layer", i) ? 2.5 : 1.1;
    ctx.stroke();
    if (isSelected("layer", i)) drawHandles(layer.polygon);
  });

  const water = S.model.water.points || [];
  if (water.length > 1) {
    const sorted = [...water].sort((a, b) => a[0] - b[0]);
    pathFrom(sorted, false);
    ctx.strokeStyle = "#0277bd"; ctx.lineWidth = 2; ctx.stroke();
    const s = toScreen(sorted[0]);
    ctx.fillStyle = "#0277bd";
    ctx.beginPath(); ctx.moveTo(s[0] - 5, s[1] - 6); ctx.lineTo(s[0] + 5, s[1] - 6);
    ctx.lineTo(s[0], s[1] + 2); ctx.fill();
  }

  S.model.line_loads.forEach((load, i) => {
    if (load.path.length < 2) return;
    pathFrom(load.path, false);
    ctx.strokeStyle = isSelected("load", i) ? "#1565c0" : "#ef6c00";
    ctx.lineWidth = 3.4; ctx.stroke();
    for (let t = 0; t <= 1.0001; t += 0.2) {
      const p = interpolate(load.path, t);
      const s = toScreen(p);
      ctx.beginPath(); ctx.moveTo(s[0], s[1] - 16); ctx.lineTo(s[0], s[1] - 3);
      ctx.strokeStyle = "#ef6c00"; ctx.lineWidth = 1.2; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(s[0] - 3, s[1] - 7); ctx.lineTo(s[0], s[1] - 2);
      ctx.lineTo(s[0] + 3, s[1] - 7); ctx.stroke();
    }
    if (isSelected("load", i)) drawHandles(load.path);
  });

  S.model.structures.forEach((st, i) => {
    if (st.path.length < 2) return;
    pathFrom(st.path, false);
    ctx.strokeStyle = isSelected("structure", i) ? "#1565c0" : "#1a237e";
    ctx.lineWidth = isSelected("structure", i) ? 6 : 4.5;
    ctx.lineCap = "round"; ctx.stroke(); ctx.lineCap = "butt";
    if (isSelected("structure", i)) drawHandles(st.path);
  });

  S.model.anchors.forEach((a, i) => {
    pathFrom([a.start, a.end], false);
    ctx.strokeStyle = isSelected("anchor", i) ? "#1565c0" : "#b71c1c";
    ctx.lineWidth = 2; ctx.setLineDash([7, 4]); ctx.stroke(); ctx.setLineDash([]);
    const e = toScreen(a.end);
    ctx.fillStyle = "#b71c1c";
    ctx.beginPath(); ctx.arc(e[0], e[1], 4, 0, 6.2832); ctx.fill();
  });

  if (S.draft.length) {
    const preview = S.hoverPoint ? [...S.draft, S.hoverPoint] : S.draft;
    pathFrom(preview, S.tool === "layer");
    ctx.strokeStyle = "#1565c0"; ctx.lineWidth = 1.8;
    ctx.setLineDash([6, 4]); ctx.stroke(); ctx.setLineDash([]);
    if (S.tool === "layer") { ctx.fillStyle = "rgba(21,101,192,0.12)"; ctx.fill(); }
    drawHandles(S.draft);
  }
}

function drawHandles(points) {
  points.forEach(p => {
    const s = toScreen(p);
    ctx.beginPath(); ctx.arc(s[0], s[1], 4, 0, 6.2832);
    ctx.fillStyle = "#ffffff"; ctx.fill();
    ctx.strokeStyle = "#1565c0"; ctx.lineWidth = 1.6; ctx.stroke();
  });
}

function interpolate(path, t) {
  const total = pathLength(path);
  let target = t * total, acc = 0;
  for (let i = 1; i < path.length; i++) {
    const seg = Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
    if (acc + seg >= target || i === path.length - 1) {
      const f = seg > 0 ? (target - acc) / seg : 0;
      return [path[i - 1][0] + f * (path[i][0] - path[i - 1][0]),
              path[i - 1][1] + f * (path[i][1] - path[i - 1][1])];
    }
    acc += seg;
  }
  return path[path.length - 1];
}

function pathLength(path) {
  let total = 0;
  for (let i = 1; i < path.length; i++)
    total += Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
  return total;
}

/* -------------------------------------------------------------- interaction */
canvas.addEventListener("mousemove", ev => {
  const rect = canvas.getBoundingClientRect();
  const raw = toWorld(ev.clientX - rect.left, ev.clientY - rect.top);
  document.getElementById("coords").textContent =
    `x ${raw[0].toFixed(2)}   y ${raw[1].toFixed(2)} m`;
  if (S.drag && S.drag.mode === "pan") {
    S.view.ox += ev.clientX - S.drag.x;
    S.view.oy -= ev.clientY - S.drag.y;
    S.drag.x = ev.clientX; S.drag.y = ev.clientY;
    draw(); return;
  }
  if (S.drag && S.drag.mode === "vertex") {
    S.drag.target[S.drag.vertex] = snap(raw);
    draw(); return;
  }
  if (S.tool !== "select" && S.draft.length) { S.hoverPoint = snap(raw); draw(); }
});

canvas.addEventListener("mousedown", ev => {
  const rect = canvas.getBoundingClientRect();
  const raw = toWorld(ev.clientX - rect.left, ev.clientY - rect.top);
  if (ev.button === 1 || ev.altKey || (S.tool === "select" && !hitVertex(raw) && !hitObject(raw))) {
    S.drag = { mode: "pan", x: ev.clientX, y: ev.clientY };
    canvas.style.cursor = "grabbing";
    return;
  }
  if (S.tool === "select") {
    const v = hitVertex(raw);
    if (v) { S.drag = v; return; }
    const obj = hitObject(raw);
    if (obj) { select(obj.kind, obj.index); }
    return;
  }
  S.draft.push(snap(raw));
  if (S.tool === "anchor" && S.draft.length === 2) finishDraft();
  draw();
});

window.addEventListener("mouseup", () => {
  if (S.drag && S.drag.mode === "vertex") pushModel();
  S.drag = null;
  canvas.style.cursor = S.tool === "select" ? "default" : "crosshair";
});

canvas.addEventListener("dblclick", () => { if (S.draft.length) finishDraft(); });

canvas.addEventListener("wheel", ev => {
  ev.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const before = toWorld(ev.clientX - rect.left, ev.clientY - rect.top);
  const factor = Math.exp(-ev.deltaY * 0.0014);
  S.view.scale = Math.max(0.4, Math.min(400, S.view.scale * factor));
  const after = toWorld(ev.clientX - rect.left, ev.clientY - rect.top);
  S.view.ox += (after[0] - before[0]) * S.view.scale;
  S.view.oy += (after[1] - before[1]) * S.view.scale;
  draw();
}, { passive: false });

window.addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT") return;
  if (ev.key === "Escape") { S.draft = []; S.hoverPoint = null; draw(); }
  if (ev.key === "Enter" && S.draft.length) finishDraft();
  if ((ev.key === "Delete" || ev.key === "Backspace") && S.selection) {
    removeSelected(); ev.preventDefault();
  }
  if (ev.key === "f") { fitView(); draw(); }
});

function hitVertex(p) {
  const tol = 8 / S.view.scale;
  const lists = [];
  S.model.layers.forEach((l, i) => lists.push([l.polygon, "layer", i]));
  S.model.structures.forEach((s, i) => lists.push([s.path, "structure", i]));
  S.model.line_loads.forEach((l, i) => lists.push([l.path, "load", i]));
  lists.push([S.model.water.points, "water", 0]);
  for (const [pts, kind, index] of lists) {
    for (let k = 0; k < pts.length; k++) {
      if (Math.hypot(pts[k][0] - p[0], pts[k][1] - p[1]) < tol) {
        select(kind, index);
        return { mode: "vertex", target: pts, vertex: k };
      }
    }
  }
  for (let i = 0; i < S.model.anchors.length; i++) {
    const a = S.model.anchors[i];
    for (const key of ["start", "end"]) {
      if (Math.hypot(a[key][0] - p[0], a[key][1] - p[1]) < tol) {
        select("anchor", i);
        return { mode: "vertex", target: a, vertex: key };
      }
    }
  }
  return null;
}

function hitObject(p) {
  const tol = 8 / S.view.scale;
  for (let i = S.model.structures.length - 1; i >= 0; i--)
    if (nearPath(S.model.structures[i].path, p, tol)) return { kind: "structure", index: i };
  for (let i = S.model.line_loads.length - 1; i >= 0; i--)
    if (nearPath(S.model.line_loads[i].path, p, tol)) return { kind: "load", index: i };
  for (let i = S.model.anchors.length - 1; i >= 0; i--)
    if (nearPath([S.model.anchors[i].start, S.model.anchors[i].end], p, tol))
      return { kind: "anchor", index: i };
  for (let i = S.model.layers.length - 1; i >= 0; i--)
    if (inPolygon(p, S.model.layers[i].polygon)) return { kind: "layer", index: i };
  return null;
}

function nearPath(path, p, tol) {
  for (let i = 1; i < path.length; i++) {
    const [a, b] = [path[i - 1], path[i]];
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const len2 = dx * dx + dy * dy;
    let t = len2 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2 : 0;
    t = Math.max(0, Math.min(1, t));
    if (Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy)) < tol) return true;
  }
  return false;
}

function inPolygon(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) &&
        p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function finishDraft() {
  const pts = S.draft;
  S.draft = []; S.hoverPoint = null;
  if (S.tool === "layer" && pts.length >= 3) {
    const i = S.model.layers.length;
    S.model.layers.push({
      name: "layer " + (i + 1), polygon: pts, mesh_size: null, material: defaultMaterial(i)
    });
    select("layer", i);
  } else if (S.tool === "structure" && pts.length >= 2) {
    S.model.structures.push({
      name: "wall " + (S.model.structures.length + 1), path: pts, r_inter: 0.67,
      section: { type: "PileSection", name: "pile row", diameter: 0.8, spacing: 1.2,
                 fck: 30, E: null, nu: 0.2, gamma: 25, rho_s: 0.01, fyk: 500,
                 stiffness_factor: 1.0 },
      interface: { kn: null, ks: null, c: null, phi: null, tensile: 0, virtual_thickness: 0.1 }
    });
    select("structure", S.model.structures.length - 1);
  } else if (S.tool === "anchor" && pts.length === 2) {
    S.model.anchors.push({
      name: "anchor " + (S.model.anchors.length + 1), start: pts[0], end: pts[1],
      properties: { EA: 200000, spacing: 2.5, prestress: 0, Fmax: 0,
                    compression: true, grout_length: 4 }
    });
    select("anchor", S.model.anchors.length - 1);
  } else if (S.tool === "load" && pts.length >= 2) {
    S.model.line_loads.push({
      name: "load " + (S.model.line_loads.length + 1), path: pts, qx: 0, qy: -20
    });
    select("load", S.model.line_loads.length - 1);
  } else if (S.tool === "water" && pts.length >= 2) {
    S.model.water.points = pts;
    select("water", 0);
  }
  setTool("select");
  pushModel();
}

/* ------------------------------------------------------------ object panels */
function select(kind, index) {
  S.selection = { kind, index };
  S.pane = "props"; showPane();
  renderTrees(); renderProps(); draw();
}

function removeSelected() {
  const { kind, index } = S.selection;
  if (kind === "layer") S.model.layers.splice(index, 1);
  if (kind === "structure") S.model.structures.splice(index, 1);
  if (kind === "anchor") S.model.anchors.splice(index, 1);
  if (kind === "load") S.model.line_loads.splice(index, 1);
  if (kind === "water") S.model.water.points = [];
  S.selection = null;
  pushModel();
}

function renderTrees() {
  const layers = document.getElementById("treeLayers");
  layers.innerHTML = "";
  S.model.layers.forEach((l, i) => {
    layers.appendChild(treeItem(l.name, "layer", i, l.material.color));
  });
  if (!S.model.layers.length) layers.innerHTML = '<div class="note">No layers yet.</div>';

  const structures = document.getElementById("treeStructures");
  structures.innerHTML = "";
  S.model.structures.forEach((s, i) => structures.appendChild(treeItem(s.name, "structure", i, "#1a237e")));
  if (!S.model.structures.length) structures.innerHTML = '<div class="note">None.</div>';

  const others = document.getElementById("treeOthers");
  others.innerHTML = "";
  S.model.anchors.forEach((a, i) => others.appendChild(treeItem(a.name, "anchor", i, "#b71c1c")));
  S.model.line_loads.forEach((l, i) => others.appendChild(treeItem(l.name, "load", i, "#ef6c00")));
  if (S.model.water.points.length) others.appendChild(treeItem("water table", "water", 0, "#0277bd"));
  if (!others.children.length) others.innerHTML = '<div class="note">None.</div>';
}

function treeItem(name, kind, index, color) {
  const div = document.createElement("div");
  div.className = "item" + (isSelected(kind, index) ? " sel" : "");
  div.innerHTML = `<span class="swatch" style="background:${color}"></span>
                   <span class="name"></span><span class="kill">&times;</span>`;
  div.querySelector(".name").textContent = name;
  div.onclick = ev => {
    if (ev.target.classList.contains("kill")) { S.selection = { kind, index }; removeSelected(); }
    else select(kind, index);
  };
  return div;
}

function field(label, value, onChange, opts = {}) {
  const wrap = document.createElement("div");
  wrap.className = "field";
  const step = opts.step !== undefined ? opts.step : "any";
  if (opts.options) {
    wrap.innerHTML = `<label>${label}</label><select>` +
      opts.options.map(o => `<option value="${o[0]}">${o[1]}</option>`).join("") + "</select>";
    const sel = wrap.querySelector("select");
    sel.value = value;
    sel.onchange = () => { onChange(sel.value); };
  } else if (opts.type === "checkbox") {
    wrap.innerHTML = `<label><input type="checkbox"> ${label}</label>`;
    const box = wrap.querySelector("input");
    box.checked = !!value;
    box.onchange = () => onChange(box.checked);
  } else {
    wrap.innerHTML = `<label>${label}</label><input type="${opts.type || "number"}" step="${step}">`;
    const input = wrap.querySelector("input");
    input.value = value === null || value === undefined ? "" : value;
    input.onchange = () => {
      const raw = input.value.trim();
      if (opts.type === "text") onChange(raw);
      else onChange(raw === "" ? null : parseFloat(raw));
    };
  }
  return wrap;
}

function group(parent, columns, fields) {
  const div = document.createElement("div");
  div.className = columns === 3 ? "grid3" : columns === 2 ? "grid2" : "";
  fields.forEach(f => div.appendChild(f));
  parent.appendChild(div);
}

function renderProps() {
  const body = document.getElementById("propsBody");
  body.innerHTML = "";
  if (!S.selection) {
    body.innerHTML = '<div class="note">Select an object on the canvas or in the list to edit its properties.</div>';
    return;
  }
  const { kind, index } = S.selection;
  const heading = document.createElement("h2");
  body.appendChild(heading);

  if (kind === "layer") {
    const layer = S.model.layers[index];
    const m = layer.material;
    heading.textContent = "Soil layer";
    body.appendChild(field("Name", layer.name, v => { layer.name = v; pushModel(); }, { type: "text" }));
    group(body, 2, [
      field("Element size [m]", layer.mesh_size, v => { layer.mesh_size = v; pushModel(); }, { step: 0.25 }),
      field("Colour", m.color, v => { m.color = v; pushModel(); }, { type: "color" })
    ]);
    body.appendChild(field("Material model", m.type, v => { m.type = v; renderProps(); pushModel(); },
      { options: [["MohrCoulomb", "Mohr-Coulomb"], ["LinearElastic", "Linear elastic"]] }));
    group(body, 2, [
      field("E [kPa]", m.E, v => { m.E = v; pushModel(); }, { step: 1000 }),
      field("Poisson's ratio", m.nu, v => { m.nu = v; pushModel(); }, { step: 0.01 })
    ]);
    if (m.type === "MohrCoulomb") {
      group(body, 3, [
        field("c' [kPa]", m.c, v => { m.c = v; pushModel(); }, { step: 1 }),
        field("&phi;' [&deg;]", m.phi, v => { m.phi = v; pushModel(); }, { step: 1 }),
        field("&psi; [&deg;]", m.psi, v => { m.psi = v; pushModel(); }, { step: 1 })
      ]);
      body.appendChild(field("Tension cut-off [kPa]", m.tension_cutoff,
        v => { m.tension_cutoff = v; pushModel(); }, { step: 1 }));
    }
    group(body, 3, [
      field("&gamma; [kN/m&sup3;]", m.gamma, v => { m.gamma = v; pushModel(); }, { step: 0.5 }),
      field("&gamma;<sub>sat</sub>", m.gamma_sat, v => { m.gamma_sat = v; pushModel(); }, { step: 0.5 }),
      field("K<sub>0</sub>", m.K0, v => { m.K0 = v; pushModel(); }, { step: 0.05 })
    ]);
    const note = document.createElement("div");
    note.className = "derived";
    const k0 = m.K0 !== null && m.K0 !== undefined ? m.K0
      : (1 - Math.sin((m.phi || 0) * Math.PI / 180));
    note.innerHTML = `Area <b>${polygonArea(layer.polygon).toFixed(1)} m&sup2;</b> &middot; ` +
      `K<sub>0</sub> in use <b>${k0.toFixed(3)}</b>` +
      (m.K0 === null ? " (Jaky, 1 &minus; sin&phi;')" : "");
    body.appendChild(note);

  } else if (kind === "structure") {
    const st = S.model.structures[index];
    heading.textContent = "Wall / pile row";
    body.appendChild(field("Name", st.name, v => { st.name = v; pushModel(); }, { type: "text" }));
    body.appendChild(field("Section type", st.section.type, v => {
      st.section = v === "PileSection"
        ? { type: "PileSection", name: "pile row", diameter: 0.8, spacing: 1.2, fck: 30,
            E: null, nu: 0.2, gamma: 25, rho_s: 0.01, fyk: 500, stiffness_factor: 1.0 }
        : { type: "WallSection", name: "wall", thickness: 0.6, fck: 30, E: null, nu: 0.2,
            gamma: 25, stiffness_factor: 1.0, EA_override: null, EI_override: null, Mp: 0 };
      renderProps(); pushModel();
    }, { options: [["PileSection", "Bored pile row"], ["WallSection", "Continuous wall"]] }));

    const sec = st.section;
    if (sec.type === "PileSection") {
      group(body, 2, [
        field("Pile diameter [m]", sec.diameter, v => { sec.diameter = v; renderProps(); pushModel(); }, { step: 0.05 }),
        field("Spacing [m]", sec.spacing, v => { sec.spacing = v; renderProps(); pushModel(); }, { step: 0.1 })
      ]);
      group(body, 2, [
        field("f<sub>ck</sub> [MPa]", sec.fck, v => { sec.fck = v; renderProps(); pushModel(); }, { step: 5 }),
        field("Reinforcement ratio", sec.rho_s, v => { sec.rho_s = v; renderProps(); pushModel(); }, { step: 0.002 })
      ]);
      group(body, 2, [
        field("f<sub>yk</sub> [MPa]", sec.fyk, v => { sec.fyk = v; renderProps(); pushModel(); }, { step: 50 }),
        field("Stiffness factor", sec.stiffness_factor, v => { sec.stiffness_factor = v; renderProps(); pushModel(); }, { step: 0.05 })
      ]);
    } else {
      group(body, 2, [
        field("Thickness [m]", sec.thickness, v => { sec.thickness = v; renderProps(); pushModel(); }, { step: 0.05 }),
        field("f<sub>ck</sub> [MPa]", sec.fck, v => { sec.fck = v; renderProps(); pushModel(); }, { step: 5 })
      ]);
      group(body, 2, [
        field("EA override [kN/m]", sec.EA_override, v => { sec.EA_override = v; renderProps(); pushModel(); }),
        field("EI override [kNm&sup2;/m]", sec.EI_override, v => { sec.EI_override = v; renderProps(); pushModel(); })
      ]);
    }
    body.appendChild(sectionSummary(sec));

    const hasInterface = !!st.interface;
    body.appendChild(field("Model soil-structure slip", hasInterface, v => {
      st.interface = v ? { kn: null, ks: null, c: null, phi: null, tensile: 0, virtual_thickness: 0.1 } : null;
      renderProps(); pushModel();
    }, { type: "checkbox" }));
    if (hasInterface) {
      body.appendChild(field("R<sub>inter</sub>", st.r_inter, v => { st.r_inter = v; pushModel(); }, { step: 0.05 }));
      const d = document.createElement("div");
      d.className = "note";
      d.innerHTML = "Interface strength is R<sub>inter</sub> times the strength of the adjacent soil; " +
        "leave the stiffness blank to derive it from the soil moduli.";
      body.appendChild(d);
    }

  } else if (kind === "anchor") {
    const a = S.model.anchors[index], p = a.properties;
    heading.textContent = "Anchor / strut";
    body.appendChild(field("Name", a.name, v => { a.name = v; pushModel(); }, { type: "text" }));
    group(body, 2, [
      field("EA [kN]", p.EA, v => { p.EA = v; pushModel(); }, { step: 10000 }),
      field("Spacing [m]", p.spacing, v => { p.spacing = v; pushModel(); }, { step: 0.5 })
    ]);
    group(body, 2, [
      field("Lock-off force [kN]", p.prestress, v => { p.prestress = v; pushModel(); }, { step: 25 }),
      field("Capacity F<sub>max</sub> [kN]", p.Fmax, v => { p.Fmax = v; pushModel(); }, { step: 50 })
    ]);
    group(body, 2, [
      field("Fixed length [m]", p.grout_length, v => { p.grout_length = v; pushModel(); }, { step: 0.5 }),
      field("Takes compression", p.compression, v => { p.compression = v; pushModel(); }, { type: "checkbox" })
    ]);
    const d = document.createElement("div");
    d.className = "derived";
    d.innerHTML = `Free length <b>${Math.hypot(a.end[0] - a.start[0], a.end[1] - a.start[1]).toFixed(2)} m</b>`;
    body.appendChild(d);

  } else if (kind === "load") {
    const l = S.model.line_loads[index];
    heading.textContent = "Line load";
    body.appendChild(field("Name", l.name, v => { l.name = v; pushModel(); }, { type: "text" }));
    group(body, 2, [
      field("q<sub>x</sub> [kN/m]", l.qx, v => { l.qx = v; pushModel(); }, { step: 5 }),
      field("q<sub>y</sub> [kN/m]", l.qy, v => { l.qy = v; pushModel(); }, { step: 5 })
    ]);
    body.appendChild(Object.assign(document.createElement("div"),
      { className: "note", textContent: "Negative q_y acts downwards." }));

  } else if (kind === "water") {
    heading.textContent = "Water table";
    const d = document.createElement("div");
    d.className = "note";
    d.textContent = "Drag the handles to shape the phreatic surface. Pore pressure is " +
      "hydrostatic below it and the horizontal gradient is carried as a seepage force.";
    body.appendChild(d);
  }
}

function sectionSummary(sec) {
  const div = document.createElement("div");
  div.className = "derived";
  const E = sec.E || 22000 * Math.pow((sec.fck + 8) / 10, 0.3) * 1000;
  let EA, EI, weight, Mp = 0;
  if (sec.type === "PileSection") {
    const A = Math.PI * sec.diameter * sec.diameter / 4;
    const I = Math.PI * Math.pow(sec.diameter, 4) / 64;
    EA = E * A / sec.spacing;
    EI = (sec.stiffness_factor || 1) * E * I / sec.spacing;
    weight = sec.gamma * A / sec.spacing;
    Mp = 0.4 * sec.diameter * (sec.rho_s * A) * sec.fyk * 1000 * 0.8 / sec.spacing;
  } else {
    EA = sec.EA_override || E * sec.thickness;
    EI = sec.EI_override || (sec.stiffness_factor || 1) * E * Math.pow(sec.thickness, 3) / 12;
    weight = sec.gamma * sec.thickness;
  }
  div.innerHTML =
    `E <b>${(E / 1e6).toFixed(1)} GPa</b> &middot; EA <b>${fmt(EA)} kN/m</b><br>` +
    `EI <b>${fmt(EI)} kNm&sup2;/m</b> &middot; weight <b>${weight.toFixed(1)} kN/m&sup2;</b>` +
    (Mp ? `<br>M<sub>p</sub> <b>${fmt(Mp)} kNm/m</b>` : "");
  return div;
}

function fmt(v) {
  if (!isFinite(v)) return "—";
  if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(2) + "e6";
  if (Math.abs(v) >= 1e3) return Math.round(v).toLocaleString();
  return v.toFixed(1);
}

function polygonArea(poly) {
  let total = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i], [x1, y1] = poly[(i + 1) % poly.length];
    total += x0 * y1 - x1 * y0;
  }
  return Math.abs(total / 2);
}

/* -------------------------------------------------------------------- stages */
function ensureStages() {
  if (!S.model.stages.length && S.model.layers.length) {
    S.model.stages = [{
      name: "1 - initial stresses", kind: "initial",
      active_layers: S.model.layers.map(l => l.name),
      active_structures: [], active_anchors: [], active_loads: [],
      water: null, reset_displacements: false, increments: 8,
      srf_min: 0.8, srf_max: 3.0, notes: ""
    }];
  }
}

function renderStages() {
  ensureStages();
  const list = document.getElementById("stageList");
  list.innerHTML = "";
  S.model.stages.forEach((stage, i) => {
    const div = document.createElement("div");
    div.className = "stage" + (S.stage === i ? " sel" : "");
    div.innerHTML =
      `<div class="row"><span class="name"></span>
       <span class="badge ${stage.kind}">${stage.kind}</span>
       <button class="kill" title="Remove">&times;</button></div>`;
    div.querySelector(".name").textContent = stage.name;
    div.onclick = ev => {
      if (ev.target.classList.contains("kill")) {
        S.model.stages.splice(i, 1); renderStages(); return;
      }
      S.stage = i; renderStages();
    };
    if (S.stage === i) div.appendChild(stageEditor(stage, i));
    list.appendChild(div);
  });
}

function stageEditor(stage, index) {
  const box = document.createElement("div");
  box.onclick = ev => ev.stopPropagation();
  box.appendChild(field("Name", stage.name, v => { stage.name = v; renderStages(); }, { type: "text" }));
  group(box, 2, [
    field("Stage type", stage.kind, v => { stage.kind = v; renderStages(); }, {
      options: [["initial", "Initial stresses"], ["plastic", "Construction step"],
                ["ssr", "Factor of safety"]]
    }),
    field("Load steps", stage.increments, v => { stage.increments = v; }, { step: 1 })
  ]);
  if (stage.kind === "ssr") {
    group(box, 2, [
      field("Search from", stage.srf_min, v => { stage.srf_min = v; }, { step: 0.1 }),
      field("Search to", stage.srf_max, v => { stage.srf_max = v; }, { step: 0.1 })
    ]);
  } else {
    box.appendChild(field("Reset displacements", stage.reset_displacements,
      v => { stage.reset_displacements = v; }, { type: "checkbox" }));
  }

  box.appendChild(checkGroup("Soil present", S.model.layers.map(l => l.name),
    stage.active_layers, v => { stage.active_layers = v; }));
  if (S.model.structures.length)
    box.appendChild(checkGroup("Structures built", S.model.structures.map(s => s.name),
      stage.active_structures, v => { stage.active_structures = v; }));
  if (S.model.anchors.length)
    box.appendChild(checkGroup("Anchors stressed", S.model.anchors.map(a => a.name),
      stage.active_anchors, v => { stage.active_anchors = v; }));
  if (S.model.line_loads.length)
    box.appendChild(checkGroup("Loads applied", S.model.line_loads.map(l => l.name),
      stage.active_loads, v => { stage.active_loads = v; }));

  const dup = document.createElement("button");
  dup.textContent = "Duplicate stage";
  dup.style.marginTop = "8px";
  dup.onclick = () => {
    const copy = JSON.parse(JSON.stringify(stage));
    copy.name = (S.model.stages.length + 1) + " - " + copy.name.replace(/^\d+\s*-\s*/, "");
    copy.kind = copy.kind === "initial" ? "plastic" : copy.kind;
    S.model.stages.splice(index + 1, 0, copy);
    S.stage = index + 1; renderStages();
  };
  box.appendChild(dup);
  return box;
}

function checkGroup(title, names, selected, onChange) {
  const wrap = document.createElement("div");
  const chosen = new Set(selected || names);
  wrap.innerHTML = `<div style="font-size:11px;color:var(--muted);margin-top:8px">${title}</div>`;
  const row = document.createElement("div");
  row.className = "checks";
  names.forEach(name => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox"><span></span>`;
    label.querySelector("span").textContent = name;
    const box = label.querySelector("input");
    box.checked = chosen.has(name);
    box.onchange = () => {
      if (box.checked) chosen.add(name); else chosen.delete(name);
      onChange(names.filter(n => chosen.has(n)));
    };
    row.appendChild(label);
  });
  wrap.appendChild(row);
  return wrap;
}

/* ------------------------------------------------------------------ results */
function renderResults() {
  const body = document.getElementById("resultsBody");
  const state = S.results;
  if (!state || !state.stages || !state.stages.length) {
    body.innerHTML = '<div class="note">Run the analysis to see results.</div>';
    return;
  }
  const summary = state.summary;
  let html = "";
  const fos = (summary ? summary.stages : []).map(s => s.factor_of_safety).filter(v => v != null);
  if (fos.length) {
    html += `<div class="fos"><b>${Math.min(...fos).toFixed(3)}</b>
             <span>factor of safety (strength reduction)</span></div>`;
  }

  html += `<div class="field"><label>Stage</label><select id="resStage">` +
    state.stages.map((s, i) => `<option value="${i}">${escapeHtml(s.name)}</option>`).join("") +
    `</select></div>`;
  html += `<div class="field"><label>Plot</label><select id="resKind">
      <option value="field">Result contours</option>
      <option value="plastic">Plastic points</option>
      <option value="deformed">Deformed mesh</option>
      <option value="vectors">Displacement directions</option>
      <option value="ssr">Strength reduction curve</option>
      <option value="forces">Structural forces</option>
      <option value="mesh">Mesh</option>
    </select></div>`;
  html += `<div class="field" id="fieldRow"><label>Quantity</label><select id="resField">` +
    (state.fields || []).map(f => `<option value="${f}">${f.replace(/_/g, " ")}</option>`).join("") +
    `</select></div>`;
  if (state.structures && state.structures.length) {
    html += `<div class="field" id="structRow"><label>Structure</label><select id="resStruct">` +
      state.structures.map(n => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join("") +
      `</select></div>`;
  }


  if (summary) {
    html += `<table><thead><tr><th>Stage</th><th>Result</th><th class="num">u<sub>max</sub> [mm]</th>
             <th class="num">FoS</th></tr></thead><tbody>`;
    summary.stages.forEach(s => {
      html += `<tr><td>${escapeHtml(s.stage)}</td>` +
        `<td class="${s.converged ? "" : "bad"}">${s.converged ? "ok" : "not converged"}</td>` +
        `<td class="num">${s.max_displacement_mm.toFixed(1)}</td>` +
        `<td class="num">${s.factor_of_safety != null ? s.factor_of_safety.toFixed(3) : ""}</td></tr>`;
    });
    html += "</tbody></table>";

    const withForces = summary.stages.filter(s => s.structures);
    if (withForces.length) {
      html += `<h2>Structural forces</h2><table><thead><tr><th>Stage</th><th>Member</th>
               <th class="num">M<sub>max</sub></th><th class="num">N<sub>max</sub></th>
               <th class="num">u [mm]</th></tr></thead><tbody>`;
      withForces.forEach(s => {
        Object.entries(s.structures).forEach(([name, item]) => {
          html += `<tr><td>${escapeHtml(s.stage)}</td><td>${escapeHtml(name)}</td>` +
            `<td class="num">${item.bending_moment_max_kNm_per_m.toFixed(0)}</td>` +
            `<td class="num">${item.axial_force_max_kN_per_m.toFixed(0)}</td>` +
            `<td class="num">${item.max_deflection_mm.toFixed(1)}</td></tr>`;
        });
      });
      html += "</tbody></table>";
    }
    const anchors = summary.stages.filter(s => s.anchor_forces_kN && Object.keys(s.anchor_forces_kN).length);
    if (anchors.length) {
      html += `<h2>Anchor forces [kN]</h2><table><tbody>`;
      anchors.forEach(s => Object.entries(s.anchor_forces_kN).forEach(([n, v]) => {
        html += `<tr><td>${escapeHtml(s.stage)}</td><td>${escapeHtml(n)}</td>
                 <td class="num">${v.toFixed(0)}</td></tr>`;
      }));
      html += "</tbody></table>";
    }
  }
  html += `<button id="btnReport" style="margin-top:10px">Write HTML report</button>`;
  body.innerHTML = html;

  const stageSel = document.getElementById("resStage");
  const kindSel = document.getElementById("resKind");
  const fieldSel = document.getElementById("resField");
  const structSel = document.getElementById("resStruct");
  if (S.resultView.stage === null || S.resultView.stage === undefined ||
      S.resultView.stage >= state.stages.length) {
    // Open on the last stage: it is the one the engineer is looking for.
    S.resultView.stage = state.stages.length - 1;
  }
  stageSel.value = S.resultView.stage;
  kindSel.value = S.resultView.kind;
  if (fieldSel) fieldSel.value = S.resultView.field;
  const refresh = () => {
    S.resultView.stage = parseInt(stageSel.value, 10);
    S.resultView.kind = kindSel.value;
    if (fieldSel) S.resultView.field = fieldSel.value;
    if (structSel) S.resultView.structure = structSel.value;
    document.getElementById("fieldRow").hidden = kindSel.value !== "field";
    if (document.getElementById("structRow"))
      document.getElementById("structRow").hidden = kindSel.value !== "forces";
    const p = new URLSearchParams({
      kind: S.resultView.kind, stage: S.resultView.stage,
      field: S.resultView.field, deformed: "1",
      name: S.resultView.structure, t: Date.now()
    });
    const url = "/api/plot?" + p.toString();
    const big = document.getElementById("bigPlot");
    big.src = url;
    big.hidden = false;
    canvas.hidden = true;
    document.getElementById("hint").textContent =
      "Showing analysis results. Switch to Properties or Stages to edit the model again.";
  };
  [stageSel, kindSel, fieldSel, structSel].forEach(el => { if (el) el.onchange = refresh; });
  refresh();

  document.getElementById("btnReport").onclick = async () => {
    setStatus("writing report…");
    const r = await post("/api/report", { out: "out/report" });
    setStatus(r.error ? r.error : "report written to " + r.path, r.error ? "error" : "ok");
  };
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/* ---------------------------------------------------------------- plumbing */
async function post(url, payload) {
  const response = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  return response.json();
}

function setStatus(message, kind) {
  const el = document.getElementById("statusMsg");
  el.textContent = message;
  el.className = "msg" + (kind ? " " + kind : "");
}

function setProgress(fraction) {
  document.getElementById("progressBar").style.width = Math.round(100 * fraction) + "%";
}

function pushModel() {
  S.model.name = document.getElementById("modelName").value || S.model.name;
  renderTrees(); renderProps(); renderStages(); draw();
}

function readSettings() {
  const size = parseFloat(document.getElementById("meshSize").value);
  S.model.mesh_size = isNaN(size) ? null : size;
  S.model.initial_stress = document.getElementById("initialStress").value;
  S.model.gamma_water = parseFloat(document.getElementById("gammaWater").value) || 9.81;
  S.model.name = document.getElementById("modelName").value || "model";
}

function writeSettings() {
  document.getElementById("modelName").value = S.model.name || "";
  document.getElementById("meshSize").value = S.model.mesh_size ?? "";
  document.getElementById("initialStress").value = S.model.initial_stress || "k0";
  document.getElementById("gammaWater").value = S.model.gamma_water ?? 9.81;
}

function setTool(tool) {
  S.tool = tool;
  S.draft = []; S.hoverPoint = null;
  document.querySelectorAll(".tools button").forEach(b =>
    b.classList.toggle("active", b.dataset.tool === tool));
  canvas.style.cursor = tool === "select" ? "default" : "crosshair";
  const hints = {
    select: "Click to select, drag handles to move points, Delete to remove. Drag the background to pan.",
    layer: "Click the corners of the soil layer. Double-click or Enter to close it.",
    structure: "Click along the line of the wall or pile row, top to bottom. Enter to finish.",
    anchor: "Click the point on the wall, then the far end of the anchor.",
    load: "Click along the loaded line. Enter to finish.",
    water: "Click along the phreatic surface, left to right. Enter to finish."
  };
  document.getElementById("hint").textContent = hints[tool];
  draw();
}

function showPane() {
  const showingResults = S.pane === "results";
  document.getElementById("bigPlot").hidden = !showingResults;
  canvas.hidden = showingResults;
  if (!showingResults) {
    document.getElementById("hint").textContent =
      "Drag to pan, scroll to zoom. Pick a drawing tool, or load an example.";
    draw();
  }
  document.getElementById("paneProps").hidden = S.pane !== "props";
  document.getElementById("paneStages").hidden = S.pane !== "stages";
  document.getElementById("paneResults").hidden = S.pane !== "results";
  document.getElementById("tabProps").classList.toggle("active", S.pane === "props");
  document.getElementById("tabStages").classList.toggle("active", S.pane === "stages");
  document.getElementById("tabResults").classList.toggle("active", S.pane === "results");
  if (S.pane === "stages") renderStages();
  if (S.pane === "results") renderResults();
}

async function pollState() {
  const state = await (await fetch("/api/state")).json();
  S.results = state;
  if (state.status === "running") {
    const fraction = state.stage_count ? state.stage_index / state.stage_count : 0;
    setProgress(fraction);
    setStatus(`running: ${state.stage_name || "…"} (${state.stage_index + 1}/${state.stage_count})`);
  } else if (state.status === "done") {
    clearInterval(S.polling); S.polling = null;
    setProgress(1);
    S.resultView.stage = null;     // open on the final stage
    const failed = state.stages.filter(s => !s.converged);
    setStatus(failed.length
      ? `finished, ${failed.length} stage(s) did not converge`
      : "analysis finished", failed.length ? "error" : "ok");
    document.getElementById("btnRun").disabled = false;
    S.pane = "results"; showPane();
  } else if (state.status === "error") {
    clearInterval(S.polling); S.polling = null;
    setProgress(0);
    setStatus(state.error, "error");
    document.getElementById("btnRun").disabled = false;
  }
}

/* ------------------------------------------------------------------- events */
document.querySelectorAll(".tools button").forEach(b =>
  b.onclick = () => setTool(b.dataset.tool));
document.getElementById("tabProps").onclick = () => { S.pane = "props"; showPane(); };
document.getElementById("tabStages").onclick = () => { S.pane = "stages"; showPane(); };
document.getElementById("tabResults").onclick = () => { S.pane = "results"; showPane(); };
document.getElementById("btnAddStage").onclick = () => {
  ensureStages();
  const last = S.model.stages[S.model.stages.length - 1];
  const copy = JSON.parse(JSON.stringify(last));
  copy.kind = "plastic";
  copy.name = (S.model.stages.length + 1) + " - new step";
  S.model.stages.push(copy);
  S.stage = S.model.stages.length - 1;
  renderStages();
};

document.getElementById("btnMesh").onclick = async () => {
  readSettings();
  setStatus("meshing…");
  const r = await post("/api/mesh", S.model);
  if (r.error || r.ok === false) {
    setStatus(r.error || ("cannot mesh: " + (r.issues || []).join("; ")), "error");
    return;
  }
  setStatus(`mesh: ${r.elements} elements, ${r.nodes} nodes, ${r.dofs} dofs, ` +
            `area ${r.area.toFixed(1)} m²`, "ok");
  S.resultView.kind = "mesh";
  S.resultView.stage = 0;
  S.results = { stages: [{ name: "mesh", kind: "mesh", converged: true }], fields: [], structures: [] };
  S.pane = "results"; showPane();
};

document.getElementById("btnRun").onclick = async () => {
  readSettings(); ensureStages();
  if (!S.model.layers.length) { setStatus("draw at least one soil layer first", "error"); return; }
  document.getElementById("btnRun").disabled = true;
  setProgress(0); setStatus("starting…");
  const r = await post("/api/run", S.model);
  if (r.error || r.ok === false) {
    setStatus(r.error || r.message, "error");
    document.getElementById("btnRun").disabled = false;
    return;
  }
  S.polling = setInterval(pollState, 700);
};

document.getElementById("btnExport").onclick = () => {
  readSettings();
  const blob = new Blob([JSON.stringify(S.model, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (S.model.name || "model").replace(/\s+/g, "_") + ".json";
  a.click();
  setStatus("model saved", "ok");
};

document.getElementById("btnImport").onclick = () => document.getElementById("fileInput").click();
document.getElementById("fileInput").onchange = ev => {
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      loadModel(JSON.parse(reader.result));
      setStatus("loaded " + file.name, "ok");
    } catch (err) { setStatus("could not read that file: " + err.message, "error"); }
  };
  reader.readAsText(file);
};

document.getElementById("exampleSelect").onchange = async ev => {
  if (!ev.target.value) return;
  const model = await (await fetch("/api/example?key=" + ev.target.value)).json();
  loadModel(model);
  setStatus("loaded example: " + ev.target.value, "ok");
  ev.target.value = "";
};

function loadModel(data) {
  S.model = Object.assign(emptyModel(), data);
  S.model.water = S.model.water || { points: [] };
  S.selection = null; S.stage = 0; S.results = null;
  writeSettings(); fitView(); pushModel();
  S.pane = "props"; showPane();
}

window.addEventListener("resize", resize);

(async function init() {
  const list = await (await fetch("/api/examples")).json();
  const sel = document.getElementById("exampleSelect");
  list.examples.forEach(e => {
    const opt = document.createElement("option");
    opt.value = e.key; opt.textContent = e.title;
    sel.appendChild(opt);
  });
  writeSettings();
  setTool("select");
  resize();
  setStatus("Ready. Load an example, or draw a soil layer to begin.");
})();
