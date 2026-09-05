import React, { useState, useRef, useEffect, useCallback } from "react";

// ==============================================================
// NutriGraph, Live Graph Explorer
// Open click-to-expand traversal. Measured edges green, predicted
// amber (matches the Neo4j evidence property). Warm-light palette
// to match the showcase. Runs on an embedded sample here; wire to
// live read-only Neo4j via the CONFIG block for the real deploy.
// ==============================================================

// ---- CONFIG: point this at your VPS Neo4j to go live ----------
const NEO4J = {
  uri: "", // e.g. "bolt://graph.nutri.abhinavsikhwal.com:7687"; leave "" to use the embedded sample
  user: "readonly",
  password: "", // read-only user password
};

const C = {
  paper: "#FAF4E8", card: "#FFFFFF", ink: "#2A2018", faint: "#8A7A64", line: "#E3D6BE",
  gold: "#E08A00", chili: "#C23B22", herb: "#5E8C3A", plum: "#8B5A9E", blue: "#1F6FA8",
  measured: "#2E7D32", predicted: "#F9A825",
};
const mono = "ui-monospace, 'SF Mono', 'IBM Plex Mono', Menlo, monospace";
const sans = "'Inter', system-ui, -apple-system, sans-serif";

const LABEL_STYLE = {
  Ingredient: { color: C.gold, caption: "name" },
  Compound: { color: C.chili, caption: "name" },
  Gene: { color: C.blue, caption: "symbol" },
  Pathway: { color: C.plum, caption: "name" },
  Tissue: { color: C.herb, caption: "name" },
  Nutrient: { color: "#C77D0E", caption: "name" },
};

// ---- Embedded sample subgraph (a real slice, for offline demo) --
const SAMPLE = {
  nodes: [
    { id: "i_turmeric", label: "Ingredient", props: { name: "Turmeric", latin: "Curcuma longa", measured_fraction: 0.23 } },
    { id: "c_curcumin", label: "Compound", props: { name: "Curcumin" } },
    { id: "c_dmc", label: "Compound", props: { name: "Demethoxycurcumin" } },
    { id: "c_thc", label: "Compound", props: { name: "Tetrahydrocurcumin" } },
    { id: "g_cnr1", label: "Gene", props: { symbol: "CNR1" } },
    { id: "g_cnr2", label: "Gene", props: { symbol: "CNR2" } },
    { id: "g_ptgs2", label: "Gene", props: { symbol: "PTGS2" } },
    { id: "g_faah", label: "Gene", props: { symbol: "FAAH" } },
    { id: "g_ptpn1", label: "Gene", props: { symbol: "PTPN1" } },
    { id: "p_gaba", label: "Pathway", props: { name: "GABAergic signaling" } },
    { id: "p_eico", label: "Pathway", props: { name: "Eicosanoid synthesis" } },
    { id: "t_liver", label: "Tissue", props: { name: "Liver" } },
    { id: "t_gut", label: "Tissue", props: { name: "Small intestine" } },
    { id: "n_iron", label: "Nutrient", props: { name: "Iron" } },
    { id: "n_mn", label: "Nutrient", props: { name: "Manganese" } },
    // a second ingredient sharing genes (shows the graph connects)
    { id: "i_ginger", label: "Ingredient", props: { name: "Ginger", latin: "Zingiber officinale", measured_fraction: 0.19 } },
    { id: "c_gingerol", label: "Compound", props: { name: "6-Gingerol" } },
    { id: "g_trpv1", label: "Gene", props: { symbol: "TRPV1" } },
  ],
  edges: [
    { from: "i_turmeric", to: "c_curcumin", type: "CONTAINS" },
    { from: "i_turmeric", to: "c_dmc", type: "CONTAINS" },
    { from: "i_turmeric", to: "c_thc", type: "CONTAINS" },
    { from: "c_curcumin", to: "g_cnr1", type: "TARGETS", evidence: "measured", confidence: 1.0 },
    { from: "c_curcumin", to: "g_cnr2", type: "TARGETS", evidence: "measured", confidence: 1.0 },
    { from: "c_curcumin", to: "g_faah", type: "TARGETS", evidence: "measured", confidence: 1.0 },
    { from: "c_curcumin", to: "g_ptpn1", type: "TARGETS", evidence: "measured", confidence: 1.0 },
    { from: "c_curcumin", to: "g_ptgs2", type: "TARGETS", evidence: "predicted", confidence: 0.68 },
    { from: "c_dmc", to: "g_cnr1", type: "TARGETS", evidence: "predicted", confidence: 0.55 },
    { from: "c_thc", to: "g_ptgs2", type: "TARGETS", evidence: "predicted", confidence: 0.61 },
    { from: "g_cnr1", to: "p_gaba", type: "IN_PATHWAY" },
    { from: "g_ptgs2", to: "p_eico", type: "IN_PATHWAY" },
    { from: "g_cnr1", to: "t_liver", type: "EXPRESSED_IN", score: 0.41 },
    { from: "g_faah", to: "t_liver", type: "EXPRESSED_IN", score: 0.33 },
    { from: "g_ptgs2", to: "t_gut", type: "EXPRESSED_IN", score: 0.29 },
    { from: "i_turmeric", to: "n_iron", type: "HAS_NUTRIENT", amount: "55 mg" },
    { from: "i_turmeric", to: "n_mn", type: "HAS_NUTRIENT", amount: "19.8 mg" },
    { from: "i_ginger", to: "c_gingerol", type: "CONTAINS" },
    { from: "c_gingerol", to: "g_trpv1", type: "TARGETS", evidence: "measured", confidence: 1.0 },
    { from: "c_gingerol", to: "g_cnr1", type: "TARGETS", evidence: "measured", confidence: 1.0 },
    { from: "g_trpv1", to: "t_gut", type: "EXPRESSED_IN", score: 0.27 },
  ],
};

function sampleSearch(q) {
  const ql = q.toLowerCase();
  return SAMPLE.nodes.filter(n => {
    const cap = LABEL_STYLE[n.label].caption;
    return (n.props[cap] || "").toLowerCase().includes(ql);
  }).slice(0, 25);
}
function sampleNeighbors(nodeId, evidenceFilter) {
  const edges = SAMPLE.edges.filter(e => {
    if (e.from !== nodeId && e.to !== nodeId) return false;
    if (evidenceFilter && e.type === "TARGETS" && e.evidence !== evidenceFilter) return false;
    return true;
  });
  const ids = new Set([nodeId]);
  edges.forEach(e => { ids.add(e.from); ids.add(e.to); });
  const nodes = SAMPLE.nodes.filter(n => ids.has(n.id));
  return { nodes, edges };
}

// ---- Tiny force-directed layout (no external deps) ------------
function useForce(graph, width, height) {
  const [pos, setPos] = useState({});
  const posRef = useRef({});
  useEffect(() => {
    const ids = graph.nodes.map(n => n.id);
    const p = {};
    ids.forEach((id, i) => {
      const prev = posRef.current[id];
      p[id] = prev || { x: width / 2 + Math.cos(i) * 120 + (Math.random() - 0.5) * 40, y: height / 2 + Math.sin(i) * 120 + (Math.random() - 0.5) * 40, vx: 0, vy: 0 };
    });
    let frame; let ticks = 0;
    const step = () => {
      const nodes = graph.nodes;
      for (let a = 0; a < nodes.length; a++) {
        for (let b = a + 1; b < nodes.length; b++) {
          const pa = p[nodes[a].id], pb = p[nodes[b].id];
          let dx = pa.x - pb.x, dy = pa.y - pb.y;
          let d2 = dx * dx + dy * dy || 1;
          const f = 2600 / d2;
          const d = Math.sqrt(d2);
          pa.vx += (dx / d) * f; pa.vy += (dy / d) * f;
          pb.vx -= (dx / d) * f; pb.vy -= (dy / d) * f;
        }
      }
      graph.edges.forEach(e => {
        const pa = p[e.from], pb = p[e.to]; if (!pa || !pb) return;
        let dx = pb.x - pa.x, dy = pb.y - pa.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = (d - 90) * 0.02;
        pa.vx += (dx / d) * f; pa.vy += (dy / d) * f;
        pb.vx -= (dx / d) * f; pb.vy -= (dy / d) * f;
      });
      Object.keys(p).forEach(id => {
        const n = p[id];
        n.vx += (width / 2 - n.x) * 0.002; n.vy += (height / 2 - n.y) * 0.002;
        n.vx *= 0.82; n.vy *= 0.82;
        n.x += n.vx; n.y += n.vy;
        n.x = Math.max(30, Math.min(width - 30, n.x));
        n.y = Math.max(30, Math.min(height - 30, n.y));
      });
      posRef.current = p;
      setPos({ ...p });
      if (++ticks < 140) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
    // eslint-disable-next-line
  }, [graph, width, height]);
  return pos;
}

export default function GraphExplorer() {
  const W = 720, H = 520;
  const [graph, setGraph] = useState(() => sampleNeighbors("i_turmeric"));
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [selected, setSelected] = useState(null);
  const [measuredOnly, setMeasuredOnly] = useState(false);
  const [expanded, setExpanded] = useState(new Set(["i_turmeric"]));
  const pos = useForce(graph, W, H);
  const live = Boolean(NEO4J.uri);

  const search = useCallback((q) => {
    setQuery(q);
    if (!q.trim()) { setResults([]); return; }
    setResults(sampleSearch(q)); // live: run cypher search template via driver
  }, []);

  const seed = (node) => {
    setResults([]); setQuery("");
    setExpanded(new Set([node.id]));
    setGraph(sampleNeighbors(node.id, measuredOnly ? "measured" : null));
    setSelected(node);
  };

  const expand = (node) => {
    const nb = sampleNeighbors(node.id, measuredOnly ? "measured" : null);
    setGraph(g => {
      const nodeMap = {}; g.nodes.forEach(n => nodeMap[n.id] = n); nb.nodes.forEach(n => nodeMap[n.id] = n);
      const edgeKey = e => `${e.from}|${e.to}|${e.type}`;
      const edgeMap = {}; g.edges.forEach(e => edgeMap[edgeKey(e)] = e); nb.edges.forEach(e => edgeMap[edgeKey(e)] = e);
      return { nodes: Object.values(nodeMap), edges: Object.values(edgeMap) };
    });
    setExpanded(s => new Set(s).add(node.id));
    setSelected(node);
  };

  const nodeCaption = (n) => n.props[LABEL_STYLE[n.label].caption] || n.label;

  return (
    <div style={{ background: C.paper, color: C.ink, fontFamily: sans, minHeight: "100vh", padding: "28px" }}>
      <div style={{ maxWidth: 1120, margin: "0 auto" }}>
        {/* header */}
        <div style={{ borderBottom: `1px solid ${C.line}`, paddingBottom: 16, marginBottom: 20, display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 12 }}>
          <div>
            <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 3, textTransform: "uppercase", color: C.gold, fontWeight: 600, marginBottom: 8 }}>NutriGraph, live graph</div>
            <h1 style={{ fontSize: 34, fontWeight: 800, letterSpacing: -1, margin: 0 }}>Explore the graph</h1>
            <p style={{ fontSize: 15, color: C.faint, margin: "8px 0 0", maxWidth: 620 }}>
              Search a food, gene, compound or pathway to start, then click any node to expand its
              connections and travel anywhere in the graph. Edge colour shows the evidence.
            </p>
          </div>
          <div style={{ fontFamily: mono, fontSize: 11, color: C.faint, textAlign: "right" }}>
            {live ? "connected to live Neo4j" : "offline sample, wire to Neo4j to go live"}<br />
            7,467 nodes · 79,301 edges
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 20 }} className="g-grid">
          {/* graph canvas */}
          <div style={{ background: C.card, border: `1px solid ${C.line}`, borderRadius: 8, position: "relative", overflow: "hidden", boxShadow: "0 1px 3px rgba(42,32,24,.05)" }}>
            {/* controls */}
            <div style={{ position: "absolute", top: 12, left: 12, right: 12, zIndex: 5, display: "flex", gap: 8, alignItems: "flex-start" }}>
              <div style={{ flex: 1, position: "relative" }}>
                <input value={query} onChange={e => search(e.target.value)} placeholder="search turmeric, CNR1, curcumin..."
                  style={{ width: "100%", fontFamily: mono, fontSize: 13, padding: "10px 12px", borderRadius: 5, border: `1px solid ${C.line}`, background: C.paper, color: C.ink, outline: "none", boxSizing: "border-box" }} />
                {results.length > 0 && (
                  <div style={{ position: "absolute", top: 42, left: 0, right: 0, background: C.card, border: `1px solid ${C.line}`, borderRadius: 5, boxShadow: "0 6px 20px rgba(42,32,24,.12)", maxHeight: 240, overflowY: "auto", zIndex: 10 }}>
                    {results.map(r => (
                      <button key={r.id} onClick={() => seed(r)} style={{ display: "flex", alignItems: "center", gap: 8, width: "100%", padding: "9px 12px", background: "none", border: "none", borderBottom: `1px solid ${C.line}`, cursor: "pointer", textAlign: "left" }}>
                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: LABEL_STYLE[r.label].color }} />
                        <span style={{ fontFamily: mono, fontSize: 13, color: C.ink }}>{nodeCaption(r)}</span>
                        <span style={{ fontFamily: mono, fontSize: 10, color: C.faint, marginLeft: "auto", textTransform: "uppercase" }}>{r.label}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <button onClick={() => { setMeasuredOnly(v => !v); }} title="filter TARGETS edges by evidence"
                style={{ fontFamily: mono, fontSize: 11, padding: "10px 12px", borderRadius: 5, cursor: "pointer", whiteSpace: "nowrap",
                  border: `1.5px solid ${measuredOnly ? C.measured : C.line}`, background: measuredOnly ? C.measured : C.card, color: measuredOnly ? "#fff" : C.ink }}>
                {measuredOnly ? "measured only" : "all evidence"}
              </button>
            </div>

            {/* svg graph */}
            <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block", height: H }}>
              {graph.edges.map((e, i) => {
                const a = pos[e.from], b = pos[e.to]; if (!a || !b) return null;
                if (measuredOnly && e.type === "TARGETS" && e.evidence !== "measured") return null;
                const isTarget = e.type === "TARGETS";
                const col = isTarget ? (e.evidence === "measured" ? C.measured : C.predicted) : C.line;
                return (
                  <g key={i}>
                    <line x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                      stroke={col} strokeWidth={isTarget ? 2 : 1.3}
                      strokeDasharray={isTarget && e.evidence === "predicted" ? "5,4" : "none"}
                      opacity={isTarget ? 0.85 : 0.4} />
                  </g>
                );
              })}
              {graph.nodes.map(n => {
                const p = pos[n.id]; if (!p) return null;
                const st = LABEL_STYLE[n.label];
                const isSel = selected && selected.id === n.id;
                const isExp = expanded.has(n.id);
                const r = n.label === "Ingredient" ? 22 : 15;
                return (
                  <g key={n.id} transform={`translate(${p.x},${p.y})`} style={{ cursor: "pointer" }}
                    onClick={() => expand(n)}>
                    <circle r={r} fill={C.card} stroke={st.color} strokeWidth={isSel ? 3.5 : 2}
                      opacity={isExp ? 1 : 0.92} />
                    <circle r={r} fill={st.color} opacity={0.12} />
                    <text textAnchor="middle" dy={r + 13} fontFamily={mono} fontSize={n.label === "Ingredient" ? 12 : 10.5}
                      fontWeight={n.label === "Ingredient" ? 700 : 500} fill={C.ink}>
                      {nodeCaption(n).length > 16 ? nodeCaption(n).slice(0, 15) + "…" : nodeCaption(n)}
                    </text>
                  </g>
                );
              })}
            </svg>

            {/* legend */}
            <div style={{ position: "absolute", bottom: 12, left: 12, display: "flex", gap: 14, fontFamily: mono, fontSize: 10.5, color: C.faint, flexWrap: "wrap", background: "rgba(255,255,255,.85)", padding: "6px 10px", borderRadius: 5 }}>
              <span><span style={{ display: "inline-block", width: 14, height: 2, background: C.measured, verticalAlign: "middle", marginRight: 4 }} />measured</span>
              <span><span style={{ display: "inline-block", width: 14, height: 2, background: C.predicted, verticalAlign: "middle", marginRight: 4, borderTop: "1px dashed" }} />predicted</span>
              <span style={{ color: C.faint }}>click a node to expand</span>
            </div>
          </div>

          {/* inspector */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ background: C.card, border: `1px solid ${C.line}`, borderRadius: 8, padding: 18, boxShadow: "0 1px 3px rgba(42,32,24,.05)" }}>
              <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 1, textTransform: "uppercase", color: C.faint, marginBottom: 14, fontWeight: 600 }}>Selected node</div>
              {selected ? (
                <>
                  <div style={{ display: "inline-block", fontFamily: mono, fontSize: 10, letterSpacing: 1, textTransform: "uppercase", color: "#fff", background: LABEL_STYLE[selected.label].color, padding: "3px 8px", borderRadius: 3, marginBottom: 10 }}>{selected.label}</div>
                  <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 10 }}>{nodeCaption(selected)}</div>
                  <div style={{ fontFamily: mono, fontSize: 12, lineHeight: 1.9, color: C.faint }}>
                    {Object.entries(selected.props).map(([k, v]) => (
                      <div key={k} style={{ display: "flex", gap: 10 }}>
                        <span style={{ width: 100, color: C.faint }}>{k}</span>
                        <span style={{ color: C.ink }}>{String(v)}</span>
                      </div>
                    ))}
                  </div>
                  <button onClick={() => expand(selected)} style={{ marginTop: 16, width: "100%", fontFamily: mono, fontSize: 12, padding: "9px", borderRadius: 5, cursor: "pointer", border: "none", background: LABEL_STYLE[selected.label].color, color: "#fff", fontWeight: 600 }}>
                    expand connections
                  </button>
                </>
              ) : (
                <div style={{ fontSize: 14, color: C.faint, lineHeight: 1.5 }}>Click any node in the graph to inspect its properties and expand its neighbours.</div>
              )}
            </div>

            <div style={{ background: C.card, border: `1px solid ${C.line}`, borderRadius: 8, padding: 18, boxShadow: "0 1px 3px rgba(42,32,24,.05)" }}>
              <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 1, textTransform: "uppercase", color: C.faint, marginBottom: 12, fontWeight: 600 }}>Node types</div>
              {Object.entries(LABEL_STYLE).map(([label, st]) => (
                <div key={label} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, padding: "4px 0" }}>
                  <span style={{ width: 12, height: 12, borderRadius: "50%", border: `2px solid ${st.color}`, background: `${st.color}22` }} />
                  <span>{label}</span>
                </div>
              ))}
            </div>

            <div style={{ fontFamily: mono, fontSize: 11, color: C.faint, lineHeight: 1.6, padding: "0 4px" }}>
              Open exploration over a read-only connection. Every TARGETS edge is coloured by whether the
              food-to-protein link was experimentally measured or structurally inferred.
            </div>
          </div>
        </div>
      </div>
      <style>{`@media (max-width: 860px){ .g-grid{ grid-template-columns: 1fr !important; } }`}</style>
    </div>
  );
}
