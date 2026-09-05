import React, { useState, useRef, useEffect, useCallback } from "react";

// ==============================================================
// NutriGraph, Live Graph Explorer (API-backed)
// Talks to the thin read-only API (which talks to Neo4j).
// Set API_BASE to your deployed API. Open click-to-expand
// traversal, measured edges green, predicted amber.
// ==============================================================

// ---- point this at your deployed read-only API ----------------
// e.g. "https://graph.nutri.abhinavsikhwal.com/api"
const API_BASE = "/api";

const C = {
  paper: "#FAF4E8", card: "#FFFFFF", ink: "#2A2018", faint: "#8A7A64", line: "#E3D6BE",
  gold: "#E08A00", chili: "#C23B22", herb: "#5E8C3A", plum: "#8B5A9E", blue: "#1F6FA8",
  measured: "#2E7D32", predicted: "#F9A825", nutrient: "#C77D0E",
};
const mono = "ui-monospace, 'SF Mono', 'IBM Plex Mono', Menlo, monospace";
const sans = "'Inter', system-ui, -apple-system, sans-serif";

const LABEL_STYLE = {
  Ingredient: { color: C.gold, caption: "name" },
  Compound: { color: C.chili, caption: "name" },
  Gene: { color: C.blue, caption: "symbol" },
  Pathway: { color: C.plum, caption: "name" },
  Tissue: { color: C.herb, caption: "name" },
  Nutrient: { color: C.nutrient, caption: "name" },
};
const nodeCaption = (n) => {
  const cap = LABEL_STYLE[n.label]?.caption || "name";
  return n.props?.[cap] || n.props?.name || n.props?.symbol || n.label;
};

// ---- API calls ------------------------------------------------
async function apiSearch(q) {
  const r = await fetch(`${API_BASE}/search?q=${encodeURIComponent(q)}`);
  if (!r.ok) throw new Error("search failed");
  const d = await r.json();
  return d.nodes || [];
}
async function apiExpand(elementId, evidence) {
  let path = `${API_BASE}/expand?id=${encodeURIComponent(elementId)}`;
  if (evidence) path += `&evidence=${encodeURIComponent(evidence)}`;
  const r = await fetch(path);
  if (!r.ok) throw new Error("expand failed");
  return await r.json(); // {nodes, edges}
}

// ---- force layout (no deps) -----------------------------------
function useForce(graph, width, height) {
  const [pos, setPos] = useState({});
  const posRef = useRef({});
  useEffect(() => {
    const p = {};
    graph.nodes.forEach((n, i) => {
      const prev = posRef.current[n.id];
      p[n.id] = prev || { x: width / 2 + Math.cos(i) * 130 + (Math.random() - 0.5) * 40, y: height / 2 + Math.sin(i) * 130 + (Math.random() - 0.5) * 40, vx: 0, vy: 0 };
    });
    let frame, ticks = 0;
    const step = () => {
      const nodes = graph.nodes;
      for (let a = 0; a < nodes.length; a++) {
        for (let b = a + 1; b < nodes.length; b++) {
          const pa = p[nodes[a].id], pb = p[nodes[b].id];
          let dx = pa.x - pb.x, dy = pa.y - pb.y;
          let d2 = dx * dx + dy * dy || 1;
          const d = Math.sqrt(d2), f = 6000 / d2;
          pa.vx += (dx / d) * f; pa.vy += (dy / d) * f;
          pb.vx -= (dx / d) * f; pb.vy -= (dy / d) * f;
        }
      }
      graph.edges.forEach(e => {
        const pa = p[e.from], pb = p[e.to]; if (!pa || !pb) return;
        let dx = pb.x - pa.x, dy = pb.y - pa.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1, f = (d - 120) * 0.015;
        pa.vx += (dx / d) * f; pa.vy += (dy / d) * f;
        pb.vx -= (dx / d) * f; pb.vy -= (dy / d) * f;
      });
      Object.keys(p).forEach(id => {
        const n = p[id];
        n.vx += (width / 2 - n.x) * 0.0015; n.vy += (height / 2 - n.y) * 0.0015;
        n.vx *= 0.84; n.vy *= 0.84; n.x += n.vx; n.y += n.vy;
        n.x = Math.max(40, Math.min(width - 40, n.x));
        n.y = Math.max(40, Math.min(height - 40, n.y));
      });
      posRef.current = p; setPos({ ...p });
      if (++ticks < 180) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
    // eslint-disable-next-line
  }, [graph, width, height]);
  return pos;
}

function mergeGraph(prev, add) {
  const nodeMap = {}; prev.nodes.forEach(n => nodeMap[n.id] = n); add.nodes.forEach(n => nodeMap[n.id] = n);
  const edgeMap = {}; prev.edges.forEach(e => edgeMap[e.id] = e); (add.edges || []).forEach(e => edgeMap[e.id] = e);
  return { nodes: Object.values(nodeMap), edges: Object.values(edgeMap) };
}

export default function GraphExplorer() {
  const W = 780, H = 600;
  const [graph, setGraph] = useState({ nodes: [], edges: [] });
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [selected, setSelected] = useState(null);
  const [hover, setHover] = useState(null);
  const [measuredOnly, setMeasuredOnly] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [showWelcome, setShowWelcome] = useState(true);
  const [showAbout, setShowAbout] = useState(false);
  const pos = useForce(graph, W, H);

  // seed with turmeric on first load
  useEffect(() => {
    (async () => {
      try {
        const found = await apiSearch("turmeric");
        if (found[0]) { await seed(found[0]); }
      } catch (e) { setErr("Could not reach the graph API. Is the API service running?"); }
    })();
    // eslint-disable-next-line
  }, []);

  const search = useCallback(async (q) => {
    setQuery(q);
    if (!q.trim()) { setResults([]); return; }
    try { setResults(await apiSearch(q)); setErr(""); }
    catch (e) { setErr("Search failed."); }
  }, []);

  async function seed(node) {
    setResults([]); setQuery(""); setLoading(true); setErr("");
    try {
      const nb = await apiExpand(node.id, measuredOnly ? "measured" : null);
      const withSeed = { nodes: [node, ...(nb.nodes || [])], edges: nb.edges || [] };
      setGraph(withSeed); setSelected(node);
    } catch (e) { setErr("Could not load node."); }
    finally { setLoading(false); }
  }

  async function expand(node) {
    setLoading(true); setErr("");
    try {
      const nb = await apiExpand(node.id, measuredOnly ? "measured" : null);
      setGraph(g => mergeGraph(g, nb)); setSelected(node);
    } catch (e) { setErr("Expand failed."); }
    finally { setLoading(false); }
  }

  return (
    <div style={{ background: C.paper, color: C.ink, fontFamily: sans, minHeight: "100vh", padding: "28px" }}>
      <div style={{ maxWidth: 1120, margin: "0 auto" }}>

        {/* Welcome card, first-load onboarding */}
        {showWelcome && (
          <div style={{ background: C.card, border: `1px solid ${C.line}`, borderLeft: `4px solid ${C.gold}`, borderRadius: 8, padding: "18px 22px", marginBottom: 20, boxShadow: "0 2px 8px rgba(42,32,24,.06)", display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 20, flexWrap: "wrap" }}>
            <div style={{ flex: 1, minWidth: 260 }}>
              <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 2, textTransform: "uppercase", color: C.gold, fontWeight: 600, marginBottom: 8 }}>How to explore</div>
              <div style={{ fontSize: 14.5, lineHeight: 1.6, color: C.ink }}>
                This is a live <b>knowledge graph</b>: foods, their molecules, the human genes and tissues they touch, and the pathways involved.
                <div style={{ display: "flex", gap: 20, marginTop: 12, flexWrap: "wrap" }}>
                  <span><b style={{ color: C.gold }}>1.</b> Search a food, gene or compound above</span>
                  <span><b style={{ color: C.gold }}>2.</b> Click any circle to grow its connections</span>
                  <span><b style={{ color: C.gold }}>3.</b> <span style={{ color: C.measured }}>Green</span> = measured, <span style={{ color: C.predicted }}>amber</span> = predicted</span>
                </div>
                <div style={{ fontSize: 13, color: C.faint, marginTop: 10 }}>Try searching <button onClick={() => { setShowWelcome(false); search("turmeric"); }} style={{ background: "none", border: "none", color: C.blue, cursor: "pointer", padding: 0, fontFamily: mono, fontSize: 13, textDecoration: "underline" }}>turmeric</button>, <button onClick={() => { setShowWelcome(false); search("CNR1"); }} style={{ background: "none", border: "none", color: C.blue, cursor: "pointer", padding: 0, fontFamily: mono, fontSize: 13, textDecoration: "underline" }}>CNR1</button>, or <button onClick={() => { setShowWelcome(false); search("curcumin"); }} style={{ background: "none", border: "none", color: C.blue, cursor: "pointer", padding: 0, fontFamily: mono, fontSize: 13, textDecoration: "underline" }}>curcumin</button>.</div>
              </div>
            </div>
            <button onClick={() => setShowWelcome(false)} style={{ fontFamily: mono, fontSize: 12, padding: "8px 14px", borderRadius: 4, cursor: "pointer", border: `1px solid ${C.line}`, background: C.paper, color: C.ink, whiteSpace: "nowrap" }}>got it</button>
          </div>
        )}

        {/* About modal */}
        {showAbout && (
          <div onClick={() => setShowAbout(false)} style={{ position: "fixed", inset: 0, background: "rgba(42,32,24,.4)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 60, padding: 20 }}>
            <div onClick={e => e.stopPropagation()} style={{ background: C.card, borderRadius: 10, maxWidth: 560, padding: 28, boxShadow: "0 20px 60px rgba(42,32,24,.25)" }}>
              <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 2, textTransform: "uppercase", color: C.gold, fontWeight: 600, marginBottom: 10 }}>What am I looking at?</div>
              <h2 style={{ fontSize: 24, fontWeight: 800, margin: "0 0 14px" }}>A living map of food and biology</h2>
              <div style={{ fontSize: 14.5, lineHeight: 1.65, color: C.ink, display: "grid", gap: 12 }}>
                <p style={{ margin: 0 }}>Each <b>circle is an entity</b>, a food, a molecule inside it, a human gene, a biological pathway, a tissue, or a nutrient. Each <b>line is a real relationship</b> between them, drawn live from a Neo4j graph database.</p>
                <p style={{ margin: 0 }}>The chain runs: <b>food → contains → compound → targets → gene → in pathway / expressed in tissue.</b> Following it shows how something you eat reaches your biology.</p>
                <p style={{ margin: 0 }}>Colour carries the honesty. On the <b>targets</b> links, <span style={{ color: C.measured, fontWeight: 600 }}>green</span> means the food-to-protein interaction was experimentally measured; <span style={{ color: C.predicted, fontWeight: 600 }}>amber</span> means it was predicted from molecular structure. Most of biology here is predicted, and the graph never hides which is which.</p>
                <p style={{ margin: 0, color: C.faint, fontSize: 13 }}>You are exploring over a read-only connection, nothing you do can change the data.</p>
              </div>
              <button onClick={() => setShowAbout(false)} style={{ marginTop: 20, fontFamily: mono, fontSize: 12, padding: "9px 16px", borderRadius: 4, cursor: "pointer", border: "none", background: C.ink, color: C.paper, fontWeight: 600 }}>start exploring</button>
            </div>
          </div>
        )}

        <div style={{ borderBottom: `1px solid ${C.line}`, paddingBottom: 16, marginBottom: 20, display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 12 }}>
          <div>
            <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 3, textTransform: "uppercase", color: C.gold, fontWeight: 600, marginBottom: 8 }}>NutriGraph, live graph</div>
            <h1 style={{ fontSize: 34, fontWeight: 800, letterSpacing: -1, margin: 0 }}>Explore the graph</h1>
            <p style={{ fontSize: 15, color: C.faint, margin: "8px 0 0", maxWidth: 640 }}>
              Search a food, gene, compound or pathway, then click any node to expand its connections and
              travel anywhere in the graph. Edge colour shows the evidence.
            </p>
          </div>
          <div style={{ fontFamily: mono, fontSize: 11, color: C.faint, textAlign: "right" }}>
            <button onClick={() => setShowAbout(true)} style={{ fontFamily: mono, fontSize: 11, color: C.blue, background: "none", border: "none", cursor: "pointer", padding: 0, textDecoration: "underline", marginBottom: 6 }}>
              what am I looking at?
            </button>
            <br />live, backed by Neo4j<br />7,467 nodes · 79,301 edges
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 20 }} className="g-grid">
          <div style={{ background: C.card, border: `1px solid ${C.line}`, borderRadius: 8, position: "relative", overflow: "hidden", boxShadow: "0 1px 3px rgba(42,32,24,.05)" }}>
            <div style={{ position: "absolute", top: 12, left: 12, right: 12, zIndex: 5, display: "flex", gap: 8, alignItems: "flex-start" }}>
              <div style={{ flex: 1, position: "relative" }}>
                <input value={query} onChange={e => search(e.target.value)} placeholder="search turmeric, CNR1, curcumin..."
                  style={{ width: "100%", fontFamily: mono, fontSize: 13, padding: "10px 12px", borderRadius: 5, border: `1px solid ${C.line}`, background: C.paper, color: C.ink, outline: "none", boxSizing: "border-box" }} />
                {results.length > 0 && (
                  <div style={{ position: "absolute", top: 42, left: 0, right: 0, background: C.card, border: `1px solid ${C.line}`, borderRadius: 5, boxShadow: "0 6px 20px rgba(42,32,24,.12)", maxHeight: 240, overflowY: "auto", zIndex: 10 }}>
                    {results.map(r => (
                      <button key={r.id} onClick={() => seed(r)} style={{ display: "flex", alignItems: "center", gap: 8, width: "100%", padding: "9px 12px", background: "none", border: "none", borderBottom: `1px solid ${C.line}`, cursor: "pointer", textAlign: "left" }}>
                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: LABEL_STYLE[r.label]?.color || C.faint }} />
                        <span style={{ fontFamily: mono, fontSize: 13, color: C.ink }}>{nodeCaption(r)}</span>
                        <span style={{ fontFamily: mono, fontSize: 10, color: C.faint, marginLeft: "auto", textTransform: "uppercase" }}>{r.label}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <button onClick={() => setMeasuredOnly(v => !v)} title="filter TARGETS edges by evidence"
                style={{ fontFamily: mono, fontSize: 11, padding: "10px 12px", borderRadius: 5, cursor: "pointer", whiteSpace: "nowrap",
                  border: `1.5px solid ${measuredOnly ? C.measured : C.line}`, background: measuredOnly ? C.measured : C.card, color: measuredOnly ? "#fff" : C.ink }}>
                {measuredOnly ? "measured only" : "all evidence"}
              </button>
            </div>

            <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block", height: H }}>
              {graph.edges.map((e) => {
                const a = pos[e.from], b = pos[e.to]; if (!a || !b) return null;
                if (measuredOnly && e.type === "TARGETS" && e.props?.evidence !== "measured") return null;
                const isTarget = e.type === "TARGETS";
                const col = isTarget ? (e.props?.evidence === "measured" ? C.measured : C.predicted) : C.line;
                return <line key={e.id} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke={col} strokeWidth={isTarget ? 2 : 1.3}
                  strokeDasharray={isTarget && e.props?.evidence === "predicted" ? "5,4" : "none"}
                  opacity={isTarget ? 0.85 : 0.4} />;
              })}
              {graph.nodes.map(n => {
                const p = pos[n.id]; if (!p) return null;
                const st = LABEL_STYLE[n.label] || { color: C.faint };
                const isSel = selected && selected.id === n.id;
                const isHover = hover === n.id;
                const r = n.label === "Ingredient" ? 20 : 12;
                const cap = nodeCaption(n);
                // Show a label if: it's an ingredient, gene, pathway, tissue, nutrient
                // (the interesting nodes), OR it's selected/hovered. Compounds only
                // label on select/hover to avoid the wall-of-text overlap.
                const alwaysLabel = n.label !== "Compound";
                const showLabel = alwaysLabel || isSel || isHover;
                const short = cap.length > 14 ? cap.slice(0, 13) + "\u2026" : cap;
                return (
                  <g key={n.id} transform={`translate(${p.x},${p.y})`} style={{ cursor: "pointer" }}
                    onClick={() => expand(n)}
                    onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(h => h === n.id ? null : h)}>
                    <circle r={r} fill={C.card} stroke={st.color} strokeWidth={isSel ? 3.5 : (isHover ? 2.5 : 1.8)} />
                    <circle r={r} fill={st.color} opacity={isHover || isSel ? 0.28 : 0.14} />
                    {showLabel && (
                      <text textAnchor="middle" dy={r + 12} fontFamily={mono}
                        fontSize={n.label === "Ingredient" ? 11.5 : 10}
                        fontWeight={n.label === "Ingredient" ? 700 : (isSel || isHover ? 700 : 400)}
                        fill={isSel || isHover ? C.ink : C.faint}
                        style={{ pointerEvents: "none" }}>
                        {short}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>

            {loading && <div style={{ position: "absolute", top: 60, right: 16, fontFamily: mono, fontSize: 11, color: C.faint }}>loading...</div>}
            {err && <div style={{ position: "absolute", top: 60, left: 16, right: 16, fontFamily: mono, fontSize: 12, color: C.chili, background: "rgba(255,255,255,.9)", padding: "8px 10px", borderRadius: 5 }}>{err}</div>}

            <div style={{ position: "absolute", bottom: 12, left: 12, right: 12, display: "flex", gap: 16, fontFamily: mono, fontSize: 10.5, color: C.faint, flexWrap: "wrap", alignItems: "center", background: "rgba(255,255,255,.9)", padding: "7px 12px", borderRadius: 5, border: `1px solid ${C.line}` }}>
              <span style={{ fontWeight: 600, color: C.ink }}>click a node to grow it</span>
              <span style={{ color: C.line }}>|</span>
              <span><span style={{ display: "inline-block", width: 14, height: 2, background: C.measured, verticalAlign: "middle", marginRight: 4 }} />measured</span>
              <span><span style={{ display: "inline-block", width: 14, height: 0, borderTop: `2px dashed ${C.predicted}`, verticalAlign: "middle", marginRight: 4 }} />predicted</span>
              <span style={{ color: C.line }}>|</span>
              <span>hover a molecule to name it</span>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ background: C.card, border: `1px solid ${C.line}`, borderRadius: 8, padding: 18, boxShadow: "0 1px 3px rgba(42,32,24,.05)" }}>
              <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 1, textTransform: "uppercase", color: C.faint, marginBottom: 14, fontWeight: 600 }}>Selected node</div>
              {selected ? (
                <>
                  <div style={{ display: "inline-block", fontFamily: mono, fontSize: 10, letterSpacing: 1, textTransform: "uppercase", color: "#fff", background: LABEL_STYLE[selected.label]?.color || C.faint, padding: "3px 8px", borderRadius: 3, marginBottom: 10 }}>{selected.label}</div>
                  <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 10 }}>{nodeCaption(selected)}</div>
                  <div style={{ fontFamily: mono, fontSize: 12, lineHeight: 1.9, color: C.faint }}>
                    {Object.entries(selected.props || {}).map(([k, v]) => (
                      <div key={k} style={{ display: "flex", gap: 10 }}>
                        <span style={{ width: 100, color: C.faint }}>{k}</span>
                        <span style={{ color: C.ink, wordBreak: "break-word" }}>{String(v)}</span>
                      </div>
                    ))}
                  </div>
                  <button onClick={() => expand(selected)} style={{ marginTop: 16, width: "100%", fontFamily: mono, fontSize: 12, padding: "9px", borderRadius: 5, cursor: "pointer", border: "none", background: LABEL_STYLE[selected.label]?.color || C.faint, color: "#fff", fontWeight: 600 }}>
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
              Open exploration over a read-only API. Every TARGETS edge is coloured by whether the
              food-to-protein link was experimentally measured or structurally inferred.
            </div>
          </div>
        </div>
      </div>
      <style>{`@media (max-width: 860px){ .g-grid{ grid-template-columns: 1fr !important; } }`}</style>
    </div>
  );
}
