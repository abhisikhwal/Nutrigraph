import React, { useState, useEffect, useCallback, useRef } from "react";

// ==============================================================
// NutriGraph, Guided Explorer
// Reveals the chain one layer at a time:
//   food -> compounds -> receptors/genes -> pathways & tissues
// Mobile-first, readable, live from Neo4j via the read-only API.
// A "raw graph" toggle keeps the force-graph for the technical crowd.
// ==============================================================

const API_BASE = "/api";

const C = {
  paper: "#FAF4E8", card: "#FFFFFF", ink: "#2A2018", faint: "#8A7A64", line: "#E3D6BE",
  gold: "#E08A00", chili: "#C23B22", herb: "#5E8C3A", plum: "#8B5A9E", blue: "#1F6FA8",
  measured: "#2E7D32", predicted: "#F9A825", nutrient: "#C77D0E",
};
const mono = "ui-monospace, 'SF Mono', 'IBM Plex Mono', Menlo, monospace";
const sans = "'Inter', system-ui, -apple-system, sans-serif";

const LABEL = {
  Ingredient: { color: C.gold, cap: "name", human: "food" },
  Compound: { color: C.chili, cap: "name", human: "compound" },
  Gene: { color: C.blue, cap: "symbol", human: "receptor / gene" },
  Pathway: { color: C.plum, cap: "name", human: "pathway" },
  Tissue: { color: C.herb, cap: "name", human: "tissue" },
  Nutrient: { color: C.nutrient, cap: "name", human: "nutrient" },
};
const cap = (n) => n?.props?.[LABEL[n.label]?.cap] || n?.props?.name || n?.props?.symbol || n?.label || "";

// which node types to offer at each step, in priority order
const NEXT_FILTER = {
  Ingredient: ["Compound", "Nutrient"],      // a food, show its compounds (and nutrients)
  Compound: ["Gene"],                          // a compound, show the genes it targets
  Gene: ["Pathway", "Tissue"],                 // a gene, show pathways + tissues
  Pathway: ["Gene"],
  Tissue: ["Gene"],
  Nutrient: ["Ingredient"],
};

// ---- API ------------------------------------------------------
async function apiSearch(q) {
  const r = await fetch(`${API_BASE}/search?q=${encodeURIComponent(q)}`);
  if (!r.ok) throw new Error("search");
  return (await r.json()).nodes || [];
}
async function apiExpand(id, evidence, type) {
  let p = `${API_BASE}/expand?id=${encodeURIComponent(id)}`;
  if (evidence) p += `&evidence=${encodeURIComponent(evidence)}`;
  if (type) p += `&type=${encodeURIComponent(type)}`;
  const r = await fetch(p);
  if (!r.ok) throw new Error("expand");
  return await r.json(); // {nodes, edges}
}

/** Fetch next-step neighbours, requesting each wanted type from the API. */
async function fetchNeighbours(node, measuredOnly) {
  const wanted = NEXT_FILTER[node.label] || Object.keys(LABEL);
  if (measuredOnly && (node.label === "Compound" || node.label === "Gene")) {
    return apiExpand(node.id, "measured");
  }
  const nodes = new Map();
  const edges = new Map();
  for (const t of wanted) {
    const nb = await apiExpand(node.id, null, t);
    for (const n of nb.nodes || []) nodes.set(n.id, n);
    for (const e of nb.edges || []) edges.set(e.id, e);
  }
  return { nodes: [...nodes.values()], edges: [...edges.values()] };
}

function kidsFromExpand(nb, node, wanted) {
  const seen = new Set();
  const kids = [];
  for (const e of nb.edges || []) {
    const otherId = e.from === node.id ? e.to : (e.to === node.id ? e.from : null);
    if (!otherId || seen.has(otherId)) continue;
    const other = (nb.nodes || []).find((n) => n.id === otherId);
    if (!other || other.id === node.id) continue;
    if (!wanted.includes(other.label)) continue;
    seen.add(otherId);
    kids.push({ node: other, edge: e });
  }
  kids.sort((a, b) => {
    const ae = a.edge?.props?.evidence === "measured" ? 0 : 1;
    const be = b.edge?.props?.evidence === "measured" ? 0 : 1;
    if (ae !== be) return ae - be;
    return cap(a.node).localeCompare(cap(b.node));
  });
  return kids.slice(0, 40);
}

// ---- small UI bits --------------------------------------------
function Chip({ node, edge, onClick, dimmed }) {
  const st = LABEL[node.label] || { color: C.faint };
  const isTarget = edge && edge.type === "TARGETS";
  const ev = edge?.props?.evidence;
  return (
    <button onClick={onClick} style={{
      display: "flex", alignItems: "center", gap: 10, width: "100%", textAlign: "left",
      background: C.card, border: `1px solid ${C.line}`, borderLeft: `4px solid ${st.color}`,
      borderRadius: 8, padding: "12px 14px", cursor: "pointer", opacity: dimmed ? 0.5 : 1,
      transition: "all .12s", boxShadow: "0 1px 2px rgba(42,32,24,.04)",
    }}
      onMouseEnter={(e) => { e.currentTarget.style.boxShadow = "0 3px 10px rgba(42,32,24,.1)"; e.currentTarget.style.transform = "translateX(2px)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.boxShadow = "0 1px 2px rgba(42,32,24,.04)"; e.currentTarget.style.transform = "none"; }}>
      <span style={{ width: 11, height: 11, borderRadius: "50%", border: `2px solid ${st.color}`, background: `${st.color}22`, flexShrink: 0 }} />
      <span style={{ flex: 1 }}>
        <span style={{ fontSize: 15, fontWeight: 600, color: C.ink }}>{cap(node)}</span>
        {node.props?.note && <span style={{ fontSize: 12.5, color: C.faint, marginLeft: 8 }}>{node.props.note}</span>}
      </span>
      {isTarget && (
        <span style={{ fontFamily: mono, fontSize: 10, fontWeight: 700, letterSpacing: 0.5, padding: "3px 8px", borderRadius: 3,
          background: ev === "measured" ? "rgba(46,125,50,.12)" : "rgba(249,168,37,.15)",
          color: ev === "measured" ? C.measured : "#B8860B" }}>
          {ev === "measured" ? "MEASURED" : "PREDICTED"}
        </span>
      )}
      <span style={{ fontFamily: mono, fontSize: 16, color: st.color, flexShrink: 0 }}>{"\u203A"}</span>
    </button>
  );
}

export default function GuidedExplorer() {
  const [trail, setTrail] = useState([]);        // breadcrumb of visited nodes
  const [current, setCurrent] = useState(null);  // node in focus
  const [children, setChildren] = useState([]);  // {node, edge} to reveal next
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [measuredOnly, setMeasuredOnly] = useState(false);
  const [err, setErr] = useState("");

  // seed with turmeric
  useEffect(() => {
    (async () => {
      try { const f = await apiSearch("turmeric"); if (f[0]) focus(f[0], true); }
      catch { setErr("Could not reach the graph. Is the service running?"); }
    })();
    // eslint-disable-next-line
  }, []);

  const search = useCallback(async (q) => {
    setQuery(q);
    if (!q.trim()) { setResults([]); return; }
    try { setResults(await apiSearch(q)); setErr(""); } catch { setErr("Search failed."); }
  }, []);

  async function focus(node, reset) {
    setLoading(true); setErr(""); setResults([]); setQuery("");
    try {
      const wanted = NEXT_FILTER[node.label] || Object.keys(LABEL);
      const nb = await fetchNeighbours(node, measuredOnly);
      const kids = kidsFromExpand(nb, node, wanted);
      setCurrent(node);
      setChildren(kids);
      setTrail((t) => (reset ? [node] : [...t, node]));
    } catch { setErr("Could not load connections."); }
    finally { setLoading(false); }
  }

  function goTo(idx) {
    const node = trail[idx];
    setTrail((t) => t.slice(0, idx + 1));
    (async () => {
      setLoading(true);
      try {
        const wanted = NEXT_FILTER[node.label] || Object.keys(LABEL);
        const nb = await fetchNeighbours(node, measuredOnly);
        setCurrent(node);
        setChildren(kidsFromExpand(nb, node, wanted));
      } finally { setLoading(false); }
    })();
  }

  const curStyle = current ? (LABEL[current.label] || { color: C.faint }) : { color: C.faint };
  const nextLabel = current ? (NEXT_FILTER[current.label] || []).map(l => LABEL[l]?.human).filter(Boolean).join(" and ") : "";
  const stepHint = {
    Ingredient: "the compounds and nutrients it carries",
    Compound: "the human genes and receptors it targets",
    Gene: "the pathways it drives and tissues it acts in",
    Pathway: "the genes in this pathway",
    Tissue: "the genes expressed here",
    Nutrient: "the foods that provide it",
  };

  return (
    <div style={{ background: C.paper, color: C.ink, fontFamily: sans, minHeight: "100vh", padding: "24px 18px 60px" }}>
      <div style={{ maxWidth: 760, margin: "0 auto" }}>
        {/* header */}
        <div style={{ marginBottom: 18 }}>
          <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 3, textTransform: "uppercase", color: C.gold, fontWeight: 600, marginBottom: 8 }}>NutriGraph, guided explorer</div>
          <h1 style={{ fontSize: 30, fontWeight: 800, letterSpacing: -1, margin: 0 }}>Follow food into the body</h1>
          <p style={{ fontSize: 15, color: C.faint, margin: "8px 0 0", lineHeight: 1.5 }}>
            Start with a food and follow the chain, one step at a time: its molecules, the receptors they reach,
            and where in the body they act. <span style={{ color: C.measured, fontWeight: 600 }}>Green</span> is
            experimentally measured, <span style={{ color: "#B8860B", fontWeight: 600 }}>amber</span> is predicted.
          </p>
        </div>

        {/* search */}
        <div style={{ position: "relative", marginBottom: 16 }}>
          <input value={query} onChange={e => search(e.target.value)} placeholder="search a food, gene or compound (try turmeric, CNR1, curcumin)"
            style={{ width: "100%", fontFamily: mono, fontSize: 14, padding: "13px 15px", borderRadius: 8, border: `1px solid ${C.line}`, background: C.card, color: C.ink, outline: "none", boxSizing: "border-box" }} />
          {results.length > 0 && (
            <div style={{ position: "absolute", top: 50, left: 0, right: 0, background: C.card, border: `1px solid ${C.line}`, borderRadius: 8, boxShadow: "0 8px 24px rgba(42,32,24,.14)", maxHeight: 280, overflowY: "auto", zIndex: 20 }}>
              {results.map(r => (
                <button key={r.id} onClick={() => focus(r, true)} style={{ display: "flex", alignItems: "center", gap: 10, width: "100%", padding: "11px 14px", background: "none", border: "none", borderBottom: `1px solid ${C.line}`, cursor: "pointer", textAlign: "left" }}>
                  <span style={{ width: 9, height: 9, borderRadius: "50%", background: LABEL[r.label]?.color || C.faint }} />
                  <span style={{ fontSize: 14, color: C.ink, fontWeight: 500 }}>{cap(r)}</span>
                  <span style={{ fontFamily: mono, fontSize: 10, color: C.faint, marginLeft: "auto", textTransform: "uppercase" }}>{LABEL[r.label]?.human || r.label}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* breadcrumb */}
        {trail.length > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap", marginBottom: 18, fontFamily: mono, fontSize: 12.5 }}>
            {trail.map((n, i) => (
              <React.Fragment key={n.id + i}>
                {i > 0 && <span style={{ color: C.faint }}>{"\u2192"}</span>}
                <button onClick={() => goTo(i)} style={{
                  background: i === trail.length - 1 ? (LABEL[n.label]?.color || C.faint) : "transparent",
                  color: i === trail.length - 1 ? "#fff" : C.ink,
                  border: `1px solid ${LABEL[n.label]?.color || C.line}`, borderRadius: 5, padding: "4px 10px", cursor: "pointer", fontWeight: 600, fontSize: 12.5,
                }}>
                  {cap(n).length > 22 ? cap(n).slice(0, 21) + "\u2026" : cap(n)}
                </button>
              </React.Fragment>
            ))}
          </div>
        )}

        {/* current focus card */}
        {current && (
          <div style={{ background: C.card, border: `1px solid ${C.line}`, borderTop: `4px solid ${curStyle.color}`, borderRadius: 10, padding: "20px 22px", marginBottom: 20, boxShadow: "0 2px 8px rgba(42,32,24,.06)" }}>
            <div style={{ fontFamily: mono, fontSize: 10.5, letterSpacing: 1.5, textTransform: "uppercase", color: curStyle.color, fontWeight: 700, marginBottom: 6 }}>{LABEL[current.label]?.human || current.label}</div>
            <div style={{ fontSize: 28, fontWeight: 800, letterSpacing: -0.5, marginBottom: 4 }}>{cap(current)}</div>
            {current.props?.latin && <div style={{ fontFamily: mono, fontStyle: "italic", color: C.faint, fontSize: 13 }}>{current.props.latin}</div>}
            {current.props?.note && <div style={{ fontSize: 14, color: C.faint, marginTop: 6 }}>{current.props.note}</div>}
            <div style={{ display: "flex", gap: 18, marginTop: 12, fontFamily: mono, fontSize: 11.5, color: C.faint, flexWrap: "wrap" }}>
              {current.props?.measured_fraction != null && <span>{Math.round(current.props.measured_fraction * 100)}% measured evidence</span>}
              {current.props?.data_status && <span>{current.props.data_status}</span>}
            </div>
          </div>
        )}

        {/* next-step reveal */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
          <div style={{ fontFamily: mono, fontSize: 12, letterSpacing: 1, textTransform: "uppercase", color: C.faint, fontWeight: 600 }}>
            {loading ? "loading..." : current ? `${stepHint[current.label] || "connections"} (${children.length})` : ""}
          </div>
          {current && current.label === "Compound" && (
            <button onClick={() => { setMeasuredOnly(v => !v); if (current) focus({ ...current }, false); }}
              style={{ fontFamily: mono, fontSize: 11, padding: "6px 12px", borderRadius: 5, cursor: "pointer", border: `1.5px solid ${measuredOnly ? C.measured : C.line}`, background: measuredOnly ? C.measured : C.card, color: measuredOnly ? "#fff" : C.ink }}>
              {measuredOnly ? "measured only" : "all evidence"}
            </button>
          )}
        </div>

        {err && <div style={{ fontFamily: mono, fontSize: 13, color: C.chili, background: "rgba(194,59,34,.08)", padding: "12px 14px", borderRadius: 8, marginBottom: 14 }}>{err}</div>}

        <div style={{ display: "grid", gap: 8 }}>
          {children.map(({ node, edge }) => (
            <Chip key={node.id} node={node} edge={edge} onClick={() => focus(node, false)} />
          ))}
          {!loading && current && children.length === 0 && (
            <div style={{ fontSize: 14, color: C.faint, padding: "20px 4px", textAlign: "center", background: C.card, border: `1px dashed ${C.line}`, borderRadius: 8 }}>
              This is a leaf of the graph, nothing further to follow from here. Use the breadcrumb to step back,
              or search for something new.
            </div>
          )}
        </div>

        {/* footer note */}
        <div style={{ marginTop: 30, paddingTop: 18, borderTop: `1px solid ${C.line}`, fontFamily: mono, fontSize: 11.5, color: C.faint, lineHeight: 1.6 }}>
          Live from a Neo4j knowledge graph over a read-only connection. 7,467 nodes, 79,301 relationships.
          Tissue links show where a gene is expressed, not proof a compound is delivered there.
        </div>
      </div>
    </div>
  );
}
