import React, { useState, useEffect, useCallback } from "react";

// ==============================================================
// NutriGraph, Food Dossier
// One food, three lenses:
//   REACHES  -> the receptors/genes it hits (measured first)
//   BODY     -> the tissues it acts in
//   SIMILAR  -> foods that share its biology
// Ranked in-database. Live from Neo4j via the read-only API.
// ==============================================================

const API = "/api";

const C = {
  paper: "#FAF4E8", card: "#FFFFFF", ink: "#2A2018", faint: "#8A7A64", line: "#E3D6BE",
  gold: "#E08A00", chili: "#C23B22", herb: "#5E8C3A", plum: "#8B5A9E", blue: "#1F6FA8",
  measured: "#2E7D32", predicted: "#C77D0E",
};
const mono = "ui-monospace, 'SF Mono', 'IBM Plex Mono', Menlo, monospace";
const sans = "'Inter', system-ui, -apple-system, sans-serif";
const STARTERS = ["Turmeric", "Ginger", "Garlic", "Coffee", "Chili", "Green tea", "Cinnamon", "Black pepper"];

async function jget(path) {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error("api");
  return r.json();
}
const capNode = (n) => n?.props?.name || n?.props?.symbol || "";

// ---- lens definitions ----------------------------------------
const LENSES = [
  { key: "reaches", label: "Reaches", color: C.blue, sub: "receptors & genes" },
  { key: "body", label: "In the body", color: C.herb, sub: "tissues" },
  { key: "similar", label: "Similar foods", color: C.chili, sub: "shared biology" },
];

function Bar({ value, max, color }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div style={{ height: 6, background: C.line, borderRadius: 3, overflow: "hidden", flex: 1 }}>
      <div style={{ height: "100%", width: `${pct}%`, background: color, borderRadius: 3 }} />
    </div>
  );
}

export default function Dossier() {
  const [food, setFood] = useState(null);      // {id, name, latin, measured_fraction}
  const [lens, setLens] = useState("reaches");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [data, setData] = useState({});         // {genes, tissues, foods, compounds}
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => { loadFood("Turmeric"); /* eslint-disable-next-line */ }, []);

  const search = useCallback(async (q) => {
    setQuery(q);
    if (!q.trim()) { setResults([]); return; }
    try { const d = await jget(`/search?q=${encodeURIComponent(q)}`); setResults(d.nodes || []); }
    catch { setErr("Search failed."); }
  }, []);

  async function loadFood(name, node) {
    setLoading(true); setErr(""); setResults([]); setQuery("");
    try {
      let f = node;
      if (!f) {
        const d = await jget(`/search?q=${encodeURIComponent(name)}`);
        f = (d.nodes || []).find(n => n.label === "Ingredient" && capNode(n).toLowerCase() === name.toLowerCase())
          || (d.nodes || []).find(n => n.label === "Ingredient") || (d.nodes || [])[0];
      }
      if (!f) { setErr("Food not found."); setLoading(false); return; }
      const foodObj = { id: f.id, name: capNode(f), latin: f.props?.latin, mf: f.props?.measured_fraction, label: f.label };
      setFood(foodObj);
      // fetch all three lenses + compounds in parallel
      const [genes, tissues, foods, compounds] = await Promise.all([
        jget(`/food/genes?id=${encodeURIComponent(f.id)}`).catch(() => ({ genes: [] })),
        jget(`/food/tissues?id=${encodeURIComponent(f.id)}`).catch(() => ({ tissues: [] })),
        jget(`/food/similar?id=${encodeURIComponent(f.id)}`).catch(() => ({ foods: [] })),
        jget(`/food/compounds?id=${encodeURIComponent(f.id)}`).catch(() => ({ compounds: [] })),
      ]);
      setData({ genes: genes.genes || [], tissues: tissues.tissues || [], foods: foods.foods || [], compounds: compounds.compounds || [] });
    } catch { setErr("Could not load this food."); }
    finally { setLoading(false); }
  }

  // headline generator per lens (grounded in the data)
  function headline() {
    if (!food) return "";
    const g = data.genes || [], t = data.tissues || [], f = data.foods || [];
    if (lens === "reaches") {
      const measured = g.filter(x => x.evidence === "measured");
      if (!g.length) return `We have no receptor data for ${food.name} yet.`;
      const top = measured.slice(0, 3).map(x => x.symbol);
      if (top.length) return `${food.name} reaches human proteins across the body. Its strongest measured targets: ${top.join(", ")}.`;
      return `${food.name}'s targets are mostly structurally predicted, ${g.length} in view, led by ${g.slice(0, 3).map(x => x.symbol).join(", ")}.`;
    }
    if (lens === "body") {
      if (!t.length) return `We have no tissue localization for ${food.name} yet.`;
      return `${food.name}'s targets concentrate in ${t.slice(0, 3).map(x => x.name).join(", ")}.`;
    }
    if (lens === "similar") {
      if (!f.length) return `No close matches found for ${food.name}.`;
      return `${food.name} shares the most biology with ${f.slice(0, 3).map(x => x.name).join(", ")}.`;
    }
    return "";
  }

  const activeLens = LENSES.find(l => l.key === lens);

  return (
    <div style={{ background: C.paper, color: C.ink, fontFamily: sans, minHeight: "100vh", padding: "24px 18px 60px" }}>
      <div style={{ maxWidth: 780, margin: "0 auto" }}>
        {/* header */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 3, textTransform: "uppercase", color: C.gold, fontWeight: 600, marginBottom: 8 }}>NutriGraph, food dossier</div>
          <h1 style={{ fontSize: 30, fontWeight: 800, letterSpacing: -1, margin: 0 }}>What is this food doing?</h1>
          <p style={{ fontSize: 15, color: C.faint, margin: "8px 0 0", lineHeight: 1.5 }}>
            Pick a food and see it three ways: the <b style={{ color: C.blue }}>receptors</b> it reaches,
            the <b style={{ color: C.herb }}>tissues</b> it acts in, and the <b style={{ color: C.chili }}>foods</b> it
            resembles. <span style={{ color: C.measured, fontWeight: 600 }}>Green</span> is measured,{" "}
            <span style={{ color: C.predicted, fontWeight: 600 }}>amber</span> is predicted.
          </p>
        </div>

        {/* search */}
        <div style={{ position: "relative", marginBottom: 12 }}>
          <input value={query} onChange={e => search(e.target.value)} placeholder="search a food (try turmeric, coffee, ginger)"
            style={{ width: "100%", fontFamily: mono, fontSize: 14, padding: "13px 15px", borderRadius: 8, border: `1px solid ${C.line}`, background: C.card, color: C.ink, outline: "none", boxSizing: "border-box" }} />
          {results.length > 0 && (
            <div style={{ position: "absolute", top: 50, left: 0, right: 0, background: C.card, border: `1px solid ${C.line}`, borderRadius: 8, boxShadow: "0 8px 24px rgba(42,32,24,.14)", maxHeight: 280, overflowY: "auto", zIndex: 20 }}>
              {results.filter(r => r.label === "Ingredient").map(r => (
                <button key={r.id} onClick={() => loadFood(capNode(r), r)} style={{ display: "flex", alignItems: "center", gap: 10, width: "100%", padding: "11px 14px", background: "none", border: "none", borderBottom: `1px solid ${C.line}`, cursor: "pointer", textAlign: "left" }}>
                  <span style={{ width: 9, height: 9, borderRadius: "50%", background: C.gold }} />
                  <span style={{ fontSize: 14, color: C.ink, fontWeight: 500 }}>{capNode(r)}</span>
                </button>
              ))}
              {results.filter(r => r.label === "Ingredient").length === 0 && (
                <div style={{ padding: "12px 14px", fontSize: 13, color: C.faint }}>No foods match. Try a food name.</div>
              )}
            </div>
          )}
        </div>

        {/* starters */}
        <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginBottom: 22, alignItems: "center" }}>
          <span style={{ fontFamily: mono, fontSize: 11, color: C.faint, marginRight: 4 }}>try</span>
          {STARTERS.map(name => (
            <button key={name} onClick={() => loadFood(name)} style={{
              fontFamily: mono, fontSize: 12.5, padding: "6px 12px", borderRadius: 20, cursor: "pointer",
              border: `1px solid ${food?.name === name ? C.gold : C.line}`, background: food?.name === name ? C.gold : C.card,
              color: food?.name === name ? "#fff" : C.ink, transition: "all .12s",
            }}>{name}</button>
          ))}
        </div>

        {food && (
          <>
            {/* food headline card */}
            <div style={{ background: C.card, border: `1px solid ${C.line}`, borderTop: `4px solid ${C.gold}`, borderRadius: 10, padding: "20px 22px", marginBottom: 18, boxShadow: "0 2px 8px rgba(42,32,24,.06)" }}>
              <div style={{ fontFamily: mono, fontSize: 10.5, letterSpacing: 1.5, textTransform: "uppercase", color: C.gold, fontWeight: 700, marginBottom: 6 }}>food</div>
              <div style={{ fontSize: 30, fontWeight: 800, letterSpacing: -0.5, marginBottom: 2 }}>{food.name}</div>
              {food.latin && <div style={{ fontFamily: mono, fontStyle: "italic", color: C.faint, fontSize: 13, marginBottom: 12 }}>{food.latin}</div>}
              <div style={{ fontSize: 16, lineHeight: 1.5, color: C.ink, marginTop: 8 }}>{loading ? "Reading the graph..." : headline()}</div>
            </div>

            {/* lens switcher */}
            <div style={{ display: "flex", gap: 8, marginBottom: 18, flexWrap: "wrap" }}>
              {LENSES.map(l => (
                <button key={l.key} onClick={() => setLens(l.key)} style={{
                  flex: "1 1 140px", fontFamily: sans, fontSize: 14, fontWeight: 600, padding: "12px 10px", borderRadius: 8, cursor: "pointer",
                  border: `1.5px solid ${lens === l.key ? l.color : C.line}`, background: lens === l.key ? l.color : C.card,
                  color: lens === l.key ? "#fff" : C.ink, transition: "all .12s", textAlign: "center",
                }}>
                  {l.label}
                  <div style={{ fontFamily: mono, fontSize: 10.5, fontWeight: 400, opacity: 0.85, marginTop: 2 }}>{l.sub}</div>
                </button>
              ))}
            </div>

            {err && <div style={{ fontFamily: mono, fontSize: 13, color: C.chili, background: "rgba(194,59,34,.08)", padding: "12px 14px", borderRadius: 8, marginBottom: 14 }}>{err}</div>}

            {/* lens content */}
            {lens === "reaches" && <ReachesLens genes={data.genes || []} compounds={data.compounds || []} loading={loading} />}
            {lens === "body" && <BodyLens tissues={data.tissues || []} loading={loading} />}
            {lens === "similar" && <SimilarLens foods={data.foods || []} loading={loading} onPick={loadFood} />}
          </>
        )}

        <div style={{ marginTop: 30, paddingTop: 18, borderTop: `1px solid ${C.line}`, fontFamily: mono, fontSize: 11.5, color: C.faint, lineHeight: 1.6 }}>
          Live from a Neo4j knowledge graph, 7,467 nodes, 79,301 relationships, read-only. Rankings computed in-database.
          Tissue links show where a gene is expressed, not proof a compound is delivered there.
        </div>
      </div>
    </div>
  );
}

// ---- LENS A: receptors/genes ---------------------------------
function ReachesLens({ genes, compounds, loading }) {
  const maxHits = Math.max(1, ...genes.map(g => g.compound_hits));
  return (
    <div>
      <SectionLabel>{loading ? "loading..." : `receptors & genes it reaches, ranked (${genes.length})`}</SectionLabel>
      <div style={{ display: "grid", gap: 7, marginBottom: 26 }}>
        {genes.map(g => (
          <div key={g.symbol} style={{ background: C.card, border: `1px solid ${C.line}`, borderLeft: `4px solid ${g.evidence === "measured" ? C.measured : C.predicted}`, borderRadius: 8, padding: "12px 14px", display: "flex", alignItems: "center", gap: 14 }}>
            <span style={{ fontFamily: mono, fontWeight: 700, fontSize: 15, width: 78, color: C.ink }}>{g.symbol}</span>
            <span style={{ display: "flex", alignItems: "center", gap: 8, flex: 1 }}>
              <Bar value={g.compound_hits} max={maxHits} color={g.evidence === "measured" ? C.measured : C.predicted} />
              <span style={{ fontFamily: mono, fontSize: 11, color: C.faint, width: 90, textAlign: "right" }}>{g.compound_hits} compound{g.compound_hits > 1 ? "s" : ""}</span>
            </span>
            <span style={{ fontFamily: mono, fontSize: 9.5, fontWeight: 700, letterSpacing: 0.5, padding: "3px 8px", borderRadius: 3, background: g.evidence === "measured" ? "rgba(46,125,50,.12)" : "rgba(199,125,14,.14)", color: g.evidence === "measured" ? C.measured : C.predicted, flexShrink: 0 }}>
              {g.evidence === "measured" ? "MEASURED" : "PREDICTED"}
            </span>
          </div>
        ))}
        {!loading && genes.length === 0 && <Empty>No receptor data for this food yet.</Empty>}
      </div>

      {compounds.length > 0 && (
        <>
          <SectionLabel>notable bioactive compounds</SectionLabel>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {compounds.map(c => (
              <span key={c.id} style={{ background: C.card, border: `1px solid ${C.line}`, borderRadius: 20, padding: "6px 13px", fontSize: 13, color: C.ink }}>
                {c.name} <span style={{ color: C.faint, fontFamily: mono, fontSize: 11 }}>· {c.genes}</span>
              </span>
            ))}
          </div>
          <div style={{ fontFamily: mono, fontSize: 11, color: C.faint, marginTop: 8 }}>the number is how many human genes each compound targets</div>
        </>
      )}
    </div>
  );
}

// ---- LENS B: tissues -----------------------------------------
function BodyLens({ tissues, loading }) {
  const max = Math.max(1, ...tissues.map(t => t.score));
  return (
    <div>
      <SectionLabel>{loading ? "loading..." : `where in the body its targets are active (${tissues.length})`}</SectionLabel>
      <div style={{ display: "grid", gap: 9 }}>
        {tissues.map(t => (
          <div key={t.name} style={{ background: C.card, border: `1px solid ${C.line}`, borderLeft: `4px solid ${C.herb}`, borderRadius: 8, padding: "13px 15px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 7 }}>
              <span style={{ fontSize: 15, fontWeight: 600 }}>{t.name}</span>
              <span style={{ fontFamily: mono, fontSize: 11, color: C.faint }}>{t.genes} gene{t.genes > 1 ? "s" : ""}</span>
            </div>
            <Bar value={t.score} max={max} color={C.herb} />
          </div>
        ))}
        {!loading && tissues.length === 0 && <Empty>No tissue localization for this food yet.</Empty>}
      </div>
      <div style={{ fontFamily: mono, fontSize: 11, color: C.faint, marginTop: 12, lineHeight: 1.5 }}>
        This shows where the target genes are expressed in the human body. It is not proof the food's compounds
        are delivered to, or act in, those tissues.
      </div>
    </div>
  );
}

// ---- LENS C: similar foods -----------------------------------
function SimilarLens({ foods, loading, onPick }) {
  const max = Math.max(1, ...foods.map(f => f.weighted));
  return (
    <div>
      <SectionLabel>{loading ? "loading..." : `foods that share the most biology (${foods.length})`}</SectionLabel>
      <div style={{ display: "grid", gap: 8 }}>
        {foods.map(f => (
          <button key={f.name} onClick={() => onPick(f.name)} style={{ background: C.card, border: `1px solid ${C.line}`, borderLeft: `4px solid ${C.chili}`, borderRadius: 8, padding: "13px 15px", cursor: "pointer", textAlign: "left", display: "flex", alignItems: "center", gap: 14 }}>
            <span style={{ fontSize: 15, fontWeight: 600, width: 150, color: C.ink }}>{f.name}</span>
            <Bar value={f.weighted} max={max} color={C.chili} />
            <span style={{ fontFamily: mono, fontSize: 11, color: C.faint, width: 120, textAlign: "right" }}>{f.shared} shared compounds</span>
            <span style={{ fontFamily: mono, fontSize: 16, color: C.chili }}>{"\u203A"}</span>
          </button>
        ))}
        {!loading && foods.length === 0 && <Empty>No close matches found for this food.</Empty>}
      </div>
      <div style={{ fontFamily: mono, fontSize: 11, color: C.faint, marginTop: 12 }}>
        Ranked by shared bioactive compounds, weighted so distinctive shared molecules count more than common ones. Tap a food to explore it.
      </div>
    </div>
  );
}

function SectionLabel({ children }) {
  return <div style={{ fontFamily: mono, fontSize: 11.5, letterSpacing: 1, textTransform: "uppercase", color: C.faint, fontWeight: 600, marginBottom: 12, marginTop: 4 }}>{children}</div>;
}
function Empty({ children }) {
  return <div style={{ fontSize: 14, color: C.faint, padding: "20px 4px", textAlign: "center", background: C.card, border: `1px dashed ${C.line}`, borderRadius: 8 }}>{children}</div>;
}
