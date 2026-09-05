import React, { useState } from "react";

// ==============================================================
// NutriGraph, warm-LIGHT edition
// "Fresh market stall in daylight", crisp white cards on warm
// cream, saturated spice ink, editorial energy. Real engine data.
// ==============================================================

const F = {
  ingredients: 695, growthFrom: 463, measuredPct: 20.4, inferredPct: 79.6,
  hitAt10: 85.8, fdrNull: 1.3, meanRedundancy: 0.905, turmericLOO: 1.87,
  compounds: 48459, genes: 1532, pathways: 4543, tissues: 68, recipeMapping: 97.58,
  sources: 27, categories: 7, namespaces: 18,
};

const INTEGRATION = [
  { cat: "Chemistry", color: "#1F6FA8", sources: "FooDB, COCONUT, HMDB, Phenol-Explorer", gives: "the bioactive molecules inside each food", ids: "InChIKey" },
  { cat: "Pharmacology", color: "#155A8A", sources: "ChEMBL, BindingDB, PharmGKB", gives: "measured compound to protein binding", ids: "UniProt to HGNC" },
  { cat: "Expression", color: "#5E8C3A", sources: "GTEx", gives: "which genes are active in which tissues", ids: "Ensembl to HGNC" },
  { cat: "Pathways", color: "#8B5A9E", sources: "Gene Ontology, Reactome, Recon3D", gives: "the biological processes genes drive", ids: "GO ID, R-HSA" },
  { cat: "Nutrition", color: "#E08A00", sources: "USDA FoodData Central", gives: "per-food nutrient composition", ids: "FDC ID to FDA DV" },
  { cat: "Recipes", color: "#C23B22", sources: "RecipeNLG, Food.com, Epicurious, +6", gives: "how foods combine in real cooking", ids: "text to species" },
  { cat: "Identity", color: "#8A7A64", sources: "HGNC, UniProt, Wikidata", gives: "the Rosetta Stone across naming systems", ids: "cross-namespace" },
];

const TURMERIC = {
  id: "SP_000052", name: "Turmeric", latin: "Curcuma longa",
  measured_fraction: 0.2264, n_targets: 1144, broadly_active: true,
  plain: "Turmeric carries thousands of plant molecules that touch human biology in roughly 1,100 places, from cannabinoid receptors to metabolism enzymes to inflammation switches. About a quarter of that is experimentally confirmed. The rest is predicted from molecular structure, and we always say which is which.",
  targets: [
    { gene: "CNR1", evidence: "measured", confidence: 1.0, moa: "AGONIST", note: "Cannabinoid receptor, mood, appetite, pain" },
    { gene: "CNR2", evidence: "measured", confidence: 1.0, note: "Cannabinoid receptor, immune signaling" },
    { gene: "FAAH", evidence: "measured", confidence: 1.0, note: "Regulates the body's own cannabinoids" },
    { gene: "PTPN1", evidence: "measured", confidence: 1.0, note: "Insulin and metabolism signaling" },
    { gene: "CES2", evidence: "measured", confidence: 1.0, note: "Drug and ester metabolism" },
    { gene: "PTGS2", evidence: "predicted", confidence: 0.68, note: "COX-2, the enzyme ibuprofen blocks" },
    { gene: "NFKB1", evidence: "predicted", confidence: 0.64, note: "Master inflammation switch" },
    { gene: "VDR", evidence: "predicted", confidence: 0.71, note: "Vitamin D receptor" },
    { gene: "TNF", evidence: "predicted", confidence: 0.58, note: "Inflammation signaling" },
  ],
  tissues: [
    { tissue: "Liver", score: 0.41 }, { tissue: "Small intestine", score: 0.29 },
    { tissue: "Pancreas", score: 0.26 }, { tissue: "Lung", score: 0.19 },
  ],
  nutrition: [
    { name: "Iron", amount: "55 mg", dv: 306 }, { name: "Manganese", amount: "19.8 mg", dv: 860 },
    { name: "Magnesium", amount: "208 mg", dv: 50 }, { name: "Fiber", amount: "22.7 g", dv: 81 },
    { name: "Potassium", amount: "2080 mg", dv: 44 },
  ],
};

const LETTUCE = {
  id: "SP_000066", name: "Lettuce", latin: "Lactuca sativa",
  measured_fraction: 0.0, n_targets: 89, broadly_active: false,
  plain: "Lettuce is famously quiet at the receptor level, with few molecules that hit human targets. But that is only half the story. It is 95% water, rich in folate, potassium and fiber. A reminder that mechanistically thin is not the same as not good for you.",
  targets: [
    { gene: "PTGS2", evidence: "predicted", confidence: 0.4, note: "COX-2" },
    { gene: "ALOX5", evidence: "predicted", confidence: 0.35, note: "Lipoxygenase, inflammation" },
  ],
  tissues: [{ tissue: "Liver", score: 0.38 }, { tissue: "Small intestine", score: 0.24 }],
  nutrition: [
    { name: "Water", amount: "94.6 g", dv: null }, { name: "Folate", amount: "136 ug", dv: 34 },
    { name: "Potassium", amount: "247 mg", dv: 5 }, { name: "Fiber", amount: "2.1 g", dv: 7 },
  ],
};

const GINGER = {
  id: "SP_000139", name: "Ginger", latin: "Zingiber officinale",
  measured_fraction: 0.19, n_targets: 812, broadly_active: true,
  plain: "Ginger's gingerols and shogaols reach hundreds of human targets, with a real measured footprint on receptors linked to nausea, pain and vascular tone. Like most spices, its signal is broad rather than sharply specific.",
  targets: [
    { gene: "TRPV1", evidence: "measured", confidence: 1.0, moa: "AGONIST", note: "Heat and capsaicin receptor" },
    { gene: "CNR1", evidence: "measured", confidence: 1.0, note: "Cannabinoid receptor" },
    { gene: "PTGS1", evidence: "measured", confidence: 1.0, note: "COX-1" },
    { gene: "HTR3A", evidence: "predicted", confidence: 0.66, note: "Serotonin receptor, nausea" },
    { gene: "PTGS2", evidence: "predicted", confidence: 0.61, note: "COX-2" },
  ],
  tissues: [{ tissue: "Liver", score: 0.39 }, { tissue: "Small intestine", score: 0.27 }, { tissue: "Stomach", score: 0.21 }],
  nutrition: [
    { name: "Potassium", amount: "415 mg", dv: 9 }, { name: "Magnesium", amount: "43 mg", dv: 10 },
    { name: "Vitamin C", amount: "5 mg", dv: 6 }, { name: "Fiber", amount: "2 g", dv: 7 },
  ],
};

const HEROES = { turmeric: TURMERIC, ginger: GINGER, lettuce: LETTUCE };

const CURRY = {
  redundancy: [
    { name: "Turmeric", change: 1.87 }, { name: "Coriander", change: 3.56 },
    { name: "Pepper", change: 3.64 }, { name: "Garlic", change: 4.35 },
    { name: "Ginger", change: 4.46 }, { name: "Cumin", change: 4.92 },
  ],
};

const C = {
  paper: "#FAF4E8", paperAlt: "#F3EAD8", card: "#FFFFFF", ink: "#2A2018", faint: "#8A7A64",
  line: "#E3D6BE", gold: "#E08A00", chili: "#C23B22", herb: "#5E8C3A", plum: "#8B5A9E",
  measured: "#1F6FA8", predicted: "#C77D0E",
};

const mono = "ui-monospace, 'SF Mono', 'IBM Plex Mono', Menlo, monospace";
const sans = "'Inter', system-ui, -apple-system, sans-serif";

function Eyebrow({ children, color }) {
  return <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 3, textTransform: "uppercase", color: color || C.faint, marginBottom: 18, fontWeight: 600 }}>{children}</div>;
}
function EvDot({ e }) {
  const m = e === "measured";
  return <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: m ? "50%" : 2, background: m ? C.measured : "transparent", border: m ? "none" : `1.5px dashed ${C.predicted}`, flexShrink: 0 }} />;
}
function Card({ title, children }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.line}`, borderRadius: 6, padding: 18, boxShadow: "0 1px 2px rgba(42,32,24,.04)" }}>
      <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 1, textTransform: "uppercase", color: C.faint, marginBottom: 14, paddingBottom: 8, borderBottom: `1px solid ${C.line}`, fontWeight: 600 }}>{title}</div>
      {children}
    </div>
  );
}

function ChainViz() {
  const nodes = [
    { label: "Turmeric", sub: "food", color: C.gold },
    { label: "Curcumin", sub: "compound", color: C.chili },
    { label: "CNR1", sub: "receptor", color: C.measured },
    { label: "GABA signal", sub: "pathway", color: C.plum },
    { label: "Liver", sub: "tissue", color: C.herb },
  ];
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 0, flexWrap: "wrap", justifyContent: "center", padding: "8px 0" }}>
      {nodes.map((n, i) => (
        <React.Fragment key={n.label}>
          <div style={{ textAlign: "center", minWidth: 92 }}>
            <div style={{ width: 64, height: 64, borderRadius: "50%", border: `2px solid ${n.color}`, background: C.card, display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 8px", fontFamily: mono, fontSize: 10.5, fontWeight: 700, color: n.color, padding: 4, lineHeight: 1.1, textAlign: "center", boxShadow: `0 2px 8px ${n.color}22` }}>
              {n.label}
            </div>
            <div style={{ fontFamily: mono, fontSize: 9.5, letterSpacing: 1, textTransform: "uppercase", color: C.faint }}>{n.sub}</div>
          </div>
          {i < nodes.length - 1 && <div style={{ width: 26, height: 2, background: `linear-gradient(90deg, ${nodes[i].color}, ${nodes[i + 1].color})`, marginBottom: 22, flexShrink: 0 }} />}
        </React.Fragment>
      ))}
    </div>
  );
}

function Hook() {
  return (
    <section style={{ padding: "13vh 0 11vh", borderBottom: `1px solid ${C.line}` }}>
      <Eyebrow color={C.gold}>NutriGraph, a molecular map of food</Eyebrow>
      <h1 style={{ fontSize: "clamp(36px, 6vw, 74px)", lineHeight: 1.03, fontWeight: 800, letterSpacing: -2, margin: 0, maxWidth: 980, color: C.ink }}>
        Every meal is a molecular message to your body.
        <span style={{ color: C.faint }}> This maps what it says,</span>
        <span style={{ color: C.gold }}> and how sure we are.</span>
      </h1>
      <p style={{ fontSize: 20, lineHeight: 1.55, color: C.ink, maxWidth: 690, marginTop: 28 }}>
        Most of the chemistry in food has never been tested against human biology. NutriGraph links{" "}
        <b>{F.ingredients} foods</b> to the receptors, pathways and tissues they touch, fills the gaps with
        validated structural inference, and labels every single claim as measured or predicted.
      </p>
      <div style={{ display: "flex", gap: 44, marginTop: 44, flexWrap: "wrap" }}>
        {[["~1,100", "targets in one spice", C.gold], [F.compounds.toLocaleString(), "bioactive compounds", C.ink], [`${F.inferredPct}%`, "inferred, and flagged", C.ink], [`${F.sources}`, "datasets unified", C.ink]].map(([n, l, col]) => (
          <div key={l}>
            <div style={{ fontFamily: mono, fontSize: 34, fontWeight: 700, color: col }}>{n}</div>
            <div style={{ fontSize: 13, color: C.faint, marginTop: 4 }}>{l}</div>
          </div>
        ))}
      </div>
      <div style={{ marginTop: 54, padding: "24px 0 8px", borderTop: `1px solid ${C.line}` }}>
        <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 2, textTransform: "uppercase", color: C.faint, marginBottom: 18, textAlign: "center", fontWeight: 600 }}>
          one thread through the graph
        </div>
        <ChainViz />
      </div>
    </section>
  );
}

function NoSuperfoods() {
  const [removed, setRemoved] = useState(null);
  const shown = removed ? CURRY.redundancy.find(r => r.name === removed).change : 0;
  return (
    <section style={{ padding: "10vh 0", borderBottom: `1px solid ${C.line}` }}>
      <Eyebrow color={C.chili}>Finding 01, there are no superfoods</Eyebrow>
      <h2 style={{ fontSize: "clamp(28px,4vw,48px)", fontWeight: 800, letterSpacing: -1, margin: "0 0 18px", maxWidth: 840, lineHeight: 1.08, color: C.ink }}>
        Take turmeric out of a curry. Watch almost nothing happen.
      </h2>
      <p style={{ fontSize: 18, lineHeight: 1.55, color: C.faint, maxWidth: 660, marginBottom: 40 }}>
        We measured what each ingredient contributes to a dish's overall biological signal, then removed each
        one and re-measured. Across <b style={{ color: C.ink }}>400 recipes</b>, pulling any single ingredient
        changed the message by almost nothing. The effect of real food is{" "}
        <b style={{ color: C.chili }}>distributed</b>, not carried by heroes. Click to remove one.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32, alignItems: "center", maxWidth: 900, background: C.card, border: `1px solid ${C.line}`, borderRadius: 8, padding: 28, boxShadow: "0 1px 3px rgba(42,32,24,.05)" }} className="nm-grid2">
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {CURRY.redundancy.map(r => (
            <button key={r.name} onClick={() => setRemoved(removed === r.name ? null : r.name)}
              style={{ fontFamily: mono, fontSize: 13, padding: "10px 15px", borderRadius: 4, cursor: "pointer",
                border: `1px solid ${removed === r.name ? C.chili : C.line}`,
                background: removed === r.name ? C.paper : C.paperAlt,
                color: removed === r.name ? C.faint : C.ink,
                textDecoration: removed === r.name ? "line-through" : "none", transition: "all .15s" }}>
              {r.name}
            </button>
          ))}
        </div>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontFamily: mono, fontSize: 76, fontWeight: 700, color: shown > 0 ? C.gold : C.faint, lineHeight: 1, transition: "color .3s" }}>
            {shown.toFixed(2)}%
          </div>
          <div style={{ fontSize: 14, color: C.faint, marginTop: 8 }}>
            {removed ? `change when ${removed} is removed` : "select an ingredient to remove"}
          </div>
          <div style={{ height: 10, background: C.paperAlt, borderRadius: 5, marginTop: 20, overflow: "hidden", border: `1px solid ${C.line}` }}>
            <div style={{ height: "100%", width: `${Math.min(100, shown * 5)}%`, background: `linear-gradient(90deg, ${C.gold}, ${C.chili})`, transition: "width .5s ease" }} />
          </div>
          <div style={{ fontFamily: mono, fontSize: 11, color: C.faint, marginTop: 10 }}>
            mean across 400 recipes, {((1 - F.meanRedundancy) * 100).toFixed(1)}% change, essentially redundant
          </div>
        </div>
      </div>
    </section>
  );
}

function Integration() {
  return (
    <section style={{ padding: "10vh 0", borderBottom: `1px solid ${C.line}` }}>
      <Eyebrow color={C.herb}>The build, heterogeneous data integration</Eyebrow>
      <h2 style={{ fontSize: "clamp(28px,4vw,48px)", fontWeight: 800, letterSpacing: -1, margin: "0 0 12px", maxWidth: 840, lineHeight: 1.08, color: C.ink }}>
        {F.sources} datasets. {F.namespaces} identifier systems. One graph.
      </h2>
      <p style={{ fontSize: 18, lineHeight: 1.55, color: C.faint, maxWidth: 680, marginBottom: 44 }}>
        The hard part was never any single dataset. It was making them speak to each other. Chemistry keyed
        on InChIKey, pharmacology on protein names, expression on Ensembl IDs, nutrition on USDA codes.
        Reconciling those namespaces is the engineering.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: 14, marginBottom: 32 }}>
        {INTEGRATION.map(s => (
          <div key={s.cat} style={{ background: C.card, border: `1px solid ${C.line}`, borderTop: `3px solid ${s.color}`, borderRadius: 6, padding: 16, boxShadow: "0 1px 2px rgba(42,32,24,.04)" }}>
            <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 1.5, textTransform: "uppercase", color: s.color, marginBottom: 8, fontWeight: 700 }}>{s.cat}</div>
            <div style={{ fontSize: 14, color: C.ink, lineHeight: 1.4, marginBottom: 10 }}>{s.gives}</div>
            <div style={{ fontFamily: mono, fontSize: 11.5, color: C.faint, marginBottom: 6, lineHeight: 1.4 }}>{s.sources}</div>
            <div style={{ fontFamily: mono, fontSize: 11, color: s.color, fontWeight: 600 }}>{s.ids}</div>
          </div>
        ))}
      </div>

      <div style={{ background: C.paperAlt, border: `1px solid ${C.line}`, borderRadius: 6, padding: "22px 24px" }}>
        <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 1.5, textTransform: "uppercase", color: C.faint, marginBottom: 16, fontWeight: 600 }}>Identifier bridges built</div>
        <div style={{ display: "grid", gap: 10 }}>
          {["BindingDB protein names to UniProt to HGNC symbols",
            "GTEx Ensembl gene IDs to HGNC",
            "Compound identity unified on InChIKey across five chemistry sources",
            "Recipe free-text to species node to FooDB to USDA FDC",
            "Dual pathway namespace, Gene Ontology plus Reactome"].map((b, i) => (
            <div key={i} style={{ fontFamily: mono, fontSize: 13, color: C.ink, display: "flex", gap: 12 }}>
              <span style={{ color: C.gold, fontWeight: 700 }}>{"->"}</span><span>{b}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Explorer() {
  const [key, setKey] = useState("turmeric");
  const [measuredOnly, setMeasuredOnly] = useState(false);
  const [sel, setSel] = useState(null);
  const d = HEROES[key];
  const targets = measuredOnly ? d.targets.filter(t => t.evidence === "measured") : d.targets;

  const rowBtn = (a) => ({ display: "flex", alignItems: "center", width: "100%", gap: 10, padding: "9px 6px", background: a ? C.paperAlt : "transparent", border: "none", borderBottom: `1px solid ${C.line}`, cursor: "pointer", textAlign: "left", color: C.ink });
  const Row = ({ k, v }) => <div style={{ display: "flex", gap: 12 }}><span style={{ color: C.faint, width: 110 }}>{k}</span><span>{v}</span></div>;

  return (
    <section style={{ padding: "10vh 0", borderBottom: `1px solid ${C.line}` }}>
      <Eyebrow color={C.gold}>Explore, one food fully mapped</Eyebrow>
      <div style={{ display: "flex", gap: 8, marginBottom: 22, flexWrap: "wrap" }}>
        {Object.keys(HEROES).map(k => (
          <button key={k} onClick={() => { setKey(k); setSel(null); setMeasuredOnly(false); }}
            style={{ fontFamily: mono, fontSize: 13, padding: "8px 16px", borderRadius: 4, cursor: "pointer", textTransform: "capitalize",
              border: `1px solid ${key === k ? C.gold : C.line}`, background: key === k ? C.gold : C.card, color: key === k ? "#fff" : C.ink, fontWeight: key === k ? 700 : 400 }}>
            {k}
          </button>
        ))}
        <span style={{ fontFamily: mono, fontSize: 12, color: C.faint, alignSelf: "center", marginLeft: 6 }}>
          a live search over all {F.ingredients} foods in the full app
        </span>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 16 }}>
        <div>
          <h2 style={{ fontSize: 46, fontWeight: 800, letterSpacing: -1.5, margin: 0, lineHeight: 1, color: C.ink }}>{d.name}</h2>
          <div style={{ fontFamily: mono, fontStyle: "italic", color: C.faint, marginTop: 6 }}>{d.latin}</div>
        </div>
        <button onClick={() => setMeasuredOnly(v => !v)}
          style={{ fontFamily: mono, fontSize: 12, padding: "10px 16px", borderRadius: 4, cursor: "pointer",
            border: `1.5px solid ${measuredOnly ? C.measured : C.line}`, background: measuredOnly ? C.measured : C.card,
            color: measuredOnly ? "#fff" : C.ink }}>
          {measuredOnly ? "showing measured only" : "show measured only"}
        </button>
      </div>

      <p style={{ fontSize: 18, lineHeight: 1.55, maxWidth: 720, margin: "22px 0 8px", color: C.ink }}>{d.plain}</p>
      <div style={{ fontFamily: mono, fontSize: 12, color: C.faint, marginBottom: 30 }}>
        evidence base {(d.measured_fraction * 100).toFixed(0)}% measured, {d.n_targets.toLocaleString()} molecular targets
        {d.broadly_active ? ", broadly active" : ", focused signal"}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 20 }} className="nm-grid2">
        <Card title={`Receptors and enzymes it touches (${targets.length}${measuredOnly ? "" : " shown"})`}>
          {targets.map(t => (
            <button key={t.gene} onClick={() => setSel(t)} style={rowBtn(sel === t)}>
              <EvDot e={t.evidence} />
              <span style={{ fontFamily: mono, fontWeight: 700, width: 66 }}>{t.gene}</span>
              <span style={{ fontSize: 13, color: C.faint, flex: 1, textAlign: "left" }}>{t.note}</span>
              {t.moa && <span style={{ fontFamily: mono, fontSize: 10, padding: "2px 6px", borderRadius: 2, background: "rgba(31,111,168,.12)", color: C.measured, fontWeight: 600 }}>{t.moa}</span>}
            </button>
          ))}
          {measuredOnly && d.targets.length > targets.length && (
            <div style={{ fontFamily: mono, fontSize: 11.5, color: C.predicted, padding: "10px 4px 2px" }}>
              {d.targets.length - targets.length} predicted target{d.targets.length - targets.length > 1 ? "s" : ""} hidden, inference not measurement
            </div>
          )}
        </Card>

        <div style={{ display: "grid", gap: 20 }}>
          <Card title="Where its targets live">
            <div style={{ fontFamily: mono, fontSize: 10.5, color: C.faint, marginBottom: 12 }}>tissue expression, not proof of delivery</div>
            {d.tissues.map(t => (
              <div key={t.tissue} style={{ marginBottom: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
                  <span>{t.tissue}</span><span style={{ fontFamily: mono, color: C.faint }}>{(t.score * 100).toFixed(0)}%</span>
                </div>
                <div style={{ height: 6, background: C.paperAlt, borderRadius: 3 }}>
                  <div style={{ height: "100%", width: `${t.score * 100}%`, background: C.herb, borderRadius: 3 }} />
                </div>
              </div>
            ))}
          </Card>
          <Card title="Nutrition per 100g">
            <div style={{ fontFamily: mono, fontSize: 10.5, color: C.faint, marginBottom: 12 }}>USDA, the other kind of meaning</div>
            {d.nutrition.map(n => (
              <div key={n.name} style={{ display: "flex", justifyContent: "space-between", fontSize: 13, padding: "5px 0", borderBottom: `1px solid ${C.line}` }}>
                <span>{n.name}</span>
                <span style={{ fontFamily: mono, color: C.faint }}>{n.amount}{n.dv ? `, ${n.dv}% DV` : ""}</span>
              </div>
            ))}
          </Card>
        </div>
      </div>

      {sel && (
        <div onClick={() => setSel(null)} style={{ position: "fixed", inset: 0, background: "rgba(42,32,24,.35)", display: "flex", alignItems: "flex-end", justifyContent: "center", zIndex: 50 }}>
          <div onClick={e => e.stopPropagation()} style={{ background: C.card, borderTop: `3px solid ${sel.evidence === "measured" ? C.measured : C.predicted}`, maxWidth: 520, width: "100%", padding: 26, borderRadius: "10px 10px 0 0", boxShadow: "0 -8px 30px rgba(42,32,24,.15)" }}>
            <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 1.5, textTransform: "uppercase", color: sel.evidence === "measured" ? C.measured : C.predicted, marginBottom: 10, fontWeight: 700 }}>
              {sel.evidence === "measured" ? "Measured evidence" : "Structural inference"}
            </div>
            <div style={{ fontSize: 26, fontWeight: 800, marginBottom: 4, color: C.ink }}>{sel.gene}</div>
            <div style={{ fontSize: 14, color: C.faint, marginBottom: 18 }}>{sel.note}</div>
            <div style={{ fontFamily: mono, fontSize: 13, lineHeight: 2, color: C.ink }}>
              <Row k="evidence" v={sel.evidence} />
              <Row k="confidence" v={sel.confidence.toFixed(2)} />
              {sel.moa && <Row k="action" v={sel.moa} />}
              <Row k="source" v={sel.evidence === "measured" ? "ChEMBL, BindingDB" : "k-NN structural inference"} />
              {sel.evidence !== "measured" && <Row k="validation" v={`${F.hitAt10}% hit@10 on held-out scaffolds`} />}
            </div>
            <button onClick={() => setSel(null)} style={{ marginTop: 20, fontFamily: mono, fontSize: 12, background: C.paper, border: `1px solid ${C.line}`, padding: "8px 14px", borderRadius: 3, cursor: "pointer", color: C.ink }}>close</button>
          </div>
        </div>
      )}
    </section>
  );
}

function Rigor() {
  const items = [
    [`${F.hitAt10}%`, "inference accuracy", "hit@10 on held-out molecular scaffolds, the model finds real targets, not memorized ones", C.gold],
    [`${F.fdrNull}%`, "false-positive rate", "random data almost never passes the significance test, the statistics are honest", C.herb],
    [`${F.inferredPct}%`, "predicted, and flagged", "most of the graph is inference, and every inferred edge declares itself", C.measured],
    [`${F.growthFrom} to ${F.ingredients}`, "ingredient universe", "grown from whole species to composites, blends and world cuisines", C.chili],
    [`${F.recipeMapping}%`, "recipe coverage", "of ingredients in 113,000 recipes resolve to the graph", C.gold],
    [`${F.tissues}`, "human tissues", "every mechanism located to where its genes are expressed", C.herb],
  ];
  return (
    <section style={{ padding: "10vh 0", borderBottom: `1px solid ${C.line}` }}>
      <Eyebrow>Under the hood, built to be checked</Eyebrow>
      <h2 style={{ fontSize: "clamp(28px,4vw,48px)", fontWeight: 800, letterSpacing: -1, margin: "0 0 40px", maxWidth: 760, lineHeight: 1.08, color: C.ink }}>
        The honest version is the impressive one.
      </h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px,1fr))", gap: 16 }}>
        {items.map(([n, l, d, col]) => (
          <div key={l} style={{ background: C.card, border: `1px solid ${C.line}`, borderRadius: 6, padding: 20, boxShadow: "0 1px 2px rgba(42,32,24,.04)" }}>
            <div style={{ fontFamily: mono, fontSize: 30, fontWeight: 700, color: col }}>{n}</div>
            <div style={{ fontSize: 14, fontWeight: 600, margin: "6px 0 8px", color: C.ink }}>{l}</div>
            <div style={{ fontSize: 13, color: C.faint, lineHeight: 1.5 }}>{d}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Footer() {
  return (
    <section style={{ padding: "10vh 0 6vh" }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 48, marginBottom: 60 }} className="nm-grid2">
        <div>
          <Eyebrow color={C.herb}>What it is</Eyebrow>
          <p style={{ fontSize: 15, lineHeight: 1.6, color: C.ink }}>
            A knowledge graph linking {F.ingredients} foods to their compounds, human receptors, pathways
            and tissues, plus full nutrition. Measured pharmacology anchors it. Validated structural
            inference fills the 80% that has never been assayed.
          </p>
        </div>
        <div>
          <Eyebrow color={C.chili}>What it is not</Eyebrow>
          <ul style={{ fontSize: 14, lineHeight: 1.7, color: C.faint, paddingLeft: 18, margin: 0 }}>
            <li>Not medical or dietary advice.</li>
            <li>No pharmacokinetics, target location is not proof a compound reaches or acts there.</li>
            <li>Receptor pharmacology plus nutrition, not fiber or microbiome effects.</li>
            <li>Relative potency, not absolute dose response.</li>
          </ul>
        </div>
      </div>
      <div style={{ borderTop: `1px solid ${C.line}`, paddingTop: 24, display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 16, alignItems: "center" }}>
        <div style={{ fontFamily: mono, fontSize: 12, color: C.faint }}>
          NutriGraph, built by Abhinav Sikhwal. Python, RDKit, Neo4j, React.
        </div>
        <button style={{ fontFamily: mono, fontSize: 13, padding: "12px 22px", borderRadius: 4, cursor: "pointer", border: "none", background: C.ink, color: C.paper, fontWeight: 700 }}>
          Explore the live graph {"->"}
        </button>
      </div>
    </section>
  );
}

export default function NutriGraph() {
  return (
    <div style={{ background: C.paper, color: C.ink, fontFamily: sans, minHeight: "100vh" }}>
      <style>{`@media (max-width: 720px){ .nm-grid2{ grid-template-columns: 1fr !important; } }`}</style>
      <div style={{ maxWidth: 1080, margin: "0 auto", padding: "0 28px" }}>
        <Hook />
        <NoSuperfoods />
        <Integration />
        <Explorer />
        <Rigor />
        <Footer />
      </div>
    </div>
  );
}
