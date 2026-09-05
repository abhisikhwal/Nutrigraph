// ==============================================================
// NutriGraph, thin read-only API
// Sits between the public browser and Neo4j. Holds the DB
// credentials server-side (never exposed). Only runs a fixed set
// of parameterized READ queries. Rejects anything else.
// Needed because Neo4j Community edition has no read-only role,
// so safety is enforced here instead.
// ==============================================================

import express from "express";
import neo4j from "neo4j-driver";

// ---- config (from environment) --------------------------------
const PORT = process.env.NUTRI_API_PORT || 8600;
const NEO4J_URI = process.env.NEO4J_URI || "bolt://localhost:7687";
const NEO4J_USER = process.env.NEO4J_USER || "neo4j";
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD; // required, set in env

if (!NEO4J_PASSWORD) {
  console.error("FATAL: set NEO4J_PASSWORD environment variable.");
  process.exit(1);
}

const driver = neo4j.driver(NEO4J_URI, neo4j.auth.basic(NEO4J_USER, NEO4J_PASSWORD), {
  maxConnectionPoolSize: 20,
});

const app = express();
app.use(express.json({ limit: "16kb" }));

// permissive CORS for the public graph page (read-only data, safe)
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

// ---- run a read query, always READ mode, always capped ---------
async function read(cypher, params) {
  const session = driver.session({ defaultAccessMode: neo4j.session.READ });
  try {
    const result = await session.run(cypher, params);
    return result.records;
  } finally {
    await session.close();
  }
}

// serialize a neo4j node to a plain object
function nodeOut(n) {
  return {
    id: n.elementId,
    label: n.labels[0],
    props: Object.fromEntries(
      Object.entries(n.properties).map(([k, v]) => [k, neo4j.isInt(v) ? v.toNumber() : v])
    ),
  };
}
function relOut(r) {
  return {
    id: r.elementId,
    type: r.type,
    from: r.startNodeElementId,
    to: r.endNodeElementId,
    props: Object.fromEntries(
      Object.entries(r.properties).map(([k, v]) => [k, neo4j.isInt(v) ? v.toNumber() : v])
    ),
  };
}

// assemble {nodes, edges} from records that RETURN n, r, m (or just n)
function graphFrom(records) {
  const nodes = new Map();
  const edges = new Map();
  for (const rec of records) {
    for (const key of rec.keys) {
      const v = rec.get(key);
      if (v == null) continue;
      if (v.labels) { const o = nodeOut(v); nodes.set(o.id, o); }
      else if (v.type && v.start !== undefined) { /* skip */ }
      else if (v.type && v.startNodeElementId) { const o = relOut(v); edges.set(o.id, o); }
    }
  }
  return { nodes: [...nodes.values()], edges: [...edges.values()] };
}

// ---- endpoints (fixed queries only, no arbitrary cypher) -------

// health
app.get("/health", (req, res) => res.json({ ok: true }));

// search by name, returns bare nodes
app.get("/search", async (req, res) => {
  const q = String(req.query.q || "").slice(0, 100);
  if (!q.trim()) return res.json({ nodes: [] });
  try {
    const recs = await read(
      "MATCH (n) WHERE toLower(coalesce(n.name, n.symbol, '')) CONTAINS toLower($q) RETURN n LIMIT 25",
      { q }
    );
    res.json({ nodes: recs.map((r) => nodeOut(r.get("n"))) });
  } catch (e) { res.status(500).json({ error: "query failed" }); }
});

// expand any node by elementId; optional ?type=Compound|Gene|Pathway|Tissue|Nutrient|Ingredient
app.get("/expand", async (req, res) => {
  const elementId = String(req.query.id || "");
  const evidence = req.query.evidence ? String(req.query.evidence) : null;
  const type = req.query.type ? String(req.query.type) : null;
  const validTypes = ["Ingredient", "Compound", "Gene", "Pathway", "Tissue", "Nutrient"];
  if (!elementId) return res.json({ nodes: [], edges: [] });
  try {
    let recs;
    if (evidence === "measured" || evidence === "predicted") {
      recs = await read(
        "MATCH (n)-[r:TARGETS]-(m) WHERE elementId(n) = $elementId AND r.evidence = $evidence RETURN n, r, m LIMIT 40",
        { elementId, evidence }
      );
    } else if (type && validTypes.includes(type)) {
      // Return neighbours of a specific type. For compounds, push generic
      // lipids (TG/CL/DG/PE...) to the back so named bioactives surface first.
      recs = await read(
        `MATCH (n)-[r]-(m)
         WHERE elementId(n) = $elementId AND $type IN labels(m)
         WITH n, r, m,
           CASE
             WHEN m.name IS NOT NULL AND (
               m.name STARTS WITH 'TG(' OR m.name STARTS WITH 'CL(' OR
               m.name STARTS WITH 'DG(' OR m.name STARTS WITH 'PE(' OR
               m.name STARTS WITH 'PC(' OR m.name STARTS WITH 'PS(' OR
               m.name STARTS WITH 'PA(' OR m.name STARTS WITH 'PI(' OR
               m.name STARTS WITH 'PG(' OR m.name STARTS WITH 'SM(' OR
               m.name STARTS WITH 'CE(' OR m.name STARTS WITH 'MG('
             ) THEN 1 ELSE 0
           END AS lipid_rank
         ORDER BY lipid_rank ASC
         RETURN n, r, m LIMIT 40`,
        { elementId, type }
      );
    } else {
      recs = await read(
        `MATCH (n)-[r]-(m)
         WHERE elementId(n) = $elementId
         WITH n, r, m,
           CASE labels(m)[0]
             WHEN 'Gene' THEN 0 WHEN 'Pathway' THEN 1 WHEN 'Tissue' THEN 2
             WHEN 'Nutrient' THEN 3 WHEN 'Ingredient' THEN 4 ELSE 5
           END AS type_rank
         ORDER BY type_rank ASC
         RETURN n, r, m LIMIT 28`,
        { elementId }
      );
    }
    res.json(graphFrom(recs));
  } catch (e) { res.status(500).json({ error: "query failed" }); }
});

// stats for the header
app.get("/stats", async (req, res) => {
  try {
    const recs = await read("MATCH (n) RETURN count(n) AS nodes", {});
    const nodeCount = recs[0].get("nodes").toNumber();
    res.json({ nodes: nodeCount });
  } catch (e) { res.status(500).json({ error: "query failed" }); }
});

// ---- LENS A: a food's genes, ranked (measured-first, then by
//      how many of the food's compounds hit that gene) ----------
app.get("/food/genes", async (req, res) => {
  const id = String(req.query.id || "");
  if (!id) return res.json({ genes: [] });
  try {
    const recs = await read(
      `MATCH (i:Ingredient)-[:CONTAINS]->(c:Compound)-[t:TARGETS]->(g:Gene)
       WHERE elementId(i) = $id
       WITH g,
            sum(CASE WHEN t.evidence = 'measured' THEN 1 ELSE 0 END) AS measured_hits,
            count(DISTINCT c) AS compound_hits,
            max(CASE WHEN t.evidence = 'measured' THEN 1 ELSE 0 END) AS has_measured
       RETURN g.symbol AS symbol, g.name AS name,
              measured_hits, compound_hits, has_measured
       ORDER BY has_measured DESC, measured_hits DESC, compound_hits DESC
       LIMIT 25`,
      { id }
    );
    res.json({
      genes: recs.map(r => ({
        symbol: r.get("symbol"),
        name: r.get("name"),
        measured_hits: r.get("measured_hits").toNumber(),
        compound_hits: r.get("compound_hits").toNumber(),
        evidence: r.get("has_measured").toNumber() > 0 ? "measured" : "predicted",
      })),
    });
  } catch (e) { res.status(500).json({ error: "query failed" }); }
});

// ---- LENS B: a food's tissues, ranked by aggregate expression --
app.get("/food/tissues", async (req, res) => {
  const id = String(req.query.id || "");
  if (!id) return res.json({ tissues: [] });
  try {
    const recs = await read(
      `MATCH (i:Ingredient)-[:CONTAINS]->(c:Compound)-[:TARGETS]->(g:Gene)-[e:EXPRESSED_IN]->(t:Tissue)
       WHERE elementId(i) = $id
       WITH t, sum(coalesce(e.score, 0)) AS total_expr, count(DISTINCT g) AS gene_count
       RETURN t.name AS name, total_expr, gene_count
       ORDER BY total_expr DESC
       LIMIT 12`,
      { id }
    );
    res.json({
      tissues: recs.map(r => ({
        name: r.get("name"),
        score: r.get("total_expr"),
        genes: r.get("gene_count").toNumber(),
      })),
    });
  } catch (e) { res.status(500).json({ error: "query failed" }); }
});

// ---- LENS C: foods most similar by shared compounds -----------
//      weighted toward distinctive compounds (a compound shared by
//      few foods counts more than one in everything).
app.get("/food/similar", async (req, res) => {
  const id = String(req.query.id || "");
  if (!id) return res.json({ foods: [] });
  try {
    const recs = await read(
      `MATCH (i:Ingredient)-[:CONTAINS]->(c:Compound)
       WHERE elementId(i) = $id
       WITH i, c, count{ (c)<-[:CONTAINS]-(:Ingredient) } AS c_freq
       MATCH (c)<-[:CONTAINS]-(other:Ingredient)
       WHERE other <> i
       WITH other,
            count(DISTINCT c) AS shared,
            sum(1.0 / c_freq) AS weighted
       WHERE shared >= 3
       RETURN other.name AS name, other.node_type AS node_type, shared, weighted
       ORDER BY weighted DESC
       LIMIT 12`,
      { id }
    );
    const rows = recs.map(r => ({
      name: r.get("name"),
      node_type: r.get("node_type"),
      shared: r.get("shared").toNumber(),
      weighted: r.get("weighted"),
    }));
    const top = rows.length ? rows[0].weighted : 1;
    for (const row of rows) row.similarity = top > 0 ? Math.round((row.weighted / top) * 100) : 0;
    res.json({ foods: rows });
  } catch (e) { res.status(500).json({ error: "query failed" }); }
});

// ---- curated compound shortlist for a food --------------------
//      hides generic lipids / CoA / phospho junk, surfaces named
//      bioactives. Ranked by how many genes each compound targets.
app.get("/food/compounds", async (req, res) => {
  const id = String(req.query.id || "");
  if (!id) return res.json({ compounds: [] });
  try {
    const recs = await read(
      `MATCH (i:Ingredient)-[:CONTAINS]->(c:Compound)
       WHERE elementId(i) = $id
         AND c.name IS NOT NULL
         AND NOT (
           c.name STARTS WITH 'TG(' OR c.name STARTS WITH 'CL(' OR
           c.name STARTS WITH 'DG(' OR c.name STARTS WITH 'PE(' OR
           c.name STARTS WITH 'PC(' OR c.name STARTS WITH 'PS(' OR
           c.name STARTS WITH 'PA(' OR c.name STARTS WITH 'PI(' OR
           c.name STARTS WITH 'PG(' OR c.name STARTS WITH 'SM(' OR
           c.name STARTS WITH 'CE(' OR c.name STARTS WITH 'MG(' OR
           c.name ENDS WITH '-CoA' OR c.name CONTAINS 'phospho'
         )
       OPTIONAL MATCH (c)-[:TARGETS]->(g:Gene)
       WITH c, count(DISTINCT g) AS gene_count
       WHERE gene_count > 0
       RETURN elementId(c) AS id, c.name AS name, gene_count
       ORDER BY gene_count DESC
       LIMIT 15`,
      { id }
    );
    res.json({
      compounds: recs.map(r => ({
        id: r.get("id"),
        name: r.get("name"),
        genes: r.get("gene_count").toNumber(),
      })),
    });
  } catch (e) { res.status(500).json({ error: "query failed" }); }
});

app.listen(PORT, "127.0.0.1", () => {
  console.log(`NutriGraph read-only API on http://127.0.0.1:${PORT}`);
});
