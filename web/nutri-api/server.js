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

// expand any node by elementId
app.get("/expand", async (req, res) => {
  const elementId = String(req.query.id || "");
  const evidence = req.query.evidence ? String(req.query.evidence) : null;
  if (!elementId) return res.json({ nodes: [], edges: [] });
  try {
    let recs;
    if (evidence === "measured" || evidence === "predicted") {
      recs = await read(
        "MATCH (n)-[r:TARGETS]-(m) WHERE elementId(n) = $elementId AND r.evidence = $evidence RETURN n, r, m LIMIT 60",
        { elementId, evidence }
      );
    } else {
      recs = await read(
        "MATCH (n)-[r]-(m) WHERE elementId(n) = $elementId RETURN n, r, m LIMIT 60",
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

app.listen(PORT, "127.0.0.1", () => {
  console.log(`NutriGraph read-only API on http://127.0.0.1:${PORT}`);
});
