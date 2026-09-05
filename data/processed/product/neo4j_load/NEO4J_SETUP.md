# Neo4j trimmed graph — public demo setup

Generated: 2026-09-05T05:37:51.200048+00:00 (build timestamp)

This package loads the **trimmed** knowledge graph for an openly explorable Neovis.js
frontend. Safety is a **read-only Neo4j user** (cannot CREATE/DELETE/SET/MERGE/DROP),
not a fixed query allow-list. Exploration uses **parameterized** Cypher templates that
work for any node.

**Deploy note:** strings such as `CHANGE_ME_READONLY_PASSWORD` and `YOUR_VPS_HOST` in
`neovis_config.json` / the Cypher below are intentional placeholders. Replace them on
the server when you create the read-only user; never commit a real password.

## Final counts (this build)

| Kind | Count |
|------|------:|
| Ingredient | 695 |
| Compound | 600 |
| Gene | 1532 |
| Pathway | 4348 |
| Tissue | 68 |
| Nutrient | 224 |
| **Nodes total** | **7467** |
| CONTAINS | 21850 |
| TARGETS | 3893 (measured 454, predicted 3439) |
| IN_PATHWAY | 10875 |
| EXPRESSED_IN | 18204 |
| HAS_NUTRIENT | 24479 |
| **Edges total** | **79301** |

Trim: top `50` compounds/ingredient;
TARGETS keeps all measured + top `20` predicted/compound;
EXPRESSED_IN top `12` tissues/gene by GTEx TPM.

**Do not** load the full 2M+ ingredient→compound layer.

---

## 1. Install Neo4j Community (VPS)

```bash
# Example: Ubuntu — follow current Neo4j Community docs for your distro
sudo apt update
# Install Neo4j Community Edition (see https://neo4j.com/docs/operations-manual/current/installation/)
# Enable and start:
sudo systemctl enable neo4j
sudo systemctl start neo4j
```

Open `http://YOUR_HOST:7474`, set the initial `neo4j` password, confirm Bolt on `7687`.

Memory (trim-sized graph is modest): in `neo4j.conf` something like
`server.memory.heap.initial_size=512m` / `server.memory.heap.max_size=1g` is enough.

Allow the Neo4j `import` directory to read the CSVs from this folder (or copy CSVs into
`$NEO4J_HOME/import/`).

---

## 2. Run the load

### Option A — Python driver (recommended)

```bash
pip install neo4j pandas pyarrow
export NEO4J_URI=bolt://127.0.0.1:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD='your-admin-password'
python scripts/product/build_neo4j_trimmed_graph.py --load
```

Re-export CSVs only (no DB):

```bash
python scripts/product/build_neo4j_trimmed_graph.py
```

### Option B — LOAD CSV

Copy `nodes_*.csv` / `edges_*.csv` into Neo4j's `import/` directory, then run the
statements in `load_csv.cypher` as the admin user.

---

## 3. Indexes (search)

Created by the loader (also in `load_csv.cypher`):

```cypher
CREATE INDEX ingredient_name IF NOT EXISTS FOR (n:Ingredient) ON (n.name);
CREATE INDEX gene_symbol IF NOT EXISTS FOR (n:Gene) ON (n.symbol);
CREATE INDEX compound_name IF NOT EXISTS FOR (n:Compound) ON (n.name);
CREATE INDEX pathway_name IF NOT EXISTS FOR (n:Pathway) ON (n.name);
CREATE INDEX ingredient_id IF NOT EXISTS FOR (n:Ingredient) ON (n.ingredient_id);
CREATE INDEX gene_name IF NOT EXISTS FOR (n:Gene) ON (n.name);
```

---

## 4. Read-only user (required for public demo)

Connect as admin and run:

```cypher
// Neo4j 5 native auth (Community)
CREATE USER readonly IF NOT EXISTS
  SET PASSWORD 'CHANGE_ME_READONLY_PASSWORD'
  SET PASSWORD CHANGE NOT REQUIRED;

GRANT ROLE reader TO readonly;
```

If your install uses the older custom-role style:

```cypher
CREATE ROLE graph_reader IF NOT EXISTS;
GRANT MATCH {*} TO graph_reader;
GRANT TRAVERSE ON GRAPH * NODES * TO graph_reader;
GRANT TRAVERSE ON GRAPH * RELATIONSHIPS * TO graph_reader;
GRANT READ {*} ON GRAPH * NODES * TO graph_reader;
GRANT READ {*} ON GRAPH * RELATIONSHIPS * TO graph_reader;
// Explicitly do NOT grant WRITE / CREATE / DELETE / SET / MERGE / DROP
CREATE USER readonly IF NOT EXISTS SET PASSWORD 'CHANGE_ME_READONLY_PASSWORD' CHANGE NOT REQUIRED;
GRANT ROLE graph_reader TO readonly;
```

Verify write is blocked:

```cypher
// As readonly — must FAIL
CREATE (n:Test {x:1});
```

```cypher
// As readonly — must SUCCEED
MATCH (n) RETURN count(n);
```

Frontend Bolt config:

| Setting | Value |
|---------|-------|
| URI | `bolt://YOUR_VPS_HOST:7687` |
| User | `readonly` |
| Password | the password above |

---

## 5. Parameterized expand queries (open exploration)

Templates live in `cypher_templates.json`. Wire Neovis / your UI to fill `$q`, `$nodeId`,
`$evidence`, `$limit` — these are **not** canned result sets.

**Search (any label):**
```cypher
MATCH (n)
WHERE toLower(coalesce(n.name, n.symbol, '')) CONTAINS toLower($q)
RETURN n
LIMIT 25
```

**Expand any node:**
```cypher
MATCH (n)-[r]-(m)
WHERE id(n) = $nodeId
RETURN n, r, m
LIMIT $limit
```

**Expand TARGETS by honesty toggle:**
```cypher
MATCH (n)-[r:TARGETS]-(m)
WHERE id(n) = $nodeId AND r.evidence = $evidence
RETURN n, r, m
LIMIT $limit
```

On Neo4j 5+, prefer `elementId(n) = $elementId` instead of numeric `id(n)`.

---

## 6. Neovis.js readiness

See `neovis_config.json` for labels, captions, relationship types, and measured vs
predicted edge colors:

| Label | Caption field |
|-------|---------------|
| Ingredient | `name` |
| Compound | `name` |
| Gene | `symbol` |
| Pathway | `name` |
| Tissue | `name` |
| Nutrient | `name` |

| Relationship | Honesty / style |
|--------------|-----------------|
| CONTAINS | Ingredient → Compound |
| TARGETS | `evidence`: measured `#2E7D32` / predicted `#F9A825`; thickness ← `confidence` |
| IN_PATHWAY | Gene → Pathway |
| EXPRESSED_IN | Gene → Tissue (`score`) |
| HAS_NUTRIENT | Ingredient → Nutrient (`amount`, `unit`) |

Click handler: run `expand_neighbors` with the clicked node's id; optional measured-only
toggle uses `expand_targets_by_evidence`.

---

## 7. Optional thin read-only API (before going fully public)

Even with a read-only DB user, put a thin API in front of Bolt if the DB port should not
be exposed:

- Authenticate / rate-limit clients
- Enforce **query timeout** (e.g. 5–10s)
- Enforce **result cap** (e.g. LIMIT ≤ 200)
- Allow only `MATCH` / `RETURN` / `WITH` / `WHERE` / `ORDER BY` / `LIMIT` (reject DDL/write)
- Pass through the same parameterized templates — still open traversal, not a fixed node set

Connection string for the API (server-side): admin or a dedicated `api_reader` with the
same reader role; browsers keep using `readonly` only if you expose Bolt via TLS + auth.

---

## Schema reminder

```
(:Ingredient)-[:CONTAINS]->(:Compound)
(:Compound)-[:TARGETS {evidence, confidence}]->(:Gene)
(:Gene)-[:IN_PATHWAY]->(:Pathway)
(:Gene)-[:EXPRESSED_IN {score}]->(:Tissue)
(:Ingredient)-[:HAS_NUTRIENT {amount, unit}]->(:Nutrient)
```

Sources are listed in `load_manifest.json`. Canonical mechanism files are never rewritten.
