#!/usr/bin/env python3
"""
Build a TRIMMED Neo4j load for the public Neovis.js demo.

Reads canonical / product files only (no mechanism recompute).
Writes CSVs + docs under data/processed/product/neo4j_load/.

Trim rules (browser-visualizable):
  - CONTAINS: top-N compounds per ingredient (default 50), ICC file order
  - TARGETS: all measured + top-K predicted per compound (default 20)
  - EXPRESSED_IN: top-T tissues per gene by GTEx TPM (default 12)
  - IN_PATHWAY / HAS_NUTRIENT: full for nodes present in the trimmed graph

Optional live load: set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD and pass --load.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/product/neo4j_load"

LOCKED_NODES = ROOT / "data/processed/product/universe_v2/ingredient_nodes_v2_locked.json"
PROFILES = ROOT / "data/processed/product/ingredient_profiles_v2.jsonl"
ICC = ROOT / "data/processed/canonical/ingredient_compound_canonical_v2.parquet"
COMPOUND_MASTER = ROOT / "data/processed/canonical/compound_master_v2.parquet"
CG_INTEGRATED = ROOT / "data/processed/integrated/compound_gene_integrated_v1.parquet"
GENE_SETS = ROOT / "data/processed/integrated/ingredient_gene_sets_v3.parquet"
PATHWAY_MAP = ROOT / "data/interim/pathways/gene_pathway_mappings.parquet"
TISSUE_PROFILES = ROOT / "data/processed/tier1/ingredient_tissue_profiles_v2.parquet"
NUTRIENTS = ROOT / "data/processed/product/nutrients/species_nutrient_profiles_production.parquet"
GTEX = ROOT / "data/raw/gtex.gct"

sys.path.insert(0, str(ROOT / "scripts/product"))
from pathway_display_names import PathwayNameResolver  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            clean = {k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames}
            w.writerow(clean)
    return len(rows)


def load_ingredients() -> tuple[list[dict], dict[str, float]]:
    locked = json.loads(LOCKED_NODES.read_text(encoding="utf-8"))["nodes"]
    measured: dict[str, float] = {}
    if PROFILES.exists():
        with PROFILES.open(encoding="utf-8") as fh:
            for line in fh:
                p = json.loads(line)
                iid = p.get("ingredient", {}).get("ingredient_id") or p.get("ingredient", {}).get(
                    "species_id"
                )
                frac = (p.get("provenance") or {}).get("measured_fraction")
                if iid is not None and frac is not None:
                    measured[str(iid)] = float(frac)

    # Fill missing from gene sets
    gs = pd.read_parquet(GENE_SETS, columns=["ingredient_id", "evidence"])
    for iid, g in gs.groupby("ingredient_id"):
        if str(iid) in measured:
            continue
        n = len(g)
        if n:
            measured[str(iid)] = float((g["evidence"] == "measured").sum()) / n

    rows = []
    for n in locked:
        iid = str(n["ingredient_id"])
        rows.append(
            {
                "ingredient_id": iid,
                "name": n.get("canonical_name") or iid,
                "latin": n.get("latin_name") or "",
                "node_type": n.get("node_type") or "",
                "data_status": n.get("data_status") or "",
                "measured_fraction": round(measured.get(iid, 0.0), 4),
            }
        )
    return rows, measured


def select_trimmed_contains(top_n: int) -> pd.DataFrame:
    """Top-N compounds per ingredient in ICC file order (trim only; no re-score)."""
    icc = pd.read_parquet(ICC)
    return icc.groupby("ingredient_id", group_keys=False).head(top_n).copy()


def build_compound_nodes(compound_ids: set[str]) -> list[dict]:
    names: dict[str, str] = {}
    if COMPOUND_MASTER.exists():
        cm = pd.read_parquet(COMPOUND_MASTER, columns=["compound_id", "name"])
        cm = cm[cm["compound_id"].astype(str).isin(compound_ids)]
        for cid, name in zip(cm["compound_id"].astype(str), cm["name"].astype(str)):
            if name and name != "nan":
                names[cid] = name
    return [
        {"compound_id": cid, "name": names.get(cid, cid)}
        for cid in sorted(compound_ids)
    ]


def build_targets(
    compound_ids: set[str], max_predicted_per_compound: int
) -> tuple[list[dict], set[str]]:
    cg = pd.read_parquet(CG_INTEGRATED)
    cg = cg[cg["compound_id"].astype(str).isin(compound_ids)].copy()
    cg["compound_id"] = cg["compound_id"].astype(str)
    cg["gene_symbol"] = cg["gene_symbol"].astype(str)
    cg["evidence"] = cg["evidence"].astype(str).str.lower()
    cg["confidence"] = pd.to_numeric(cg["confidence_weight"], errors="coerce").fillna(0.0)

    measured = cg[cg["evidence"] == "measured"]
    predicted = cg[cg["evidence"] == "predicted"].sort_values(
        ["compound_id", "confidence"], ascending=[True, False]
    )
    predicted = predicted.groupby("compound_id", group_keys=False).head(max_predicted_per_compound)
    kept = pd.concat([measured, predicted], ignore_index=True)
    kept = kept.drop_duplicates(subset=["compound_id", "gene_symbol", "evidence"], keep="first")

    edges = [
        {
            "compound_id": r.compound_id,
            "gene_symbol": r.gene_symbol,
            "evidence": r.evidence,
            "confidence": round(float(r.confidence), 6),
        }
        for r in kept.itertuples(index=False)
    ]
    genes = set(kept["gene_symbol"].tolist())
    return edges, genes


def build_genes(extra_symbols: set[str]) -> tuple[list[dict], set[str]]:
    gs = pd.read_parquet(GENE_SETS, columns=["gene_symbol"])
    symbols = set(gs["gene_symbol"].astype(str)) | set(extra_symbols)
    # Prefer the product gene set size as the primary roster
    primary = sorted(set(gs["gene_symbol"].astype(str)))
    # Include any target-only genes so TARGETS edges resolve
    for s in sorted(extra_symbols - set(primary)):
        primary.append(s)
    rows = [{"gene_symbol": s, "name": s, "symbol": s} for s in primary]
    return rows, set(primary)


def build_pathways(gene_symbols: set[str]) -> tuple[list[dict], list[dict]]:
    gp = pd.read_parquet(PATHWAY_MAP)
    gp = gp[gp["gene_symbol"].astype(str).isin(gene_symbols)].copy()
    resolver = PathwayNameResolver()
    pathway_ids = sorted(gp["pathway_id"].astype(str).unique())
    nodes = []
    for pid in pathway_ids:
        r = resolver.resolve(pid)
        nodes.append(
            {
                "pathway_id": r["pathway"],
                "name": r["pathway_name"],
                "database": "reactome"
                if str(pid).startswith("R-HSA")
                else ("go" if "GO:" in str(pid) or str(pid).startswith("GO:") else "other"),
            }
        )
    # Map original pathway_id strings to stable ids for edges
    id_map = {pid: resolver.resolve(pid)["pathway"] for pid in pathway_ids}
    edges = []
    seen = set()
    for r in gp.itertuples(index=False):
        g = str(r.gene_symbol)
        pid = id_map[str(r.pathway_id)]
        key = (g, pid)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"gene_symbol": g, "pathway_id": pid})
    return nodes, edges


def build_tissues_and_expression(
    gene_symbols: set[str], top_tissues: int
) -> tuple[list[dict], list[dict]]:
    tp = pd.read_parquet(TISSUE_PROFILES, columns=["tissue"])
    tissue_names = sorted(tp["tissue"].astype(str).unique())
    tissue_nodes = [{"tissue_id": t, "name": t.replace("_", " ")} for t in tissue_names]
    tissue_set = set(tissue_names)

    if not GTEX.exists():
        return tissue_nodes, []

    edges: list[dict] = []
    with GTEX.open(encoding="utf-8") as fh:
        fh.readline()  # version
        fh.readline()  # dims
        header = fh.readline().rstrip("\n").split("\t")
        # header: Name, Description, tissue...
        tissues = header[2:]
        # Align GTEx column names to tissue profile names (already underscored)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            symbol = parts[1]
            if symbol not in gene_symbols:
                continue
            scores = []
            for tissue, val in zip(tissues, parts[2:]):
                if tissue not in tissue_set:
                    continue
                try:
                    tpm = float(val)
                except ValueError:
                    continue
                if tpm > 0:
                    scores.append((tissue, tpm))
            scores.sort(key=lambda x: x[1], reverse=True)
            for tissue, tpm in scores[:top_tissues]:
                edges.append(
                    {
                        "gene_symbol": symbol,
                        "tissue_id": tissue,
                        "score": round(tpm, 6),
                    }
                )
    return tissue_nodes, edges


def build_nutrients(ingredient_ids: set[str]) -> tuple[list[dict], list[dict]]:
    nut = pd.read_parquet(NUTRIENTS)
    # species_id aligns with ingredient_id for SP_*; ING_* may lack nutrients
    nut = nut[nut["species_id"].astype(str).isin(ingredient_ids)].copy()
    nodes = []
    seen = set()
    for r in nut[["nutrient_id", "nutrient_name", "nutrient_group"]].drop_duplicates().itertuples(
        index=False
    ):
        nid = str(r.nutrient_id)
        if nid in seen:
            continue
        seen.add(nid)
        nodes.append(
            {
                "nutrient_id": nid,
                "name": str(r.nutrient_name),
                "group": str(r.nutrient_group) if pd.notna(r.nutrient_group) else "",
            }
        )
    edges = []
    for r in nut.itertuples(index=False):
        edges.append(
            {
                "ingredient_id": str(r.species_id),
                "nutrient_id": str(r.nutrient_id),
                "amount": "" if pd.isna(r.amount) else round(float(r.amount), 6),
                "unit": "" if pd.isna(r.unit) else str(r.unit),
            }
        )
    return nodes, edges


def write_docs(out: Path, counts: dict[str, Any]) -> None:
    templates = {
        "search_by_name": {
            "description": "Search any node by name (Gene.name == symbol).",
            "cypher": (
                "MATCH (n)\n"
                "WHERE toLower(coalesce(n.name, n.symbol, '')) CONTAINS toLower($q)\n"
                "RETURN n\n"
                "LIMIT 25"
            ),
            "params": {"q": "turmeric"},
        },
        "expand_neighbors": {
            "description": "Expand ANY clicked node (works for all labels).",
            "cypher": (
                "MATCH (n)-[r]-(m)\n"
                "WHERE id(n) = $nodeId\n"
                "RETURN n, r, m\n"
                "LIMIT $limit"
            ),
            "params": {"nodeId": 123, "limit": 50},
            "neo4j5_note": "Prefer elementId(n) = $elementId on Neo4j 5+.",
            "cypher_neo4j5": (
                "MATCH (n)-[r]-(m)\n"
                "WHERE elementId(n) = $elementId\n"
                "RETURN n, r, m\n"
                "LIMIT $limit"
            ),
        },
        "expand_targets_by_evidence": {
            "description": "Expand TARGETS filtered by honesty (measured|predicted).",
            "cypher": (
                "MATCH (n)-[r:TARGETS]-(m)\n"
                "WHERE id(n) = $nodeId AND r.evidence = $evidence\n"
                "RETURN n, r, m\n"
                "LIMIT $limit"
            ),
            "params": {"nodeId": 123, "evidence": "measured", "limit": 50},
        },
        "expand_from_ingredient_id": {
            "description": "Seed view from a known ingredient_id (search → expand).",
            "cypher": (
                "MATCH (i:Ingredient {ingredient_id: $ingredientId})-[r]-(m)\n"
                "RETURN i, r, m\n"
                "LIMIT $limit"
            ),
            "params": {"ingredientId": "SP_000042", "limit": 50},
        },
        "expand_from_gene_symbol": {
            "description": "Seed view from a gene symbol.",
            "cypher": (
                "MATCH (g:Gene {symbol: $symbol})-[r]-(m)\n"
                "RETURN g, r, m\n"
                "LIMIT $limit"
            ),
            "params": {"symbol": "TNF", "limit": 50},
        },
    }
    (out / "cypher_templates.json").write_text(
        json.dumps(templates, indent=2) + "\n", encoding="utf-8"
    )

    neovis = {
        "serverUrl": "bolt://YOUR_VPS_HOST:7687",
        "serverUser": "readonly",
        "serverPassword": "CHANGE_ME_READONLY_PASSWORD",
        "initialCypher": (
            "MATCH (n) WHERE toLower(coalesce(n.name, n.symbol, '')) "
            "CONTAINS toLower($q) RETURN n LIMIT 25"
        ),
        "labels": {
            "Ingredient": {
                "caption": "name",
                "size": "measured_fraction",
                "community": "node_type",
            },
            "Compound": {"caption": "name"},
            "Gene": {"caption": "symbol"},
            "Pathway": {"caption": "name"},
            "Tissue": {"caption": "name"},
            "Nutrient": {"caption": "name"},
        },
        "relationships": {
            "CONTAINS": {"caption": False, "thickness": 1},
            "TARGETS": {
                "caption": "evidence",
                "thickness": "confidence",
                "color_by_property": {
                    "property": "evidence",
                    "map": {
                        "measured": "#2E7D32",
                        "predicted": "#F9A825",
                    },
                },
            },
            "IN_PATHWAY": {"caption": False},
            "EXPRESSED_IN": {"caption": False, "thickness": "score"},
            "HAS_NUTRIENT": {"caption": False},
        },
        "notes": [
            "Neovis.js does not natively map relationship color by property in all versions; "
            "apply measured/predicted colors in a post-query style hook or via vis.js edge options "
            "when assembling the network from Cypher results.",
            "Connect as the read-only Neo4j user only.",
            "Use parameterized expand templates from cypher_templates.json on node click.",
        ],
        "click_to_expand": {
            "query": templates["expand_neighbors"]["cypher"],
            "limit": 50,
            "evidence_toggle_query": templates["expand_targets_by_evidence"]["cypher"],
        },
    }
    (out / "neovis_config.json").write_text(json.dumps(neovis, indent=2) + "\n", encoding="utf-8")

    setup = f"""# Neo4j trimmed graph — public demo setup

Generated: {counts.get("generated_at")}

This package loads the **trimmed** knowledge graph for an openly explorable Neovis.js
frontend. Safety is a **read-only Neo4j user** (cannot CREATE/DELETE/SET/MERGE/DROP),
not a fixed query allow-list. Exploration uses **parameterized** Cypher templates that
work for any node.

## Final counts (this build)

| Kind | Count |
|------|------:|
| Ingredient | {counts["nodes"]["Ingredient"]} |
| Compound | {counts["nodes"]["Compound"]} |
| Gene | {counts["nodes"]["Gene"]} |
| Pathway | {counts["nodes"]["Pathway"]} |
| Tissue | {counts["nodes"]["Tissue"]} |
| Nutrient | {counts["nodes"]["Nutrient"]} |
| **Nodes total** | **{counts["nodes"]["total"]}** |
| CONTAINS | {counts["edges"]["CONTAINS"]} |
| TARGETS | {counts["edges"]["TARGETS"]} (measured {counts["edges"]["TARGETS_measured"]}, predicted {counts["edges"]["TARGETS_predicted"]}) |
| IN_PATHWAY | {counts["edges"]["IN_PATHWAY"]} |
| EXPRESSED_IN | {counts["edges"]["EXPRESSED_IN"]} |
| HAS_NUTRIENT | {counts["edges"]["HAS_NUTRIENT"]} |
| **Edges total** | **{counts["edges"]["total"]}** |

Trim: top `{counts["trim"]["compounds_per_ingredient"]}` compounds/ingredient;
TARGETS keeps all measured + top `{counts["trim"]["predicted_targets_per_compound"]}` predicted/compound;
EXPRESSED_IN top `{counts["trim"]["tissues_per_gene"]}` tissues/gene by GTEx TPM.

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
GRANT MATCH {{*}} TO graph_reader;
GRANT TRAVERSE ON GRAPH * NODES * TO graph_reader;
GRANT TRAVERSE ON GRAPH * RELATIONSHIPS * TO graph_reader;
GRANT READ {{*}} ON GRAPH * NODES * TO graph_reader;
GRANT READ {{*}} ON GRAPH * RELATIONSHIPS * TO graph_reader;
// Explicitly do NOT grant WRITE / CREATE / DELETE / SET / MERGE / DROP
CREATE USER readonly IF NOT EXISTS SET PASSWORD 'CHANGE_ME_READONLY_PASSWORD' CHANGE NOT REQUIRED;
GRANT ROLE graph_reader TO readonly;
```

Verify write is blocked:

```cypher
// As readonly — must FAIL
CREATE (n:Test {{x:1}});
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
(:Compound)-[:TARGETS {{evidence, confidence}}]->(:Gene)
(:Gene)-[:IN_PATHWAY]->(:Pathway)
(:Gene)-[:EXPRESSED_IN {{score}}]->(:Tissue)
(:Ingredient)-[:HAS_NUTRIENT {{amount, unit}}]->(:Nutrient)
```

Sources are listed in `load_manifest.json`. Canonical mechanism files are never rewritten.
"""
    (out / "NEO4J_SETUP.md").write_text(setup, encoding="utf-8")


def write_load_csv_cypher(out: Path) -> None:
    cypher = """// LOAD CSV script for Neo4j Community (admin user)
// Copy CSV files into $NEO4J_HOME/import/ first.
// Paths below assume files are named as in this package.

// --- constraints / indexes ---
CREATE CONSTRAINT ingredient_id IF NOT EXISTS FOR (n:Ingredient) REQUIRE n.ingredient_id IS UNIQUE;
CREATE CONSTRAINT compound_id IF NOT EXISTS FOR (n:Compound) REQUIRE n.compound_id IS UNIQUE;
CREATE CONSTRAINT gene_symbol IF NOT EXISTS FOR (n:Gene) REQUIRE n.symbol IS UNIQUE;
CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (n:Pathway) REQUIRE n.pathway_id IS UNIQUE;
CREATE CONSTRAINT tissue_id IF NOT EXISTS FOR (n:Tissue) REQUIRE n.tissue_id IS UNIQUE;
CREATE CONSTRAINT nutrient_id IF NOT EXISTS FOR (n:Nutrient) REQUIRE n.nutrient_id IS UNIQUE;

CREATE INDEX ingredient_name IF NOT EXISTS FOR (n:Ingredient) ON (n.name);
CREATE INDEX compound_name IF NOT EXISTS FOR (n:Compound) ON (n.name);
CREATE INDEX gene_name IF NOT EXISTS FOR (n:Gene) ON (n.name);
CREATE INDEX pathway_name IF NOT EXISTS FOR (n:Pathway) ON (n.name);

// --- nodes ---
LOAD CSV WITH HEADERS FROM 'file:///nodes_ingredient.csv' AS row
MERGE (n:Ingredient {ingredient_id: row.ingredient_id})
SET n.name = row.name,
    n.latin = row.latin,
    n.node_type = row.node_type,
    n.data_status = row.data_status,
    n.measured_fraction = toFloat(row.measured_fraction);

LOAD CSV WITH HEADERS FROM 'file:///nodes_compound.csv' AS row
MERGE (n:Compound {compound_id: row.compound_id})
SET n.name = row.name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_gene.csv' AS row
MERGE (n:Gene {symbol: row.symbol})
SET n.name = row.name, n.gene_symbol = row.gene_symbol;

LOAD CSV WITH HEADERS FROM 'file:///nodes_pathway.csv' AS row
MERGE (n:Pathway {pathway_id: row.pathway_id})
SET n.name = row.name, n.database = row.database;

LOAD CSV WITH HEADERS FROM 'file:///nodes_tissue.csv' AS row
MERGE (n:Tissue {tissue_id: row.tissue_id})
SET n.name = row.name;

LOAD CSV WITH HEADERS FROM 'file:///nodes_nutrient.csv' AS row
MERGE (n:Nutrient {nutrient_id: row.nutrient_id})
SET n.name = row.name, n.group = row.group;

// --- edges ---
LOAD CSV WITH HEADERS FROM 'file:///edges_contains.csv' AS row
MATCH (a:Ingredient {ingredient_id: row.ingredient_id})
MATCH (b:Compound {compound_id: row.compound_id})
MERGE (a)-[:CONTAINS]->(b);

LOAD CSV WITH HEADERS FROM 'file:///edges_targets.csv' AS row
MATCH (a:Compound {compound_id: row.compound_id})
MATCH (b:Gene {symbol: row.gene_symbol})
MERGE (a)-[r:TARGETS]->(b)
SET r.evidence = row.evidence, r.confidence = toFloat(row.confidence);

LOAD CSV WITH HEADERS FROM 'file:///edges_in_pathway.csv' AS row
MATCH (a:Gene {symbol: row.gene_symbol})
MATCH (b:Pathway {pathway_id: row.pathway_id})
MERGE (a)-[:IN_PATHWAY]->(b);

LOAD CSV WITH HEADERS FROM 'file:///edges_expressed_in.csv' AS row
MATCH (a:Gene {symbol: row.gene_symbol})
MATCH (b:Tissue {tissue_id: row.tissue_id})
MERGE (a)-[r:EXPRESSED_IN]->(b)
SET r.score = toFloat(row.score);

LOAD CSV WITH HEADERS FROM 'file:///edges_has_nutrient.csv' AS row
MATCH (a:Ingredient {ingredient_id: row.ingredient_id})
MATCH (b:Nutrient {nutrient_id: row.nutrient_id})
MERGE (a)-[r:HAS_NUTRIENT]->(b)
SET r.amount = CASE WHEN row.amount = '' THEN null ELSE toFloat(row.amount) END,
    r.unit = row.unit;
"""
    (out / "load_csv.cypher").write_text(cypher, encoding="utf-8")


def load_into_neo4j(out: Path, uri: str, user: str, password: str, clear: bool) -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise SystemExit("Install neo4j driver: pip install neo4j") from e

    driver = GraphDatabase.driver(uri, auth=(user, password))

    def run_batches(session, query: str, rows: list[dict], batch: int = 500) -> None:
        for i in range(0, len(rows), batch):
            chunk = rows[i : i + batch]
            session.run(query, {"rows": chunk})

    def read_csv_rows(name: str) -> list[dict]:
        path = out / name
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    with driver.session() as session:
        if clear:
            session.run("MATCH (n) DETACH DELETE n")

        for stmt in [
            "CREATE CONSTRAINT ingredient_id IF NOT EXISTS FOR (n:Ingredient) REQUIRE n.ingredient_id IS UNIQUE",
            "CREATE CONSTRAINT compound_id IF NOT EXISTS FOR (n:Compound) REQUIRE n.compound_id IS UNIQUE",
            "CREATE CONSTRAINT gene_symbol_uq IF NOT EXISTS FOR (n:Gene) REQUIRE n.symbol IS UNIQUE",
            "CREATE CONSTRAINT pathway_id IF NOT EXISTS FOR (n:Pathway) REQUIRE n.pathway_id IS UNIQUE",
            "CREATE CONSTRAINT tissue_id IF NOT EXISTS FOR (n:Tissue) REQUIRE n.tissue_id IS UNIQUE",
            "CREATE CONSTRAINT nutrient_id IF NOT EXISTS FOR (n:Nutrient) REQUIRE n.nutrient_id IS UNIQUE",
            "CREATE INDEX ingredient_name IF NOT EXISTS FOR (n:Ingredient) ON (n.name)",
            "CREATE INDEX compound_name IF NOT EXISTS FOR (n:Compound) ON (n.name)",
            "CREATE INDEX gene_name IF NOT EXISTS FOR (n:Gene) ON (n.name)",
            "CREATE INDEX pathway_name IF NOT EXISTS FOR (n:Pathway) ON (n.name)",
        ]:
            session.run(stmt)

        ings = read_csv_rows("nodes_ingredient.csv")
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (n:Ingredient {ingredient_id: row.ingredient_id})
            SET n.name = row.name, n.latin = row.latin, n.node_type = row.node_type,
                n.data_status = row.data_status,
                n.measured_fraction = toFloat(row.measured_fraction)
            """,
            ings,
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (n:Compound {compound_id: row.compound_id})
            SET n.name = row.name
            """,
            read_csv_rows("nodes_compound.csv"),
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (n:Gene {symbol: row.symbol})
            SET n.name = row.name, n.gene_symbol = row.gene_symbol
            """,
            read_csv_rows("nodes_gene.csv"),
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (n:Pathway {pathway_id: row.pathway_id})
            SET n.name = row.name, n.database = row.database
            """,
            read_csv_rows("nodes_pathway.csv"),
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (n:Tissue {tissue_id: row.tissue_id})
            SET n.name = row.name
            """,
            read_csv_rows("nodes_tissue.csv"),
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MERGE (n:Nutrient {nutrient_id: row.nutrient_id})
            SET n.name = row.name, n.group = row.group
            """,
            read_csv_rows("nodes_nutrient.csv"),
        )

        run_batches(
            session,
            """
            UNWIND $rows AS row
            MATCH (a:Ingredient {ingredient_id: row.ingredient_id})
            MATCH (b:Compound {compound_id: row.compound_id})
            MERGE (a)-[:CONTAINS]->(b)
            """,
            read_csv_rows("edges_contains.csv"),
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MATCH (a:Compound {compound_id: row.compound_id})
            MATCH (b:Gene {symbol: row.gene_symbol})
            MERGE (a)-[r:TARGETS]->(b)
            SET r.evidence = row.evidence, r.confidence = toFloat(row.confidence)
            """,
            read_csv_rows("edges_targets.csv"),
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MATCH (a:Gene {symbol: row.gene_symbol})
            MATCH (b:Pathway {pathway_id: row.pathway_id})
            MERGE (a)-[:IN_PATHWAY]->(b)
            """,
            read_csv_rows("edges_in_pathway.csv"),
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MATCH (a:Gene {symbol: row.gene_symbol})
            MATCH (b:Tissue {tissue_id: row.tissue_id})
            MERGE (a)-[r:EXPRESSED_IN]->(b)
            SET r.score = toFloat(row.score)
            """,
            read_csv_rows("edges_expressed_in.csv"),
        )
        run_batches(
            session,
            """
            UNWIND $rows AS row
            MATCH (a:Ingredient {ingredient_id: row.ingredient_id})
            MATCH (b:Nutrient {nutrient_id: row.nutrient_id})
            MERGE (a)-[r:HAS_NUTRIENT]->(b)
            SET r.amount = CASE WHEN row.amount = '' THEN null ELSE toFloat(row.amount) END,
                r.unit = row.unit
            """,
            read_csv_rows("edges_has_nutrient.csv"),
        )

    driver.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-compounds", type=int, default=50)
    ap.add_argument("--max-predicted-targets", type=int, default=20)
    ap.add_argument("--top-tissues", type=int, default=12)
    ap.add_argument("--load", action="store_true", help="Load into Neo4j using env credentials")
    ap.add_argument("--clear", action="store_true", help="DETACH DELETE all nodes before load")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("Loading ingredients…")
    ingredients, _ = load_ingredients()
    ingredient_ids = {r["ingredient_id"] for r in ingredients}

    print(f"Selecting top-{args.top_compounds} compounds per ingredient…")
    contains_df = select_trimmed_contains(args.top_compounds)
    # Only ingredients in locked universe
    contains_df = contains_df[contains_df["ingredient_id"].astype(str).isin(ingredient_ids)]
    compound_ids = set(contains_df["compound_id"].astype(str))

    print("Building compound nodes + TARGETS…")
    compounds = build_compound_nodes(compound_ids)
    targets, target_genes = build_targets(compound_ids, args.max_predicted_targets)

    print("Building genes / pathways / tissues / nutrients…")
    genes, gene_symbols = build_genes(target_genes)
    pathways, in_pathway = build_pathways(gene_symbols)
    tissues, expressed = build_tissues_and_expression(gene_symbols, args.top_tissues)
    nutrients, has_nutrient = build_nutrients(ingredient_ids)

    contains_edges = [
        {
            "ingredient_id": str(r.ingredient_id),
            "compound_id": str(r.compound_id),
        }
        for r in contains_df.itertuples(index=False)
    ]

    # Write CSVs
    n_ing = _write_csv(
        out / "nodes_ingredient.csv",
        ingredients,
        ["ingredient_id", "name", "latin", "node_type", "data_status", "measured_fraction"],
    )
    n_comp = _write_csv(out / "nodes_compound.csv", compounds, ["compound_id", "name"])
    n_gene = _write_csv(out / "nodes_gene.csv", genes, ["gene_symbol", "symbol", "name"])
    n_path = _write_csv(out / "nodes_pathway.csv", pathways, ["pathway_id", "name", "database"])
    n_tiss = _write_csv(out / "nodes_tissue.csv", tissues, ["tissue_id", "name"])
    n_nut = _write_csv(out / "nodes_nutrient.csv", nutrients, ["nutrient_id", "name", "group"])

    e_contains = _write_csv(
        out / "edges_contains.csv", contains_edges, ["ingredient_id", "compound_id"]
    )
    e_targets = _write_csv(
        out / "edges_targets.csv",
        targets,
        ["compound_id", "gene_symbol", "evidence", "confidence"],
    )
    e_path = _write_csv(
        out / "edges_in_pathway.csv", in_pathway, ["gene_symbol", "pathway_id"]
    )
    e_expr = _write_csv(
        out / "edges_expressed_in.csv",
        expressed,
        ["gene_symbol", "tissue_id", "score"],
    )
    e_nut = _write_csv(
        out / "edges_has_nutrient.csv",
        has_nutrient,
        ["ingredient_id", "nutrient_id", "amount", "unit"],
    )

    targets_m = sum(1 for t in targets if t["evidence"] == "measured")
    targets_p = sum(1 for t in targets if t["evidence"] == "predicted")
    node_total = n_ing + n_comp + n_gene + n_path + n_tiss + n_nut
    edge_total = e_contains + e_targets + e_path + e_expr + e_nut

    counts = {
        "generated_at": _utc_now(),
        "scope": "trimmed visualizable graph for public Neovis demo",
        "trim": {
            "compounds_per_ingredient": args.top_compounds,
            "predicted_targets_per_compound": args.max_predicted_targets,
            "tissues_per_gene": args.top_tissues,
            "note": "Full 2M+ CONTAINS layer NOT loaded",
        },
        "nodes": {
            "Ingredient": n_ing,
            "Compound": n_comp,
            "Gene": n_gene,
            "Pathway": n_path,
            "Tissue": n_tiss,
            "Nutrient": n_nut,
            "total": node_total,
        },
        "edges": {
            "CONTAINS": e_contains,
            "TARGETS": e_targets,
            "TARGETS_measured": targets_m,
            "TARGETS_predicted": targets_p,
            "IN_PATHWAY": e_path,
            "EXPRESSED_IN": e_expr,
            "HAS_NUTRIENT": e_nut,
            "total": edge_total,
        },
        "sources": {
            "ingredients": str(LOCKED_NODES.relative_to(ROOT)),
            "profiles_measured_fraction": str(PROFILES.relative_to(ROOT)),
            "contains": str(ICC.relative_to(ROOT)),
            "targets": str(CG_INTEGRATED.relative_to(ROOT)),
            "genes": str(GENE_SETS.relative_to(ROOT)),
            "pathways": str(PATHWAY_MAP.relative_to(ROOT)),
            "tissues": str(TISSUE_PROFILES.relative_to(ROOT)),
            "expression": str(GTEX.relative_to(ROOT)),
            "nutrients": str(NUTRIENTS.relative_to(ROOT)),
        },
        "honesty": {
            "property": "TARGETS.evidence",
            "values": ["measured", "predicted"],
            "confidence_property": "TARGETS.confidence",
        },
        "safety": {
            "model": "read-only Neo4j user + open parameterized MATCH templates",
            "not": "fixed canned query set / limited node allow-list",
        },
    }
    (out / "load_manifest.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    write_docs(out, counts)
    write_load_csv_cypher(out)

    print(json.dumps(counts["nodes"], indent=2))
    print(json.dumps(counts["edges"], indent=2))
    print(f"Wrote package -> {out}")

    if args.load:
        uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")
        if not password:
            raise SystemExit("NEO4J_PASSWORD required for --load")
        print(f"Loading into {uri} as {user}…")
        load_into_neo4j(out, uri, user, password, clear=args.clear)
        print("Load complete.")


if __name__ == "__main__":
    main()
