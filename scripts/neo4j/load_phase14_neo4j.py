"""
Load Phase14 Neo4j CSVs (nodes.csv, edges.csv) into a Neo4j database.
Uses batched UNWIND for nodes and edges; MATCH by label+id to avoid cartesian product.
Supports --wipe, --database; prints progress and post-load sanity stats.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Batch sizes for UNWIND (tune for 5k–20k)
DEFAULT_NODE_BATCH_SIZE = 10_000
DEFAULT_EDGE_BATCH_SIZE = 10_000


def _require_driver():
    try:
        import neo4j
        return neo4j
    except ImportError:
        raise ImportError(
            "Neo4j Python driver not installed. Install with: pip install neo4j\n"
            "Then set NEO4J_PASSWORD or pass --password."
        ) from None


def _read_csv_headers(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        row = next(r, None)
        return list(row) if row else []


def _read_nodes(run_dir: Path) -> List[Dict[str, Any]]:
    nodes_path = run_dir / "neo4j" / "nodes.csv"
    if not nodes_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(nodes_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: v for k, v in row.items() if v is not None and str(v).strip() != ""})
    return rows


def _read_edges(run_dir: Path) -> List[Dict[str, Any]]:
    edges_path = run_dir / "neo4j" / "edges.csv"
    if not edges_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(edges_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: v for k, v in row.items() if v is not None and str(v).strip() != ""})
    return rows


def _id_col(nodes_headers: List[str]) -> Optional[str]:
    for h in (":ID", "node_id", "id"):
        if h in nodes_headers:
            return h
    return nodes_headers[0] if nodes_headers else None


def _label_col(nodes_headers: List[str]) -> Optional[str]:
    for h in (":LABEL", "label", ":LABEL(S)"):
        if h in nodes_headers:
            return h
    return None


def _start_end_type(edges_headers: List[str]) -> Tuple[str, str, str]:
    start = ":START_ID" if ":START_ID" in edges_headers else "source_id"
    end = ":END_ID" if ":END_ID" in edges_headers else "target_id"
    etype = ":TYPE" if ":TYPE" in edges_headers else "edge_type"
    return start, end, etype


def _batches(items: List[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def create_constraints_indexes(
    driver, database: str, node_labels: Set[str], wipe: bool
) -> None:
    """Create uniqueness constraints / indexes per label. If wipe, DETACH DELETE by label."""
    with driver.session(database=database) as session:
        if wipe:
            for label in node_labels:
                try:
                    session.run(f"MATCH (n:{label}) DETACH DELETE n")
                    logger.info("Wiped label %s", label)
                except Exception as e:
                    logger.warning("Wipe %s: %s", label, e)
        for label in node_labels:
            try:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                )
            except Exception as e:
                logger.debug("Constraint %s: %s", label, e)
            try:
                session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.id)")
            except Exception as e:
                logger.debug("Index %s: %s", label, e)
    logger.info("Constraints/indexes ensured for %s", sorted(node_labels))


def load_nodes_batched_no_apoc(
    driver,
    database: str,
    nodes: List[Dict[str, Any]],
    id_col: str,
    label_col: str,
    batch_size: int,
) -> int:
    """Variant without APOC: SET each property explicitly from batch row."""
    if not nodes:
        return 0
    by_label: Dict[str, List[Dict]] = {}
    for row in nodes:
        nid = row.get(id_col)
        label = (row.get(label_col) or "Node").strip()
        if not nid or not label:
            continue
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(row)

    total = 0
    with driver.session(database=database) as session:
        for label, group in by_label.items():
            prop_keys = sorted(set(k for r in group for k in r.keys() if k not in (id_col, label_col)))
            n_batches = (len(group) + batch_size - 1) // batch_size
            start_t = time.perf_counter()
            for batch_idx, batch in enumerate(_batches(group, batch_size)):
                batch_data = []
                for row in batch:
                    nid = row.get(id_col)
                    if not nid:
                        continue
                    rec = {"id": nid}
                    for k in prop_keys:
                        rec[k] = row.get(k)
                    batch_data.append(rec)
                if not batch_data:
                    continue
                set_clause = ", ".join([f"n.{k} = row.{k}" for k in prop_keys])
                query = f"UNWIND $batch AS row MERGE (n:{label} {{id: row.id}}) SET {set_clause}"
                session.run(query, {"batch": batch_data})
                total += len(batch_data)
                elapsed = time.perf_counter() - start_t
                done = batch_idx + 1
                remaining = n_batches - done
                eta = (elapsed / done) * remaining if done else 0
                logger.info(
                    "Nodes %s: batch %d/%d, %d so far, elapsed %.1fs, ~%.1fs remaining",
                    label, done, n_batches, total, elapsed, eta,
                )
    return total


def load_edges_batched(
    driver,
    database: str,
    edges: List[Dict[str, Any]],
    start_col: str,
    end_col: str,
    type_col: str,
    id_to_label: Dict[str, str],
    batch_size: int,
) -> int:
    """Load edges with UNWIND. MATCH (a:StartLabel {id}), (b:EndLabel {id}) per batch to use indexes and avoid cartesian."""
    if not edges:
        return 0
    # Group by (start_label, end_label, rel_type) so each query uses indexed MATCH on both sides
    key_to_edges: Dict[Tuple[str, str, str], List[Dict]] = {}
    for row in edges:
        start_id = row.get(start_col)
        end_id = row.get(end_col)
        rel_type = (row.get(type_col) or "RELATES").strip().replace(" ", "_").replace("-", "_")
        if not start_id or not end_id or not rel_type:
            continue
        start_label = id_to_label.get(start_id)
        end_label = id_to_label.get(end_id)
        if not start_label or not end_label:
            continue
        key = (start_label, end_label, rel_type)
        if key not in key_to_edges:
            key_to_edges[key] = []
        key_to_edges[key].append(row)

    total = 0
    with driver.session(database=database) as session:
        for (start_label, end_label, rel_type), group in key_to_edges.items():
            n_batches = (len(group) + batch_size - 1) // batch_size
            start_t = time.perf_counter()
            all_extra = set()
            for row in group:
                all_extra.update(k for k in row.keys() if k not in (start_col, end_col, type_col))
            extra_props = sorted(all_extra)
            for batch_idx, batch in enumerate(_batches(group, batch_size)):
                batch_data = []
                for row in batch:
                    rec = {"start_id": row[start_col], "end_id": row[end_col]}
                    for k in extra_props:
                        v = row.get(k)
                        if v is not None and str(v).strip() != "":
                            rec[k] = v
                    batch_data.append(rec)
                if not batch_data:
                    continue
                set_parts = [f"r.{k} = row.{k}" for k in extra_props]
                set_clause = " SET " + ", ".join(set_parts) if set_parts else ""
                query = (
                    f"UNWIND $batch AS row "
                    f"MATCH (a:{start_label} {{id: row.start_id}}) "
                    f"MATCH (b:{end_label} {{id: row.end_id}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b){set_clause}"
                )
                session.run(query, {"batch": batch_data})
                total += len(batch_data)
                elapsed = time.perf_counter() - start_t
                done = batch_idx + 1
                remaining = n_batches - done
                eta = (elapsed / done) * remaining if done else 0
                logger.info(
                    "Edges %s->%s %s: batch %d/%d, %d so far, elapsed %.1fs, ~%.1fs remaining",
                    start_label, end_label, rel_type, done, n_batches, total, elapsed, eta,
                )
    return total


def run_sanity_queries(driver, database: str) -> None:
    """Run node/rel counts, labels, relationship types and print."""
    with driver.session(database=database) as session:
        r = session.run("MATCH (n) RETURN count(n) AS c")
        node_count = r.single()["c"] or 0
        r = session.run("MATCH ()-[r]->() RETURN count(r) AS c")
        rel_count = r.single()["c"] or 0
        labels: List[str] = []
        rel_types: List[str] = []
        try:
            r = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
            labels = [row["label"] for row in r]
        except Exception as e:
            logger.warning("db.labels() not available: %s", e)
        try:
            r = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType")
            rel_types = [row["relationshipType"] for row in r]
        except Exception as e:
            logger.warning("db.relationshipTypes() not available: %s", e)
    print("\n--- Sanity (post-load) ---")
    print(f"Node count: {node_count}")
    print(f"Relationship count: {rel_count}")
    print(f"Labels: {labels}")
    print(f"Relationship types: {rel_types}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Load Phase14 Neo4j CSVs into Neo4j")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to phase14 run directory (contains neo4j/)")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--user", type=str, default="neo4j")
    parser.add_argument("--password", type=str, default=None, help="Or set NEO4J_PASSWORD")
    parser.add_argument("--database", type=str, default="neo4j")
    parser.add_argument("--wipe", action="store_true", help="Delete existing Phase14 nodes by label before load")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--node-batch-size", type=int, default=DEFAULT_NODE_BATCH_SIZE)
    parser.add_argument("--edge-batch-size", type=int, default=DEFAULT_EDGE_BATCH_SIZE)
    args = parser.parse_args()

    run_dir = (args.repo_root / args.run_dir.replace("\\", "/")).resolve()
    if not run_dir.exists():
        logger.error("Run dir not found: %s", run_dir)
        return 1
    neo_dir = run_dir / "neo4j"
    if not neo_dir.exists():
        logger.error("neo4j/ not found under %s", run_dir)
        return 1
    nodes_path = neo_dir / "nodes.csv"
    edges_path = neo_dir / "edges.csv"
    if not nodes_path.exists() or not edges_path.exists():
        logger.error("nodes.csv or edges.csv missing in %s", neo_dir)
        return 1

    password = args.password or os.environ.get("NEO4J_PASSWORD")
    if not password:
        logger.error("Provide --password or set NEO4J_PASSWORD")
        return 1

    neo4j = _require_driver()
    driver = neo4j.GraphDatabase.driver(args.neo4j_uri, auth=(args.user, password))
    try:
        t0 = time.perf_counter()
        nodes = _read_nodes(run_dir)
        edges = _read_edges(run_dir)
        nodes_headers = _read_csv_headers(nodes_path)
        edges_headers = _read_csv_headers(edges_path)
        id_col = _id_col(nodes_headers) or ":ID"
        label_col = _label_col(nodes_headers) or ":LABEL"
        start_col, end_col, type_col = _start_end_type(edges_headers)

        node_labels: Set[str] = set()
        for row in nodes:
            lb = row.get(label_col)
            if lb:
                node_labels.add(lb.strip())
        id_to_label = {row[id_col]: (row.get(label_col) or "Node").strip() for row in nodes if row.get(id_col)}

        logger.info("Discovered node labels: %s", sorted(node_labels))
        create_constraints_indexes(driver, args.database, node_labels, args.wipe)

        n_loaded = load_nodes_batched_no_apoc(
            driver, args.database, nodes, id_col, label_col, args.node_batch_size
        )
        logger.info("Loaded %d nodes", n_loaded)

        e_loaded = load_edges_batched(
            driver, args.database, edges, start_col, end_col, type_col,
            id_to_label, args.edge_batch_size,
        )
        logger.info("Loaded %d edges", e_loaded)

        run_sanity_queries(driver, args.database)

        elapsed = time.perf_counter() - t0
        print(f"\nNodes loaded: {n_loaded}")
        print(f"Edges loaded: {e_loaded}")
        print(f"Total time: {elapsed:.1f}s")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
