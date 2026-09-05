#!/usr/bin/env python3
"""
Pathway ID -> human-readable display name resolution for product profiles.

Sources (priority order):
  1. Reactome R-HSA-*  — ReactomePathways.txt (Homo sapiens)
  2. GO bracket format — embedded name in enrichment pathway_id string
  3. GO term lookup    — go-basic.obo name field (fallback / validation)
  4. pathway_category_map_v2 — supplemental Reactome names if present
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

REACTOME_PATHWAYS = ROOT / "data/raw/pathways/ReactomePathways.txt"
GO_OBO = ROOT / "data/raw/pathways/go-basic.obo"
PATHWAY_MAP_V2 = ROOT / "data/processed/tier1/pathway_category_map_v2.parquet"
REACTOME_HIERARCHY = ROOT / "data/processed/tier1/reactome_category_hierarchy_v1.json"

GO_BRACKET_RE = re.compile(r"^(?P<name>.+?)\s+\[(?P<go_id>GO:\d+)\]\s*$")
REACTOME_ID_RE = re.compile(r"^R-HSA-\d+$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reactome_human_names(path: Path = REACTOME_PATHWAYS) -> dict[str, str]:
    names: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            pid, name, species = parts
            if species == "Homo sapiens":
                names[pid] = name
    return names


def load_go_term_names(path: Path = GO_OBO) -> dict[str, str]:
    terms: dict[str, str] = {}
    current_id: str | None = None
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line == "[Term]":
                current_id = None
                continue
            if not line:
                continue
            if line.startswith("id:"):
                current_id = line.split(":", 1)[1].strip()
            elif current_id and line.startswith("name:"):
                terms[current_id] = line.split(":", 1)[1].strip()
    return terms


def load_pathway_map_v2_reactome_names(path: Path = PATHWAY_MAP_V2) -> dict[str, str]:
    """Supplement Reactome names from category map if pathway_id is R-HSA (usually redundant)."""
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["pathway_id"])
    extra: dict[str, str] = {}
    for pid in df["pathway_id"].astype(str).unique():
        if REACTOME_ID_RE.match(pid):
            # pathway_category_map does not store Reactome display names for R-HSA rows
            continue
    return extra


def parse_pathway_id(raw: str) -> tuple[str, str | None, str]:
    """
    Return (stable_id, embedded_name_or_none, id_type).
    id_type: reactome | go | other
    """
    raw = str(raw).strip()
    if REACTOME_ID_RE.match(raw):
        return raw, None, "reactome"
    m = GO_BRACKET_RE.match(raw)
    if m:
        return m.group("go_id"), m.group("name").strip(), "go"
    if raw.startswith("GO:"):
        return raw, None, "go"
    return raw, None, "other"


class PathwayNameResolver:
    def __init__(
        self,
        reactome_names: dict[str, str] | None = None,
        go_names: dict[str, str] | None = None,
    ) -> None:
        self.reactome_names = reactome_names or load_reactome_human_names()
        self.go_names = go_names or load_go_term_names()
        self._cache: dict[str, dict[str, Any]] = {}

    def resolve(self, raw_pathway_id: str) -> dict[str, Any]:
        if raw_pathway_id in self._cache:
            return self._cache[raw_pathway_id]

        stable_id, embedded_name, id_type = parse_pathway_id(raw_pathway_id)
        source = "unresolved"
        display_name: str | None = None

        if id_type == "reactome":
            display_name = self.reactome_names.get(stable_id)
            if display_name:
                source = "reactome_pathways_txt"
        elif id_type == "go":
            if embedded_name:
                display_name = embedded_name
                source = "go_enrichment_string"
            if stable_id in self.go_names:
                obo_name = self.go_names[stable_id]
                if not display_name:
                    display_name = obo_name
                    source = "go_basic_obo"
                elif display_name != obo_name:
                    source = "go_enrichment_string+obo_validated"
            elif not display_name:
                display_name = stable_id
                source = "go_id_fallback"
        else:
            display_name = raw_pathway_id
            source = "passthrough"

        resolved = display_name is not None and display_name != stable_id
        if id_type == "reactome" and not resolved:
            display_name = stable_id

        result = {
            "pathway": stable_id,
            "pathway_name": display_name or stable_id,
            "pathway_id_type": id_type,
            "name_source": source,
            "resolved": resolved or (id_type == "go" and bool(embedded_name)),
        }
        self._cache[raw_pathway_id] = result
        return result

    def make_pathway_entry(self, raw_pathway_id: str, base_entry: dict[str, Any]) -> dict[str, Any]:
        resolved = self.resolve(raw_pathway_id)
        out = dict(base_entry)
        out["pathway"] = resolved["pathway"]
        out["pathway_name"] = resolved["pathway_name"]
        return out


def build_resolution_report(pathway_ids: list[str]) -> dict[str, Any]:
    resolver = PathwayNameResolver()
    reactome_ids: list[str] = []
    go_ids: list[str] = []
    other_ids: list[str] = []
    unresolved: list[dict[str, str]] = []
    by_source: dict[str, int] = defaultdict(int)

    for raw in sorted(set(pathway_ids)):
        r = resolver.resolve(raw)
        by_source[r["name_source"]] += 1
        pid_type = r["pathway_id_type"]
        if pid_type == "reactome":
            reactome_ids.append(raw)
        elif pid_type == "go":
            go_ids.append(raw)
        else:
            other_ids.append(raw)
        if not r["resolved"] and pid_type == "reactome":
            unresolved.append({"pathway_id": raw, "reason": r["name_source"]})

    n_total = len(set(pathway_ids))
    n_resolved = n_total - len(unresolved)

    return {
        "n_pathways_total": n_total,
        "n_resolved": n_resolved,
        "n_unresolved": len(unresolved),
        "resolution_rate": round(n_resolved / n_total, 6) if n_total else 1.0,
        "reactome_count": len(reactome_ids),
        "go_count": len(go_ids),
        "other_count": len(other_ids),
        "by_name_source": dict(sorted(by_source.items())),
        "unresolved": unresolved,
        "lookup_inputs": {
            "reactome_pathways_txt": {
                "path": str(REACTOME_PATHWAYS.relative_to(ROOT)),
                "sha256": sha256_file(REACTOME_PATHWAYS),
            },
            "go_basic_obo": {
                "path": str(GO_OBO.relative_to(ROOT)),
                "sha256": sha256_file(GO_OBO),
            },
            "pathway_category_map_v2": {
                "path": str(PATHWAY_MAP_V2.relative_to(ROOT)),
                "sha256": sha256_file(PATHWAY_MAP_V2),
            },
        },
    }


def enrich_pathway_list(
    entries: list[dict[str, Any]],
    resolver: PathwayNameResolver,
    raw_ids: list[str],
) -> list[dict[str, Any]]:
    """Pair profile entries with original raw pathway ids from enrichment."""
    enriched: list[dict[str, Any]] = []
    for raw_id, entry in zip(raw_ids, entries):
        base = {k: v for k, v in entry.items() if k not in ("pathway_name",)}
        enriched.append(resolver.make_pathway_entry(raw_id, base))
    return enriched
