# Phase14 Neo4j — Analysis query pack

## Diagnosis: why earlier queries felt off

1. **Shared-compound query was too broad and double-counted:** Pairs (A,B) and (B,A) were both counted; self-pairs (A,A) could appear; no canonical ordering. **Fix:** Enforce `i1.id < i2.id` for ingredient pairs, exclude self-pairs, optional filter to compounds that target ≥1 gene and/or filter hub compounds (high ingredient degree).

2. **Showcase grouped by category only:** Query 2 aggregated over all interactions per category, producing one giant ingredient list per category. **Fix:** Group per **Interaction** node: return one row per (interaction_id, ingredient_pair, category, top_pathways, mechanistic_score).

3. **Unknown pathway dominated:** Pathways with `display_name = "Unknown pathway"` were included in explanations. **Fix:** Demo/analysis versions exclude these; report counts in `reports/unknown_pathway_report.json`.

4. **mechanistic_score appeared null:** The score lived only on MEDIATED_BY edges; AFFECTS (INT→CAT) did not carry it. **Fix:** Export now adds `mechanistic_score` to AFFECTS from `pair_category_mediation`; regenerate Neo4j export. Query via `(i)-[a:AFFECTS]->(cat)` with `a.mechanistic_score` or via MEDIATED_BY.

---

## A) Shared-compound analysis (deduped, optional specificity)

**Canonical pair:** enforce `ing1.id < ing2.id`. Exclude self-pairs. Optional: only compounds that target ≥1 gene; exclude hub compounds (e.g. ingredient_degree > threshold).

**Specificity:** `specificity = 1 / ingredient_degree` (ingredient_degree = number of ingredients linked to that compound). Weighted shared-compound score = sum(specificity) over shared compounds.

### Recommended: ingredient pair → shared compound count + top 10 biologically connected

```cypher
// Ingredient pair -> shared compound count (canonical order, no self-pairs)
// Optional: only compounds that target at least 1 gene
MATCH (i1:Ingredient)-[:HAS_COMPOUND]->(c:Compound)<-[:HAS_COMPOUND]-(i2:Ingredient)
WHERE i1.id < i2.id
  AND (c)-[:TARGETS]->(:Gene)  // optional: biologically connected
WITH i1, i2, c,
     size((c)<-[:HAS_COMPOUND]-(:Ingredient)) AS ingredient_degree
WITH i1, i2, c,
     1.0 / CASE WHEN size((c)<-[:HAS_COMPOUND]-(:Ingredient)) = 0 THEN 1 ELSE size((c)<-[:HAS_COMPOUND]-(:Ingredient)) END AS specificity
WITH i1, i2,
     count(c) AS shared_count,
     sum(specificity) AS weighted_shared_score,
     collect({ compound: c.display_name, specificity: specificity })[0..10] AS top_compounds
RETURN i1.display_name AS ingredient_a,
       i2.display_name AS ingredient_b,
       shared_count,
       weighted_shared_score,
       top_compounds
ORDER BY weighted_shared_score DESC
LIMIT 25;
```

### Simpler: shared compound count only (canonical, no self, with gene filter)

```cypher
MATCH (i1:Ingredient)-[:HAS_COMPOUND]->(c:Compound)<-[:HAS_COMPOUND]-(i2:Ingredient)
WHERE i1.id < i2.id
  AND EXISTS((c)-[:TARGETS]->(:Gene))
WITH i1, i2, count(c) AS shared_count
RETURN i1.display_name, i2.display_name, shared_count
ORDER BY shared_count DESC
LIMIT 25;
```

### Exclude hub compounds (e.g. degree > 50)

```cypher
MATCH (i1:Ingredient)-[:HAS_COMPOUND]->(c:Compound)<-[:HAS_COMPOUND]-(i2:Ingredient)
WHERE i1.id < i2.id
WITH c, count(DISTINCT i1) + count(DISTINCT i2) AS approx_degree
// Approximate: count ingredients per compound
MATCH (c)<-[:HAS_COMPOUND]-(ing:Ingredient)
WITH c, count(DISTINCT ing) AS ingredient_degree
WHERE ingredient_degree <= 50
MATCH (i1:Ingredient)-[:HAS_COMPOUND]->(c)<-[:HAS_COMPOUND]-(i2:Ingredient)
WHERE i1.id < i2.id
WITH i1, i2, count(c) AS shared_count
RETURN i1.display_name, i2.display_name, shared_count
ORDER BY shared_count DESC
LIMIT 25;
```

---

## B) Showcase: per-interaction grouping (no giant category lists)

Return one row per **interaction**: interaction_id, ingredient_pair, category, top_pathways, mechanistic_score. Use `collect` scoped per interaction.

### Interaction → ingredients, category, top pathways, mechanistic_score (excl. Unknown pathway)

```cypher
MATCH (i:Interaction)-[:HAS_INGREDIENT]->(ing:Ingredient),
      (i)-[a:AFFECTS]->(cat:Category)
WITH i, cat, a.mechanistic_score AS score,
     collect(DISTINCT ing.display_name) AS ingredients
ORDER BY ing.id
WITH i, cat, score, ingredients
MATCH (i)-[:MEDIATED_BY]->(p:Pathway)
WHERE p.display_name <> "Unknown pathway"
WITH i, cat, score, ingredients,
     collect(DISTINCT p.display_name)[0..5] AS top_pathways
RETURN i.id AS interaction_id,
       ingredients[0] + " + " + ingredients[1] AS ingredient_pair,
       cat.display_name AS category,
       top_pathways,
       score AS mechanistic_score
ORDER BY score DESC
LIMIT 25;
```

### With mechanistic_score from AFFECTS (preferred when available)

```cypher
MATCH (i:Interaction)-[:HAS_INGREDIENT]->(ing:Ingredient),
      (i)-[a:AFFECTS]->(cat:Category)
WHERE a.mechanistic_score IS NOT NULL
WITH i, cat, a.mechanistic_score AS score,
     collect(DISTINCT ing.display_name) AS ingredients
ORDER BY ing.id
WITH i, cat, score, ingredients
OPTIONAL MATCH (i)-[:MEDIATED_BY]->(p:Pathway)
WHERE p.display_name <> "Unknown pathway"
WITH i, cat, score, ingredients,
     collect(DISTINCT p.display_name) AS pathways
RETURN i.id AS interaction_id,
       ingredients[0] + " + " + ingredients[1] AS ingredient_pair,
       cat.display_name AS category,
       pathways[0..5] AS top_pathways,
       score AS mechanistic_score
ORDER BY score DESC
LIMIT 25;
```

---

## C) Unknown pathway: filter in queries + report

- **Report:** `reports/unknown_pathway_report.json` — `mediated_by_to_unknown_pathway`, `interactions_only_unknown_pathway`, `fraction_only_unknown`.
- **In queries:** Exclude with `WHERE p.display_name <> "Unknown pathway"` (or `p.display_name IS NULL OR p.display_name <> "Unknown pathway"`).

---

## D) mechanistic_score in Neo4j

- **Where it is now:** On **AFFECTS** (Interaction → Category) and on **MEDIATED_BY** (Interaction → Pathway). Export enriches AFFECTS from `pair_category_mediation`.
- **Validation:** `reports/mechanistic_score_validation.json` — `affects_with_mechanistic_score`, sample rows.
- **Regenerate export** so AFFECTS get the score:  
  `python scripts/phase14/regenerate_neo4j_export.py --latest`

---

## E) 10 analysis-grade Cypher queries

### 1. Top interactions by mechanistic_score (from AFFECTS)

```cypher
MATCH (i:Interaction)-[a:AFFECTS]->(cat:Category)
WHERE a.mechanistic_score IS NOT NULL
RETURN i.id, cat.display_name, a.mechanistic_score
ORDER BY a.mechanistic_score DESC
LIMIT 20;
```

### 2. Interaction → ingredients → category (one row per interaction–category)

```cypher
MATCH (i:Interaction)-[:HAS_INGREDIENT]->(ing:Ingredient),
      (i)-[a:AFFECTS]->(cat:Category)
WITH i, cat, a.mechanistic_score,
     collect(DISTINCT ing.display_name) AS ingredients
RETURN i.id AS interaction_id, ingredients, cat.display_name AS category, a.mechanistic_score
ORDER BY a.mechanistic_score DESC
LIMIT 25;
```

### 3. Interaction → top pathways (excluding Unknown)

```cypher
MATCH (i:Interaction)-[:MEDIATED_BY]->(p:Pathway)
WHERE p.display_name <> "Unknown pathway"
WITH i, collect(DISTINCT p.display_name)[0..5] AS top_pathways
RETURN i.id, top_pathways
LIMIT 25;
```

### 4. Ingredient pair → shared compounds (deduped + weighted)

```cypher
MATCH (c:Compound)<-[:HAS_COMPOUND]-(ing:Ingredient)
WITH c, count(DISTINCT ing) AS ingredient_degree
WHERE ingredient_degree >= 1
WITH c, 1.0 / ingredient_degree AS specificity
MATCH (i1:Ingredient)-[:HAS_COMPOUND]->(c)<-[:HAS_COMPOUND]-(i2:Ingredient)
WHERE i1.id < i2.id
WITH i1, i2, count(c) AS n_shared, sum(specificity) AS weighted_shared_score
RETURN i1.display_name, i2.display_name, n_shared, weighted_shared_score
ORDER BY weighted_shared_score DESC
LIMIT 20;
```

### 5. Ingredient → top genes

```cypher
MATCH (ing:Ingredient)-[:HAS_COMPOUND]->(c:Compound)-[:TARGETS]->(g:Gene)
WITH ing, g, count(c) AS n_compounds
ORDER BY n_compounds DESC
WITH ing, collect(g.display_name)[0..10] AS top_genes
RETURN ing.display_name, top_genes
LIMIT 20;
```

### 6. Gene → top pathways

```cypher
MATCH (g:Gene)-[:IN_PATHWAY]->(p:Pathway)
WHERE p.display_name <> "Unknown pathway"
WITH g, p, count(*) AS cnt
ORDER BY cnt DESC
WITH g, collect(p.display_name)[0..10] AS top_pathways
RETURN g.display_name, top_pathways
LIMIT 20;
```

### 7. Category → top interactions

```cypher
MATCH (i:Interaction)-[a:AFFECTS]->(cat:Category)
WHERE a.mechanistic_score IS NOT NULL
WITH cat, i, a.mechanistic_score
ORDER BY a.mechanistic_score DESC
WITH cat, collect(i.id)[0..15] AS top_interactions
RETURN cat.display_name, top_interactions
LIMIT 15;
```

### 8. Pathways most frequently mediating interactions

```cypher
MATCH (i:Interaction)-[:MEDIATED_BY]->(p:Pathway)
WHERE p.display_name <> "Unknown pathway"
WITH p, count(i) AS mediation_count
ORDER BY mediation_count DESC
RETURN p.display_name, mediation_count
LIMIT 20;
```

### 9. Compounds most reused across interactions

```cypher
MATCH (i:Interaction)-[:HAS_INGREDIENT]->(ing:Ingredient)-[:HAS_COMPOUND]->(c:Compound)
WITH c, count(DISTINCT i) AS interaction_count
ORDER BY interaction_count DESC
RETURN c.display_name, interaction_count
LIMIT 20;
```

### 10. Interactions with low-quality explanation (only Unknown pathway)

```cypher
MATCH (i:Interaction)-[:MEDIATED_BY]->(p:Pathway)
WITH i, collect(p) AS pathways,
     collect(CASE WHEN p.display_name = "Unknown pathway" THEN 1 ELSE 0 END) AS flags
WITH i, pathways,
     size([x IN flags WHERE x = 1]) AS n_unknown,
     size(pathways) AS total
WHERE total > 0 AND n_unknown = total
RETURN i.id
LIMIT 50;
```

---

## Commands after fix

1. **Regenerate Neo4j export** (adds mechanistic_score to AFFECTS, writes validation + Unknown pathway report):
   ```bash
   python scripts/phase14/regenerate_neo4j_export.py --latest
   ```

2. **Cypher:** Use the queries above. Prefer AFFECTS for mechanistic_score when you need per (interaction, category); use MEDIATED_BY for pathway-level scores. Exclude Unknown pathway in demo/analysis with `WHERE p.display_name <> "Unknown pathway"`.
