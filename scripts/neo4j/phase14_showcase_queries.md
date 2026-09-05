# Phase14 Mediation Graph — Showcase Cypher Queries

Use these after loading `neo4j/nodes.csv` and `neo4j/edges.csv`. Assume node property `id` = `:ID`; edges use `:START_ID`, `:END_ID`, `:TYPE`. **For analysis-grade queries (shared compounds deduped, per-interaction grouping, Unknown pathway excluded, mechanistic_score on AFFECTS), see `phase14_analysis_queries.md`.**

---

## 1. Top interactions by mechanistic_score

*After export fix: mechanistic_score is on AFFECTS (Interaction→Category) and MEDIATED_BY (Interaction→Pathway). Prefer AFFECTS for per-category score.*

```cypher
MATCH (i:Interaction)-[a:AFFECTS]->(cat:Category)
WHERE a.mechanistic_score IS NOT NULL
RETURN i.id AS interaction_id, cat.display_name AS category, a.mechanistic_score
ORDER BY a.mechanistic_score DESC
LIMIT 20;
```

---

## 2. Interaction → ingredients → category (one row per interaction–category)

*Group per interaction so you don't get one giant list per category.*

```cypher
MATCH (i:Interaction)-[:HAS_INGREDIENT]->(ing:Ingredient),
      (i)-[a:AFFECTS]->(cat:Category)
WITH i, cat, a.mechanistic_score,
     collect(DISTINCT ing.display_name) AS ingredients
RETURN i.id AS interaction_id, ingredients, cat.display_name AS category, a.mechanistic_score
ORDER BY a.mechanistic_score DESC
LIMIT 25;
```

---

## 3. Ingredient → compound → gene

```cypher
MATCH (ing:Ingredient)-[:HAS_COMPOUND]->(c:Compound)-[:TARGETS]->(g:Gene)
RETURN ing.display_name AS ingredient,
       c.display_name AS compound,
       g.display_name AS gene
LIMIT 25;
```

---

## 4. Ingredient → compound → gene → pathway → category

```cypher
MATCH (ing:Ingredient)-[:HAS_COMPOUND]->(c:Compound)-[:TARGETS]->(g:Gene),
      (g)-[:IN_PATHWAY]->(p:Pathway)-[:MAPS_TO_CATEGORY]->(cat:Category)
RETURN ing.display_name AS ingredient,
       c.display_name AS compound,
       g.display_name AS gene,
       p.display_name AS pathway,
       cat.display_name AS category
LIMIT 20;
```

---

## 5. Interaction → mediated pathway → category (excl. Unknown pathway)

```cypher
MATCH (i:Interaction)-[:MEDIATED_BY]->(p:Pathway)-[:MAPS_TO_CATEGORY]->(c:Category)
WHERE p.display_name <> "Unknown pathway"
RETURN i.id AS interaction_id, p.display_name AS pathway, c.display_name AS category
LIMIT 25;
```

---

## 6. Top ingredients by compound count

```cypher
MATCH (ing:Ingredient)-[:HAS_COMPOUND]->(c:Compound)
WITH ing, count(c) AS compound_count
ORDER BY compound_count DESC
RETURN ing.display_name AS ingredient, compound_count
LIMIT 20;
```

---

## 7. Top compounds by gene count

```cypher
MATCH (c:Compound)-[:TARGETS]->(g:Gene)
WITH c, count(g) AS gene_count
ORDER BY gene_count DESC
RETURN c.display_name AS compound, gene_count
LIMIT 20;
```

---

## 8. Top genes by pathway count

```cypher
MATCH (g:Gene)-[:IN_PATHWAY]->(p:Pathway)
WITH g, count(p) AS pathway_count
ORDER BY pathway_count DESC
RETURN g.display_name AS gene, pathway_count
LIMIT 20;
```

---

## Import note

If your loader maps CSV columns to properties as-is, use the same property names: `name`, `display_name`, `source`, and ensure node `id` is set from the `:ID` column so that relationships can resolve correctly.
