# Phase14.5 Analysis pack v2 — category-balanced and richer

Use with **confirmed** graph (`neo4j/`) or **expanded discovery** graph (`neo4j_expanded/`). Prefer excluding `Unknown pathway` where relevant.

---

## 1. Category-balanced top interactions

Top interactions **per category** so Apoptosis does not dominate the list.

```cypher
MATCH (i:Interaction)-[a:AFFECTS]->(cat:Category)
WHERE a.mechanistic_score IS NOT NULL
WITH cat, i, a.mechanistic_score AS score
ORDER BY cat.display_name, score DESC
WITH cat.display_name AS category, collect({ interaction_id: i.id, score: score })[0..5] AS top5
RETURN category, top5
ORDER BY category;
```

---

## 2. Ingredient pairs ranked by shared gene-linked compounds (deduped, weighted)

```cypher
MATCH (i1:Ingredient)-[:HAS_COMPOUND]->(c:Compound)-[:TARGETS]->(:Gene)
MATCH (c)<-[:HAS_COMPOUND]-(i2:Ingredient)
WHERE i1.id < i2.id
WITH c, count(DISTINCT i1) + count(DISTINCT i2) AS deg
WITH c, 1.0 / CASE WHEN deg = 0 THEN 1 ELSE deg END AS spec
MATCH (i1:Ingredient)-[:HAS_COMPOUND]->(c)<-[:HAS_COMPOUND]-(i2:Ingredient)
WHERE i1.id < i2.id
WITH i1, i2, count(c) AS n_shared, sum(spec) AS weighted
RETURN i1.display_name, i2.display_name, n_shared, weighted
ORDER BY weighted DESC
LIMIT 25;
```

---

## 3. Ingredient pairs ranked by shared pathways (excluding Unknown)

```cypher
MATCH (i1:Ingredient)-[:HAS_COMPOUND]->(c:Compound)-[:TARGETS]->(g:Gene)-[:IN_PATHWAY]->(p:Pathway)
WHERE p.display_name <> "Unknown pathway"
MATCH (p)<-[:IN_PATHWAY]-(g2:Gene)<-[:TARGETS]-(c2:Compound)<-[:HAS_COMPOUND]-(i2:Ingredient)
WHERE i1.id < i2.id AND c = c2 AND g = g2
WITH i1, i2, count(DISTINCT p) AS shared_pathways
RETURN i1.display_name, i2.display_name, shared_pathways
ORDER BY shared_pathways DESC
LIMIT 25;
```

*Simpler variant: shared pathways via any compound–gene path.*

```cypher
MATCH (i1:Ingredient)-[:HAS_COMPOUND]->(c:Compound)-[:TARGETS]->(g:Gene)-[:IN_PATHWAY]->(p:Pathway)
WHERE p.display_name <> "Unknown pathway"
WITH i1, p
MATCH (i2:Ingredient)-[:HAS_COMPOUND]->(:Compound)-[:TARGETS]->(:Gene)-[:IN_PATHWAY]->(p)
WHERE i1.id < i2.id
WITH i1, i2, count(DISTINCT p) AS shared_pathways
RETURN i1.display_name, i2.display_name, shared_pathways
ORDER BY shared_pathways DESC
LIMIT 25;
```

---

## 4. Ingredients with most gene targets

```cypher
MATCH (ing:Ingredient)-[:HAS_COMPOUND]->(c:Compound)-[:TARGETS]->(g:Gene)
WITH ing, count(DISTINCT g) AS gene_count
ORDER BY gene_count DESC
RETURN ing.display_name, gene_count
LIMIT 20;
```

---

## 5. Pathways most frequently mediating interactions (excluding Unknown)

```cypher
MATCH (i:Interaction)-[:MEDIATED_BY]->(p:Pathway)
WHERE p.display_name <> "Unknown pathway"
WITH p, count(i) AS mediation_count
ORDER BY mediation_count DESC
RETURN p.display_name, mediation_count
LIMIT 20;
```

---

## 6. Compounds most frequently reused across interactions

```cypher
MATCH (i:Interaction)-[:HAS_INGREDIENT]->(ing:Ingredient)-[:HAS_COMPOUND]->(c:Compound)
WITH c, count(DISTINCT i) AS interaction_count
ORDER BY interaction_count DESC
RETURN c.display_name, interaction_count
LIMIT 20;
```

---

## 7. Low-confidence / weak-explanation interactions

Interactions that have only Unknown pathway or no MEDIATED_BY.

```cypher
MATCH (i:Interaction)-[:MEDIATED_BY]->(p:Pathway)
WITH i, collect(p) AS pathways,
     size([x IN collect(p) WHERE x.display_name = "Unknown pathway"]) AS n_unknown,
     size(collect(p)) AS total
WHERE total > 0 AND n_unknown = total
RETURN i.id AS interaction_id, "only_unknown_pathway" AS reason
LIMIT 50;
```

```cypher
OPTIONAL MATCH (i:Interaction)-[:MEDIATED_BY]->(p:Pathway)
WITH i, count(p) AS pathway_count
WHERE pathway_count = 0
RETURN i.id AS interaction_id, "no_mediated_pathway" AS reason
LIMIT 50;
```

---

## 8. Category diversity report

Count of interactions and mean mechanistic_score per category.

```cypher
MATCH (i:Interaction)-[a:AFFECTS]->(cat:Category)
WHERE a.mechanistic_score IS NOT NULL
WITH cat.display_name AS category,
     count(DISTINCT i) AS interaction_count,
     avg(a.mechanistic_score) AS mean_score
RETURN category, interaction_count, mean_score
ORDER BY interaction_count DESC;
```
