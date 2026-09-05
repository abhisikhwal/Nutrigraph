// LOAD CSV script for Neo4j Community (admin user)
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
