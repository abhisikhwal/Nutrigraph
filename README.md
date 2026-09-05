# NutriGraph

**A molecular evidence graph built from 27 heterogeneous chemical, biological, nutritional and culinary datasets, connecting food chemistry to human biology.**

**Entity resolution, molecular machine learning and evidence provenance connect food → compounds → genes → pathways → tissues while keeping measured and inferred relationships explicitly separate.**

**Built by Abhinav Sikhwal**  
[LinkedIn](https://www.linkedin.com/in/abhinav-sikhwal/) · [Portfolio](https://abhinavsikhwal.com) · abhisikhwal@gmail.com

---

## Explore NutriGraph

| Interface | URL |
|---|---|
| **Food Explorer** | [nutri.abhinavsikhwal.com](https://nutri.abhinavsikhwal.com) |
| **Knowledge Graph** | [graph.nutri.abhinavsikhwal.com](https://graph.nutri.abhinavsikhwal.com) |

---

## From food to biology

Foods contain thousands of chemical compounds, but many of those molecules have never been tested against human protein targets.

That creates a large missing-data problem between:

**what is present in food** and **what those molecules may interact with biologically**.

NutriGraph connects those layers.

It integrates food chemistry, measured bioactivity, molecular structure, gene targets, pathways, tissue expression, nutrition and culinary data into a common graph:

```text
Food → Compound → Gene → Pathway → Tissue
  │
  └────────────→ Nutrient
```

For compounds without measured target data, NutriGraph uses molecular similarity to infer candidate human targets.

Crucially, those predictions are not presented as experimental facts.

Every biological relationship carries its evidence type:

```text
evidence = measured | predicted
```

This makes it possible to ask not only:

> What biological targets are associated with this food?

but also:

> Which relationships are experimentally measured, and which are computationally inferred?

---

## At a glance

| Metric | Value |
|---|---:|
| Culinary ingredients | **695** |
| Food-associated compounds | **~48,459** |
| Human gene targets | **1,532** |
| Measured ingredient→gene relationships | **20.4%** |
| Inferred ingredient→gene relationships | **79.6%** |
| Scaffold-split target inference Hit@10 | **85.8%** |
| Recipe mapping coverage | **97.6%** |
| Integrated datasets | **27** |
| Reconciled identifier namespaces | **18** |
| Trimmed Neo4j graph | **~7.5k nodes / ~79k edges** |

---

## The core idea

NutriGraph combines two different kinds of biological evidence.

### Measured biology

Experimentally observed compound→target relationships are integrated from bioactivity resources including **ChEMBL** and **BindingDB**.

Food chemistry data from sources including **FooDB**, **COCONUT** and supporting metabolite datasets connects those compounds back to foods and ingredients.

These relationships form the experimentally supported portion of the graph.

### Inferred biology

Many food-associated compounds do not have measured human target data.

For these compounds, NutriGraph uses molecular fingerprints and k-nearest-neighbour inference to identify structurally related compounds with known biological activity and infer candidate targets.

To reduce overly optimistic validation caused by structurally similar molecules appearing across train and test sets, the inference pipeline is evaluated using a **Murcko-scaffold split**.

The resulting model achieved:

**85.8% Hit@10**

on the scaffold-held-out evaluation.

This is a retrieval-style measure of whether a known target appears within the top ten inferred targets. It does not mean that 85.8% of individual biological predictions are experimentally correct.

Predicted relationships remain explicitly labelled as predictions throughout the downstream graph.

---

## Evidence, not just edges

A predicted molecular interaction and an experimentally measured interaction should not look identical in a biological knowledge graph.

NutriGraph therefore propagates provenance through the graph rather than collapsing every relationship into a generic edge.

Where available, relationships retain information such as:

- whether the underlying evidence is **measured** or **predicted**
- the source of the evidence
- prediction confidence
- the compound through which an ingredient→gene relationship was established

That distinction is preserved as relationships are rolled up into ingredient-level biological profiles and exposed through the graph.

```text
measured evidence ≠ inferred evidence
```

NutriGraph does not treat structural similarity as experimental proof.

---

## Data integration

**NutriGraph is fundamentally a heterogeneous data-integration and entity-resolution problem.** Its 27 source datasets use different schemas, identifiers and naming conventions for compounds, proteins, genes, foods, tissues and pathways.

The pipeline reconciles **18 identifier namespaces** through canonical IDs and crosswalks, including InChIKey, UniProt, HGNC, Ensembl and USDA FoodData Central identifiers.

This allows otherwise disconnected chemical, biological, nutritional and culinary datasets to be joined into one provenance-aware graph.

| Domain | Sources | Canonical identity / mapping |
|---|---|---|
| Food chemistry | FooDB, COCONUT, supporting HMDB | **InChIKey** |
| Bioactivity | ChEMBL, BindingDB | compound → **UniProt** |
| Genes | UniProt, HGNC, Ensembl | **HGNC symbol** |
| Tissue expression | GTEx | gene → tissue |
| Pathways | Gene Ontology, Reactome | gene → pathway |
| Nutrition | USDA FoodData Central | food / species |
| Culinary data | Recipe corpora | ingredient / species |

These mappings allow evidence from otherwise disconnected databases to converge on the same food, compound and gene entities while retaining source provenance.

---

## From raw data to graph

```text
                 Raw datasets
                      │
                      ▼
              Identity resolution
           Food · Compound · Gene
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
  Food chemistry          Measured bioactivity
                                  │
                                  ▼
                       Structural inference
                         for missing targets
                                  │
                                  ▼
                     Evidence-aware food→gene
                              profiles
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
          Pathways             Tissues             Nutrition
                                                      │
                                                      ▼
                                                   Recipes
                                  │
                                  ▼
                              NutriGraph
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              Food Explorer              Neo4j Graph
```

The complete methodology is documented in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Pipeline

The project is built as a sequence of data-integration and inference stages.

### 1. Food chemistry corpus

Build canonical food→compound relationships from sources including FooDB and supporting natural-product chemistry datasets.

### 2. Measured bioactivity

Integrate experimentally measured compound→target relationships from ChEMBL and BindingDB and reconcile protein identifiers to human gene symbols.

### 3. Structural target inference

Expand coverage for compounds without measured target data using molecular fingerprints and k-nearest-neighbour structural inference.

Validation is performed using Murcko-scaffold separation to reduce structural leakage.

### 4. Ingredient gene profiles

Roll compound-level evidence up to ingredient→gene relationships while preserving:

- evidence type
- confidence
- source provenance

### 5. Biological enrichment

Connect ingredient-associated genes to biological pathways and downstream enrichment profiles.

### 6. Culinary universe expansion

Expand the core species universe using recipe corpora, producing a final live universe of **695 culinary ingredients**.

### 7. Nutrition

Integrate USDA FoodData Central composition data and relative nutrient contribution helpers.

### 8. Recipe layer

Map recipe ingredients into the canonical food universe and generate recipe-level analysis and demonstration data.

### 9. Knowledge graph

Generate a trimmed Neo4j representation for interactive exploration.

The public graph contains approximately:

**7.5k nodes / ~79k edges**

while the larger upstream datasets remain outside the repository.

---

## Engineering scope

NutriGraph combines several engineering problems in one end-to-end system:

- **Heterogeneous data integration** across 27 scientific and culinary datasets
- **Entity resolution and identifier normalization** across 18 namespaces
- **ETL and data pipelines** for large biological, chemical, nutritional and recipe datasets
- **Molecular machine learning** using RDKit fingerprints and structural similarity
- **Leakage-aware evaluation** using Murcko-scaffold splits
- **Knowledge graph construction** and graph data modelling in Neo4j
- **Evidence provenance**, separating measured observations from ML inference
- **Biological enrichment** across genes, pathways and tissue expression
- **Product data layers** supporting React-based exploration interfaces

---

## Tech stack

### Data and machine learning

- Python
- pandas
- pyarrow
- scikit-learn
- NetworkX

### Cheminformatics

- RDKit
- molecular fingerprints
- Murcko scaffolds
- structural similarity search

### Knowledge graph

- Neo4j
- Cypher
- Neovis.js

### Web interfaces

- React
- Vite

### Local interface

- Streamlit

---

## Repository structure

```text
scripts/product/          # product builds, profiles, showcase and Neo4j load
scripts/thread2/          # structural target inference
scripts/tier1/            # enrichment and tissue profiles

data/processed/product/   # small committed product deliverables
docs/                     # methodology, architecture and deployment docs
licenses/                 # dataset licensing registry
src/                      # shared Python library code

web/nutri-showcase/       # React/Vite Food Explorer
web/nutri-graph/          # React/Vite knowledge-graph explorer

streamlit_app/            # optional local Streamlit interface
```

Raw source databases live under `data/raw/` and are gitignored. The full raw-data footprint is **60GB+**.

---

## Running locally

### 1. Create the environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate food-genome
```

Or using the supplied Python requirements:

```bash
pip install -r requirements.txt
pip install -e .
```

Create a local environment file:

```bash
cp .env.example .env
```

Add required credentials locally.

**Never commit `.env` or real database credentials.**

### 2. Obtain the source data

Raw databases are intentionally not committed to Git.

Download the required sources into `data/raw/` using the instructions in:

[`docs/dataset_sources.md`](docs/dataset_sources.md)

The full pipeline depends on external datasets including:

- FooDB
- COCONUT
- ChEMBL
- BindingDB
- USDA FoodData Central
- GTEx
- Gene Ontology
- Reactome
- recipe corpora
- supporting identity and metabolite sources

The complete rebuild is multi-stage, disk-heavy and subject to the licences of the upstream datasets.

### 3. Build the product layer

Once canonical upstream edges have been generated, the main product build includes:

```bash
python scripts/product/live_universe_v2.py
python scripts/product/build_showcase_bundle.py
python scripts/product/build_neo4j_trimmed_graph.py
```

Structural inference code is under:

```text
scripts/thread2/
```

Enrichment and tissue processing are under:

```text
scripts/tier1/
```

### 4. Neo4j

For an administrative local Neo4j instance:

```bash
export NEO4J_URI=bolt://127.0.0.1:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD='…'

python scripts/product/build_neo4j_trimmed_graph.py --load
```

Read-only graph configuration and Neovis wiring are documented in:

[`data/processed/product/neo4j_load/NEO4J_SETUP.md`](data/processed/product/neo4j_load/NEO4J_SETUP.md)

Real Neo4j credentials must never be committed.

### 5. Web interfaces

The repository contains two React/Vite interfaces:

```text
web/nutri-showcase/
web/nutri-graph/
```

Deployment documentation:

[`docs/DEPLOY_GUIDE.md`](docs/DEPLOY_GUIDE.md)

An optional Streamlit interface can also be run locally:

```bash
streamlit run streamlit_app/app.py
```

---

## Data sources and licences

**The code in this repository is MIT licensed. The underlying datasets are not necessarily MIT licensed.**

Always consult the original dataset licence before downloading, using or redistributing source data.

| Dataset | Role | Licence / terms | Source |
|---|---|---|---|
| FooDB | Food→compound | CC BY-NC 4.0 | https://foodb.ca/ |
| COCONUT | Natural-product compounds | See COCONUT terms | https://coconut.naturalproducts.net/ |
| ChEMBL | Measured bioactivity | CC BY-SA 3.0 | https://www.ebi.ac.uk/chembl/ |
| BindingDB | Measured binding | Verify upstream | https://www.bindingdb.org/ |
| USDA FoodData Central | Nutrition | Public domain, US government | https://fdc.nal.usda.gov/ |
| GTEx | Gene–tissue expression | GTEx / dbGaP terms | https://gtexportal.org/ |
| Gene Ontology | Biological annotations | CC BY 4.0 | http://geneontology.org/ |
| Reactome | Pathways | CC BY 4.0 | https://reactome.org/ |
| HGNC | Gene symbols | HGNC terms | https://www.genenames.org/ |
| UniProt | Protein↔gene mapping | CC BY 4.0 | https://www.uniprot.org/ |
| PharmGKB / ClinPGx | Pharmacogenomic crosswalks | PharmGKB terms | https://www.pharmgkb.org/ |
| RecipeNLG | Recipes | MIT | https://recipenlg.cs.put.poznan.pl/ |
| Food.com / other recipes | Culinary corpora | Respective source terms | — |
| HMDB | Supporting metabolite data | HMDB terms | https://hmdb.ca/ |
| PubChem | Compound identity | Public domain | https://pubchem.ncbi.nlm.nih.gov/ |
| Wikidata | Food labels | CC0 | https://www.wikidata.org/ |

Full dataset registry:

[`licenses/datasets_registry.csv`](licenses/datasets_registry.csv)

Product dataset roster:

[`data/processed/product/showcase/dataset_roster.json`](data/processed/product/showcase/dataset_roster.json)

---

## Limitations

NutriGraph represents **molecular evidence and computational inference**, not clinical outcomes.

- Structural similarity can suggest candidate molecular targets, but it does not establish that an interaction occurs in a human after eating a food.
- The system does not model absorption, distribution, metabolism or excretion (**ADME**).
- It does not model clinical efficacy.
- Nutrient and contribution analyses are relative rather than estimates of therapeutic dose.
- Individual predicted compound→target relationships may be incorrect despite aggregate validation performance.
- Pathway and tissue associations can inherit uncertainty from upstream relationships.
- Coverage is constrained by the underlying public and licensed datasets.
- A molecular association should not be interpreted as evidence that a food prevents, diagnoses or treats disease.

**NutriGraph is a research and exploration system, not medical advice or a clinical decision-support tool.**

---

## License

Code released under the **MIT License**.

Dataset usage remains subject to the licences and terms of the respective upstream sources.

MIT © 2026 Abhinav Sikhwal

See [`LICENSE`](LICENSE).

---

**Abhinav Sikhwal**  
[LinkedIn](https://www.linkedin.com/in/abhinav-sikhwal/) · [Portfolio](https://abhinavsikhwal.com) · abhisikhwal@gmail.com
