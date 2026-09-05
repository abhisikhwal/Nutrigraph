# NutriGraph

**A knowledge graph linking 695 foods to human molecular targets — honest about what's measured vs inferred.**

**Built by Abhinav Sikhwal**  
[LinkedIn](https://www.linkedin.com/in/abhinav-sikhwal/) · [Portfolio](https://abhinavsikhwal.com) · abhisikhwal@gmail.com

| Live demo | URL |
|-----------|-----|
| Showcase | [nutri.abhinavsikhwal.com](https://nutri.abhinavsikhwal.com) *(placeholder — confirm after deploy)* |
| Interactive graph | [graph.nutri.abhinavsikhwal.com](https://graph.nutri.abhinavsikhwal.com) *(placeholder — confirm after deploy)* |

---

## The problem

Most food chemistry is pharmacologically uncharacterized — the so-called **dark metabolome**. Public databases know which compounds sit in turmeric or cabbage, but for the majority of those molecules there is no measured binding assay against a human protein. NutriGraph bridges that gap with structural inference, then **labels every edge** so you can see what was measured in a lab and what was predicted.

## What it does

```
Food → Compound → Gene → Pathway → Tissue
                ↘ Nutrient (USDA composition)
```

- **695** culinary ingredients in the live universe  
- **~48k** compounds in the food–chemistry layer  
- **1,532** human gene targets in the product gene set  
- **79.6%** of ingredient→gene edges are **inferred** (and flagged); **20.4%** are **measured**  
- **85.8% hit@10** on a hard Murcko-scaffold validation split for k-NN target inference  
- **27** external datasets · **18** ID namespaces reconciled  

Honesty is a first-class property: `evidence = measured | predicted` travels with the graph (including the public Neo4j demo).

## Headline numbers

| Metric | Value |
|--------|------:|
| Ingredients (live universe) | 695 |
| Compounds (ICC chemistry layer) | ~48,459 |
| Genes (product set) | 1,532 |
| Inferred vs measured (ingredient→gene) | 79.6% / 20.4% |
| Scaffold-split hit@10 | 85.8% |
| Recipe mapping coverage | 97.6% |
| Trimmed Neo4j demo graph | ~7.5k nodes / ~79k edges |
| Datasets integrated | 27 |

## Data-integration story

This is less a single model and more an **identity and evidence stack**:

| Layer | What gets unified |
|-------|-------------------|
| Chemistry | FooDB + COCONUT (+ supporting HMDB) → **InChIKey** compound master |
| Bioactivity | ChEMBL + BindingDB assays → UniProt → **HGNC** symbols |
| Genetics / expression | Ensembl (GTEx) → HGNC; tissue attribution |
| Pathways | GO + Reactome gene→pathway maps |
| Nutrition | USDA FDC food IDs → locked species composition |
| Culinary | Recipe corpora → fuzzy string → species / ingredient IDs |

The engineering work is the crosswalks (InChIKey, UniProt↔HGNC, Ensembl↔HGNC, FDC, GO/Reactome) plus keeping **measured** and **predicted** edges separable end-to-end.

## Architecture

```
┌─────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ Raw corpora │ → │ Canonical edges  │ → │ Product layer   │
│ FooDB/ChEMBL│   │ ICC, compound→   │   │ profiles, indexes│
│ BindingDB…  │   │ gene, gene sets  │   │ showcase JSON   │
└─────────────┘   └────────┬─────────┘   └────────┬────────┘
                           │                      │
              ┌────────────▼──────────┐           │
              │ Structural inference  │           │
              │ (k-NN / scaffolds)    │           │
              └────────────┬──────────┘           │
                           │                      │
              ┌────────────▼──────────┐   ┌───────▼────────┐
              │ Enrichment + tissues  │ → │ Neo4j trimmed  │
              │ + nutrition + recipes │   │ public demo    │
              └───────────────────────┘   └────────────────┘
```

**Pipeline stages (ordered):**

1. **Corpus** — food→compound (FooDB) and measured compound→gene (ChEMBL/BindingDB)  
2. **Structural inference** — expand dark compounds via fingerprint k-NN; scaffold-validated confidence  
3. **Ingredient gene sets** — roll up to foods with `evidence` + confidence  
4. **Enrichment** — pathway enrichment with weight-permutation FDR calibration  
5. **Universe expansion** — 463 species → 695 culinary nodes from recipe corpora  
6. **Nutrition + dose** — USDA FDC composition and relative contribution helpers  
7. **Recipe layer** — mapping, landscape, demo recipes  
8. **Neo4j** — trimmed visualizable graph + read-only user for open exploration  

Details: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Tech stack

- **Python** — pandas, pyarrow, scikit-learn, NetworkX  
- **Chemistry** — RDKit (fingerprints, Murcko scaffolds)  
- **Graph** — Neo4j (trimmed demo) + Cypher templates for Neovis.js  
- **Web** — React/Vite showcase & graph UIs *(deployed separately; see live links)*  
- **Legacy demo** — Streamlit (`streamlit_app/`)

## Repository layout (intended)

```
scripts/product/          # product builds (profiles, showcase, Neo4j load)
scripts/thread2/          # structural inference
scripts/tier1/            # enrichment / tissue profiles
data/processed/product/   # small deliverables committed (showcase, neo4j_load)
docs/                     # methodology & architecture
licenses/                 # dataset registry
src/                      # shared library code
web/nutri-showcase/       # React/Vite showcase site
web/nutri-graph/          # React/Vite live graph explorer
streamlit_app/            # optional local demo
```

Raw dumps live under `data/raw/` (gitignored, ~60GB+). Rebuild from sources below.

## How to run

### 1. Environment

```bash
conda env create -f environment.yml
conda activate food-genome
# or: pip install -r requirements.txt
pip install -e .
cp .env.example .env   # add keys locally; never commit .env
```

### 2. Obtain raw data

Do **not** expect raw databases in git. Download into `data/raw/` per [`docs/dataset_sources.md`](docs/dataset_sources.md) and the table below (FooDB, ChEMBL, BindingDB, USDA FDC, GTEx, GO, Reactome, recipe corpora, etc.). Respect each license — especially **non-commercial** sources.

### 3. Pipeline (high level)

Full rebuild is multi-stage and disk-heavy. Typical product path after canonical edges exist:

```bash
# Structural inference / integration (see scripts/thread2/)
# Enrichment / tissues (see scripts/tier1/)
python scripts/product/live_universe_v2.py
python scripts/product/build_showcase_bundle.py
python scripts/product/build_neo4j_trimmed_graph.py
```

Neo4j load (admin machine only):

```bash
export NEO4J_URI=bolt://127.0.0.1:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD='…'   # from your .env — not committed
python scripts/product/build_neo4j_trimmed_graph.py --load
```

Read-only user + Neovis wiring: `data/processed/product/neo4j_load/NEO4J_SETUP.md`.

### 4. Web apps

- Showcase: `web/nutri-showcase/` · Graph: `web/nutri-graph/`  
- Deploy notes: [`docs/DEPLOY_GUIDE.md`](docs/DEPLOY_GUIDE.md)  
- Local Streamlit (optional): `streamlit run streamlit_app/app.py`

Read-only Neo4j credentials used by the graph demo are **deploy-time placeholders** (`CHANGE_ME_READONLY_PASSWORD` in `data/processed/product/neo4j_load/`). Replace them on the server; never commit a real password.

## Data sources & licenses

Code is MIT. **Data is not.** Always check the upstream license before redistributing dumps.

| Dataset | Role | License (registry) | Link |
|---------|------|--------------------|------|
| FooDB | Food→compound | CC BY-NC 4.0 | https://foodb.ca/ |
| COCONUT | Natural-product compounds | See COCONUT terms | https://coconut.naturalproducts.net/ |
| ChEMBL | Measured bioactivity | CC BY-SA 3.0 | https://www.ebi.ac.uk/chembl/ |
| BindingDB | Measured binding | Verify upstream | https://www.bindingdb.org/ |
| USDA FoodData Central | Nutrition | Public domain (US gov) | https://fdc.nal.usda.gov/ |
| GTEx | Gene–tissue expression | GTEx / dbGaP terms | https://gtexportal.org/ |
| Gene Ontology | Pathways | CC BY 4.0 | http://geneontology.org/ |
| Reactome | Pathways | CC BY 4.0 | https://reactome.org/ |
| HGNC | Gene symbols | HGNC terms | https://www.genenames.org/ |
| UniProt | Protein↔gene | CC BY 4.0 | https://www.uniprot.org/ |
| PharmGKB / ClinPGx | PGx crosswalks | PharmGKB terms | https://www.pharmgkb.org/ |
| RecipeNLG | Recipes | MIT | https://recipenlg.cs.put.poznan.pl/ |
| Food.com / other recipes | Culinary corpora | Respective Kaggle/source terms | — |
| HMDB | Metabolites (supporting) | HMDB terms | https://hmdb.ca/ |
| PubChem | Compound identity | Public domain | https://pubchem.ncbi.nlm.nih.gov/ |
| Wikidata | Food labels | CC0 | https://www.wikidata.org/ |

Full registry: [`licenses/datasets_registry.csv`](licenses/datasets_registry.csv) · roster: `data/processed/product/showcase/dataset_roster.json`.

## Honest limitations

- **No pharmacokinetics** — no absorption, distribution, metabolism, or excretion modeling.  
- **Receptor pharmacology + nutrition**, not clinical efficacy.  
- **Relative**, not absolute, dose/contribution framing.  
- Inference is validated statistically; individual predicted edges can still be wrong.  
- **Not medical advice.** Do not use this graph to diagnose, treat, or prescribe.

## License

MIT © 2026 Abhinav Sikhwal — see [`LICENSE`](LICENSE).

---

**Built by Abhinav Sikhwal**  
[LinkedIn](https://www.linkedin.com/in/abhinav-sikhwal/) · [Portfolio](https://abhinavsikhwal.com) · abhisikhwal@gmail.com

*Portfolio project: end-to-end food chemistry → molecular targets, with measured/predicted honesty baked into the product graph.*
