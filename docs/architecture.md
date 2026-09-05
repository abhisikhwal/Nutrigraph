# Architecture Documentation

## System Overview

NutriGraph is structured as a modular, multi-phase pipeline that links food chemistry to molecular targets and biological activity (with an earlier environmental-origin research track retained in the codebase).

### High-Level Architecture

```
Environmental Data → Ingredient Origins → Compound Chemistry → Bioactivity → Transcriptomics
        ↓                    ↓                    ↓                ↓              ↓
  [Satellite/            [Ingredient         [Chemical         [Protein      [Gene
   Climate/Soil]          Master]           Database]         Targets]     Expression]
                              ↓                    ↓                ↓              ↓
                         [Origin-Chemistry Model]         [Pathway Enrichment]
                                        ↓                          ↓
                                    [End-to-End Prediction]
```

---

## Data Flow

### Phase 1-2: Data Ingestion
- **Input**: Raw datasets (USDA, Wikidata, FooDB, ChEMBL, etc.)
- **Process**: Download, parse, standardize
- **Output**: Canonical tables (ingredient_master, compound_master)

### Phase 3-4: Knowledge Graph Construction
- **Input**: Canonical tables
- **Process**: Build mappings (ingredient→compound, compound→target)
- **Output**: Relational tables and networks

### Phase 5-7: Feature Engineering
- **Input**: Earth observation data, transcriptomics
- **Process**: Extract features, compute signatures
- **Output**: Feature matrices

### Phase 8: Modeling
- **Input**: Origin features + compound concentrations
- **Process**: Train ML model (gradient boosting)
- **Output**: Compound shift predictor

### Phase 9: Inference
- **Input**: Ingredient list + origin location
- **Process**: Full pipeline (ingredients → compounds → targets → pathways)
- **Output**: Predicted biological effects

---

## Module Organization

### `src/data/`
Data acquisition and processing
- `downloaders.py`: Generic download utilities
- `parsers.py`: Dataset-specific parsers
- `validators.py`: Schema validation

### `src/features/`
Feature engineering
- `compound_fingerprints.py`: Molecular fingerprints (Morgan, MACCS)

### `src/models/`
Machine learning models
- `compound_shift.py`: Origin → chemistry model
- `pathway_predictor.py`: Target → pathway enrichment

### `src/graph/`
Network analysis
- `cooccurrence.py`: Ingredient co-occurrence from recipes
- `network_analysis.py`: Centrality, communities

### `src/utils/`
Shared utilities
- `io.py`: Data I/O (Parquet, CSV, JSON)
- `logging_config.py`: Logging setup
- `ontology.py`: Name matching, synonyms

---

## Data Schemas

All data follows strict schemas defined in `schemas/*.json`:

### Key Tables
1. **ingredient_master**: Canonical ingredient ontology
2. **compound_master**: Chemical entities (PubChem CIDs)
3. **target_master**: Protein targets (UniProt)
4. **pathway_master**: Biological pathways (Reactome)
5. **signature_library**: Transcriptomic profiles (LINCS)

### Mapping Tables
- `ingredient_compound_map`: Which compounds are in which ingredients
- `compound_target_map`: Bioactivity data
- `target_pathway_map`: Pathway annotations
- `recipe_ingredient_map`: Recipe compositions

---

## Identifier Standards

See `schemas/canonical_ids.md` for full details.

**Primary IDs:**
- Ingredients: Custom `ING_00001`
- Compounds: PubChem CID (integer)
- Targets: UniProt accession (string)
- Pathways: Reactome ID (`R-HSA-XXXXXX`)
- Signatures: Source-specific (LINCS, GEO)

---

## Technology Stack

### Core Libraries
- **Data**: pandas, numpy, pyarrow (Parquet)
- **Chemistry**: RDKit (fingerprints, SMILES)
- **Biology**: Biopython, bioservices (API access)
- **ML**: scikit-learn, PyTorch (optional)
- **Geospatial**: geopandas, rasterio, xarray
- **Network**: NetworkX, igraph

### Workflow Orchestration
- **Makefile**: Simple sequential execution
- **Prefect** (optional): DAG-based workflows
- **DVC** (optional): Data version control

---

## Design Principles

1. **Modularity**: Each phase can run independently
2. **Idempotency**: Re-running scripts doesn't break things
3. **Configuration over hardcoding**: All paths in YAML
4. **Schema validation**: Catch errors early
5. **Logging over printing**: Track execution
6. **Parquet over CSV**: Faster, smaller, typed

---

## Scalability Considerations

### Current Implementation
- Designed for single-machine execution
- Optimized for datasets up to ~10M rows
- Uses Parquet for efficient storage

### Future Scaling
- **Distributed computing**: Spark/Dask for larger datasets
- **Cloud storage**: S3/GCS for raw data
- **Containerization**: Docker for reproducibility
- **Kubernetes**: For production deployment

---

## Testing Strategy

### Unit Tests
- Test individual functions/classes
- Mock external dependencies
- Located in `tests/test_*/`

### Integration Tests
- Test full pipeline phases
- Use small test datasets
- Validate outputs against schemas

### Data Quality Checks
- Schema validation after each phase
- Foreign key constraints
- ID format validation

---

## Documentation Standards

### Code Documentation
- Docstrings for all public functions
- Type hints where applicable
- Inline comments for complex logic

### Pipeline Documentation
- README for setup instructions
- This file for architecture
- `dataset_sources.md` for data provenance

---

**Last updated**: 2026-02-01
