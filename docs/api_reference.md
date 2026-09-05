# API Reference

Quick reference for key functions and classes in the NutriGraph codebase.

---

## Data Module

### `src.data.downloaders.DatasetDownloader`

Generic downloader for datasets with rate limiting and retry logic.

```python
from src.data.downloaders import DatasetDownloader

downloader = DatasetDownloader("usda_fooddata")
downloader.verify_license()
downloader.download_file(url, output_path)
```

**Methods:**
- `download_file(url, output_path)`: Download single file
- `api_request(endpoint, params)`: Make API request with rate limiting
- `verify_license()`: Check license compliance

---

### `src.data.validators.SchemaValidator`

Validate DataFrames against JSON schemas.

```python
from src.data.validators import SchemaValidator

validator = SchemaValidator()
result = validator.validate_dataframe(df, 'ingredient_schema')

if not result['valid']:
    print(result['errors'])
```

**Methods:**
- `validate_dataframe(df, schema_name)`: Validate against schema
- `check_foreign_keys(df, fk_col, ref_df, ref_col)`: Validate FK constraints
- `check_id_format(df, id_col, pattern)`: Validate ID patterns

---

## Features Module

### `src.features.compound_fingerprints.MorganFingerprintGenerator`

Generate Morgan (ECFP) fingerprints for compounds.

```python
from src.features.compound_fingerprints import MorganFingerprintGenerator

generator = MorganFingerprintGenerator(radius=2, n_bits=2048)
fp = generator.smiles_to_fingerprint(smiles)
similarity = generator.compute_tanimoto_similarity(fp1, fp2)
```

**Methods:**
- `smiles_to_fingerprint(smiles)`: Convert SMILES to fingerprint
- `generate_fingerprints(df, smiles_col)`: Batch generation
- `compute_tanimoto_similarity(fp1, fp2)`: Calculate similarity

---

## Models Module

### `src.models.compound_shift.CompoundShiftPredictor`

Predict compound concentration shifts based on environmental origin.

```python
from src.models.compound_shift import CompoundShiftPredictor

model = CompoundShiftPredictor(model_type="gradient_boosting")
metrics = model.train(X_train, y_train)
predictions = model.predict(X_test)
importance = model.get_feature_importance()
```

**Methods:**
- `train(X, y, **params)`: Train model
- `predict(X)`: Make predictions
- `get_feature_importance()`: Get feature importances
- `save_model(path)` / `load_model(path)`: Persistence

---

### `src.models.pathway_predictor.PathwayPredictor`

Perform pathway enrichment analysis.

```python
from src.models.pathway_predictor import PathwayPredictor

predictor = PathwayPredictor()
enrichment = predictor.enrich_pathways(
    target_list=['P35354', 'P12345'],
    target_pathway_map=df
)
```

**Methods:**
- `enrich_pathways(target_list, target_pathway_map)`: Run enrichment
- `map_compounds_to_pathways(compound_ids, ...)`: Compounds → pathways

---

## Graph Module

### `src.graph.cooccurrence.CooccurrenceNetwork`

Build ingredient co-occurrence networks from recipes.

```python
from src.graph.cooccurrence import CooccurrenceNetwork

network = CooccurrenceNetwork()
edgelist = network.build_cooccurrence_matrix(recipe_ingredient_map)
edgelist = network.calculate_pmi(edgelist, recipe_ingredient_map)
strong_pairs = network.find_strong_pairs(edgelist, min_pmi=2.0)
```

**Methods:**
- `build_cooccurrence_matrix(recipe_ingredient_map)`: Build matrix
- `calculate_pmi(edgelist, recipe_ingredient_map)`: Add PMI scores
- `find_strong_pairs(edgelist, min_cooccurrence, min_pmi)`: Filter

---

### `src.graph.network_analysis.NetworkAnalyzer`

Analyze network properties.

```python
from src.graph.network_analysis import NetworkAnalyzer

analyzer = NetworkAnalyzer()
G = analyzer.edgelist_to_graph(edgelist, 'source', 'target')
centrality = analyzer.calculate_centrality(G, method='degree')
communities = analyzer.detect_communities(G)
```

**Methods:**
- `edgelist_to_graph(edgelist, source_col, target_col)`: Convert to NetworkX
- `calculate_centrality(G, method)`: Compute centrality
- `detect_communities(G, method)`: Community detection
- `get_network_stats(G)`: Basic statistics

---

## Utilities

### `src.utils.io`

I/O utilities for data loading/saving.

```python
from src.utils.io import load_parquet, save_parquet, load_config

config = load_config('config/paths.yaml')
df = load_parquet('data/processed/ingredient_master.parquet')
save_parquet(df, 'output.parquet', compression='snappy')
```

**Functions:**
- `load_config(path)`: Load YAML config
- `load_parquet(path, columns=None)`: Load Parquet file
- `save_parquet(df, path, compression='snappy')`: Save Parquet file

---

### `src.utils.ontology.NameMatcher`

Match ingredient names with synonym handling.

```python
from src.utils.ontology import NameMatcher

matcher = NameMatcher(similarity_threshold=0.85)
match = matcher.match_with_synonyms('tumeric', reference_df)  # → 'turmeric'
results = matcher.batch_match(query_names, reference_df)
```

**Methods:**
- `normalize_name(name)`: Clean and normalize
- `exact_match(query, candidates)`: Find exact match
- `fuzzy_match(query, candidates)`: Fuzzy matching
- `match_with_synonyms(query, reference_df)`: Match with synonyms
- `batch_match(query_names, reference_df)`: Batch matching

---

### `src.utils.logging_config`

Configure logging.

```python
from src.utils.logging_config import setup_logging

setup_logging(level="INFO", log_file="logs/pipeline.log")
```

**Function:**
- `setup_logging(level, log_file=None, format_string=None)`: Configure logging

---

## Configuration Files

### `config/paths.yaml`
All file paths (never hardcode!)

### `config/datasets.yaml`
Dataset metadata (URLs, licenses, API endpoints)

### `config/model_params.yaml`
Hyperparameters for models

---

## Schema Validation

All output tables should validate against schemas in `schemas/*.json`:
- `ingredient_schema.json`
- `compound_schema.json`
- `recipe_schema.json`
- `signature_schema.json`

---

**Last updated**: 2026-02-01
