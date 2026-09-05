# Canonical ID Standards

This document defines the primary identifier systems used throughout NutriGraph. Consistent IDs are critical for data integration across 15+ heterogeneous datasets.

---

## Ingredients

### Primary ID
- **Format**: Custom `ingredient_id` (string)
- **Pattern**: `ING_` + 5-digit zero-padded number
- **Example**: `ING_00001`, `ING_00042`, `ING_12345`
- **Rationale**: No single external database covers all ingredients (foods, spices, additives)

### External Mappings
Maintain mappings to external databases where available:
- **FDC ID** (USDA FoodData Central): Integer, e.g., `167512`
- **Wikidata QID**: String, e.g., `Q23501` (turmeric)
- **NCBI Taxonomy ID**: Integer, e.g., `4179` (Curcuma longa)
- **ITIS TSN** (Integrated Taxonomic Information System): Integer

### Naming Convention
- **canonical_name**: Preferred English name, lowercase, singular
- **scientific_name**: Binomial nomenclature when applicable
- **synonyms**: Array of alternative names (including regional spellings)

**Example**:
```json
{
  "ingredient_id": "ING_00042",
  "canonical_name": "turmeric",
  "scientific_name": "Curcuma longa",
  "synonyms": ["haldi", "curcuma", "indian saffron"],
  "fdc_id": 168408,
  "wikidata_qid": "Q23501",
  "taxonomy_id": 4179
}
```

---

## Compounds (Chemical Entities)

### Primary ID
- **Format**: **PubChem CID** (Compound Identifier)
- **Type**: Integer
- **Example**: `969516` (curcumin)
- **Rationale**: PubChem is the largest open chemistry database with comprehensive coverage

### Secondary Identifiers
- **InChIKey**: 14-character hash for structure deduplication
  - Example: `VFLDPWHFBUODDF-UHFFFAOYSA-N` (curcumin)
  - Use for merging data across sources
- **SMILES**: Canonical SMILES string (from PubChem)
  - Example: `COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O`
  - Store canonical form for consistency

### Alternative IDs (map but don't use as primary)
- ChEMBL ID: `CHEMBL2334026`
- CAS Registry Number: `458-37-7` (not always unique)
- ChemSpider ID: Integer
- KEGG Compound ID: `C10443` (limited coverage)

### Compound Naming
- **iupac_name**: IUPAC systematic name
- **common_names**: Array of trivial/trade names
- **molecular_formula**: `C21H20O6`

**Example**:
```json
{
  "pubchem_cid": 969516,
  "inchikey": "VFLDPWHFBUODDF-UHFFFAOYSA-N",
  "canonical_smiles": "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O",
  "iupac_name": "(1E,6E)-1,7-bis(4-hydroxy-3-methoxyphenyl)hepta-1,6-diene-3,5-dione",
  "common_names": ["curcumin", "diferuloylmethane"],
  "molecular_formula": "C21H20O6",
  "molecular_weight": 368.38,
  "chembl_id": "CHEMBL2334026"
}
```

---

## Targets (Proteins)

### Primary ID
- **Format**: **UniProt Accession** (UniProtKB)
- **Type**: String (6-10 alphanumeric characters)
- **Example**: `P35354` (PTGS2/COX-2)
- **Rationale**: UniProt is the gold standard for protein nomenclature

### Secondary Mappings
- **Ensembl Gene ID**: `ENSG00000073756`
- **Entrez Gene ID**: Integer, e.g., `5743`
- **HGNC Symbol**: Human gene symbol, e.g., `PTGS2`
- **PDB ID**: Protein structure ID (if available)

### Target Naming
- **protein_name**: Official protein name
- **gene_symbol**: Standard gene symbol (HGNC for human)
- **organism**: NCBI Taxonomy ID (default: `9606` for human)

**Example**:
```json
{
  "uniprot_accession": "P35354",
  "protein_name": "Prostaglandin G/H synthase 2",
  "gene_symbol": "PTGS2",
  "gene_synonyms": ["COX-2", "COX2"],
  "organism_id": 9606,
  "organism_name": "Homo sapiens",
  "ensembl_id": "ENSG00000073756",
  "entrez_id": 5743
}
```

---

## Pathways

### Primary ID
- **Format**: **Reactome Stable ID**
- **Pattern**: `R-HSA-` + 6-7 digit number (human pathways)
- **Example**: `R-HSA-2162123` (Synthesis of Prostaglandins)
- **Rationale**: Reactome provides curated, versioned, stable identifiers

### Secondary IDs
- **GO ID** (Gene Ontology): `GO:0006633` (biological process)
  - Use for pathway enrichment analysis
  - Note: GO terms are broader than Reactome pathways
- **KEGG Pathway ID**: e.g., `hsa00590` (Arachidonic acid metabolism)
  - **Caution**: KEGG has restrictive licensing; use sparingly

### Pathway Metadata
- **pathway_name**: Human-readable name
- **category**: Top-level Reactome category
- **species**: NCBI Taxonomy ID

**Example**:
```json
{
  "reactome_id": "R-HSA-2162123",
  "pathway_name": "Synthesis of Prostaglandins (PG) and Thromboxanes (TX)",
  "category": "Metabolism",
  "species_id": 9606,
  "go_terms": ["GO:0006633"],
  "kegg_id": "hsa00590"
}
```

---

## Recipes

### Primary ID
- **Format**: Custom `recipe_id`
- **Pattern**: `{source}_{original_id}`
- **Example**: `recipenlg_0001234`, `recipe1m_5a3b2c1d`
- **Rationale**: Preserve source provenance

### Recipe Metadata
- **recipe_name**: Title/name of recipe
- **source_dataset**: Origin database
- **url**: Link to original recipe (if available)
- **cuisine**: Optional classification

**Example**:
```json
{
  "recipe_id": "recipenlg_0001234",
  "recipe_name": "Golden Milk (Turmeric Latte)",
  "source_dataset": "RecipeNLG",
  "url": "https://...",
  "cuisine": "Indian",
  "ingredients": ["ING_00042", "ING_00123", "ING_00456"]
}
```

---

## Signatures (Transcriptomic Profiles)

### Primary ID
- **Format**: Source-specific signature ID
- **LINCS**: `sig_id` from LINCS metadata (e.g., `CPC014_VCAP_24H:BRD-K01507359`)
- **GEO**: GEO accession (e.g., `GSE92742`)
- **Rationale**: Preserve original IDs for traceability

### Gene Identifiers in Signatures
- **Use Entrez Gene ID** (integer) for consistency with pathway databases
- **Map from**: Ensembl, gene symbols, probe IDs
- **Example**: `5743` (PTGS2/COX-2)

### Signature Metadata
- **cell_line**: Cell type or tissue
- **treatment**: Compound/intervention
- **dose**: Concentration (with units)
- **duration**: Treatment time
- **platform**: L1000, RNA-seq, microarray

**Example**:
```json
{
  "signature_id": "CPC014_VCAP_24H:BRD-K01507359",
  "source": "LINCS_L1000",
  "cell_line": "VCAP",
  "treatment": "curcumin",
  "pubchem_cid": 969516,
  "dose_um": 10.0,
  "duration_hours": 24,
  "platform": "L1000",
  "num_genes": 978
}
```

---

## Geographic Locations (Origin Features)

### Primary ID
- **Format**: **(latitude, longitude, precision_meters)**
- **Type**: Tuple of floats + integer
- **Example**: `(28.6139, 77.2090, 1000)` for New Delhi region
- **Precision**: Spatial resolution (e.g., 1000m for regional, 10000m for country-level)

### Administrative IDs
- **ISO Country Code**: 2-letter (e.g., `IN` for India)
- **GADM Code**: Global Administrative Areas database
- **GAUL Code**: FAO Global Administrative Unit Layers

### Example
```json
{
  "location_id": "LOC_00123",
  "latitude": 28.6139,
  "longitude": 77.2090,
  "precision_m": 1000,
  "country_iso": "IN",
  "country_name": "India",
  "admin_level_1": "Delhi",
  "gadm_code": "IND.7_1"
}
```

---

## Cross-Referencing Best Practices

1. **Always store primary ID**: Use the canonical ID as the main key
2. **Store mappings separately**: Create mapping tables (e.g., `compound_target_map.parquet`)
3. **Track ID provenance**: Note which database each external ID comes from
4. **Handle ID changes**: Some databases update IDs; version your mappings
5. **NULL handling**: Use `null` (not `NA`, `None`, `-1`) for missing mappings

---

## ID Validation

Each schema JSON file includes validation rules:
- **Required fields**: Must be present
- **Format patterns**: Regex for ID formats
- **Type constraints**: Integer, string, array
- **Foreign key checks**: Referenced IDs must exist in master tables

See `schemas/*.json` for programmatic validation schemas.

---

**Last updated**: 2026-02-01
