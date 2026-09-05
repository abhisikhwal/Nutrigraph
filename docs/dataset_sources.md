# Dataset Sources and Access Methods

Complete reference for all datasets used in the NutriGraph project.

---

## Ingredient Ontology

### USDA FoodData Central
- **URL**: https://fdc.nal.usda.gov/
- **Type**: US government nutritional database
- **Access**: API (requires key) or bulk download
- **Coverage**: ~800,000 foods
- **License**: Public Domain
- **Update frequency**: Quarterly
- **Citation**: U.S. Department of Agriculture, Agricultural Research Service. FoodData Central, 2024. fdc.nal.usda.gov.

### Wikidata
- **URL**: https://www.wikidata.org/
- **Type**: Open knowledge graph
- **Access**: SPARQL query endpoint
- **Coverage**: Universal (100M+ entities)
- **License**: CC0
- **Update frequency**: Real-time
- **Citation**: Wikidata contributors. Wikidata. https://www.wikidata.org/

---

## Food Chemistry

### FooDB
- **URL**: https://foodb.ca/
- **Type**: Food compound database
- **Access**: Bulk download
- **Coverage**: 70,000+ compounds, 900+ foods
- **License**: ⚠️ CC BY-NC 4.0 (Non-Commercial)
- **Update frequency**: Irregular
- **Citation**: FooDB Version 1.0, https://foodb.ca

### FlavorDB
- **URL**: https://cosylab.iiitd.edu.in/flavordb/
- **Type**: Flavor compounds and odor molecules
- **Access**: Web download
- **Coverage**: 25,000+ flavor molecules
- **License**: ⚠️ UNCLEAR - verify before use
- **Citation**: Garg et al. (2018) FlavorDB: a database of flavor molecules. Nucleic Acids Res.

### PubChem
- **URL**: https://pubchem.ncbi.nlm.nih.gov/
- **Type**: Chemical compound database
- **Access**: API, FTP
- **Coverage**: 110M+ compounds
- **License**: Public Domain
- **Update frequency**: Weekly
- **Citation**: Kim et al. (2023) PubChem 2023 update. Nucleic Acids Res.

---

## Bioactivity

### ChEMBL
- **URL**: https://www.ebi.ac.uk/chembl/
- **Type**: Bioactivity database (compound-target-assay)
- **Access**: FTP, API
- **Coverage**: 2.4M compounds, 15,000+ targets, 20M+ activities
- **License**: CC BY-SA 3.0
- **Update frequency**: Quarterly
- **Citation**: Gaulton et al. (2017) The ChEMBL database in 2017. Nucleic Acids Res.

### BindingDB
- **URL**: https://www.bindingdb.org/
- **Type**: Protein-ligand binding affinities
- **Access**: Bulk download
- **Coverage**: 2.8M binding data, 1M compounds, 9,000 targets
- **License**: ⚠️ CHECK - unclear for commercial use
- **Citation**: Gilson et al. (2016) BindingDB in 2015. Nucleic Acids Res.

---

## Culinary Data

### RecipeNLG
- **URL**: https://recipenlg.cs.put.poznan.pl/
- **Type**: Recipe dataset (NLG task)
- **Access**: Bulk download
- **Coverage**: 2.2M recipes
- **License**: MIT (fully open)
- **Citation**: Bień et al. (2020) RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation. ACL Workshop.

### Recipe1M+
- **URL**: http://pic2recipe.csail.mit.edu/
- **Type**: Recipe dataset with images
- **Access**: Request from authors
- **Coverage**: 1M recipes
- **License**: ⚠️ Research only (requires data use agreement)
- **Citation**: Marin et al. (2019) Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images. TPAMI.

---

## Pathways & Ontologies

### Reactome
- **URL**: https://reactome.org/
- **Type**: Curated pathway database
- **Access**: API, bulk download
- **Coverage**: 2,700+ human pathways
- **License**: CC BY 4.0
- **Update frequency**: Quarterly
- **Citation**: Gillespie et al. (2022) The reactome pathway knowledgebase 2022. Nucleic Acids Res.

### Gene Ontology (GO)
- **URL**: http://geneontology.org/
- **Type**: Gene function ontology
- **Access**: OBO/OWL download, API
- **Coverage**: Universal gene function annotations
- **License**: CC BY 4.0
- **Update frequency**: Daily
- **Citation**: Gene Ontology Consortium (2023) The Gene Ontology knowledgebase in 2023. Genetics.

---

## Transcriptomics

### LINCS L1000
- **URL**: https://clue.io/
- **Type**: Large-scale gene expression profiles
- **Access**: API (requires account), bulk download
- **Coverage**: 1.3M signatures, 20,000+ compounds
- **License**: Open (attribution requested)
- **Citation**: Subramanian et al. (2017) A Next Generation Connectivity Map. Cell.

### GEO (Gene Expression Omnibus)
- **URL**: https://www.ncbi.nlm.nih.gov/geo/
- **Type**: Public genomics data repository
- **Access**: FTP, API
- **Coverage**: 6M+ samples
- **License**: Varies by dataset (mostly open)
- **Citation**: Barrett et al. (2013) NCBI GEO: archive for functional genomics data sets. Nucleic Acids Res.

---

## Earth Observation

### Copernicus Sentinel
- **URL**: https://dataspace.copernicus.eu/
- **Type**: Satellite imagery (Sentinel-1, Sentinel-2)
- **Access**: API (requires account)
- **Coverage**: Global, 10m resolution (optical)
- **License**: Free and open
- **Attribution**: "Contains modified Copernicus Sentinel data [year]"
- **Citation**: ESA Copernicus Sentinel Data.

### ERA5 Climate Reanalysis
- **URL**: https://cds.climate.copernicus.eu/
- **Type**: Climate reanalysis data
- **Access**: API (requires CDS account)
- **Coverage**: Global, hourly, 1979-present
- **License**: Copernicus license (open with attribution)
- **Citation**: Hersbach et al. (2020) The ERA5 global reanalysis. QJRMS.

### SoilGrids
- **URL**: https://soilgrids.org/
- **Type**: Global soil properties
- **Access**: API, WCS service
- **Coverage**: Global, 250m resolution
- **License**: CC BY 4.0
- **Citation**: Poggio et al. (2021) SoilGrids 2.0. SOIL.

---

## Metabolomics (Optional)

### MetaboLights
- **URL**: https://www.ebi.ac.uk/metabolights/
- **Type**: Metabolomics studies repository
- **Access**: API, FTP
- **Coverage**: 6,000+ studies
- **License**: CC BY 4.0 (individual studies may vary)
- **Citation**: Haug et al. (2020) MetaboLights: open-access repository for metabolomics. Nucleic Acids Res.

---

## Protein Structure (Optional)

### Protein Data Bank (PDB)
- **URL**: https://www.rcsb.org/
- **Type**: 3D protein structures
- **Access**: FTP, API
- **Coverage**: 200,000+ structures
- **License**: Open
- **Citation**: Berman et al. (2000) The Protein Data Bank. Nucleic Acids Res.

---

## Access Instructions

### Getting API Keys

1. **USDA FoodData Central**: https://fdc.nal.usda.gov/api-key-signup.html
2. **LINCS L1000**: https://clue.io/connectopedia/api_access
3. **Copernicus Sentinel**: https://dataspace.copernicus.eu/
4. **ERA5**: https://cds.climate.copernicus.eu/api-how-to

### Rate Limits

- **Wikidata SPARQL**: ~1 request/second (unofficial)
- **PubChem**: 5 requests/second (unauthenticated)
- **ChEMBL**: No strict limit, be respectful
- **LINCS**: Documented in API docs

---

## Data Use Agreements

Some datasets require explicit agreements:
- **Recipe1M+**: Email authors for data use agreement
- **BindingDB**: Verify terms on website
- **FlavorDB**: Contact authors to clarify license

---

## Citation Best Practices

When publishing, include:
1. Dataset name and version
2. Access date
3. URL
4. Primary citation (from above)

Example:
```
We obtained bioactivity data from ChEMBL version 33 (accessed 2024-10-15)
[Gaulton et al. 2017]. Climate data was sourced from ERA5 reanalysis 
[Hersbach et al. 2020], containing modified Copernicus data (2020-2024).
```

---

**Last updated**: 2026-02-01  
**Maintained by**: Abhinav Sikhwal
