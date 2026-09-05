# Dataset License Compliance Notes

## Overview
This document provides detailed guidance on using datasets in the NutriGraph project, with special attention to redistribution rights and commercial use restrictions.

---

## ✅ FULLY OPEN (Public Domain or Permissive Licenses)

### USDA FoodData Central
- **License**: Public Domain (US Government)
- **Commercial use**: ✅ Allowed
- **Redistribution**: ✅ Allowed
- **Attribution**: Not required, but recommended
- **Notes**: Fully open for any purpose

### Wikidata
- **License**: CC0 (Creative Commons Zero)
- **Commercial use**: ✅ Allowed
- **Redistribution**: ✅ Allowed
- **Attribution**: Not required
- **Notes**: Fully open, no restrictions

### PubChem
- **License**: Public Domain (NIH/NCBI)
- **Commercial use**: ✅ Allowed
- **Redistribution**: ✅ Allowed
- **Attribution**: Not required, but cite in publications
- **Notes**: Fully open

### ChEMBL
- **License**: CC BY-SA 3.0
- **Commercial use**: ✅ Allowed
- **Redistribution**: ✅ Allowed with attribution
- **Attribution**: **Required** - cite ChEMBL publication
- **Share-alike**: If you create derivatives, must use same license
- **Citation**: Gaulton et al. (2017) Nucleic Acids Res.

### RecipeNLG
- **License**: MIT
- **Commercial use**: ✅ Allowed
- **Redistribution**: ✅ Allowed
- **Attribution**: Required (copyright notice)
- **Notes**: Fully open for commercial use

### LINCS L1000
- **License**: Open (Broad Institute)
- **Commercial use**: ✅ Allowed
- **Redistribution**: ✅ Allowed
- **Attribution**: Cite Subramanian et al. (2017)
- **Notes**: Requires free account for API access

### Reactome
- **License**: CC BY 4.0
- **Commercial use**: ✅ Allowed
- **Redistribution**: ✅ Allowed with attribution
- **Attribution**: **Required**
- **Citation**: Gillespie et al. (2022) Nucleic Acids Res.

### Gene Ontology
- **License**: CC BY 4.0
- **Commercial use**: ✅ Allowed
- **Redistribution**: ✅ Allowed with attribution
- **Attribution**: **Required**
- **Citation**: Gene Ontology Consortium

### Copernicus Sentinel & ERA5
- **License**: Copernicus Open Access Hub
- **Commercial use**: ✅ Allowed
- **Redistribution**: ✅ Allowed with attribution
- **Attribution**: **Required** - "Contains modified Copernicus Sentinel data [year]"
- **Notes**: Free and open, must attribute ESA/Copernicus

### SoilGrids
- **License**: CC BY 4.0
- **Commercial use**: ✅ Allowed
- **Redistribution**: ✅ Allowed with attribution
- **Attribution**: **Required** - cite ISRIC
- **Citation**: Poggio et al. (2021)

### MetaboLights
- **License**: CC BY 4.0
- **Commercial use**: ✅ Allowed
- **Redistribution**: ✅ Allowed with attribution
- **Attribution**: **Required**
- **Notes**: Individual studies may have additional restrictions

---

## ⚠️ RESTRICTED OR UNCLEAR LICENSES

### FooDB
- **License**: CC BY-NC 4.0 (Non-Commercial)
- **Commercial use**: ❌ **NOT ALLOWED**
- **Redistribution**: ⚠️ Allowed for non-commercial use only
- **Attribution**: **Required**
- **CRITICAL**: Cannot be used in commercial products or services
- **Workaround**: Use only for research; replace with PubChem/ChEMBL for production

### FlavorDB
- **License**: ⚠️ **UNKNOWN/UNCLEAR**
- **Commercial use**: ❌ **VERIFY FIRST**
- **Redistribution**: ❌ **VERIFY FIRST**
- **Action required**: Contact authors before use
- **Email**: (Find contact on website)
- **Status**: DO NOT USE until license is clarified

### Recipe1M+
- **License**: Research use only (requires data use agreement)
- **Commercial use**: ❌ **NOT ALLOWED**
- **Redistribution**: ❌ **NOT ALLOWED**
- **Attribution**: Required
- **Notes**: Must sign data use agreement; strictly for academic research

### BindingDB
- **License**: ⚠️ **CHECK WEBSITE**
- **Commercial use**: ❌ **VERIFY FIRST**
- **Redistribution**: ❌ **VERIFY FIRST**
- **Status**: License terms not clearly stated; verify before use

---

## Redistribution Guidelines

### ✅ CAN Redistribute (with proper attribution)
- USDA FoodData Central
- Wikidata
- PubChem
- ChEMBL (with CC BY-SA)
- RecipeNLG
- LINCS L1000
- Reactome
- Gene Ontology
- Copernicus/Sentinel
- SoilGrids
- MetaboLights

### ⚠️ CANNOT Redistribute
- FooDB (non-commercial restriction)
- Recipe1M+ (research agreement)
- FlavorDB (unknown license)
- BindingDB (unclear)

---

## Commercial Use Summary

### ✅ Safe for Commercial Products
Start with these datasets for MVP/production:
1. USDA FoodData Central
2. Wikidata
3. PubChem
4. ChEMBL
5. RecipeNLG (not Recipe1M+)
6. LINCS L1000
7. Reactome
8. Gene Ontology
9. Copernicus data
10. SoilGrids

### ❌ NOT Safe for Commercial Use
Do NOT use these in commercial products:
1. **FooDB** - Use PubChem/ChEMBL instead
2. **Recipe1M+** - Use RecipeNLG instead
3. **FlavorDB** - Verify or replace with open alternatives

---

## Action Items

### Before First Data Download
- [ ] Review this document
- [ ] Update `datasets_registry.csv` with actual download dates
- [ ] Verify FlavorDB license (contact authors)
- [ ] Verify BindingDB license (check website)
- [ ] Decide: use FooDB (research only) or skip (production)?

### Before Commercial Deployment
- [ ] Remove or replace FooDB data
- [ ] Remove or replace Recipe1M+ data
- [ ] Ensure all remaining datasets allow commercial use
- [ ] Add attribution statements to documentation
- [ ] Include dataset citations in acknowledgments

### Attribution Template for Publications
```
This work uses data from:
- USDA FoodData Central (https://fdc.nal.usda.gov/)
- ChEMBL (Gaulton et al., 2017, doi:10.1093/nar/gkw1074)
- Reactome (Gillespie et al., 2022, doi:10.1093/nar/gkab1028)
- LINCS L1000 (Subramanian et al., 2017, doi:10.1016/j.cell.2017.10.049)
- Gene Ontology Consortium (http://geneontology.org/)
- Copernicus Sentinel data [year]
- (Add others as used)
```

---

## Legal Disclaimer

**This document is for guidance only and does not constitute legal advice.**

Always:
1. Read the original license terms on each dataset's website
2. Contact dataset maintainers if terms are unclear
3. Consult legal counsel for commercial use questions
4. Document all license verification steps

License terms may change over time. Periodically re-verify.

---

**Last updated**: 2026-02-01  
**Reviewed by**: (Add your name after verification)
