#!/usr/bin/env python3
"""Human-friendly effect and body-region theme definitions for retrieval layer."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

# Body-region themes -> GTEx tissue labels (68-tissue vocabulary).
BODY_REGION_THEMES: dict[str, dict[str, Any]] = {
    "liver": {
        "label": "Liver",
        "description": "Hepatic compartments and portal tract",
        "tissues": [
            "Liver",
            "Liver_Hepatocyte",
            "Liver_Mixed_Cell",
            "Liver_Portal_Tract",
        ],
    },
    "gut": {
        "label": "Gut / digestive tract",
        "description": "Stomach, intestine, colon, esophagus",
        "tissues": [
            "Stomach",
            "Stomach_Mucosa",
            "Stomach_Muscularis",
            "Stomach_Mixed_Cell",
            "Small_Intestine_Terminal_Ileum",
            "Small_Intestine_Terminal_Ileum_Mixed_Cell",
            "Small_Intestine_Terminal_Ileum_Lymphoid_Aggregate",
            "Colon_Sigmoid",
            "Colon_Transverse",
            "Colon_Transverse_Mucosa",
            "Colon_Transverse_Muscularis",
            "Colon_Transverse_Mixed_Cell",
            "Esophagus_Mucosa",
            "Esophagus_Muscularis",
            "Esophagus_Gastroesophageal_Junction",
        ],
    },
    "brain": {
        "label": "Brain / CNS",
        "description": "GTEx brain regions and spinal cord",
        "tissues": [
            "Brain_Cortex",
            "Brain_Frontal_Cortex_BA9",
            "Brain_Hippocampus",
            "Brain_Hypothalamus",
            "Brain_Amygdala",
            "Brain_Caudate_basal_ganglia",
            "Brain_Putamen_basal_ganglia",
            "Brain_Nucleus_accumbens_basal_ganglia",
            "Brain_Substantia_nigra",
            "Brain_Cerebellum",
            "Brain_Cerebellar_Hemisphere",
            "Brain_Anterior_cingulate_cortex_BA24",
            "Brain_Spinal_cord_cervical_c-1",
        ],
    },
    "heart": {
        "label": "Heart",
        "description": "Cardiac tissue compartments",
        "tissues": ["Heart_Left_Ventricle", "Heart_Atrial_Appendage"],
    },
    "kidney": {
        "label": "Kidney",
        "description": "Renal cortex and medulla",
        "tissues": ["Kidney_Cortex", "Kidney_Medulla"],
    },
    "lung": {
        "label": "Lung",
        "description": "Pulmonary tissue",
        "tissues": ["Lung"],
    },
    "adipose": {
        "label": "Adipose / fat",
        "description": "Subcutaneous and visceral fat depots",
        "tissues": ["Adipose_Subcutaneous", "Adipose_Visceral_Omentum"],
    },
    "muscle": {
        "label": "Skeletal muscle",
        "description": "Muscle tissue",
        "tissues": ["Muscle_Skeletal"],
    },
    "skin": {
        "label": "Skin",
        "description": "Sun-exposed and protected skin",
        "tissues": ["Skin_Not_Sun_Exposed_Suprapubic", "Skin_Sun_Exposed_Lower_leg"],
    },
    "blood": {
        "label": "Blood / immune cells",
        "description": "Whole blood and lymphocytes",
        "tissues": ["Whole_Blood", "Cells_EBV-transformed_lymphocytes", "Spleen"],
    },
    "pancreas": {
        "label": "Pancreas",
        "description": "Islets, acini, and mixed pancreatic tissue",
        "tissues": ["Pancreas", "Pancreas_Islets", "Pancreas_Acini", "Pancreas_Mixed_Cell"],
    },
    "reproductive": {
        "label": "Reproductive organs",
        "description": "Uterus, ovary, prostate, testis, etc.",
        "tissues": [
            "Uterus",
            "Ovary",
            "Prostate",
            "Testis",
            "Vagina",
            "Fallopian_Tube",
            "Cervix_Ectocervix",
            "Cervix_Endocervix",
            "Breast_Mammary_Tissue",
        ],
    },
    "endocrine": {
        "label": "Endocrine glands",
        "description": "Thyroid, adrenal, pituitary",
        "tissues": ["Thyroid", "Adrenal_Gland", "Pituitary"],
    },
    "nerve": {
        "label": "Peripheral nerve",
        "description": "Peripheral nervous tissue",
        "tissues": ["Nerve_Tibial"],
    },
}

# Broad retrieval intents may include sub-themes at scoring time (pathway union).
# Sub-themes remain queryable on their own for precise lookups.
THEME_RETRIEVAL_EXPANSIONS: dict[str, dict[str, Any]] = {
    "inflammation_immune": {
        "include_sub_themes": ["eicosanoid_prostaglandin"],
        "description": (
            "Broad inflammation intent spans prostaglandin/leukotriene/COX axis "
            "(eicosanoid_prostaglandin sub-theme) in addition to core immune/inflammatory pathways."
        ),
    },
}

# Effect/system themes: keyword rules applied to pathway_name + category_name (case-insensitive).
# Pathways may match multiple themes (multi-assign).
EFFECT_THEME_RULES: list[dict[str, Any]] = [
    {
        "theme_id": "inflammation_immune",
        "label": "Inflammation & immune response",
        "keywords": [
            r"\binflamm",
            r"\bcytokine",
            r"\binterleukin\b",
            r"\btnf\b",
            r"\bimmune response",
            r"\bleukocyte",
            r"\bmacrophage",
            r"\bantigen",
            r"\bcomplement\b",
            r"\binterferon",
            r"\bnf-kappa",
            r"\bnf-kb",
            r"\btraf6",
            r"\bikk\b",
        ],
        "pathway_ids": [
            "GO:0006954",  # inflammatory response
            "GO:0019371",  # cyclooxygenase pathway
            "GO:0043123",  # positive regulation of canonical NF-kappaB
            "GO:0043124",  # negative regulation of canonical NF-kappaB
            "GO:1901224",  # positive regulation of non-canonical NF-kappaB
            "R-HSA-445989",  # TAK1-dependent IKK and NF-kappa-B activation
            "R-HSA-933542",  # TRAF6 mediated NF-kB activation
        ],
        "category_ids": [],
    },
    {
        "theme_id": "eicosanoid_prostaglandin",
        "label": "Eicosanoid / prostaglandin / COX",
        "keywords": [
            r"\bprostaglandin",
            r"\beicosanoid",
            r"\bcyclooxygenase",
            r"\bcox\b",
            r"\blipoxygenase",
            r"\bleukotriene",
            r"\bthromboxane",
            r"\bepoxygenase",
        ],
        "pathway_ids": ["GO:0019371", "R-HSA-2142670", "R-HSA-2142690"],
        "category_ids": [],
    },
    {
        "theme_id": "xenobiotic_detox",
        "label": "Detoxification & xenobiotic metabolism",
        "keywords": [
            r"\bxenobiotic",
            r"\bdrug adme",
            r"\bbiological oxidation",
            r"\bcytochrome p450",
            r"\bcyp\d",
            r"\bphase i\b",
            r"\bphase ii\b",
            r"\bglucuronid",
            r"\bsulfat",
            r"\b detox",
        ],
        "category_ids": ["GO:0006805", "R-HSA-9748784", "R-HSA-211859"],
    },
    {
        "theme_id": "neurotransmitter_brain",
        "label": "Neurotransmitter & brain signaling",
        "keywords": [
            r"\bgaba",
            r"\bdopamin",
            r"\bserotonin",
            r"\bach\b",
            r"\bglutamat",
            r"\bsynaptic",
            r"\bneuron",
            r"\bneurotrans",
            r"\bcholinergic",
            r"\badrenergic",
        ],
    },
    {
        "theme_id": "endocannabinoid",
        "label": "Endocannabinoid signaling",
        "keywords": [r"\bcannabinoid", r"\bcnr[12]", r"\bfatty acid amide"],
        "pathway_ids": [],
    },
    {
        "theme_id": "gpcr_signaling",
        "label": "GPCR & receptor signaling",
        "keywords": [
            r"\bg protein-coupled",
            r"\bgpcr",
            r"\breceptor signaling pathway",
            r"\bsignaling by gpcr",
        ],
        "category_ids": [],
    },
    {
        "theme_id": "metabolism_energy",
        "label": "General metabolism & energy",
        "keywords": [
            r"\bmetabolic process",
            r"\bmetabolism",
            r"\batp\b",
            r"\bglycol",
            r"\bgluconeogen",
            r"\bfatty acid metabolic",
            r"\benergy derivation",
        ],
        "category_ids": ["GO:0008152", "R-HSA-1430728"],
    },
    {
        "theme_id": "lipid_metabolism",
        "label": "Lipid metabolism",
        "keywords": [
            r"\blipid metabolic",
            r"\blipid biosynth",
            r"\bcholesterol",
            r"\btriglyceride",
            r"\bphospholipid",
            r"\bsteroid",
        ],
        "category_ids": ["GO:0006629"],
    },
    {
        "theme_id": "steroid_hormone",
        "label": "Steroid & hormone metabolism",
        "keywords": [
            r"\bsteroid",
            r"\bestrogen",
            r"\btestosterone",
            r"\bprogesterone",
            r"\bhormone",
            r"\bandrogen",
        ],
    },
    {
        "theme_id": "oxidative_stress",
        "label": "Oxidative stress & redox",
        "keywords": [
            r"\boxidative",
            r"\bredox",
            r"\breactive oxygen",
            r"\bantioxidant",
            r"\bperoxidase",
        ],
    },
    {
        "theme_id": "cardiovascular",
        "label": "Cardiovascular & vascular",
        "keywords": [
            r"\bvascular",
            r"\bheart",
            r"\bcardiac",
            r"\bblood pressure",
            r"\bangiotensin",
            r"\bhemostasis",
            r"\bcoagul",
        ],
        "category_ids": ["R-HSA-109582"],
    },
    {
        "theme_id": "digestion_absorption",
        "label": "Digestion & nutrient absorption",
        "keywords": [
            r"\bdigest",
            r"\babsorption",
            r"\bintestinal",
            r"\bnutrient transport",
        ],
        "category_ids": ["R-HSA-8963743"],
    },
    {
        "theme_id": "cell_cycle_dna",
        "label": "Cell cycle & DNA repair",
        "keywords": [
            r"\bcell cycle",
            r"\bdna repair",
            r"\bdna replication",
            r"\bmitotic",
            r"\bchromosome",
        ],
        "category_ids": ["R-HSA-1640170", "R-HSA-73894", "R-HSA-69306"],
    },
    {
        "theme_id": "apoptosis_cell_death",
        "label": "Apoptosis & cell death",
        "keywords": [r"\bapoptosis", r"\bcell death", r"\bnecroptosis", r"\bautophagy"],
        "category_ids": ["R-HSA-9612973"],
    },
    {
        "theme_id": "immune_cytokine",
        "label": "Cytokine & interleukin signaling",
        "keywords": [
            r"\bcytokine",
            r"\binterleukin",
            r"\binterferon",
            r"\btoll-like",
            r"\bnf-kappa",
        ],
    },
    {
        "theme_id": "vitamin_micronutrient",
        "label": "Vitamin & micronutrient response",
        "keywords": [
            r"\bvitamin",
            r"\bretino",
            r"\bfolate",
            r"\bresponse to vitamin",
        ],
    },
    {
        "theme_id": "ion_transport",
        "label": "Ion transport & membrane",
        "keywords": [
            r"\bion transport",
            r"\bchloride transmembrane",
            r"\bcalcium signaling",
            r"\bpotassium",
            r"\bsodium",
        ],
    },
    {
        "theme_id": "pain_nociception",
        "label": "Pain & nociception",
        "keywords": [r"\bpain", r"\bnocicept", r"\banalges", r"\bopioid"],
    },
    {
        "theme_id": "muscle_contraction",
        "label": "Muscle contraction & motility",
        "keywords": [r"\bmuscle contraction", r"\bactin", r"\bmyosin", r"\bmotility"],
    },
    {
        "theme_id": "wnt_developmental",
        "label": "Wnt & developmental signaling",
        "keywords": [r"\bwnt", r"\bdevelopmental", r"\bmorphogen"],
        "category_ids": ["R-HSA-1266738"],
    },
    {
        "theme_id": "biological_regulation",
        "label": "Biological regulation & homeostasis",
        "keywords": [
            r"\bbiological regulation",
            r"\bhomeostasis",
            r"\bresponse to stimulus",
            r"\bregulation of",
        ],
    },
]

# Curated cuisine -> characteristic ingredient strings (resolved to species at build time).
CUISINE_SEED: dict[str, dict[str, Any]] = {
    "indian": {
        "label": "Indian",
        "ingredient_strings": [
            "turmeric", "cumin", "coriander", "cardamom", "ginger", "black pepper",
            "cinnamon", "clove", "fenugreek", "mustard", "curry", "garam masala",
            "chili", "garlic", "onion", "basmati rice", "lentil", "ghee",
        ],
        "recipe_source_proxy": "indian_food",
    },
    "italian": {
        "label": "Italian",
        "ingredient_strings": [
            "basil", "oregano", "tomato", "garlic", "olive oil", "parmesan",
            "rosemary", "thyme", "parsley", "mozzarella", "pasta", "wine",
            "balsamic vinegar", "prosciutto",
        ],
    },
    "mexican": {
        "label": "Mexican",
        "ingredient_strings": [
            "cumin", "chili", "coriander", "lime", "avocado", "tomato", "garlic",
            "oregano", "cilantro", "jalapeno", "black bean", "corn", "cocoa",
        ],
    },
    "thai": {
        "label": "Thai",
        "ingredient_strings": [
            "ginger", "lemongrass", "basil", "cilantro", "lime", "coconut",
            "fish sauce", "turmeric", "garlic", "chili", "galangal", "kaffir lime",
        ],
    },
    "mediterranean": {
        "label": "Mediterranean",
        "ingredient_strings": [
            "olive oil", "garlic", "tomato", "basil", "oregano", "rosemary",
            "thyme", "lemon", "feta", "chickpea", "eggplant", "caper",
        ],
    },
    "japanese": {
        "label": "Japanese",
        "ingredient_strings": [
            "soy sauce", "miso", "ginger", "wasabi", "seaweed", "rice",
            "sesame", "tofu", "mirin", "sake", "dashi",
        ],
    },
    "chinese": {
        "label": "Chinese",
        "ingredient_strings": [
            "ginger", "garlic", "soy sauce", "sesame oil", "star anise",
            "sichuan pepper", "rice", "bok choy", "tofu", "green onion",
        ],
    },
}


def compile_theme_patterns() -> list[dict[str, Any]]:
    compiled: list[dict[str, Any]] = []
    for rule in EFFECT_THEME_RULES:
        patterns = [re.compile(k, re.IGNORECASE) for k in rule.get("keywords", [])]
        compiled.append({**rule, "_patterns": patterns})
    return compiled


def match_effect_themes(
    text: str,
    pathway_id: str | None = None,
    category_id: str | None = None,
    compiled_rules: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return theme_ids matching text, explicit pathway_id, or category_id."""
    compiled_rules = compiled_rules or compile_theme_patterns()
    matched: list[str] = []
    haystack = text or ""
    for rule in compiled_rules:
        theme_id = rule["theme_id"]
        if pathway_id and pathway_id in rule.get("pathway_ids", []):
            matched.append(theme_id)
            continue
        if category_id and category_id in rule.get("category_ids", []):
            matched.append(theme_id)
            continue
        if any(p.search(haystack) for p in rule["_patterns"]):
            matched.append(theme_id)
    return matched


def expanded_theme_pathway_ids(
    theme_id: str,
    theme_pathways: dict[str, set[str]],
) -> set[str]:
    """Return pathway IDs for scoring/retrieval, including configured sub-theme unions."""
    pids = set(theme_pathways.get(theme_id, set()))
    expansion = THEME_RETRIEVAL_EXPANSIONS.get(theme_id, {})
    for sub_tid in expansion.get("include_sub_themes", []):
        pids.update(theme_pathways.get(sub_tid, set()))
    return pids


def audit_theme_fragmentation(
    theme_pathways: dict[str, set[str]],
    pathway_names: dict[str, str],
    compiled_rules: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Surface intent-relevant axes split across themes or left unthemed.
    Heuristic: keyword families that match pathways in multiple themes or in none.
    """
    compiled_rules = compiled_rules or compile_theme_patterns()
    axis_patterns: list[tuple[str, re.Pattern[str]]] = [
        ("prostaglandin/eicosanoid", re.compile(r"prostaglandin|eicosanoid|leukotriene|cyclooxygenase", re.I)),
        ("nf-kb", re.compile(r"nf-kappa|nf-kb|traf6|ikk", re.I)),
        ("cytokine/interleukin", re.compile(r"cytokine|interleukin|interferon", re.I)),
        ("xenobiotic/cyp", re.compile(r"xenobiotic|cytochrome p450|cyp\d", re.I)),
        ("gaba/dopamine", re.compile(r"gaba|dopamin|serotonin", re.I)),
        ("cannabinoid", re.compile(r"cannabinoid|cnr[12]", re.I)),
    ]
    findings: list[dict[str, Any]] = []
    pid_to_themes: dict[str, set[str]] = defaultdict(set)
    for tid, pids in theme_pathways.items():
        for pid in pids:
            pid_to_themes[pid].add(tid)

    for axis_name, pattern in axis_patterns:
        matching_pids = [
            pid for pid, name in pathway_names.items() if pattern.search(name or "")
        ]
        if not matching_pids:
            continue
        theme_counts: dict[str, int] = defaultdict(int)
        unthemed = 0
        for pid in matching_pids:
            themes = pid_to_themes.get(pid, set())
            if not themes:
                unthemed += 1
            for tid in themes:
                theme_counts[tid] += 1
        if len(theme_counts) > 1 or unthemed > 0:
            findings.append(
                {
                    "axis": axis_name,
                    "n_pathways_matched": len(matching_pids),
                    "n_unthemed": unthemed,
                    "themes_with_hits": dict(sorted(theme_counts.items(), key=lambda x: -x[1])),
                    "sample_unthemed": [
                        {"pathway_id": pid, "pathway_name": pathway_names.get(pid, pid)}
                        for pid in matching_pids
                        if pid not in pid_to_themes
                    ][:5],
                }
            )
    return findings
