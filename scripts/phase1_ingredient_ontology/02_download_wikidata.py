#!/usr/bin/env python3
"""
Query Wikidata for food ingredients, scientific names, and multilingual labels.

SPARQL endpoint: https://query.wikidata.org/sparql
License: CC0 (fully open)

Queries for:
- Food items (Q2095 subclasses)
- Spices (Q42527 subclasses)
- Scientific names (taxonomic bindings)
- Synonyms in multiple languages

Output:
- data/raw/wikidata/food_items.json
- data/interim/wikidata/ingredient_synonyms.parquet
"""

from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import json
import logging
import yaml
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/paths.yaml") as f:
    paths = yaml.safe_load(f)

RAW_DIR = Path(paths["raw_data"]) / "wikidata"
INTERIM_DIR = Path(paths["interim_data"]) / "wikidata"
RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

def query_wikidata(sparql_query):
    """Execute SPARQL query against Wikidata."""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", "Global-Food-Genome-Project/0.1.0 (Research)")
    
    try:
        results = sparql.query().convert()
        return results
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return None

def get_spices_and_herbs():
    """
    Get spices and herbs with scientific names and labels.
    """
    query = """
    SELECT DISTINCT ?item ?itemLabel ?scientificName ?enLabel ?hiLabel ?esLabel
    WHERE {
      # Spices or herbs
      { ?item wdt:P279* wd:Q42527 . }  # Subclass of spice
      UNION
      { ?item wdt:P279* wd:Q207123 . } # Subclass of herb
      
      # Get scientific name if available
      OPTIONAL { ?item wdt:P225 ?scientificName . }
      
      # Get labels in different languages
      OPTIONAL { ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en") }
      OPTIONAL { ?item rdfs:label ?hiLabel . FILTER(LANG(?hiLabel) = "hi") }
      OPTIONAL { ?item rdfs:label ?esLabel . FILTER(LANG(?esLabel) = "es") }
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 500
    """
    
    logger.info("Querying Wikidata for spices and herbs...")
    results = query_wikidata(query)
    
    if not results:
        return []
    
    bindings = results.get("results", {}).get("bindings", [])
    
    items = []
    for binding in bindings:
        item = {
            'wikidata_id': binding['item']['value'].split('/')[-1],
            'label': binding.get('itemLabel', {}).get('value'),
            'scientific_name': binding.get('scientificName', {}).get('value'),
            'label_en': binding.get('enLabel', {}).get('value'),
            'label_hi': binding.get('hiLabel', {}).get('value'),
            'label_es': binding.get('esLabel', {}).get('value'),
        }
        items.append(item)
    
    logger.info(f"Retrieved {len(items)} spices/herbs from Wikidata")
    return items

def get_food_items():
    """
    Get general food items.
    """
    query = """
    SELECT DISTINCT ?item ?itemLabel ?scientificName
    WHERE {
      ?item wdt:P279* wd:Q2095 .  # Subclass of food
      OPTIONAL { ?item wdt:P225 ?scientificName . }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 1000
    """
    
    logger.info("Querying Wikidata for food items...")
    results = query_wikidata(query)
    
    if not results:
        return []
    
    bindings = results.get("results", {}).get("bindings", [])
    
    items = []
    for binding in bindings:
        item = {
            'wikidata_id': binding['item']['value'].split('/')[-1],
            'label': binding.get('itemLabel', {}).get('value'),
            'scientific_name': binding.get('scientificName', {}).get('value'),
        }
        items.append(item)
    
    logger.info(f"Retrieved {len(items)} food items from Wikidata")
    return items

def main():
    # Get spices/herbs (high priority for your project)
    spices = get_spices_and_herbs()
    time.sleep(2)  # Rate limiting
    
    # Get general foods
    foods = get_food_items()
    
    # Save raw JSON
    all_items = spices + foods
    output_json = RAW_DIR / "food_items.json"
    with open(output_json, 'w') as f:
        json.dump(all_items, f, indent=2)
    logger.info(f"Saved raw data to {output_json}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_items)
    df = df.drop_duplicates(subset=['wikidata_id'])
    
    # Save as Parquet
    output_parquet = INTERIM_DIR / "wikidata_foods.parquet"
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved {len(df)} unique items to {output_parquet}")
    
    logger.info(f"✅ Wikidata download complete")

if __name__ == "__main__":
    main()
