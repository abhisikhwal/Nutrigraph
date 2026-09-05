"""
Apply approved gene-name normalizations to compound_gene_expanded_canonical.csv.
Writes compound_gene_expanded_canonical_normalized.csv; does not modify the original.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/processed/canonical/compound_gene_expanded_canonical.csv"
OUT = ROOT / "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv"

# Explicit user-approved renames (includes AMBIGUOUS overrides).
EXPLICIT_RENAMES: dict[str, str] = {
    "CASPASE-9": "CASP9",
    "CASPASE-8": "CASP8",
    "CASPASE-4": "CASP4",
    "CASPASE-5": "CASP5",
    "ARGINASE-1": "ARG1",
    "CEREBLON": "CRBN",
    "CHYMASE": "CMA1",
    "EXPORTIN-1": "XPO1",
    "MENIN": "MEN1",
    "WNT-3A": "WNT3A",
    "HUNTINGTIN": "HTT",
    "RENIN": "REN",
    "NEPRILYSIN": "MME",
    "GALECTIN-3": "LGALS3",
    "SORTILIN": "SORT1",
    "DYNAMIN-1": "DNM1",
    "MDR1": "ABCB1",
}

# Remaining RESOLVED-UNIQUE renames from proposed_gene_normalization.csv (unambiguous).
ADDITIONAL_RENAMES: dict[str, str] = {
    "AXIN-2": "AXIN2",
    "CASPASE-2": "CASP2",
    "CASPASE-7": "CASP7",
    "CYCLIN-C": "CCNC",
    "CYCLIN-K": "CCNK",
    "CYCLIN-O": "CCNO",
    "ELONGIN-C": "ELOC",
    "GALECTIN-1": "LGALS1",
    "GALECTIN-8": "LGALS8",
    "GASTRICSIN": "PGC",
    "LEGUMAIN": "LGMN",
    "MYOGLOBIN": "MB",
    "MYOSIN-2": "MYH2",
    "MYOSIN-9": "MYH9",
    "PEREGRIN": "BRPF1",
    "PROMOTILIN": "MLN",
    "S100-B": "S100B",
    "UTROPHIN": "UTRN",
}

# Identity-confirmed HGNC symbols (no symbol change; provenance column only).
IDENTITY_CONFIRMED: frozenset[str] = frozenset(
    {
        "ABCC2",
        "APOA5",
        "APOE",
        "CETP",
        "CYP2E1",
        "FURIN",
        "FUT2",
        "GSTM1",
        "GSTT1",
        "LIPC",
        "MDM4",
        "MTHFR",
        "NAT1",
        "NAT2",
        "NEDD8",
        "PIM2",
        "SHBG",
        "SLC22A1",
        "SULT1A3",
        "SULT1E1",
        "SULT2A1",
        "TAS2R16",
        "TCF7L2",
        "UGT1A1",
        "UGT1A3",
        "UGT2B15",
    }
)

# Must remain unchanged.
EXCLUDED: frozenset[str] = frozenset(
    {
        "E-SELECTIN",
        "INTEGRASE",
        "TRYPSIN",
        "LANA1",
        "D3-HUMAN",
        "MDRC4",
        "CYP1A",
        "CYCLIN-Y",
        "E6",
        "HEPARANASE",
        "NINEIN",
        "PLECTIN",
    }
)

RENAMES = {**ADDITIONAL_RENAMES, **EXPLICIT_RENAMES}


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source not found: {SRC}", file=sys.stderr)
        return 1

    overlap = set(RENAMES) & EXCLUDED
    if overlap:
        print(f"ERROR: rename/exclude overlap: {sorted(overlap)}", file=sys.stderr)
        return 1

    df = pd.read_csv(SRC)
    if "gene_symbol" not in df.columns:
        print("ERROR: gene_symbol column missing", file=sys.stderr)
        return 1

    genes_in_file = set(df["gene_symbol"].astype(str).str.strip().unique())

    # Validate every rename key exists.
    missing_keys = sorted(k for k in RENAMES if k not in genes_in_file)
    if missing_keys:
        print(f"ERROR: approved rename keys not in source file: {missing_keys}", file=sys.stderr)
        return 1

    # Validate identity-confirmed symbols present when they appear in unmapped set context.
    missing_identity = sorted(g for g in IDENTITY_CONFIRMED if g in genes_in_file)
    # all identity symbols that exist are fine; warn if none exist (unlikely)
    if not missing_identity:
        print("WARNING: none of the 26 identity-confirmed symbols found in file", file=sys.stderr)

    # Ensure excluded names are not in rename map.
    for name in EXCLUDED:
        if name in RENAMES:
            print(f"ERROR: excluded name {name} appears in rename map", file=sys.stderr)
            return 1

    df = df.copy()
    df["raw_gene_name"] = df["gene_symbol"].astype(str).str.strip()

    changed_mask = df["gene_symbol"].astype(str).str.strip().isin(RENAMES)
    n_rows = len(df)
    n_changed_rows = int(changed_mask.sum())

    # Apply renames row-wise for approved keys only.
    applied: list[dict] = []
    for raw, new in sorted(RENAMES.items()):
        mask = df["gene_symbol"].astype(str).str.strip() == raw
        n = int(mask.sum())
        if n == 0:
            print(f"ERROR: expected rows for {raw} but found 0", file=sys.stderr)
            return 1
        if df.loc[mask, "gene_symbol"].astype(str).str.strip().nunique() != 1:
            print(f"ERROR: inconsistent gene_symbol values for {raw}", file=sys.stderr)
            return 1
        df.loc[mask, "gene_symbol"] = new
        applied.append({"raw_name": raw, "new_symbol": new, "n_rows": n, "source": "explicit" if raw in EXPLICIT_RENAMES else "additional_resolved_unique"})

    # Post-apply: excluded names must still appear unchanged as gene_symbol.
    for name in EXCLUDED:
        if name in set(df["gene_symbol"].astype(str).str.strip()):
            still = df[df["gene_symbol"].astype(str).str.strip() == name]
            if not (still["raw_gene_name"].astype(str).str.strip() == name).all():
                print(f"ERROR: excluded name {name} has modified raw_gene_name", file=sys.stderr)
                return 1
        # also check none were renamed away incorrectly
        wrongly_renamed = df[(df["raw_gene_name"].astype(str).str.strip() == name) & (df["gene_symbol"].astype(str).str.strip() != name)]
        if len(wrongly_renamed):
            print(f"ERROR: excluded name {name} was renamed", file=sys.stderr)
            return 1

    # Renamed rows must preserve raw name.
    renamed_rows = df[df["raw_gene_name"] != df["gene_symbol"]]
    bad = renamed_rows[~renamed_rows["raw_gene_name"].isin(RENAMES.keys())]
    if len(bad):
        print(f"ERROR: unexpected renames outside approved map: {bad['raw_gene_name'].unique()[:10]}", file=sys.stderr)
        return 1

    # No unapproved renames.
    for raw in renamed_rows["raw_gene_name"].unique():
        expected = RENAMES.get(raw)
        actual = renamed_rows.loc[renamed_rows["raw_gene_name"] == raw, "gene_symbol"].iloc[0]
        if expected != actual:
            print(f"ERROR: {raw} mapped to {actual}, expected {expected}", file=sys.stderr)
            return 1

    # Write output (gene_symbol column order: keep raw_gene_name after gene_symbol).
    cols = list(df.columns)
    cols.remove("raw_gene_name")
    gene_idx = cols.index("gene_symbol")
    cols.insert(gene_idx + 1, "raw_gene_name")
    df = df[cols]
    df.to_csv(OUT, index=False)

    sample = renamed_rows.drop_duplicates(subset=["raw_gene_name"]).head(10)[
        ["compound_id", "raw_gene_name", "gene_symbol", "source"]
    ]

    report = {
        "source_file": str(SRC),
        "output_file": str(OUT),
        "total_rows": n_rows,
        "rows_renamed": n_changed_rows,
        "rows_unchanged": n_rows - n_changed_rows,
        "unique_genes_renamed": len(RENAMES),
        "identity_confirmed_carried": sorted(g for g in IDENTITY_CONFIRMED if g in genes_in_file),
        "explicit_renames_applied": {k: v for k, v in sorted(EXPLICIT_RENAMES.items())},
        "additional_renames_applied": {k: v for k, v in sorted(ADDITIONAL_RENAMES.items())},
        "excluded_still_present": {n: int((df["gene_symbol"] == n).sum()) for n in sorted(EXCLUDED) if n in genes_in_file},
        "sample_renamed_rows": sample.to_dict(orient="records"),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
