#!/usr/bin/env python3
"""
k-NN target-inference accuracy stratified by nearest-neighbor Tanimoto similarity.

Calibrates confidence-tier hit rates on the Part B held-out evaluation setup.
RECON ONLY — metrics only, no production model, no graph writes.

Usage (from repo root):
    python scripts/thread2/knn_similarity_validation.py
    python scripts/thread2/knn_similarity_validation.py --max-compounds 30000

Output: data/processed/thread2/recon/knn_similarity_validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Reuse Part B / Part A helpers (same Tanimoto, split, k-NN logic).
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import feasibility_recon as fr  # noqa: E402

ROOT = fr.ROOT
OUT_DIR = fr.OUT_DIR
OUT_PATH = OUT_DIR / "knn_similarity_validation.json"

DEFAULT_MAX_COMPOUNDS = fr.DEFAULT_PART_B_MAX
DEFAULT_SEED = 42
DEFAULT_TEST_BATCH = fr.PART_B_TEST_BATCH
K_NEIGHBORS = fr.K_NEIGHBORS
METRIC_KS = fr.PART_B_KS
SCAFFOLD_TEST_SAMPLE = 5_000

SIMILARITY_BANDS: list[tuple[float, float, str]] = [
    (0.3, 0.4, "[0.3-0.4)"),
    (0.4, 0.5, "[0.4-0.5)"),
    (0.5, 0.6, "[0.5-0.6)"),
    (0.6, 0.7, "[0.6-0.7)"),
    (0.7, 0.8, "[0.7-0.8)"),
    (0.8, 0.9, "[0.8-0.9)"),
    (0.9, 1.0, "[0.9-1.0]"),
]


def require_rdkit_scaffold() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: F401

        fr.configure_rdkit_logging()
        return True
    except ImportError:
        print(
            "ERROR: RDKit required for Murcko scaffold split.\n"
            "  pip install rdkit",
            file=sys.stderr,
        )
        return False


def similarity_band(sim: float) -> str:
    if sim < 0.3:
        return "below_0.3"
    for lo, hi, label in SIMILARITY_BANDS:
        if label.endswith("]"):
            if lo <= sim <= hi:
                return label
        elif lo <= sim < hi:
            return label
    return "below_0.3"


def aggregate_metrics(records: list[dict[str, Any]], metric_keys: tuple[str, ...]) -> dict[str, float]:
    if not records:
        return {k: 0.0 for k in metric_keys}
    return {k: float(np.mean([r["metrics"][k] for r in records])) for k in metric_keys}


def summarize_by_band(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[similarity_band(rec["max_nn_tanimoto"])].append(rec)

    metric_keys = tuple(
        f"{name}@{k}" for k in METRIC_KS for name in ("hit", "precision", "recall")
    )
    per_band: dict[str, Any] = {}
    for _, _, label in SIMILARITY_BANDS:
        rows = grouped.get(label, [])
        per_band[label] = {
            "n": len(rows),
            **{k: aggregate_metrics(rows, (k,))[k] for k in metric_keys},
        }
    per_band["below_0.3"] = {
        "n": len(grouped.get("below_0.3", [])),
        **{
            k: aggregate_metrics(grouped.get("below_0.3", []), (k,))[k]
            for k in metric_keys
        },
    }
    return per_band


def tier_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    tiers = {
        "predicted_high_ge_0.7": [r for r in records if r["max_nn_tanimoto"] >= 0.7],
        "predicted_moderate_0.5_to_0.7": [
            r for r in records if 0.5 <= r["max_nn_tanimoto"] < 0.7
        ],
        "predicted_low_0.3_to_0.5": [
            r for r in records if 0.3 <= r["max_nn_tanimoto"] < 0.5
        ],
        "withhold_lt_0.3": [r for r in records if r["max_nn_tanimoto"] < 0.3],
    }
    out: dict[str, Any] = {}
    for name, rows in tiers.items():
        out[name] = {
            "n": len(rows),
            "hit@10": aggregate_metrics(rows, ("hit@10",))["hit@10"],
            "hit@5": aggregate_metrics(rows, ("hit@5",))["hit@5"],
            "hit@1": aggregate_metrics(rows, ("hit@1",))["hit@1"],
        }
    return out


def recommend_confidence_tiers(tier_stats: dict[str, Any]) -> dict[str, Any]:
    high = tier_stats["predicted_high_ge_0.7"]["hit@10"]
    mod = tier_stats["predicted_moderate_0.5_to_0.7"]["hit@10"]
    low = tier_stats["predicted_low_0.3_to_0.5"]["hit@10"]
    withhold = tier_stats["withhold_lt_0.3"]["hit@10"]

    def _reliability(hit: float) -> str:
        if hit >= 0.75:
            return "high"
        if hit >= 0.45:
            return "moderate"
        if hit >= 0.20:
            return "low"
        return "unreliable"

    return {
        "evidence_based_thresholds": {
            "predicted_high": {
                "max_nn_tanimoto_gte": 0.7,
                "measured_hit_at_10": high,
                "reliability": _reliability(high),
                "guidance": (
                    f"Nearest-neighbor similarity >= 0.7: hit@10 = {high:.1%}. "
                    "Suitable for high-confidence predicted edges when combined with score ranking."
                ),
            },
            "predicted_moderate": {
                "max_nn_tanimoto_range": [0.5, 0.7],
                "measured_hit_at_10": mod,
                "reliability": _reliability(mod),
                "guidance": (
                    f"NN similarity 0.5-0.7: hit@10 = {mod:.1%}. "
                    "Use moderate confidence; validate high-impact predictions."
                ),
            },
            "predicted_low": {
                "max_nn_tanimoto_range": [0.3, 0.5],
                "measured_hit_at_10": low,
                "reliability": _reliability(low),
                "guidance": (
                    f"NN similarity 0.3-0.5: hit@10 = {low:.1%}. "
                    "Low confidence — prefer withholding or manual review."
                ),
            },
            "withhold": {
                "max_nn_tanimoto_lt": 0.3,
                "measured_hit_at_10": withhold,
                "reliability": _reliability(withhold),
                "guidance": (
                    f"NN similarity < 0.3: hit@10 = {withhold:.1%}. "
                    "Do not emit predicted target edges."
                ),
            },
        },
        "note": (
            "Tier boundaries use measured k-NN hit@10 on the Part B random compound-level split. "
            "Distinctive/dark compounds concentrate in the 0.4-0.6 NN-similarity operating zone."
        ),
    }


def evaluate_knn_with_nn_sim(
    test_fps: np.ndarray,
    train_fps: np.ndarray,
    train_genes: list[set[str]],
    test_genes: list[set[str]],
    test_batch: int,
    stage: str,
) -> list[dict[str, Any]]:
    n_test = len(test_fps)
    print(
        f"[{stage}] batched float32 Tanimoto k-NN: {n_test:,} test x {len(train_fps):,} train "
        f"(batch={test_batch}, k={K_NEIGHBORS})",
        flush=True,
    )
    timer = fr.StageTimer(stage)
    records: list[dict[str, Any]] = []

    for start in range(0, n_test, test_batch):
        end = min(start + test_batch, n_test)
        sims = fr.tanimoto_similarity_matrix(test_fps[start:end], train_fps)
        max_nn = sims.max(axis=1)
        for i, row_idx in enumerate(range(start, end)):
            ranked = fr.knn_predict_from_similarities(sims[i], train_genes, K_NEIGHBORS)
            metrics = fr.ranking_metrics(test_genes[row_idx], ranked, METRIC_KS)
            records.append({"max_nn_tanimoto": float(max_nn[i]), "metrics": metrics})
        elapsed = time.perf_counter() - timer.t0
        rate = end / elapsed if elapsed > 0 else 0.0
        remaining = (n_test - end) / rate if rate > 0 else 0.0
        print(
            f"[{stage}] {end:,}/{n_test:,} scored, "
            f"elapsed {elapsed:.0f}s, est remaining {remaining:.0f}s",
            flush=True,
        )

    timer.done(f"{n_test:,} compounds")
    return records


def timing_probe(
    test_fps: np.ndarray,
    train_fps: np.ndarray,
    train_genes: list[set[str]],
    test_genes: list[set[str]],
    test_batch: int,
) -> dict[str, float]:
    n_probe = min(test_batch, len(test_fps))
    print(f"[probe] timing first {n_probe} test compounds on actual code path...", flush=True)
    t0 = time.perf_counter()
    sims = fr.tanimoto_similarity_matrix(test_fps[:n_probe], train_fps)
    max_nn = sims.max(axis=1)
    for i in range(n_probe):
        ranked = fr.knn_predict_from_similarities(sims[i], train_genes, K_NEIGHBORS)
        fr.ranking_metrics(test_genes[i], ranked, METRIC_KS)
    probe_s = time.perf_counter() - t0
    est_random_s = probe_s * (len(test_fps) / n_probe) if n_probe else 0.0
    print(
        f"[probe] batch={probe_s:.2f}s -> est random-split k-NN {est_random_s / 60:.1f} min "
        f"({len(test_fps):,} test)",
        flush=True,
    )
    return {
        "probe_n": n_probe,
        "probe_sec": probe_s,
        "estimated_random_split_knn_sec": est_random_s,
    }


def compute_murcko_scaffolds(compounds: pd.DataFrame) -> list[str | None]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    print(f"[scaffold] computing Murcko scaffolds for {len(compounds):,} compounds...", flush=True)
    timer = fr.StageTimer("scaffold")
    scaffolds: list[str | None] = []
    for i, row in compounds.iterrows():
        if (len(scaffolds) + 1) % 5_000 == 0:
            print(f"  [scaffold] {len(scaffolds) + 1:,}/{len(compounds):,}", flush=True)
        mol = fr.mol_from_smiles_or_inchi(row.get("smiles"), row.get("inchi"))
        if mol is None:
            scaffolds.append(None)
            continue
        try:
            scaffolds.append(
                MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            )
        except Exception:
            scaffolds.append(None)
    n_ok = sum(1 for s in scaffolds if s)
    timer.done(f"{n_ok:,} scaffolds ok, {len(scaffolds) - n_ok:,} failed")
    return scaffolds


def murcko_train_test_indices(
    n: int,
    scaffolds: list[str | None],
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Bemis-Murcko style: whole scaffold groups go to train or test (no leakage)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, scaf in enumerate(scaffolds):
        key = scaf if scaf else f"__missing_{i}"
        groups[key].append(i)

    rng = np.random.RandomState(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    train_idx: list[int] = []
    test_idx: list[int] = []
    n_train_target = int(n * (1.0 - test_fraction))
    for _, idxs in group_items:
        if len(train_idx) + len(idxs) <= n_train_target:
            train_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    train_scafs = {scaffolds[i] for i in train_idx if scaffolds[i]}
    test_scafs = {scaffolds[i] for i in test_idx if scaffolds[i]}
    overlap = train_scafs & test_scafs
    if overlap:
        raise RuntimeError(f"Scaffold leakage: {len(overlap)} scaffolds in both train and test")
    return train_idx, test_idx


def load_part_b_dataset(max_compounds: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, dict[str, set[str]]]:
    compounds, gene_sets = fr.load_corpus_labels()
    if max_compounds > 0 and len(compounds) > max_compounds:
        compounds = compounds.sample(max_compounds, random_state=seed)
        print(f"[data] subsampled to {len(compounds):,} compounds (Part B parity)", flush=True)

    fps = fr.load_part_b_fingerprints(compounds)
    return compounds.reset_index(drop=True), fps, gene_sets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="k-NN accuracy stratified by nearest-neighbor Tanimoto (RECON)"
    )
    parser.add_argument("--max-compounds", type=int, default=DEFAULT_MAX_COMPOUNDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-batch", type=int, default=DEFAULT_TEST_BATCH)
    parser.add_argument(
        "--scaffold-test-sample",
        type=int,
        default=SCAFFOLD_TEST_SAMPLE,
        help="Max scaffold-split test compounds for k-NN (0 = use all)",
    )
    args = parser.parse_args()

    if not require_rdkit_scaffold():
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_t0 = time.perf_counter()
    timing: dict[str, float] = {}

    print("=== k-NN similarity-band validation (RECON) ===", flush=True)
    print(f"Output: {OUT_PATH}", flush=True)

    t0 = time.perf_counter()
    compounds, fps_all, gene_sets = load_part_b_dataset(args.max_compounds, args.seed)
    timing["load_data_sec"] = time.perf_counter() - t0

    inchikeys = compounds["inchikey"].astype(str).tolist()
    print("[split] random compound-level 80/20 (same as Part B)...", flush=True)
    t0 = time.perf_counter()
    train_ik, test_ik = train_test_split(inchikeys, test_size=0.2, random_state=args.seed)
    ik_to_row = {k: i for i, k in enumerate(inchikeys)}
    train_idx = [ik_to_row[k] for k in train_ik]
    test_idx = [ik_to_row[k] for k in test_ik]
    train_fps = fps_all[train_idx]
    test_fps = fps_all[test_idx]
    train_genes = [gene_sets[k] for k in train_ik]
    test_genes = [gene_sets[k] for k in test_ik]
    timing["random_split_sec"] = time.perf_counter() - t0
    print(
        f"[split] train={len(train_ik):,}, test={len(test_ik):,} "
        f"({timing['random_split_sec']:.1f}s)",
        flush=True,
    )

    probe = timing_probe(test_fps, train_fps, train_genes, test_genes, args.test_batch)
    timing.update(probe)

    t0 = time.perf_counter()
    random_records = evaluate_knn_with_nn_sim(
        test_fps, train_fps, train_genes, test_genes, args.test_batch, "random"
    )
    timing["random_knn_sec"] = time.perf_counter() - t0

    per_band = summarize_by_band(random_records)
    tier_stats = tier_summary(random_records)
    random_hit10 = aggregate_metrics(random_records, ("hit@10",))["hit@10"]

    op_04_05 = per_band["[0.4-0.5)"]
    op_05_06 = per_band["[0.5-0.6)"]
    print("\n=== Operating zone (distinctive-relevant NN similarity) ===", flush=True)
    print(
        f"  [0.4-0.5): n={op_04_05['n']:,}, hit@10={op_04_05['hit@10']:.1%}",
        flush=True,
    )
    print(
        f"  [0.5-0.6): n={op_05_06['n']:,}, hit@10={op_05_06['hit@10']:.1%}",
        flush=True,
    )
    print(
        f"  combined [0.4-0.6): hit@10="
        f"{aggregate_metrics([r for r in random_records if 0.4 <= r['max_nn_tanimoto'] < 0.6], ('hit@10',))['hit@10']:.1%}",
        flush=True,
    )

    print("\n=== Per-band calibration table (random split) ===", flush=True)
    for _, _, label in SIMILARITY_BANDS:
        row = per_band[label]
        print(
            f"  {label}: n={row['n']:,}, hit@1={row['hit@1']:.3f}, "
            f"hit@5={row['hit@5']:.3f}, hit@10={row['hit@10']:.3f}, "
            f"p@10={row['precision@10']:.3f}, r@10={row['recall@10']:.3f}",
            flush=True,
        )

    # --- Scaffold split (leakage honesty check) ---
    t0 = time.perf_counter()
    scaffolds = compute_murcko_scaffolds(compounds)
    sc_train_idx, sc_test_idx = murcko_train_test_indices(
        len(compounds), scaffolds, test_fraction=0.2, seed=args.seed
    )
    timing["scaffold_prep_sec"] = time.perf_counter() - t0

    sc_train_fps = fps_all[sc_train_idx]
    sc_test_fps_full = fps_all[sc_test_idx]
    sc_train_genes = [gene_sets[inchikeys[i]] for i in sc_train_idx]
    sc_test_genes_full = [gene_sets[inchikeys[i]] for i in sc_test_idx]

    if args.scaffold_test_sample > 0 and len(sc_test_idx) > args.scaffold_test_sample:
        rng = np.random.RandomState(args.seed)
        pick = rng.choice(len(sc_test_idx), size=args.scaffold_test_sample, replace=False)
        sc_eval_idx = [sc_test_idx[i] for i in sorted(pick)]
        sc_test_fps = fps_all[sc_eval_idx]
        sc_test_genes = [gene_sets[inchikeys[i]] for i in sc_eval_idx]
        scaffold_sample_note = (
            f"Evaluated k-NN on {len(sc_eval_idx):,} / {len(sc_test_idx):,} scaffold-test "
            f"compounds (sample cap={args.scaffold_test_sample:,})"
        )
    else:
        sc_test_fps = sc_test_fps_full
        sc_test_genes = sc_test_genes_full
        scaffold_sample_note = f"Evaluated all {len(sc_test_idx):,} scaffold-test compounds"

    print(f"\n[scaffold] {scaffold_sample_note}", flush=True)
    t0 = time.perf_counter()
    scaffold_records = evaluate_knn_with_nn_sim(
        sc_test_fps,
        sc_train_fps,
        sc_train_genes,
        sc_test_genes,
        args.test_batch,
        "scaffold",
    )
    timing["scaffold_knn_sec"] = time.perf_counter() - t0
    scaffold_hit10 = aggregate_metrics(scaffold_records, ("hit@10",))["hit@10"]
    scaffold_per_band = summarize_by_band(scaffold_records)
    scaffold_tier_stats = tier_summary(scaffold_records)

    print("\n=== Per-band calibration table (scaffold split) ===", flush=True)
    for _, _, label in SIMILARITY_BANDS:
        row = scaffold_per_band[label]
        print(
            f"  {label}: n={row['n']:,}, hit@1={row['hit@1']:.3f}, "
            f"hit@5={row['hit@5']:.3f}, hit@10={row['hit@10']:.3f}, "
            f"p@10={row['precision@10']:.3f}, r@10={row['recall@10']:.3f}",
            flush=True,
        )

    print("\n=== Leakage honesty check ===", flush=True)
    print(f"  Random-split hit@10:   {random_hit10:.1%}", flush=True)
    print(f"  Scaffold-split hit@10: {scaffold_hit10:.1%}", flush=True)
    if scaffold_hit10 > 0:
        rel = (random_hit10 / scaffold_hit10 - 1) * 100
        print(
            f"  Inflation (random - scaffold): {random_hit10 - scaffold_hit10:.1%} "
            f"({rel:.0f}% relative)",
            flush=True,
        )

    confidence = recommend_confidence_tiers(tier_stats)

    report: dict[str, Any] = {
        "config": {
            "max_compounds": args.max_compounds,
            "seed": args.seed,
            "test_batch": args.test_batch,
            "k_neighbors": K_NEIGHBORS,
            "split": "compound-level 80/20 (Part B parity)",
            "fingerprints": "reused Part A corpus checkpoint (no recomputation)",
            "tanimoto": "batched float32 matmul (feasibility_recon.tanimoto_similarity_matrix)",
            "scaffold_test_sample": args.scaffold_test_sample,
        },
        "timing_sec": {k: round(v, 2) for k, v in timing.items()},
        "total_wall_sec": round(time.perf_counter() - total_t0, 2),
        "random_split": {
            "train_n": len(train_ik),
            "test_n": len(test_ik),
            "overall_hit_at_10": random_hit10,
            "per_band": per_band,
            "operating_zone": {
                "band_0.4_0.5": {
                    "n": op_04_05["n"],
                    "hit_at_10": op_04_05["hit@10"],
                    "hit_at_5": op_04_05["hit@5"],
                    "hit_at_1": op_04_05["hit@1"],
                },
                "band_0.5_0.6": {
                    "n": op_05_06["n"],
                    "hit_at_10": op_05_06["hit@10"],
                    "hit_at_5": op_05_06["hit@5"],
                    "hit_at_1": op_05_06["hit@1"],
                },
                "combined_0.4_0.6": {
                    "n": sum(
                        1 for r in random_records if 0.4 <= r["max_nn_tanimoto"] < 0.6
                    ),
                    "hit_at_10": aggregate_metrics(
                        [r for r in random_records if 0.4 <= r["max_nn_tanimoto"] < 0.6],
                        ("hit@10",),
                    )["hit@10"],
                },
            },
            "tier_summary": tier_stats,
        },
        "scaffold_split": {
            "train_n": len(sc_train_idx),
            "test_n_full": len(sc_test_idx),
            "test_n_evaluated": len(sc_test_fps),
            "sample_note": scaffold_sample_note,
            "overall_hit_at_10": scaffold_hit10,
            "random_split_hit_at_10": random_hit10,
            "hit_at_10_inflation_random_minus_scaffold": random_hit10 - scaffold_hit10,
            "per_band": scaffold_per_band,
            "tier_summary": scaffold_tier_stats,
        },
        "confidence_tier_recommendation": confidence,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote {OUT_PATH}", flush=True)
    print(f"Total wall time: {report['total_wall_sec']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
