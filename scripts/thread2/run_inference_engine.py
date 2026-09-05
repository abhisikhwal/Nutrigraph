#!/usr/bin/env python3
"""
Path 2 k-NN inference engine: predict gene targets for dark ingredient compounds.

Produces tiered predicted compound→gene edges (separate from measured corpus).
Does NOT integrate into graph, enrichment, or canonical files.

Usage (from repo root):
    python scripts/thread2/run_inference_engine.py
    python scripts/thread2/run_inference_engine.py --no-resume

Outputs: data/processed/thread2/inference/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import feasibility_recon as fr  # noqa: E402

ROOT = fr.ROOT
RECON_DIR = fr.OUT_DIR
INFERENCE_DIR = ROOT / "data/processed/thread2/inference"

CORPUS_META = RECON_DIR / "fingerprints_corpus_meta.parquet"
CORPUS_FPS = RECON_DIR / "fingerprints_corpus_fps.npz"
DARK_META = RECON_DIR / "fingerprints_dark_meta.parquet"
DARK_FPS = RECON_DIR / "fingerprints_dark_fps.npz"
MEASURED_GENE_PATH = fr.COMPOUND_GENE

EDGES_OUT = INFERENCE_DIR / "predicted_compound_gene_v1.parquet"
WITHHELD_OUT = INFERENCE_DIR / "withheld_compounds_v1.parquet"
REPORT_OUT = INFERENCE_DIR / "inference_engine_report.json"

CKPT_PATH = INFERENCE_DIR / "inference_checkpoint.npz"
EDGES_BATCH_GLOB = "_predicted_edges_batch_*.parquet"
WITHHELD_BATCH_GLOB = "_withheld_batch_*.parquet"

K_NEIGHBORS = fr.K_NEIGHBORS
DARK_BATCH = fr.NN_DARK_BATCH
CORPUS_CHUNK = fr.CORPUS_BATCH
CHECKPOINT_EVERY = fr.NN_CHECKPOINT_EVERY

TIER_THRESHOLDS = {
    "predicted_high": 0.7,
    "predicted_moderate": 0.5,
    "predicted_low": 0.3,
}


def confidence_tier(max_nn_tanimoto: float) -> str:
    if max_nn_tanimoto >= TIER_THRESHOLDS["predicted_high"]:
        return "predicted_high"
    if max_nn_tanimoto >= TIER_THRESHOLDS["predicted_moderate"]:
        return "predicted_moderate"
    if max_nn_tanimoto >= TIER_THRESHOLDS["predicted_low"]:
        return "predicted_low"
    return "withheld"


def load_corpus_gene_labels() -> dict[str, set[str]]:
    print("[load] corpus gene labels from compound_target_corpus_v1...", flush=True)
    timer = fr.StageTimer("load:corpus-labels")
    df = pd.read_parquet(fr.CORPUS_PATH, columns=["compound_inchikey", "gene_symbol"])
    gene_sets = (
        df.groupby("compound_inchikey")["gene_symbol"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )
    timer.done(f"{len(gene_sets):,} labeled corpus compounds")
    return {str(k): v for k, v in gene_sets.items()}


def load_fingerprint_checkpoints() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    for path in (CORPUS_META, CORPUS_FPS, DARK_META, DARK_FPS):
        if not path.exists():
            raise FileNotFoundError(f"Missing Part A checkpoint: {path}")

    print("[load] Part A corpus fingerprint checkpoint...", flush=True)
    t0 = time.perf_counter()
    corpus_meta = pd.read_parquet(CORPUS_META)
    corpus_fps = np.load(CORPUS_FPS)["fps"]
    print(
        f"[load] corpus: {len(corpus_meta):,} compounds ({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    print("[load] Part A dark fingerprint checkpoint...", flush=True)
    t0 = time.perf_counter()
    dark_meta = pd.read_parquet(DARK_META)
    dark_fps = np.load(DARK_FPS)["fps"]
    print(
        f"[load] dark: {len(dark_meta):,} compounds "
        f"(distinctive: {int(dark_meta['is_distinctive'].sum()):,}, "
        f"{time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    return corpus_meta, corpus_fps, dark_meta, dark_fps


def batched_topk_neighbors(
    query_fps: np.ndarray,
    corpus_fps: np.ndarray,
    k: int,
    corpus_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k corpus indices and Tanimoto per query row (chunked float32 matmul)."""
    n_q = len(query_fps)
    n_c = len(corpus_fps)
    qf = np.asarray(query_fps, dtype=np.uint8).astype(np.float32)
    q_counts = np.asarray(query_fps, dtype=np.uint8).sum(axis=1, dtype=np.int32, keepdims=True)
    top_sims = np.full((n_q, k), -1.0, dtype=np.float32)
    top_idx = np.full((n_q, k), -1, dtype=np.int64)

    for lo in range(0, n_c, corpus_chunk):
        hi = min(lo + corpus_chunk, n_c)
        c_u8 = np.asarray(corpus_fps[lo:hi], dtype=np.uint8)
        cf = c_u8.astype(np.float32)
        intersection = (qf @ cf.T).astype(np.int32)
        c_counts = c_u8.sum(axis=1, dtype=np.int32)
        union = q_counts + c_counts - intersection
        block = np.divide(
            intersection,
            union,
            where=union > 0,
            out=np.zeros(intersection.shape, dtype=np.float32),
        )

        if block.shape[1] <= k:
            local_sims = block
            local_idx = np.broadcast_to(np.arange(lo, hi, dtype=np.int64), (n_q, hi - lo))
        else:
            pick = np.argpartition(block, -k, axis=1)[:, -k:]
            local_sims = np.take_along_axis(block, pick, axis=1)
            local_idx = pick.astype(np.int64) + lo

        merged_sims = np.concatenate([top_sims, local_sims], axis=1)
        merged_idx = np.concatenate([top_idx, local_idx], axis=1)
        pick = np.argpartition(merged_sims, -k, axis=1)[:, -k:]
        row_ix = np.arange(n_q)[:, None]
        top_sims = np.take_along_axis(merged_sims, pick, axis=1)
        top_idx = np.take_along_axis(merged_idx, pick, axis=1)
        order = np.argsort(top_sims, axis=1)[:, ::-1]
        top_sims = np.take_along_axis(top_sims, order, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)

    return top_sims, top_idx


def infer_genes_with_provenance(
    neighbor_indices: np.ndarray,
    neighbor_sims: np.ndarray,
    corpus_genes: list[set[str]],
    corpus_inchikeys: list[str],
) -> tuple[dict[str, float], dict[str, tuple[str, float]]]:
    """Tanimoto-weighted votes + best-neighbor provenance per gene."""
    gene_votes: dict[str, float] = {}
    gene_neighbor: dict[str, tuple[str, float]] = {}
    for idx, sim in zip(neighbor_indices, neighbor_sims):
        if idx < 0 or sim < 0:
            continue
        neighbor_ik = corpus_inchikeys[int(idx)]
        weight = float(sim)
        for gene in corpus_genes[int(idx)]:
            gene_votes[gene] = gene_votes.get(gene, 0.0) + weight
            prev = gene_neighbor.get(gene)
            if prev is None or weight > prev[1]:
                gene_neighbor[gene] = (neighbor_ik, weight)
    return gene_votes, gene_neighbor


def process_dark_batch(
    batch_meta: pd.DataFrame,
    batch_fps: np.ndarray,
    corpus_fps: np.ndarray,
    corpus_inchikeys: list[str],
    corpus_genes: list[set[str]],
    corpus_chunk: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    top_sims, top_idx = batched_topk_neighbors(batch_fps, corpus_fps, K_NEIGHBORS, corpus_chunk)
    edge_rows: list[dict[str, Any]] = []
    withheld_rows: list[dict[str, Any]] = []

    dark_iks = batch_meta["inchikey"].astype(str).to_numpy()
    is_distinctive = batch_meta["is_distinctive"].to_numpy(dtype=bool)

    for i in range(len(dark_iks)):
        max_nn = float(top_sims[i, 0]) if top_sims.shape[1] else -1.0
        tier = confidence_tier(max_nn)

        if tier == "withheld":
            withheld_rows.append(
                {
                    "dark_compound_inchikey": dark_iks[i],
                    "max_nn_tanimoto": max_nn,
                    "is_distinctive": bool(is_distinctive[i]),
                    "reason": "max_nn_tanimoto below 0.3 calibrated withhold threshold",
                }
            )
            continue

        votes, provenance = infer_genes_with_provenance(
            top_idx[i], top_sims[i], corpus_genes, corpus_inchikeys
        )
        for gene, score in votes.items():
            nn_ik, nn_sim = provenance[gene]
            edge_rows.append(
                {
                    "dark_compound_inchikey": dark_iks[i],
                    "predicted_gene_symbol": gene,
                    "prediction_score": float(score),
                    "max_nn_tanimoto": max_nn,
                    "confidence_tier": tier,
                    "nearest_neighbor_inchikey": nn_ik,
                    "nearest_neighbor_tanimoto": float(nn_sim),
                    "is_distinctive": bool(is_distinctive[i]),
                }
            )

    return edge_rows, withheld_rows


def write_batch_parquet(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        return
    pd.DataFrame(rows)[columns].to_parquet(path, index=False)


def concat_batch_parquets(pattern: str, columns: list[str]) -> pd.DataFrame:
    paths = sorted(INFERENCE_DIR.glob(pattern))
    if not paths:
        return pd.DataFrame(columns=columns)
    parts = [pd.read_parquet(p) for p in paths]
    return pd.concat(parts, ignore_index=True)


def tier_counts_for_compounds(
    compound_tiers: dict[str, str],
    distinctive_iks: set[str],
) -> dict[str, Any]:
    all_counter = Counter(compound_tiers.values())
    dist_counter = Counter(
        tier for ik, tier in compound_tiers.items() if ik in distinctive_iks
    )
    return {
        "all_dark_compounds": dict(all_counter),
        "distinctive_compounds": dict(dist_counter),
        "distinctive_n": len(distinctive_iks),
        "distinctive_with_predicted_edge": sum(
            1
            for ik in distinctive_iks
            if compound_tiers.get(ik) in ("predicted_high", "predicted_moderate", "predicted_low")
        ),
        "distinctive_fully_withheld": sum(
            1 for ik in distinctive_iks if compound_tiers.get(ik) == "withheld"
        ),
    }


def build_report(
    edges_df: pd.DataFrame,
    withheld_df: pd.DataFrame,
    dark_meta: pd.DataFrame,
    measured_genes: set[str],
    timing: dict[str, float],
    probe: dict[str, float],
) -> dict[str, Any]:
    distinctive_iks = set(
        dark_meta.loc[dark_meta["is_distinctive"], "inchikey"].astype(str)
    )
    dark_iks = dark_meta["inchikey"].astype(str).tolist()

    compound_tiers: dict[str, str] = {}
    for ik in dark_iks:
        compound_tiers[ik] = "withheld"
    if len(withheld_df):
        for _, r in withheld_df.iterrows():
            compound_tiers[str(r["dark_compound_inchikey"])] = "withheld"
    if len(edges_df):
        for ik, max_nn in edges_df.groupby("dark_compound_inchikey")["max_nn_tanimoto"].first().items():
            compound_tiers[str(ik)] = confidence_tier(float(max_nn))

    n_with_edges = sum(
        1
        for t in compound_tiers.values()
        if t in ("predicted_high", "predicted_moderate", "predicted_low")
    )
    n_withheld = sum(1 for t in compound_tiers.values() if t == "withheld")

    predicted_genes = set(edges_df["predicted_gene_symbol"].astype(str)) if len(edges_df) else set()
    overlap = predicted_genes & measured_genes
    novel = predicted_genes - measured_genes

    tier_edge_counts = (
        edges_df["confidence_tier"].value_counts().to_dict() if len(edges_df) else {}
    )

    dist_edges = edges_df[edges_df["is_distinctive"]] if len(edges_df) else edges_df
    dist_gene_counts = (
        dist_edges["predicted_gene_symbol"].value_counts().head(20).to_dict()
        if len(dist_edges)
        else {}
    )
    all_gene_counts = (
        edges_df["predicted_gene_symbol"].value_counts().head(20).to_dict()
        if len(edges_df)
        else {}
    )

    # Sample 10 distinctive compounds with predictions for eyeball review
    samples: list[dict[str, Any]] = []
    if len(dist_edges):
        sample_iks = (
            dist_edges["dark_compound_inchikey"]
            .drop_duplicates()
            .head(10)
            .astype(str)
            .tolist()
        )
        for ik in sample_iks:
            sub = dist_edges[dist_edges["dark_compound_inchikey"] == ik].sort_values(
                "prediction_score", ascending=False
            )
            samples.append(
                {
                    "dark_compound_inchikey": ik,
                    "max_nn_tanimoto": float(sub["max_nn_tanimoto"].iloc[0]),
                    "confidence_tier": str(sub["confidence_tier"].iloc[0]),
                    "n_predicted_genes": int(len(sub)),
                    "top_predictions": [
                        {
                            "gene": str(r["predicted_gene_symbol"]),
                            "prediction_score": float(r["prediction_score"]),
                            "nearest_neighbor_inchikey": str(r["nearest_neighbor_inchikey"]),
                            "nearest_neighbor_tanimoto": float(r["nearest_neighbor_tanimoto"]),
                        }
                        for _, r in sub.head(8).iterrows()
                    ],
                }
            )

    return {
        "summary": {
            "total_dark_compounds_processed": len(dark_iks),
            "compounds_with_predicted_edges": n_with_edges,
            "compounds_fully_withheld": n_withheld,
            "total_predicted_edges": int(len(edges_df)),
            "unique_predicted_genes": int(len(predicted_genes)),
            "measured_gene_vocabulary_size": len(measured_genes),
            "predicted_genes_also_in_measured_layer": len(overlap),
            "predicted_genes_not_in_measured_layer": len(novel),
            "predicted_gene_overlap_fraction": (
                len(overlap) / len(predicted_genes) if predicted_genes else 0.0
            ),
        },
        "tier_breakdown_compounds": tier_counts_for_compounds(compound_tiers, distinctive_iks),
        "tier_breakdown_edges": tier_edge_counts,
        "gene_frequency_top20_all_predictions": all_gene_counts,
        "gene_frequency_top20_distinctive_predictions": dist_gene_counts,
        "distinctive_deconvergence_note": (
            "Compare gene_frequency_top20_distinctive_predictions to measured-layer common genes "
            "(e.g. CA family). High concentration on the same top genes suggests limited de-convergence."
        ),
        "distinctive_prediction_samples": samples,
        "timing_sec": {k: round(v, 2) for k, v in timing.items()},
        "probe": probe,
        "calibrated_tiers": {
            "predicted_high": {"min_max_nn_tanimoto": 0.7, "calibrated_hit_at_10": 0.99},
            "predicted_moderate": {"min_max_nn_tanimoto": 0.5, "calibrated_hit_at_10": 0.97},
            "predicted_low": {"min_max_nn_tanimoto": 0.3, "calibrated_hit_at_10": 0.69},
            "withheld": {"max_max_nn_tanimoto": 0.3, "calibrated_hit_at_10": 0.32},
        },
        "outputs": {
            "predicted_edges": str(EDGES_OUT),
            "withheld_compounds": str(WITHHELD_OUT),
            "note": "PREDICTED edges only — never merge with measured corpus without explicit tier tagging",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Path 2 k-NN inference engine")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Restart inference")
    parser.add_argument("--dark-batch", type=int, default=DARK_BATCH)
    parser.add_argument("--corpus-chunk", type=int, default=CORPUS_CHUNK)
    parser.add_argument(
        "--max-dark",
        type=int,
        default=0,
        help="Limit dark compounds processed (0 = all; for smoke tests only)",
    )
    args = parser.parse_args()

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    total_t0 = time.perf_counter()
    timing: dict[str, float] = {}

    print("=== Path 2 k-NN Inference Engine ===", flush=True)
    print(f"Output directory: {INFERENCE_DIR}", flush=True)

    t0 = time.perf_counter()
    corpus_gene_map = load_corpus_gene_labels()
    corpus_meta, corpus_fps, dark_meta, dark_fps = load_fingerprint_checkpoints()
    if args.max_dark > 0:
        dark_meta = dark_meta.iloc[: args.max_dark].reset_index(drop=True)
        dark_fps = dark_fps[: args.max_dark]
        print(f"[data] limited to first {len(dark_meta):,} dark compounds (--max-dark)", flush=True)
    timing["load_sec"] = time.perf_counter() - t0

    corpus_inchikeys = corpus_meta["inchikey"].astype(str).tolist()
    corpus_genes = [corpus_gene_map.get(ik, set()) for ik in corpus_inchikeys]
    n_missing_labels = sum(1 for g in corpus_genes if not g)
    if n_missing_labels:
        print(f"[warn] {n_missing_labels:,} corpus compounds have no gene labels", flush=True)

    measured_genes = set(
        pd.read_csv(MEASURED_GENE_PATH, usecols=["gene_symbol"])["gene_symbol"].astype(str)
    )
    print(f"[load] measured gene vocabulary (canonical layer): {len(measured_genes):,}", flush=True)

    start_idx = 0
    if args.resume and CKPT_PATH.exists():
        ck = np.load(CKPT_PATH)
        start_idx = int(ck["next_idx"])
        print(f"[infer] resuming from dark compound index {start_idx:,}", flush=True)
    elif not args.resume:
        for path in INFERENCE_DIR.glob("_predicted_edges_batch_*.parquet"):
            path.unlink()
        for path in INFERENCE_DIR.glob("_withheld_batch_*.parquet"):
            path.unlink()
        if CKPT_PATH.exists():
            CKPT_PATH.unlink()

    n_dark = len(dark_meta)
    edge_cols = [
        "dark_compound_inchikey",
        "predicted_gene_symbol",
        "prediction_score",
        "max_nn_tanimoto",
        "confidence_tier",
        "nearest_neighbor_inchikey",
        "nearest_neighbor_tanimoto",
        "is_distinctive",
    ]
    withheld_cols = [
        "dark_compound_inchikey",
        "max_nn_tanimoto",
        "is_distinctive",
        "reason",
    ]

    # Timing probe: one full dark batch on the actual inference path (NN + voting + tiering)
    probe: dict[str, float] = {}
    _edge_probe: list[dict[str, Any]] = []
    _wh_probe: list[dict[str, Any]] = []
    probe_end = start_idx
    if n_dark > start_idx:
        probe_end = min(start_idx + args.dark_batch, n_dark)
        probe_n = probe_end - start_idx
        print(
            f"[probe] timing first full batch ({probe_n} compounds): "
            f"top-k NN + gene voting + tiering...",
            flush=True,
        )
        t_probe = time.perf_counter()
        _edge_probe, _wh_probe = process_dark_batch(
            dark_meta.iloc[start_idx:probe_end],
            dark_fps[start_idx:probe_end],
            corpus_fps,
            corpus_inchikeys,
            corpus_genes,
            args.corpus_chunk,
        )
        probe_s = time.perf_counter() - t_probe
        n_batches = int(np.ceil((n_dark - start_idx) / args.dark_batch))
        est_infer_s = probe_s * n_batches
        probe = {
            "probe_compounds": probe_n,
            "probe_sec": probe_s,
            "probe_sec_per_compound": probe_s / probe_n if probe_n else 0.0,
            "estimated_inference_sec": est_infer_s,
            "estimated_inference_min": est_infer_s / 60,
            "n_batches": n_batches,
            "note": "Gene voting is negligible vs top-k NN search (same matmul path as Part A + local top-10 merge)",
        }
        print(
            f"[probe] batch={probe_s:.2f}s ({probe_n/probe_s:.1f} dark/s) -> "
            f"est inference {est_infer_s / 60:.1f} min "
            f"({n_batches} batches x {probe_s:.1f}s, {n_dark - start_idx:,} compounds)",
            flush=True,
        )

    print(
        f"[infer] k={K_NEIGHBORS}, dark_batch={args.dark_batch}, "
        f"corpus_chunk={args.corpus_chunk}, corpus={len(corpus_fps):,}",
        flush=True,
    )
    infer_t0 = time.perf_counter()
    batch_num = 0
    probe_end = min(start_idx + args.dark_batch, n_dark) if n_dark > start_idx else start_idx
    for q_start in range(start_idx, n_dark, args.dark_batch):
        q_end = min(q_start + args.dark_batch, n_dark)
        if batch_num == 0 and q_start == start_idx:
            if q_end <= probe_end:
                print(
                    f"[infer] batch 1 already timed by probe ({q_end - q_start:,} compounds) — saving results",
                    flush=True,
                )
                write_batch_parquet(
                    INFERENCE_DIR / f"_predicted_edges_batch_{q_end:06d}.parquet",
                    _edge_probe,
                    edge_cols,
                )
                write_batch_parquet(
                    INFERENCE_DIR / f"_withheld_batch_{q_end:06d}.parquet",
                    _wh_probe,
                    withheld_cols,
                )
                batch_num += 1
                if batch_num % CHECKPOINT_EVERY == 0 or q_end == n_dark:
                    np.savez_compressed(CKPT_PATH, next_idx=q_end)
                elapsed = time.perf_counter() - infer_t0
                print(
                    f"[infer] {q_end:,}/{n_dark:,} dark compounds, "
                    f"+{len(_edge_probe):,} edges, +{len(_wh_probe):,} withheld, "
                    f"elapsed {elapsed:.0f}s (from probe)",
                    flush=True,
                )
                continue
            print(f"[infer] processing batch 1 ({q_end - q_start:,} compounds)...", flush=True)

        edge_rows, withheld_rows = process_dark_batch(
            dark_meta.iloc[q_start:q_end],
            dark_fps[q_start:q_end],
            corpus_fps,
            corpus_inchikeys,
            corpus_genes,
            args.corpus_chunk,
        )
        write_batch_parquet(
            INFERENCE_DIR / f"_predicted_edges_batch_{q_end:06d}.parquet",
            edge_rows,
            edge_cols,
        )
        write_batch_parquet(
            INFERENCE_DIR / f"_withheld_batch_{q_end:06d}.parquet",
            withheld_rows,
            withheld_cols,
        )

        batch_num += 1
        elapsed = time.perf_counter() - infer_t0
        done = q_end - start_idx
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (n_dark - q_end) / rate if rate > 0 else 0.0
        print(
            f"[infer] {q_end:,}/{n_dark:,} dark compounds, "
            f"+{len(edge_rows):,} edges, +{len(withheld_rows):,} withheld, "
            f"elapsed {elapsed:.0f}s, est remaining {remaining:.0f}s",
            flush=True,
        )

        if batch_num % CHECKPOINT_EVERY == 0 or q_end == n_dark:
            np.savez_compressed(CKPT_PATH, next_idx=q_end)

    timing["inference_sec"] = time.perf_counter() - infer_t0

    print("[write] concatenating batch parquet files...", flush=True)
    t0 = time.perf_counter()
    edges_df = concat_batch_parquets(EDGES_BATCH_GLOB, edge_cols)
    withheld_df = concat_batch_parquets(WITHHELD_BATCH_GLOB, withheld_cols)
    edges_df.to_parquet(EDGES_OUT, index=False)
    withheld_df.to_parquet(WITHHELD_OUT, index=False)
    for path in INFERENCE_DIR.glob("_predicted_edges_batch_*.parquet"):
        path.unlink()
    for path in INFERENCE_DIR.glob("_withheld_batch_*.parquet"):
        path.unlink()

    if CKPT_PATH.exists():
        CKPT_PATH.unlink()
    timing["write_sec"] = time.perf_counter() - t0

    timing["total_sec"] = time.perf_counter() - total_t0
    report = build_report(edges_df, withheld_df, dark_meta, measured_genes, timing, probe)

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Inference summary ===", flush=True)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print("\n=== Distinctive compound tier breakdown ===", flush=True)
    print(json.dumps(report["tier_breakdown_compounds"]["distinctive_compounds"], indent=2), flush=True)
    print(f"\nWrote {EDGES_OUT}", flush=True)
    print(f"Wrote {WITHHELD_OUT}", flush=True)
    print(f"Wrote {REPORT_OUT}", flush=True)
    print(f"Total wall time: {timing['total_sec']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
