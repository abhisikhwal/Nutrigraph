#!/usr/bin/env python3
"""
Thread 2 feasibility + model-comparison recon.

Evaluates whether dark ingredient compounds are structurally predictable from the
structure->target corpus, and compares similarity k-NN vs a learned multi-label model
on a held-out labeled subset.

RECON ONLY — no canonical writes, no graph, no production model.

Usage (from repo root):
    python scripts/thread2/feasibility_recon.py
    python scripts/thread2/feasibility_recon.py --part-b-max-compounds 30000
    python scripts/thread2/feasibility_recon.py --skip-part-b

Outputs: data/processed/thread2/recon/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/processed/thread2/recon"

CORPUS_PATH = ROOT / "data/processed/corpus/compound_target_corpus_v1.parquet"
INGREDIENT_COMPOUNDS = ROOT / "data/processed/canonical/ingredient_compound_canonical.csv"
COMPOUND_GENE = ROOT / "data/processed/canonical/compound_gene_expanded_canonical_normalized.csv"
FOODB_COMPOUNDS = ROOT / "data/interim/foodb/compounds.parquet"

FP_RADIUS = 2
FP_NBITS = 2048
SIM_THRESHOLDS = (0.7, 0.5, 0.3)
K_NEIGHBORS = 10
PART_B_KS = (1, 5, 10)
DEFAULT_PART_B_MAX = 50_000
CORPUS_BATCH = 20_000
NN_DARK_BATCH = 500
NN_CHECKPOINT_EVERY = 2  # checkpoint every N dark batches
PART_B_TEST_BATCH = 100
PART_B_MIN_GENE_POSITIVES = 10
PART_B_MAX_GENES = 500

_MORGAN_GENERATOR = None


def configure_rdkit_logging() -> None:
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass


def get_morgan_generator():
    global _MORGAN_GENERATOR
    if _MORGAN_GENERATOR is None:
        from rdkit.Chem import rdFingerprintGenerator

        _MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
            radius=FP_RADIUS,
            fpSize=FP_NBITS,
        )
    return _MORGAN_GENERATOR


@dataclass
class StageTimer:
    name: str
    t0: float = field(default_factory=time.perf_counter)

    def done(self, msg: str = "") -> float:
        elapsed = time.perf_counter() - self.t0
        suffix = f" — {msg}" if msg else ""
        print(f"[{self.name}] done in {elapsed:.1f}s{suffix}", flush=True)
        return elapsed


def require_rdkit():
    try:
        from rdkit import Chem, DataStructs  # noqa: F401
        from rdkit.Chem import rdFingerprintGenerator  # noqa: F401
        configure_rdkit_logging()
        return True
    except ImportError:
        print(
            "ERROR: RDKit is required.\n"
            "  pip install rdkit\n"
            "  # or: conda install -c conda-forge rdkit",
            file=sys.stderr,
        )
        return False


def mol_from_smiles_or_inchi(smiles: str | None, inchi: str | None):
    from rdkit import Chem

    if smiles and str(smiles).strip():
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is not None:
            return mol
    if inchi and str(inchi).strip():
        try:
            return Chem.MolFromInchi(str(inchi).strip())
        except Exception:
            return None
    return None


def morgan_fp_array(mol, generator=None) -> np.ndarray | None:
    from rdkit import DataStructs

    if mol is None:
        return None
    if generator is None:
        generator = get_morgan_generator()
    bv = generator.GetFingerprint(mol)
    arr = np.zeros((FP_NBITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def tanimoto_self_similarity(fp: np.ndarray) -> float:
    """Sanity check helper: fingerprint vs itself should be 1.0."""
    q = np.asarray(fp, dtype=np.uint8).astype(np.float32)
    on = float(q @ q)
    if on == 0:
        return 0.0
    return on / on


def estimate_nn_peak_memory_mb(dark_batch: int, corpus_chunk: int) -> float:
    """Transient peak for one dark batch against one corpus chunk."""
    sims_mb = dark_batch * corpus_chunk * 4 / 1e6
    query_mb = dark_batch * FP_NBITS * 4 / 1e6  # float32 query block
    chunk_mb = corpus_chunk * FP_NBITS * 4 / 1e6  # float32 corpus chunk
    return sims_mb + query_mb + chunk_mb


def prepare_corpus_for_nn(corpus_fps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reuse stored uint8 fingerprints and precompute on-bit counts once."""
    corpus = np.asarray(corpus_fps, dtype=np.uint8)
    corpus_counts = corpus.sum(axis=1, dtype=np.int32)
    return corpus, corpus_counts


def max_tanimoto_for_dark_batch(
    query_fps: np.ndarray,
    corpus_fps: np.ndarray,
    corpus_counts: np.ndarray,
    corpus_chunk_size: int,
) -> np.ndarray:
    """Max Tanimoto of each query row against all corpus rows (chunked, no full matrix)."""
    q = np.asarray(query_fps, dtype=np.uint8)
    q_counts = q.sum(axis=1, dtype=np.int32)
    qf = q.astype(np.float32)
    n_q = len(q)
    n_c = len(corpus_fps)
    batch_max = np.full(n_q, -1.0, dtype=np.float32)

    for lo in range(0, n_c, corpus_chunk_size):
        hi = min(lo + corpus_chunk_size, n_c)
        cf = corpus_fps[lo:hi].astype(np.float32)
        intersection = (qf @ cf.T).astype(np.int32)
        union = q_counts[:, np.newaxis] + corpus_counts[lo:hi] - intersection
        sims = np.divide(
            intersection,
            union,
            where=union > 0,
            out=np.zeros((n_q, hi - lo), dtype=np.float32),
        )
        batch_max = np.maximum(batch_max, sims.max(axis=1))
    return batch_max


def tanimoto_similarity_vector(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Tanimoto similarity of one query against each corpus row."""
    q = np.asarray(query, dtype=np.uint8).astype(np.float32)
    c = np.asarray(corpus, dtype=np.uint8).astype(np.float32)
    intersection = (c @ q).astype(np.int32)
    union = int(np.asarray(query, dtype=np.uint8).sum()) + np.asarray(corpus, dtype=np.uint8).sum(
        axis=1, dtype=np.int32
    ) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            intersection,
            union,
            where=union > 0,
            out=np.zeros(corpus.shape[0], dtype=np.float64),
        )


def tanimoto_similarity_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Batched Tanimoto matrix via float32 matmul (same approach as Part A NN)."""
    q_u8 = np.asarray(queries, dtype=np.uint8)
    c_u8 = np.asarray(corpus, dtype=np.uint8)
    q = q_u8.astype(np.float32)
    c = c_u8.astype(np.float32)
    intersection = (q @ c.T).astype(np.int32)
    q_counts = q_u8.sum(axis=1, dtype=np.int32, keepdims=True)
    c_counts = c_u8.sum(axis=1, dtype=np.int32)
    union = q_counts + c_counts - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            intersection,
            union,
            where=union > 0,
            out=np.zeros(intersection.shape, dtype=np.float32),
        )


def load_corpus_compounds() -> pd.DataFrame:
    df = pd.read_parquet(
        CORPUS_PATH,
        columns=["compound_inchikey", "compound_smiles", "compound_inchi"],
    )
    out = (
        df.drop_duplicates("compound_inchikey")
        .rename(
            columns={
                "compound_inchikey": "inchikey",
                "compound_smiles": "smiles",
                "compound_inchi": "inchi",
            }
        )
        .reset_index(drop=True)
    )
    return out


def load_dark_compounds() -> pd.DataFrame:
    ic = pd.read_csv(INGREDIENT_COMPOUNDS, usecols=["ingredient_id", "compound_id"])
    cg = pd.read_csv(COMPOUND_GENE, usecols=["compound_id"])
    mapped = set(cg["compound_id"].astype(str))
    dark_ids = set(ic["compound_id"].astype(str)) - mapped
    comp_counts = ic.groupby("compound_id")["ingredient_id"].nunique()
    distinctive = set(comp_counts[comp_counts == 1].index.astype(str))

    # Structure from FoodDB interim (columns are swapped: smiles=inchikey, inchikey=InChI)
    fb = pd.read_parquet(FOODB_COMPOUNDS, columns=["smiles", "inchikey"])
    fb = fb.rename(columns={"smiles": "inchikey_lookup", "inchikey": "inchi"})
    fb = fb.drop_duplicates("inchikey_lookup")

    base = pd.DataFrame({"inchikey": sorted(dark_ids)})
    base["is_distinctive"] = base["inchikey"].isin(distinctive)
    base = base.merge(fb, left_on="inchikey", right_on="inchikey_lookup", how="left")

    # Fallback: corpus structures for dark compounds that appear in corpus
    corp = load_corpus_compounds().rename(
        columns={"smiles": "corpus_smiles", "inchi": "corpus_inchi"}
    )
    base = base.merge(corp, on="inchikey", how="left")
    base["smiles"] = base["corpus_smiles"]
    base["inchi"] = base["inchi"].fillna(base["corpus_inchi"])
    return base.drop(columns=["inchikey_lookup", "corpus_smiles", "corpus_inchi"], errors="ignore")


def compute_fingerprints(
    df: pd.DataFrame,
    label: str,
    out_meta: Path,
    out_fps: Path,
    resume: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    if resume and out_meta.exists() and out_fps.exists():
        meta = pd.read_parquet(out_meta)
        fps = np.load(out_fps)["fps"]
        print(f"[fp:{label}] resumed {len(meta):,} fingerprints from checkpoint", flush=True)
        return meta, fps

    print(f"[fp:{label}] computing Morgan ECFP4 ({FP_NBITS}-bit) for {len(df):,} compounds", flush=True)
    timer = StageTimer(f"fp:{label}")
    generator = get_morgan_generator()
    keys: list[str] = []
    fps_list: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    failed = 0
    for i, row in df.iterrows():
        if (i + 1) % 5000 == 0:
            print(f"  [{label}] {i + 1:,}/{len(df):,} processed ({len(fps_list):,} ok, {failed:,} fail)", flush=True)
        mol = mol_from_smiles_or_inchi(row.get("smiles"), row.get("inchi"))
        fp = morgan_fp_array(mol, generator)
        if fp is None:
            failed += 1
            continue
        keys.append(str(row["inchikey"]))
        fps_list.append(fp)
        meta_rows.append(row.to_dict())

    meta = pd.DataFrame(meta_rows).reset_index(drop=True)
    fps = np.vstack(fps_list).astype(np.uint8)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(out_meta, index=False)
    np.savez_compressed(out_fps, fps=fps)
    timer.done(f"{len(meta):,} ok, {failed:,} failed")
    return meta, fps


def max_similarity_batched(
    query_fps: np.ndarray,
    corpus_fps: np.ndarray,
    batch_size: int,
    checkpoint_path: Path | None,
    start_idx: int = 0,
) -> np.ndarray:
    n_q = len(query_fps)
    n_c = len(corpus_fps)
    max_sims = np.full(n_q, -1.0, dtype=np.float32)
    if checkpoint_path and checkpoint_path.exists():
        ck = np.load(checkpoint_path)
        max_sims = ck["max_sims"]
        start_idx = int(ck["next_idx"])
        print(f"[nn] resumed from query index {start_idx:,}", flush=True)

    n_corpus_chunks = (n_c + batch_size - 1) // batch_size
    peak_mb = estimate_nn_peak_memory_mb(NN_DARK_BATCH, batch_size)
    print(
        f"[nn] starting: {n_q:,} dark x {n_c:,} corpus "
        f"(dark_batch={NN_DARK_BATCH:,}, corpus_chunks={n_corpus_chunks} x {batch_size:,})",
        flush=True,
    )
    print("[nn] precomputing corpus on-bit counts (one-time)...", flush=True)
    prep_t0 = time.perf_counter()
    corpus_matrix, corpus_counts = prepare_corpus_for_nn(corpus_fps)
    corpus_mb = corpus_matrix.nbytes / 1e6
    print(
        f"[nn] corpus ready in {time.perf_counter() - prep_t0:.1f}s — "
        f"reusing {corpus_mb:.0f} MB uint8 matrix, peak ~{peak_mb:.0f} MB per dark batch",
        flush=True,
    )
    if n_q > 0:
        self_sim = tanimoto_self_similarity(query_fps[0])
        print(f"[nn] sanity check: fingerprint vs itself tanimoto={self_sim:.4f}", flush=True)

    search_t0 = time.perf_counter()
    batch_num = 0
    for q_start in range(start_idx, n_q, NN_DARK_BATCH):
        q_end = min(q_start + NN_DARK_BATCH, n_q)
        if batch_num == 0:
            print(
                f"[nn] processing dark batch 1 ({q_end - q_start:,} compounds x "
                f"{n_c:,} corpus, {n_corpus_chunks} corpus chunks)...",
                flush=True,
            )
        max_sims[q_start:q_end] = max_tanimoto_for_dark_batch(
            query_fps[q_start:q_end],
            corpus_matrix,
            corpus_counts,
            batch_size,
        )
        batch_num += 1
        elapsed = time.perf_counter() - search_t0
        done = q_end - start_idx
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (n_q - q_end) / rate if rate > 0 else 0.0
        print(
            f"[nn] {q_end:,}/{n_q:,} dark processed, "
            f"elapsed {elapsed:.0f}s, est remaining {remaining:.0f}s",
            flush=True,
        )
        if checkpoint_path and (batch_num % NN_CHECKPOINT_EVERY == 0 or q_end == n_q):
            np.savez_compressed(checkpoint_path, max_sims=max_sims, next_idx=q_end)

    if checkpoint_path:
        np.savez_compressed(checkpoint_path, max_sims=max_sims, next_idx=n_q)
    print(
        f"[nn] complete in {time.perf_counter() - search_t0:.1f}s "
        f"({(n_q - start_idx) / max(time.perf_counter() - search_t0, 1e-9):.0f} dark/s)",
        flush=True,
    )
    return max_sims


def threshold_breakdown(sims: np.ndarray, is_distinctive: np.ndarray | None = None) -> dict[str, Any]:
    def _one(arr: np.ndarray) -> dict[str, float]:
        n = len(arr)
        if n == 0:
            return {"n": 0}
        out = {"n": n}
        out["ge_0.7"] = float((arr >= 0.7).mean())
        out["ge_0.5"] = float((arr >= 0.5).mean())
        out["ge_0.3"] = float((arr >= 0.3).mean())
        out["lt_0.3"] = float((arr < 0.3).mean())
        out["median"] = float(np.median(arr))
        out["mean"] = float(np.mean(arr))
        out["p25"] = float(np.percentile(arr, 25))
        out["p75"] = float(np.percentile(arr, 75))
        return out

    result = {"all": _one(sims)}
    if is_distinctive is not None:
        result["distinctive"] = _one(sims[is_distinctive])
        result["non_distinctive"] = _one(sims[~is_distinctive])
    return result


def run_part_a(args: argparse.Namespace) -> dict[str, Any]:
    print("\n=== PART A: Structural feasibility gate ===", flush=True)
    corpus_df = load_corpus_compounds()
    dark_df = load_dark_compounds()
    print(f"[A] corpus compounds: {len(corpus_df):,}", flush=True)
    print(f"[A] dark compounds: {len(dark_df):,} (distinctive: {dark_df['is_distinctive'].sum():,})", flush=True)

    corpus_meta, corpus_fps = compute_fingerprints(
        corpus_df,
        "corpus",
        OUT_DIR / "fingerprints_corpus_meta.parquet",
        OUT_DIR / "fingerprints_corpus_fps.npz",
        resume=args.resume,
    )
    dark_meta, dark_fps = compute_fingerprints(
        dark_df,
        "dark",
        OUT_DIR / "fingerprints_dark_meta.parquet",
        OUT_DIR / "fingerprints_dark_fps.npz",
        resume=args.resume,
    )

    nn_ckpt = OUT_DIR / "part_a_nn_checkpoint.npz"
    if (OUT_DIR / "part_a_nearest_neighbor.parquet").exists() and args.resume:
        nn_df = pd.read_parquet(OUT_DIR / "part_a_nearest_neighbor.parquet")
        print(f"[A] resumed NN results ({len(nn_df):,} rows)", flush=True)
    else:
        max_sims = max_similarity_batched(
            dark_fps,
            corpus_fps,
            batch_size=args.nn_batch_size,
            checkpoint_path=nn_ckpt,
        )
        nn_df = dark_meta[["inchikey", "is_distinctive"]].copy()
        nn_df["max_tanimoto"] = max_sims
        nn_df.to_parquet(OUT_DIR / "part_a_nearest_neighbor.parquet", index=False)
        if nn_ckpt.exists():
            nn_ckpt.unlink()

    breakdown = threshold_breakdown(
        nn_df["max_tanimoto"].to_numpy(),
        nn_df["is_distinctive"].to_numpy(),
    )
    reachable_05 = breakdown["distinctive"]["ge_0.5"] if "distinctive" in breakdown else 0

    part_a = {
        "corpus_compounds_with_fp": int(len(corpus_meta)),
        "dark_compounds_with_fp": int(len(dark_meta)),
        "dark_parse_failures": int(len(dark_df) - len(dark_meta)),
        "similarity_breakdown": breakdown,
        "verdict": {
            "distinctive_reachable_ge_0.5_fraction": breakdown.get("distinctive", {}).get("ge_0.5"),
            "distinctive_isolated_lt_0.3_fraction": breakdown.get("distinctive", {}).get("lt_0.3"),
            "summary": (
                "Structural inference is credible for a meaningful subset"
                if reachable_05 and reachable_05 >= 0.15
                else "Most distinctive dark compounds appear structurally isolated; Path 2 will be partial at best"
            ),
        },
    }
    with open(OUT_DIR / "part_a_summary.json", "w", encoding="utf-8") as f:
        json.dump(part_a, f, indent=2)
    print(json.dumps(part_a["verdict"], indent=2), flush=True)
    return part_a


def load_corpus_labels() -> tuple[pd.DataFrame, dict[str, set[str]]]:
    print("[B] loading corpus labels from parquet...", flush=True)
    timer = StageTimer("B:labels")
    df = pd.read_parquet(
        CORPUS_PATH,
        columns=["compound_inchikey", "gene_symbol", "compound_smiles", "compound_inchi"],
    )
    gene_sets = (
        df.groupby("compound_inchikey")["gene_symbol"]
        .apply(lambda symbols: set(symbols.astype(str)))
        .to_dict()
    )
    compounds = (
        df.drop_duplicates("compound_inchikey")
        .rename(
            columns={
                "compound_inchikey": "inchikey",
                "compound_smiles": "smiles",
                "compound_inchi": "inchi",
            }
        )[["inchikey", "smiles", "inchi"]]
        .reset_index(drop=True)
    )
    compounds["n_targets"] = compounds["inchikey"].astype(str).map(
        lambda ik: len(gene_sets[str(ik)])
    )
    timer.done(f"{len(compounds):,} compounds, {len(gene_sets):,} label sets")
    return compounds, gene_sets


def load_part_b_fingerprints(compounds: pd.DataFrame) -> np.ndarray:
    """Reuse Part A corpus fingerprint checkpoint — no RDKit recomputation."""
    meta_path = OUT_DIR / "fingerprints_corpus_meta.parquet"
    fps_path = OUT_DIR / "fingerprints_corpus_fps.npz"
    if not meta_path.exists() or not fps_path.exists():
        raise FileNotFoundError(
            "Part A corpus fingerprint checkpoint missing. Run Part A first "
            f"(expected {meta_path} and {fps_path})."
        )
    print(
        f"[B] aligning {len(compounds):,} compounds to Part A corpus fingerprint checkpoint...",
        flush=True,
    )
    timer = StageTimer("B:fp-align")
    meta = pd.read_parquet(meta_path)
    fps = np.load(fps_path)["fps"]
    ik_to_idx = {str(ik): i for i, ik in enumerate(meta["inchikey"])}
    keys = compounds["inchikey"].astype(str).tolist()
    missing = [k for k in keys if k not in ik_to_idx]
    if missing:
        raise RuntimeError(
            f"{len(missing):,} Part B compounds missing from corpus fingerprints "
            f"(first: {missing[0]})"
        )
    aligned = fps[[ik_to_idx[k] for k in keys]]
    timer.done(f"{len(aligned):,} fingerprints aligned")
    return aligned


def select_part_b_genes(
    train_gene_list: list[set[str]],
    min_positives: int,
    max_genes: int,
) -> tuple[list[str], np.ndarray]:
    """Keep genes with enough train positives; cap to top-N by frequency."""
    all_genes = sorted({g for gs in train_gene_list for g in gs})
    gene_to_col = {g: i for i, g in enumerate(all_genes)}
    y = np.zeros((len(train_gene_list), len(all_genes)), dtype=np.int8)
    for i, gs in enumerate(train_gene_list):
        for g in gs:
            y[i, gene_to_col[g]] = 1
    pos_counts = y.sum(axis=0)
    eligible = np.flatnonzero(pos_counts >= min_positives)
    if len(eligible) > max_genes:
        top = eligible[np.argsort(pos_counts[eligible])[::-1][:max_genes]]
        eligible = np.sort(top)
    selected_genes = [all_genes[i] for i in eligible]
    return selected_genes, y[:, eligible]


def ranking_metrics(true_genes: set[str], ranked: list[str], ks: tuple[int, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        pred = ranked[:k]
        pred_set = set(pred)
        hits = len(true_genes & pred_set)
        out[f"precision@{k}"] = hits / k if k else 0.0
        out[f"recall@{k}"] = hits / len(true_genes) if true_genes else 0.0
        out[f"hit@{k}"] = 1.0 if hits > 0 else 0.0
    return out


def knn_predict_from_similarities(
    sims_row: np.ndarray,
    train_genes: list[set[str]],
    k: int,
) -> list[str]:
    order = np.argsort(sims_row)[::-1][:k]
    gene_scores: dict[str, float] = {}
    for idx in order:
        w = float(sims_row[idx])
        for g in train_genes[idx]:
            gene_scores[g] = gene_scores.get(g, 0.0) + w
    return sorted(gene_scores, key=gene_scores.get, reverse=True)


def knn_predict(
    test_fp: np.ndarray,
    train_fps: np.ndarray,
    train_genes: list[set[str]],
    k: int,
) -> list[str]:
    sims = tanimoto_similarity_vector(test_fp, train_fps)
    return knn_predict_from_similarities(sims, train_genes, k)


def evaluate_knn_batched(
    test_fps: np.ndarray,
    train_fps: np.ndarray,
    train_genes: list[set[str]],
    test_genes: list[set[str]],
    k: int,
    ks: tuple[int, ...],
    test_batch: int,
) -> tuple[dict[str, float], float]:
    """Batched float32 Tanimoto k-NN (same matmul path as Part A)."""
    n_test = len(test_fps)
    print(
        f"[B:knn] batched float32 Tanimoto: {n_test:,} test x {len(train_fps):,} train "
        f"(batch={test_batch})",
        flush=True,
    )
    timer = StageTimer("B:knn")
    knn_accum: dict[str, list[float]] = {}
    for start in range(0, n_test, test_batch):
        end = min(start + test_batch, n_test)
        sims = tanimoto_similarity_matrix(test_fps[start:end], train_fps)
        for i, row_idx in enumerate(range(start, end)):
            ranked = knn_predict_from_similarities(sims[i], train_genes, k)
            metrics = ranking_metrics(test_genes[row_idx], ranked, ks)
            for mk, mv in metrics.items():
                knn_accum.setdefault(mk, []).append(mv)
        elapsed = time.perf_counter() - timer.t0
        rate = end / elapsed if elapsed > 0 else 0.0
        remaining = (n_test - end) / rate if rate > 0 else 0.0
        print(
            f"[B:knn] {end:,}/{n_test:,} test compounds scored, "
            f"elapsed {elapsed:.0f}s, est remaining {remaining:.0f}s",
            flush=True,
        )
    knn_metrics = {mk: float(np.mean(vals)) for mk, vals in knn_accum.items()}
    elapsed = timer.done()
    return knn_metrics, elapsed


def run_part_b(args: argparse.Namespace) -> dict[str, Any]:
    print("\n=== PART B: Held-out approach comparison ===", flush=True)
    try:
        from sklearn.linear_model import SGDClassifier
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("ERROR: scikit-learn required.  pip install scikit-learn", file=sys.stderr)
        return {}

    compounds, gene_sets = load_corpus_labels()
    if args.part_b_max_compounds > 0 and len(compounds) > args.part_b_max_compounds:
        compounds = compounds.sample(args.part_b_max_compounds, random_state=args.seed)
        print(f"[B] subsampled to {len(compounds):,} compounds for tractable recon", flush=True)

    fps_b = load_part_b_fingerprints(compounds)
    inchikeys = compounds["inchikey"].astype(str).tolist()

    print("[B] train/test split (80/20)...", flush=True)
    split_t0 = time.perf_counter()
    train_ik, test_ik = train_test_split(inchikeys, test_size=0.2, random_state=args.seed)
    ik_to_row = {k: i for i, k in enumerate(inchikeys)}
    train_idx = [ik_to_row[k] for k in train_ik]
    test_idx = [ik_to_row[k] for k in test_ik]
    train_fps = fps_b[train_idx]
    test_fps = fps_b[test_idx]
    train_gene_list = [gene_sets[k] for k in train_ik]
    test_gene_list = [gene_sets[k] for k in test_ik]
    print(
        f"[B] split done in {time.perf_counter() - split_t0:.1f}s — "
        f"train={len(train_ik):,}, test={len(test_ik):,}",
        flush=True,
    )

    selected_genes, y_train = select_part_b_genes(
        train_gene_list,
        min_positives=args.part_b_min_gene_positives,
        max_genes=args.part_b_max_genes,
    )
    print(
        f"[B] gene label matrix: {y_train.shape[1]:,} genes "
        f"(>= {args.part_b_min_gene_positives} train positives, "
        f"cap {args.part_b_max_genes:,})",
        flush=True,
    )

    results: dict[str, Any] = {
        "train_n": len(train_ik),
        "test_n": len(test_ik),
        "n_genes": len(selected_genes),
        "gene_filter": {
            "min_train_positives": args.part_b_min_gene_positives,
            "max_genes": args.part_b_max_genes,
        },
        "fingerprints": "reused Part A corpus checkpoint (fingerprints_corpus_*.parquet/npz)",
        "knn_similarity": "batched float32 Tanimoto matmul (tanimoto_similarity_matrix)",
    }

    knn_metrics, knn_elapsed = evaluate_knn_batched(
        test_fps,
        train_fps,
        train_gene_list,
        test_gene_list,
        K_NEIGHBORS,
        PART_B_KS,
        args.part_b_test_batch,
    )
    results["knn"] = {
        "metrics": knn_metrics,
        "runtime_sec": knn_elapsed,
        "k_neighbors": K_NEIGHBORS,
        "test_batch": args.part_b_test_batch,
    }

    print(
        f"[B:lr] training One-vs-Rest SGD (log-loss) on {len(train_ik):,} x {FP_NBITS} "
        f"for {y_train.shape[1]:,} genes...",
        flush=True,
    )
    lr_timer = StageTimer("B:lr")
    clf = OneVsRestClassifier(
        SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            tol=1e-3,
            random_state=args.seed,
        ),
        n_jobs=-1,
    )
    clf.fit(train_fps.astype(np.float32), y_train)
    print(f"[B:lr] fitting done in {time.perf_counter() - lr_timer.t0:.1f}s — scoring test set...", flush=True)
    probas = clf.predict_proba(test_fps.astype(np.float32))
    if isinstance(probas, list):
        score_mat = np.column_stack([p[:, 1] if p.ndim == 2 else p for p in probas])
    else:
        score_mat = probas

    print(f"[B:lr] computing ranking metrics for {len(test_ik):,} test compounds...", flush=True)
    metric_t0 = time.perf_counter()
    lr_accum: dict[str, list[float]] = {}
    for i in range(len(test_ik)):
        if (i + 1) % 2000 == 0:
            elapsed = time.perf_counter() - metric_t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            remaining = (len(test_ik) - (i + 1)) / rate if rate > 0 else 0.0
            print(
                f"[B:lr] metrics {i + 1:,}/{len(test_ik):,}, "
                f"elapsed {elapsed:.0f}s, est remaining {remaining:.0f}s",
                flush=True,
            )
        scores = score_mat[i]
        order = np.argsort(scores)[::-1]
        ranked = [selected_genes[j] for j in order]
        metrics = ranking_metrics(test_gene_list[i], ranked, PART_B_KS)
        for mk, mv in metrics.items():
            lr_accum.setdefault(mk, []).append(mv)
    lr_metrics = {mk: float(np.mean(vals)) for mk, vals in lr_accum.items()}
    lr_elapsed = lr_timer.done()
    results["learned_logreg_ovr"] = {
        "metrics": lr_metrics,
        "runtime_sec": lr_elapsed,
        "model": "OneVsRest(SGDClassifier(log_loss))",
        "note": "SGD log-loss feasibility surrogate; genes capped for tractability",
    }

    with open(OUT_DIR / "part_b_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)
    return results


def run_part_c(part_a: dict[str, Any], part_b: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    print("\n=== PART C: Synthesis ===", flush=True)
    nn_path = OUT_DIR / "part_a_nearest_neighbor.parquet"
    nn_df = pd.read_parquet(nn_path) if nn_path.exists() else pd.DataFrame()

    # Distinctive compounds that are reachable AND would get credible top-10 hit proxy
    hit10 = part_b.get("knn", {}).get("metrics", {}).get("hit@10", 0.0)
    credible_threshold = 0.5
    if len(nn_df):
        reachable = nn_df["max_tanimoto"] >= credible_threshold
        distinctive_reachable = int((nn_df["is_distinctive"] & reachable).sum())
        distinctive_total = int(nn_df["is_distinctive"].sum())
    else:
        distinctive_reachable = distinctive_total = 0

    estimated_fixable = int(distinctive_reachable * hit10)

    synthesis = {
        "feasibility": {
            "dark_structurally_reachable_ge_0.5": part_a.get("similarity_breakdown", {}).get("all", {}).get("ge_0.5"),
            "distinctive_reachable_ge_0.5": part_a.get("similarity_breakdown", {}).get("distinctive", {}).get("ge_0.5"),
            "distinctive_isolated_lt_0.3": part_a.get("similarity_breakdown", {}).get("distinctive", {}).get("lt_0.3"),
            "verdict": part_a.get("verdict", {}).get("summary"),
        },
        "approach_comparison": {
            "knn_hit_at_10": hit10,
            "learned_hit_at_10": part_b.get("learned_logreg_ovr", {}).get("metrics", {}).get("hit@10"),
            "knn_runtime_sec": part_b.get("knn", {}).get("runtime_sec"),
            "learned_runtime_sec": part_b.get("learned_logreg_ovr", {}).get("runtime_sec"),
            "better_approach": (
                "knn"
                if hit10 >= part_b.get("learned_logreg_ovr", {}).get("metrics", {}).get("hit@10", 0)
                else "learned_logreg_ovr"
            ),
        },
        "collapse_impact_estimate": {
            "distinctive_total": distinctive_total,
            "distinctive_structurally_reachable_ge_0.5": distinctive_reachable,
            "held_out_hit_at_10_rate": hit10,
            "estimated_distinctive_with_credible_prediction": estimated_fixable,
            "note": "Rough upper bound: reachable distinctive * held-out hit@10 from k-NN recon",
        },
        "limitations": [
            "Predictions for Tanimoto < 0.3 should be treated as low-confidence or withheld.",
            "k-NN inherits neighbor annotation bias; rare targets will be under-predicted.",
            "Part B uses a quick logistic model — production should not reuse without re-tuning.",
            "Gene labels are assay-derived; measured vs predicted edges must be tagged separately in the graph.",
            "Scaffold leakage not fully controlled; use Murcko scaffold splits for production training.",
        ],
        "confidence_scoring_recommendation": {
            "measured": "corpus or BindingDB/ChEMBL direct edges",
            "predicted_high": "max_tanimoto >= 0.7 and prediction score in top decile",
            "predicted_moderate": "0.5 <= max_tanimoto < 0.7",
            "predicted_low": "0.3 <= max_tanimoto < 0.5",
            "withhold": "max_tanimoto < 0.3",
        },
    }
    with open(OUT_DIR / "synthesis_report.json", "w", encoding="utf-8") as f:
        json.dump(synthesis, f, indent=2)
    print(json.dumps(synthesis, indent=2), flush=True)
    return synthesis


def main() -> int:
    parser = argparse.ArgumentParser(description="Thread 2 feasibility + model comparison recon")
    parser.add_argument("--skip-part-a", action="store_true")
    parser.add_argument("--skip-part-b", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore checkpoints")
    parser.add_argument("--part-b-max-compounds", type=int, default=DEFAULT_PART_B_MAX)
    parser.add_argument("--part-b-test-batch", type=int, default=PART_B_TEST_BATCH)
    parser.add_argument("--part-b-min-gene-positives", type=int, default=PART_B_MIN_GENE_POSITIVES)
    parser.add_argument("--part-b-max-genes", type=int, default=PART_B_MAX_GENES)
    parser.add_argument("--nn-batch-size", type=int, default=CORPUS_BATCH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not require_rdkit():
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_t0 = time.perf_counter()
    print(f"Output directory: {OUT_DIR}", flush=True)

    part_a: dict[str, Any] = {}
    part_b: dict[str, Any] = {}
    if not args.skip_part_a:
        part_a = run_part_a(args)
    elif (OUT_DIR / "part_a_summary.json").exists():
        part_a = json.loads((OUT_DIR / "part_a_summary.json").read_text(encoding="utf-8"))

    if not args.skip_part_b:
        part_b = run_part_b(args)
    elif (OUT_DIR / "part_b_metrics.json").exists():
        part_b = json.loads((OUT_DIR / "part_b_metrics.json").read_text(encoding="utf-8"))

    if part_a and part_b:
        run_part_c(part_a, part_b, args)

    print(f"\nTotal elapsed: {time.perf_counter() - total_t0:.1f}s", flush=True)
    print("Bring back the JSON files under data/processed/thread2/recon/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
