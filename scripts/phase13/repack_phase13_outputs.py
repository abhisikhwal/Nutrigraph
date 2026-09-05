"""
Repack Phase13B Parquet outputs into Windows-friendly portable Parquet + CSV.
Usage: python scripts/phase13/repack_phase13_outputs.py --input <run_dir>

Run in WSL (or wherever the run dir lives); then copy the dir to Windows.
Read on Windows with pd.read_parquet(.../atlas_confirmed_portable.parquet) or CSV fallback.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.parquet_portable import repack_parquet_file


def main() -> None:
    p = argparse.ArgumentParser(description="Repack Phase13B Parquet outputs to portable format + CSV")
    p.add_argument("--input", type=Path, required=True, help="Phase13B run directory (e.g. .../phase13_interactions_v3_*_b_gpu_stable)")
    args = p.parse_args()
    run_dir = args.input.resolve()
    if not run_dir.is_dir():
        print(f"Error: not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    pairs = [
        (run_dir / "atlas_confirmed.parquet", run_dir / "atlas_confirmed_portable.parquet", run_dir / "atlas_confirmed.csv"),
        (run_dir / "bootstrap_stability.parquet", run_dir / "bootstrap_stability_portable.parquet", run_dir / "bootstrap_stability.csv"),
    ]
    for src, dst_portable, csv_path in pairs:
        if not src.exists():
            print(f"Skip (missing): {src.name}")
            continue
        try:
            repack_parquet_file(src, dst_portable, csv_path)
            print(f"Repacked: {src.name} -> {dst_portable.name}, {csv_path.name}")
        except Exception as e:
            print(f"Failed {src.name}: {e}", file=sys.stderr)

    kg_dir = run_dir / "kg"
    if kg_dir.is_dir():
        for name in ("kg_nodes.parquet", "kg_edges.parquet"):
            src = kg_dir / name
            if not src.exists():
                print(f"Skip (missing): kg/{name}")
                continue
            dst_portable = kg_dir / name.replace(".parquet", "_portable.parquet")
            csv_path = kg_dir / name.replace(".parquet", ".csv")
            try:
                repack_parquet_file(src, dst_portable, csv_path)
                print(f"Repacked: kg/{name} -> kg/{dst_portable.name}, kg/{csv_path.name}")
            except Exception as e:
                print(f"Failed kg/{name}: {e}", file=sys.stderr)
    else:
        print("No kg/ directory; skipping KG repack.")

    print("Done.")


if __name__ == "__main__":
    main()
