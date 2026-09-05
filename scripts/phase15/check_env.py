"""
Phase15 environment check. Prints Python, pip, import status for node2vec, pykeen, torch, torch_geometric.
Exits non-zero with actionable install commands if any required package is missing.
"""
from __future__ import annotations

import sys
import subprocess


def main() -> int:
    print("=== Phase15 environment check ===\n")
    print("sys.executable:", sys.executable)
    print("Python version:", sys.version)
    if sys.version_info >= (3, 13):
        print("WARNING: Python 3.13 may have compatibility issues. Recommended: Python 3.11.")
    try:
        pip_out = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        print("pip:", pip_out.stdout.strip() if pip_out.stdout else pip_out.stderr or "?")
    except Exception as e:
        print("pip: (could not run)", e)

    # Jupyter kernel hint
    try:
        import IPython
        print("Jupyter/IPython: detected (kernel likely active)")
    except ImportError:
        print("Jupyter: not in notebook (or IPython not installed)")

    checks = []
    # node2vec
    try:
        from node2vec import Node2Vec
        print("node2vec: OK")
        checks.append(("node2vec", True))
    except Exception as e:
        print("node2vec: MISSING:", e)
        checks.append(("node2vec", False))

    # pykeen
    try:
        import pykeen
        print("pykeen: OK")
        checks.append(("pykeen", True))
    except ImportError:
        print("pykeen: MISSING")
        checks.append(("pykeen", False))

    # torch
    try:
        import torch
        print("torch: OK", getattr(torch, "__version__", ""))
        checks.append(("torch", True))
    except ImportError:
        print("torch: MISSING")
        checks.append(("torch", False))

    # torch_geometric
    try:
        import torch_geometric
        print("torch_geometric: OK")
        checks.append(("torch_geometric", True))
    except ImportError:
        print("torch_geometric: MISSING")
        checks.append(("torch_geometric", False))

    missing = [name for name, ok in checks if not ok]
    if missing:
        print("\n--- Why 'MISSING' when pip says already satisfied? ---")
        print("  This script runs with THIS Python:", sys.executable)
        print("  If your terminal 'pip install' used a different Python (e.g. base conda),")
        print("  but Jupyter/notebook uses another kernel, that kernel does NOT see those packages.")
        print("  Fix: install into the SAME Python that runs the notebook/script:")
        print("  ", sys.executable, "-m pip install <package>")
        print("\n--- Install commands (run with the Python above) ---")
        if "node2vec" in missing:
            print("  ", sys.executable, "-m pip install node2vec gensim")
        if "pykeen" in missing:
            print("  ", sys.executable, "-m pip install pykeen")
        if "torch" in missing or "torch_geometric" in missing:
            print("  ", sys.executable, "-m pip install torch --index-url https://download.pytorch.org/whl/cpu")
            print("  ", sys.executable, "-m pip install torch-geometric")
        print("\nOr: ", sys.executable, "-m pip install -r requirements-phase15.txt")
        return 1
    print("\nAll Phase15 dependencies OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
