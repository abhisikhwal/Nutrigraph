# Phase14 Readiness Check — Notebook Snippet

Paste the following into a notebook cell to run the Phase14 readiness check and print the dashboard plus `RERUN_PHASE14: YES/NO`.

```python
# Ensure repo root is on path
import sys
from pathlib import Path
repo_root = Path(".").resolve()  # run from the repository root
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.phase14.readiness import (
    run_readiness,
    load_latest_run_dir,
    print_readiness_dashboard,
)

rerun, reasons, payload = run_readiness(repo_root=repo_root)
last_run_dir = load_latest_run_dir(repo_root=repo_root)
print_readiness_dashboard(
    current_inputs=payload["current_input_fingerprints"],
    last_run_dir=last_run_dir,
    last_run_inputs=payload.get("last_run_input_fingerprints") or {},
    last_metrics=payload.get("last_run_metrics") or {},
    rerun=rerun,
    reasons=reasons,
)
# Result: RERUN_PHASE14: YES or NO
```

To write the result to `reports/readiness_check.json`, run from the repo root:

```bash
python scripts/phase14/run_readiness_check.py --write
```
