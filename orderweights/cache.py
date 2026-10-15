"""Caching of slow simulations.

`cached(name, compute)` returns the stored result results/<name>.pkl if it exists, and
otherwise runs `compute()`, stores its result and returns it. Set RECOMPUTE = True (or pass
recompute=True) to ignore the stored results and run every simulation again.
"""
import pickle
import time
from pathlib import Path

__all__ = ["cached", "RESULTS", "FIGURES", "TABLES", "RECOMPUTE"]

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
RECOMPUTE = False


def cached(name, compute, recompute=None):
    recompute = RECOMPUTE if recompute is None else recompute
    path = RESULTS / f"{name}.pkl"
    if path.exists() and not recompute:
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    out = compute()
    RESULTS.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"[{name}] computed in {time.time() - t0:.0f} s and stored in results/{name}.pkl")
    return out
