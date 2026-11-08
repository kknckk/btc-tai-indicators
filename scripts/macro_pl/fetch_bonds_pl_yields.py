#!/usr/bin/env python3
"""Download PL bond yields 2/5/7/10Y from MF public CSV (not GPW Benchmark SPW)."""
from experiments.macro_pl import fetch_bonds_pl_yields


if __name__ == "__main__":
    out = fetch_bonds_pl_yields(refresh=True)
    for name, s in out.items():
        print(f"{name}: {len(s)} rows, last={s.iloc[-1]:.4f} @ {s.index[-1].date()}")
