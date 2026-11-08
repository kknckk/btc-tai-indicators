#!/usr/bin/env python3
"""Download NBP reference rate (RPP) via official API."""
from experiments.macro_pl import fetch_nbp_reference_rate


if __name__ == "__main__":
    s = fetch_nbp_reference_rate(refresh=True)
    print(f"nbp_rate: {len(s)} rows, last={s.iloc[-1]:.2f} @ {s.index[-1].date()}")
