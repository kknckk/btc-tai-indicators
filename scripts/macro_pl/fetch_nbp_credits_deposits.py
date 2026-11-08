#!/usr/bin/env python3
"""Download NBP monetary statistics (credits + household deposits) from XLS bulletin."""
from experiments.macro_pl import fetch_nbp_credits_deposits


if __name__ == "__main__":
    out = fetch_nbp_credits_deposits(refresh=True)
    for name, s in out.items():
        print(f"{name}: {len(s)} months, last={s.iloc[-1]:.0f} @ {s.index[-1].date()}")
