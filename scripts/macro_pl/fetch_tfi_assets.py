#!/usr/bin/env python3
"""Download TFI sector net assets from IZFiA (not Analizy.pl)."""
from experiments.macro_pl import fetch_tfi_assets


if __name__ == "__main__":
    s = fetch_tfi_assets(refresh=True)
    print(f"tfi_assets: {len(s)} months, last={s.iloc[-1]:.0f} @ {s.index[-1].date()}")
