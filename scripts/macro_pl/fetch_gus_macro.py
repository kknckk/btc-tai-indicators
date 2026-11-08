"""Load GUS CPI y/y and PMI into macro_pl cache (CSV or BDL API)."""
from experiments.macro_pl import fetch_gus_macro


if __name__ == "__main__":
    out = fetch_gus_macro(refresh=True)
    for name, s in out.items():
        print(f"{name}: {len(s)} rows, last={s.iloc[-1]:.2f} @ {s.index[-1].date()}")
