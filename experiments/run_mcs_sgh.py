"""Zbiorczy MCS dla serii SGH z zapisanych ramek predykcji.

Czyta results/preds_sgh/<stem>__<model>_h<h>.parquet (zapisuje je
evaluation/preds_io.py po kazdym horyzoncie runu — dzieki temu merge CPU+GPU
sumie sie w katalogu, nie w pamieci jednego procesu). Dla kazdej
(pary stem, horyzont) wyrównuje ramki po datach (wspolne originy) i liczy
Model Confidence Set z uwzglednieniem horyzontu (patrz: review bloku,
poprawiony parametr h w evaluation/mcs.py).

Wynik: results/sgh_mcs.json + results/sgh_mcs.md

Uzycie: python -m experiments.run_mcs_sgh
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.mcs import mcs

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / "results" / "preds_sgh"
OUT = ROOT / "results"


def _load_grouped() -> dict[tuple[str, int], dict[str, pd.DataFrame]]:
    """Parquety -> {(stem_z_horyzontem, model) -> frame}; stem bez suffiksu _hN."""
    grouped: dict[tuple[str, int], dict[str, pd.DataFrame]] = defaultdict(dict)
    if not PREDS.is_dir():
        return grouped
    for fp in sorted(PREDS.glob("*.parquet")):
        m = re.match(r"(.+)__(\w+?)_h(\d+)\.parquet$", fp.name)
        if not m:
            continue
        stem, model, h = m.group(1), m.group(2), int(m.group(3))
        grouped[(stem, h)][model] = pd.read_parquet(fp)
    return grouped


def _align_losses(frames: dict[str, pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
    """Kwadraty bledow na wspolnych originach (przeciecie dat wszystkich modeli)."""
    models = sorted(frames)
    if len(models) < 2:
        raise ValueError("mcs wymaga >= 2 modeli na zbiorze")
    common = None
    for m in models:
        f = frames[m]
        idx = f.index if len(f) else pd.DatetimeIndex([])
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) < 10:
        raise ValueError(f"za malo wspolnych originow ({0 if common is None else len(common)})")
    losses = np.column_stack(
        [(frames[m]["y_true"] - frames[m]["y_pred"]).reindex(common).to_numpy() ** 2 for m in models]
    )
    return losses, models


def main() -> int:
    grouped = _load_grouped()
    results: dict[str, dict] = {}
    lines = ["# SGH — Model Confidence Set (wspolny, po scaleniu CPU/GPU)", ""]
    for (stem, h), frames in sorted(grouped.items()):
        try:
            losses, models = _align_losses(frames)
            res = mcs({m: losses[:, i] for i, m in enumerate(models)}, alpha=0.10, n_bootstrap=1000, h=h, seed=42)
            conf = res["confidence_set"]
            elim = [m for m in models if m not in conf]
            results.setdefault(stem, {})[str(h)] = {
                "models": models,
                "mcs_set": conf,
                "eliminated": elim,
                "elimination_p": res["elimination_p"],
                "n_origins": int(losses.shape[0]),
            }
            lines.append(
                f"**{stem} h={h}** ({losses.shape[0]} originow): w zestawie {sorted(conf)}"
                + (f"; wyeliminowane: {sorted(elim)}" if elim else "")
            )
        except Exception as e:  # noqa: BLE001
            results.setdefault(stem, {})[str(h)] = {"error": str(e)[:200]}
            lines.append(f"**{stem} h={h}**: n/a ({e})")
    (OUT / "sgh_mcs.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "sgh_mcs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MCS: {len(grouped)} grup (stem,h) -> results/sgh_mcs.json/.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
