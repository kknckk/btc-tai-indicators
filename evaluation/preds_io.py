"""Zapis ramiek predykcji per-origin dla serii SGH (wejscie dla MCS/DM zbiorczych).

Runnerek po kazdym horyzoncie ma w pamieci `frames[model]` z kolumnami
y_true/y_pred/price — do tej pory byly zapisywane tylko zagregowane metryki.
Ten modul persistuje ramki do results/preds_sgh/<stem>__<model>_h<h>.parquet,
dzieki czemu MCS/porownania parowe mozna liczyc zbiorczo po scaleniu CPU/GPU,
bez ponownego treningu."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_pred_frames(frames: dict[str, pd.DataFrame], dir_path: Path, stem: str, h: int) -> list[str]:
    dir_path.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for m, fr in frames.items():
        try:
            out = fr.copy()
            if "y_true" not in out.columns or "y_pred" not in out.columns:
                continue
            fp = dir_path / f"{stem}__{m}_h{h}.parquet"
            out.to_parquet(fp)
            written.append(fp.name)
        except Exception as e:  # noqa: BLE001 — zapis nie moze zabic runa
            print(f"warn: pred frame {stem}/{m}/h{h} not saved: {e}")
    return written
