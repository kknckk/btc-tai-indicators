#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=== Rozpoczęcie lokalnego pipeline'u analitycznego ==="

# Jeśli używasz venv, odkomentuj:
# source venv/bin/activate

echo "1. Obliczanie pochodnych wskaźników (Log-returns, RV, NetFlows)..."
python3 compute_derived_metrics.py

echo "2. Uruchamianie modeli ekonometrycznych (GARCH, PCA, ARDL)..."
python3 run_econometrics.py
python3 run_paper3_ardl_v2.py
python3 run_paper1_garch_v2.py



echo "3. Uruchamianie symulacji strategii (MVRV_Z, NUPL)..."
python3 run_strategies.py
python3 run_strategies_v2.py


echo "4. Uruchamianie modeli predykcyjnych (ML / Prophet / EWI / Deep Learning / Classification)..."
python3 run_ml_dl.py
python3 run_paper10_ewi_v2.py
python3 run_paper6_dl_v2.py
python3 run_paper7_ml_v2.py




echo "=== Zakończono pomyślnie! Wyniki w folderze 'results/' ==="
