#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=== Rozpoczęcie lokalnego pipeline'u analitycznego ==="

# Jeśli używasz venv, odkomentuj:
# source venv/bin/activate

echo "1. Obliczanie pochodnych wskaźników (Log-returns, RV, NetFlows)..."
python3 compute_derived_metrics.py

echo "2. Uruchamianie modeli ekonometrycznych (GARCH, PCA)..."
python3 run_econometrics.py

echo "3. Uruchamianie symulacji strategii (MVRV_Z, NUPL)..."
python3 run_strategies.py

echo "4. Uruchamianie modeli predykcyjnych (ML / Prophet)..."
python3 run_ml_dl.py

echo "=== Zakończono pomyślnie! Wyniki w folderze 'results/' ==="
