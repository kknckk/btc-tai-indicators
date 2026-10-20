# Szczegółowy Plan Implementacji: Modele AI Predykcji Kursów (Akcje, Obligacje, BTC On-Chain)

## 1. Kontekst i Cel

Repozytorium `btc-tai-indicators` (pełna wersja; prywatne `tai-btc-indicators` jest okrojone z śladami AI) zawiera:
- Dashboard Next.js + FastAPI na Cloud Run + BigQuery (serverless GCP).
- 38+ wskaźników on-chain (CoinMetrics, BGeometrics, derived).
- `analytical_pipeline/` z reimplementacją 6 prac (GARCH, ARDL, strategies expanding-window, EWI, DL volatility, ML classification z Boruta) — zero look-ahead bias, walk-forward, koszty transakcyjne, Diebold-Mariano.

Cel: Rozszerzyć analizę o 10 świeżych, obiecujących prac na temat modeli predykcji szeregów czasowych (TSFM, Transformery, multimodal LLM), zreprodukować wyniki na danych BTC/on-chain + equity/bonds, wdrożyć na GCP, raportować agentom kodowania (Cursor, Devin).

## 2. 10 Świeżych i Obiecujących Artykułów (2025-2026)

1. **Saidd (2026)** — *A Controlled Comparison of Deep Learning Architectures for Multi-Horizon Financial Forecasting* (arXiv:2603.16886). 918 eksperymentów: ModernTCN (#1, 75% wins), PatchTST (#2) > LSTM. Crypto/forex/equity, h=4/24h. Architecture > params.
2. **Rahimikia, Ni, Wang (2025)** — *Re(Visiting) Time Series Foundation Models in Finance* (arXiv:2511.18578). Zero-shot TSFM (Chronos, TimesFM) słabe; domain pre-training lepsze; CatBoost często wygrywa ekonomicznie.
3. **Noguer i Alonso & Franklin (2026)** — Pretrained TSFMs (TimeGPT, TimesFM-2.5, Moirai-2.0, Chronos) vs NBEATS/PatchTST na US equities. TSFM dominują ranking, ale gains vs random-walk małe.
4. **Zhao et al. (2026)** — *DualCast* (arXiv:2609.38197). Dual-path LLM (frozen Qwen + financial vocab) + news revision (LoRA + RL). Najlepszy MAPE na long horizons.
5. **ChanFormer** (Nguyen et al., 2026) — Channel-wise Transformer dla multi-asset (crypto+stocks). 29% niższy error vs baselines, Sharpe ~1.1-1.27 w backtest.
6. **Kelly, Kuznetsov, Malamud, Xu (2025/2026)** — *Artificial Intelligence Asset Pricing Models* (NBER WP 33351). Transformer w SDF, cross-asset information sharing. Duże redukcje pricing errors.
7. **Fu (2025)** — *The New Quant* (arXiv:2510.05533). Survey LLM w predykcji i tradingu: sentiment, RAG, agentic systems.
8. **Time-LLM adaptations for Bitcoin** (2025, Knowledge-Based Systems). Structural improvements + non-stationarity handling: 50% lepszy SMAPE na BTC.
9. **TPGR** (2026, Eng. App. AI). Temporal Prior Graph Representation + GNN dla multivariate financial TS. 14-32% niższy MSE.
10. **LLM + Sentiment + DL** (2026, Information Sciences). LLM (Llama/Vicuna/Mistral) zero-shot sentiment + LSTM/Transformer/GAN na meme stocks.

Dodatkowo z istniejącego pipeline: Rafi et al. (volatility DL), Omole & Enke (classification), Näsman (timing), etc.

## 3. Plan Eksperymentów i Reprodukcji

### Faza 0: Setup (1-2 dni)
- Fork/clone `btc-tai-indicators`.
- Środowisko: Python 3.11, venv, requirements (torch, transformers, chronos, timesfm, xgboost, arch, statsmodels, boruta, nixtla, etc.).
- Dane: Istniejące CSV on-chain + yfinance (BTC-USD, SPY, bonds yields via FRED/Treasury), news (FinBERT/LLM).
- Walidacja: Walk-forward / purged CV / expanding window (jak w istniejących skryptach). Embargo, no leakage.

### Faza 1: Baseline'y i istniejące paper'y (2-3 dni)
- Uruchom `analytical_pipeline/run_all.sh`.
- Porównaj z HAR-RV, GARCH, XGBoost, naive.
- Metryki: RMSE/MAE/MAPE, R² OOS, directional accuracy, Sharpe, MaxDD, turnover + costs (10-20 bps).

### Faza 2: Nowe architektury (5-10 dni, GPU)
- **ModernTCN & PatchTST** (z Saidd 2026): Implementacja w PyTorch na BTC + multi-asset. Horyzonty 1/7/30d.
- **TSFM zero-shot + fine-tune**: TimesFM, Chronos (HuggingFace), Moirai. Continual pre-training na on-chain + returns.
- **ChanFormer / iTransformer**: Multi-asset (BTC + alt + equity).
- **DualCast / Time-LLM**: Frozen LLM + financial tokens + news/sentiment.
- **AIPM-style**: Transformer attention na cross-sectional features (on-chain + macro).
- **Graph (TPGR)**: Dynamic graphs z korelacji/on-chain zależności + GNN.
- Ablacje: Z/bez on-chain features, z/bez sentiment, różne window sizes.

### Faza 3: Ekonomiczna ewaluacja i strategie (3-5 dni)
- Backtesty long/short/flat z position sizing, costs, slippage.
- Regime analysis (bull/bear/halving).
- Uncertainty: Quantile, Bayesian, ensembles.
- Diebold-Mariano, Model Confidence Set.

### Oczekiwane wyniki
- ModernTCN/PatchTST: 10-30% niższy RMSE vs LSTM na short horizons; directional ~50-55%.
- Domain-pretrained TSFM: Lepsze Sharpe niż zero-shot, ale często gorsze od XGBoost + dobre features.
- Multimodal (news + on-chain): Lift w volatile periods, ale ryzyko hallucination.
- Strategie expanding-window (jak Paper 4): Redukcja MaxDD vs B&H, pozytywny Sharpe po costs.
- Realistycznie: Małe, niestabilne edge'y; sukces = robust OOS + positive risk-adjusted po costs.

## 4. Złożoności Pamięciowe i Obliczeniowe

- **LSTM/GRU**: O(T * H), niska pamięć, wolny training. ~1-2 GB VRAM dla seq=60, batch=64.
- **PatchTST / ModernTCN**: Patch reduces complexity. ModernTCN ~230k params, efektywny. Training: godziny na A100 dla pełnego walk-forward.
- **TSFM (Chronos/TimesFM 200M-500M)**: Zero-shot inference szybki (CPU/GPU). Fine-tune: 10-50 GB VRAM, godziny-dni. LoRA: 4-10 GPU-hours.
- **ChanFormer / multi-asset**: Wyższa pamięć (cross-channel), ale inference <2s.
- **LLM dual-path (DualCast 8B)**: Frozen backbone + LoRA ~50M params. Inference wolniejszy, ale news conditioning wartościowy.
- **Graph/TPGR**: Graph construction CPU-minutes, training single GPU.
- **GCP**: Vertex AI / Cloud Run Jobs dla batch; BigQuery dla danych; Cloud Storage dla artefaktów. Koszt: ~10-100$ za pełny cykl eksperymentów (zależnie od GPU).
- **Ogólnie**: Preferuj walk-forward na rolling windows aby kontrolować compute. Parallelizacja seeds/folds.

## 5. Wdrożenie do GCP

1. **Dane**: Rozszerz BigQuery dataset `btc_indicators` o derived features, predictions, results JSON.
2. **Training**: Vertex AI Custom Training lub Cloud Run Jobs z GPU (A100/L4). Dockerfile w `analytical_pipeline/`.
3. **Inference API**: Rozszerz FastAPI o endpointy `/predict` (załaduj best models z GCS).
4. **Scheduler**: Cloud Scheduler + Cloud Functions dla daily retrain/inference.
5. **Monitoring**: Cloud Logging + custom metrics (prediction drift, Sharpe live).
6. **Dashboard**: Next.js — dodaj wykresy equity curves, alerty EWI, model rankings.

## 6. Raport dla Agentów Kodowania (Cursor & Devin)

### Instrukcje dla Cursor/Devin:
- **Priorytet**: Reprodukcja + zero leakage. Zawsze fit scalers/Boruta tylko na train.
- **Struktura kodu**:
  - `experiments/` — notebooki/skrypty per paper/model.
  - `models/` — implementacje (ModernTCN.py, patchtst.py, tsfm_wrapper.py).
  - `evaluation/` — walk_forward.py, metrics.py (DM test, economic).
  - `gcp/` — deployment scripts, Dockerfiles.
- **Kod**: Modularny, type hints, logging, config YAML (horyzont, window, seeds).
- **Testy**: Unit dla no-leakage, integration dla pipeline.
- **Repro**: Seed=42, requirements.lock, data versioning (DVC lub GCS).
- **Oczekiwane outputy**: JSON results + plots (equity, residuals) + summary table Markdown.
- **Zadania sekwencyjne**:
  1. Setup env + run existing pipeline.
  2. Implement ModernTCN/PatchTST baseline.
  3. Wrap TSFM (Chronos/TimesFM) zero-shot + fine-tune.
  4. Multimodal extension.
  5. GCP deploy + API.
  6. Dashboard updates.
- **Constraints**: Koszty <50$ na eksperyment, preferuj CPU gdzie możliwe, early stopping, mixed precision.

## 7. Harmonogram i Milestone'y

- Tydzień 1: Setup + baseline'y + 2-3 nowe modele (ModernTCN, PatchTST).
- Tydzień 2: TSFM + multimodal + ablacje.
- Tydzień 3: Backtesty ekonomiczne + GCP deployment.
- Tydzień 4: Raport, dashboard, final review + push results.

## 8. Ryzyka i Mitigacje

- Overfitting / leakage → Rygorystyczny walk-forward + audit jak w AUDIT_V2.md.
- Concept drift → Expanding/rolling windows, regime detection.
- Compute cost → Spot instances, smaller models first, LoRA.
- Reproducibility → Seeds, locked deps, data snapshots.

**Następny krok**: Uruchom istniejący pipeline, zaimplementuj ModernTCN/PatchTST na BTC data, push results.
