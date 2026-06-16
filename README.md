# BTC TAI Indicators (On-Chain Dashboard & API)

## Project Context for Future AI Agents ("Vibe Coding")
**Dear AI Agent:** This repository contains an on-chain analytics dashboard and REST API for Bitcoin network indicators. The goal of this project is to serve historical and live metrics (such as Realized Cap, SOPR, MVRV Z-Score, Active Supply, CDD, Median Fees, etc.) using a serverless Google Cloud Platform (GCP) architecture without the need for hosting a fully synced 600GB+ Bitcoin node.

### Scope & Architecture
- **Data Ingestion (`/data_ingestion`)**: Python scripts designed to fetch historical and daily data. We rely on free sources like the CoinMetrics Community CSV dataset (`btc.csv`) and `mempool.space` API. We plan to store everything in **Google BigQuery**.
- **REST API (`/api`)**: A Python **FastAPI** application designed to be deployed on **Google Cloud Run**. It reads directly from BigQuery using parameterized queries to serve historical time-series data to the frontend.
- **Frontend Dashboard (`/dashboard`)**: A **Next.js** (App Router, Tailwind CSS, TypeScript) application. It features premium, dark-mode UI with **TradingView Lightweight Charts** for plotting metrics like MVRV Z-Score.

### What Has Been Done So Far
1. **Repository Structure**: Defined the monorepo structure with `/api`, `/dashboard`, and `/data_ingestion`.
2. **Data Ingestion Skeltons**: Created `coinmetrics_fetcher.py` to parse and push historical CSV data into BigQuery. Created `mempool_fetcher.py` outline for daily fee fetching.
3. **API Implementation**: Built a FastAPI server (`main.py`) with a complete BigQuery client integration. It exposes `/api/v1/metrics/available` and `/api/v1/metrics/values`. Dockerfile provided for Cloud Run deployment.
4. **Dashboard Implementation**: Bootstrapped Next.js. Created a `Chart.tsx` component using `lightweight-charts`. Built an `ApiTester.tsx` component to make REST calls visually from the browser. The main page is styled using Tailwind CSS and `lucide-react` icons.

### What Needs To Be Done Next (Action Plan for You)
1. **Execute Data Ingestion**: The scripts in `data_ingestion` need to be actually executed on a machine or GCP Cloud Function against a real GCP project with BigQuery enabled.
2. **GCP Project Setup**: Provision BigQuery datasets (`btc_indicators`), Cloud Run services, and Cloud Scheduler for daily syncs. Currently, these scripts use placeholder `GCP_PROJECT_ID`.
3. **Advanced Metrics (Group 3 & 4 Disclaimers)**: 
   - **Exchange Flows (Group 4)**: Computing napływy/odpływy z giełd (Exchange In/Out) za pomocą BigQuery jest niemożliwe bez zewnętrznych **Etykiet Adresów** (Address Labels). Jednakże z powodzeniem wyciągnęliśmy te wskaźniki (`FlowInExUSD`, `FlowOutExUSD`) korzystając z darmowej paczki `btc.csv` od **CoinMetrics**.
   - **LTH/STH Split & UTXO (Group 3)**: Samodzielne odtworzenie wieku UTXO dla całego blockchaina z BigQuery to gigantyczne i kosztowne wyzwanie inżynieryjne. Na szczęście omijamy ten problem korzystając z darmowego API **BGeometrics**, z którego ściągamy wskaźniki `LTH_MVRV`, `STH_MVRV`, `SOPR` oraz `CDD`.
4. **Data Sources Pivot & Derived Metrics**: Z uwagi na fakt, że CoinMetrics usunęło ze swojego darmowego API wiele podstawowych wskaźników (np. `TxMeanByte`, `DiffMean`, `FeeMedUSD`), nasz pipeline został zmuszony do wyliczania braków we własnym zakresie:
   - `bq_easy_metrics.py` odpytuje bezpośrednio BigQuery dla darmowych/prostych wskaźników (oraz parsuje historyczną trudność `DiffMean` z hexadecymalnego pola `bits` w blokach).
   - `coinmetrics_fetcher.py` pełni rolę workera pobierającego to co zostało darmowe w CoinMetrics (np. Exchange Flows, kapitalizacje).
   - `bgeometrics_fetcher.py` zasila nas zaawansowanymi wskaźnikami zachowania inwestorów (LTH/STH).
   - `compute_derived_metrics.py` na koniec przetwarza pobrane CSV i **wylicza matematycznie** utracone wskaźniki bez konieczności płacenia (np. "Naiwne NVT" ze stosunku kapitalizacji do transferów, średnią wartość transferu przez podzielenie surowego `TxTfrValUSD` przez liczność, czy też wylicza "Naiwną Prędkość"). Skrypt dokonuje też aliasingu `IssTotNtv` do `IssContNtv` (oba oznaczają ciągłą nową inflację bazową BTC).
   Wszystkie te skrypty generują ostatecznie proste pliki `.csv` podawane przez REST API z Cloud Run.
5. **Dashboard Wiring**: Connect the Next.js frontend directly to the deployed Cloud Run API endpoint (currently it defaults to `localhost:8080` in `ApiTester.tsx`).

## Deployment

### Backend (GCP Cloud Run)
```bash
cd api
gcloud run deploy btc-api --source . --region europe-central2 --allow-unauthenticated
```

### Frontend
```bash
cd dashboard
npm run build
npm start
```
