-- Schemat BigQuery dla prognoz TSFM.
--
-- Brak hardkodowanego project ID: uruchamiaj z projektem z env, np.
--   bq --project_id="$GCP_PROJECT_ID" query --use_legacy_sql=false < gcp/bigquery_schema.sql
-- Dataset można zmienić przez podmianę `btc_indicators` (BQ_DATASET w jobie).
--
-- Tabele:
--   predictions – prognozy punktowe i kwantylowe (p10/p50/p90) per krok,
--   model_runs  – metadane uruchomień (status, artefakt, metryki, commit),
--   predictions_latest – widok z najnowszym runem per model/as_of/step.

CREATE SCHEMA IF NOT EXISTS `btc_indicators`
OPTIONS (location = 'EU', description = 'BTC on-chain indicators + prognozy TSFM');

-- ---------------------------------------------------------------------------
-- Prognozy: jeden wiersz = jeden krok horyzontu jednego runu.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `btc_indicators.predictions` (
  run_id        STRING    NOT NULL OPTIONS (description = 'ID runu, np. infer-20260524T030000Z'),
  model         STRING    NOT NULL OPTIONS (description = 'stub | chronos | chronos-bolt | timesfm | moirai'),
  mode          STRING    NOT NULL OPTIONS (description = 'artifact (z artefaktu) | stub (fallback)'),
  as_of         DATE      NOT NULL OPTIONS (description = 'Ostatnia obserwacja użyta w kontekście'),
  forecast_date DATE      NOT NULL OPTIONS (description = 'Data prognozowana'),
  step          INT64     NOT NULL OPTIONS (description = 'Krok horyzontu: 1..h'),
  point         FLOAT64   OPTIONS (description = 'Prognoza punktowa (mediana p50)'),
  p10           FLOAT64   OPTIONS (description = 'Kwantyl 10%'),
  p50           FLOAT64   OPTIONS (description = 'Mediana'),
  p90           FLOAT64   OPTIONS (description = 'Kwantyl 90%'),
  last_price    FLOAT64   OPTIONS (description = 'Cena w dniu as_of'),
  created_at    TIMESTAMP NOT NULL OPTIONS (description = 'UTC czas zapisu')
)
PARTITION BY as_of
CLUSTER BY model, step
OPTIONS (
  description = 'Prognozy cen BTC (zero-shot TSFM lub jawny stub)'
);

-- ---------------------------------------------------------------------------
-- Metadane runów: audyt, monitoring dryfu, porównania modeli.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `btc_indicators.model_runs` (
  run_id       STRING    NOT NULL OPTIONS (description = 'ID runu (klucz idempotencji)'),
  model        STRING    NOT NULL,
  mode         STRING    NOT NULL OPTIONS (description = 'artifact | stub'),
  artifact_uri STRING    OPTIONS (description = 'gs://... artefaktu modelu (jeśli użyty)'),
  horizon      INT64,
  as_of        DATE,
  n_history    INT64     OPTIONS (description = 'Liczba obserwacji w kontekście'),
  metrics_json STRING    OPTIONS (description = 'JSON: metryki offline (RMSE/MAE/DirAcc) lub powód fallbacku'),
  status       STRING    OPTIONS (description = 'ok | failed'),
  error        STRING    OPTIONS (description = 'Komunikat błędu przy status=failed'),
  git_commit   STRING    OPTIONS (description = 'Commit obrazu/kodu (CI)'),
  started_at   TIMESTAMP,
  finished_at  TIMESTAMP
)
PARTITION BY DATE(started_at)
CLUSTER BY model, status
OPTIONS (
  description = 'Rejestr uruchomień treningu/inferencji TSFM'
);

-- ---------------------------------------------------------------------------
-- Widok pomocniczy: najnowszy run per model/as_of/step.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW `btc_indicators.predictions_latest` AS
SELECT * EXCEPT (rn)
FROM (
  SELECT
    p.*,
    ROW_NUMBER() OVER (
      PARTITION BY p.model, p.as_of, p.step
      ORDER BY p.created_at DESC
    ) AS rn
  FROM `btc_indicators.predictions` AS p
)
WHERE rn = 1;

-- ---------------------------------------------------------------------------
-- Przykładowe zapytania (do weryfikacji po wdrożeniu)
-- ---------------------------------------------------------------------------
-- Ostatnie prognozy:
-- SELECT model, as_of, step, forecast_date, point, p10, p90
-- FROM `btc_indicators.predictions_latest`
-- ORDER BY as_of DESC, model, step
-- LIMIT 100;
--
-- Realizacja prognoz (po naliczeniu cen rzeczywistych):
-- SELECT p.model, p.as_of, p.step,
--        SAFE_DIVIDE(p.point - c.PriceUSD, c.PriceUSD) * 100 AS err_pct
-- FROM `btc_indicators.predictions_latest` p
-- JOIN `btc_indicators.coinmetrics_daily` c ON DATE(c.time) = p.forecast_date
-- WHERE p.as_of >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY);
