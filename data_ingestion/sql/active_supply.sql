-- data_ingestion/sql/active_supply.sql
-- Zapytanie obliczające Podaż Aktywną (Active Supply) w oknach 30d, 180d, 1yr
-- Liczy sumę wartości wyjść (outputs) utworzonych w danym oknie, które nie zostały wydane
-- do dnia @target_date.

WITH unspent_outputs AS (
  SELECT
    o.transaction_hash,
    o.index,
    o.value,
    DATE(o.block_timestamp) AS creation_date
  FROM
    `bigquery-public-data.crypto_bitcoin.outputs` AS o
  LEFT JOIN
    `bigquery-public-data.crypto_bitcoin.inputs` AS i
  ON
    o.transaction_hash = i.spent_transaction_hash
    AND o.index = i.spent_output_index
    AND DATE(i.block_timestamp) <= @target_date
  WHERE
    DATE(o.block_timestamp) <= @target_date
    AND i.spent_transaction_hash IS NULL -- Gwarantuje, że nie zostało to wydane
)

SELECT
  @target_date AS date,
  SUM(CASE WHEN DATE_DIFF(@target_date, creation_date, DAY) <= 30 THEN value ELSE 0 END) / 100000000.0 AS SplyAct30d,
  SUM(CASE WHEN DATE_DIFF(@target_date, creation_date, DAY) <= 180 THEN value ELSE 0 END) / 100000000.0 AS SplyAct180d,
  SUM(CASE WHEN DATE_DIFF(@target_date, creation_date, DAY) <= 365 THEN value ELSE 0 END) / 100000000.0 AS SplyAct1yr
FROM
  unspent_outputs
