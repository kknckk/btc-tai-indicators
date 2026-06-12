-- data_ingestion/sql/cdd.sql
-- Zapytanie obliczające Coin Days Destroyed (CDD) dla konkretnego dnia
-- Przeznaczone do wywoływania z Cloud Functions (parametr @target_date = wczoraj)

SELECT
  DATE(i.block_timestamp) AS date,
  SUM(
    DATE_DIFF(DATE(i.block_timestamp), DATE(o.block_timestamp), DAY) * (o.value / 100000000.0)
  ) AS cdd
FROM
  `bigquery-public-data.crypto_bitcoin.inputs` AS i
JOIN
  `bigquery-public-data.crypto_bitcoin.outputs` AS o
ON
  i.spent_transaction_hash = o.transaction_hash
  AND i.spent_output_index = o.index
WHERE
  DATE(i.block_timestamp) = @target_date
GROUP BY
  date
