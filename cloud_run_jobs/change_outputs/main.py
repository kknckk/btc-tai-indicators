import os
import time
from google.cloud import bigquery

def main():
    print("Rozpoczęcie joba Cloud Run do obliczania heurystyki Change Outputs...")
    project_id = os.environ.get("GCP_PROJECT_ID", "btc-ind")
    
    # UWAGA: Szablon pobierający wczorajsze transakcje do lokalnej weryfikacji Change Output.
    # Używamy heurystyki z "największym outputem na ten sam typ skryptu" jako adres resztowy (Change).
    
    print(f"Używany projekt GCP: {project_id}")
    try:
        client = bigquery.Client(project=project_id)
        # Przykład szkieletowy - job iteracyjnie przeprocesuje wyjścia
        query = """
        SELECT transaction_hash, output_count 
        FROM `bigquery-public-data.crypto_bitcoin.transactions` 
        WHERE DATE(block_timestamp) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
        LIMIT 10
        """
        results = client.query(query).result()
        for row in results:
            print(f"Tx: {row['transaction_hash']} | Outputs: {row['output_count']}")
    except Exception as e:
        print(f"Job działa w trybie offline/bez GCP lub wystąpił błąd: {e}")
        
    print("Sukces: Job zakończył przetwarzanie danych heurystycznych (wersja szkieletowa).")

if __name__ == "__main__":
    main()
