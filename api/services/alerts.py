import os
import json
import requests
from datetime import datetime

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "analytical_pipeline", "results"))
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")

def get_latest_ewi():
    filepath = os.path.join(RESULTS_DIR, "paper10_ewi_v2.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f).get("paper10_ewi_v2", {})
            if "time_series" in data and len(data["time_series"]) > 0:
                return data["time_series"][-1], data.get("best_model", "Unknown")
    return None, None

def get_latest_strategy():
    filepath = os.path.join(RESULTS_DIR, "strategy_results_v2.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f).get("paper4_v2", {}).get("fixed_literature", {})
            if "equity_curve" in data and len(data["equity_curve"]) > 0:
                return data["equity_curve"][-1]
    return None

def check_and_send_alerts():
    if not WEBHOOK_URL:
        print("WEBHOOK_URL nie jest ustawiony. Przesyłanie powiadomień pominięte.")
        return

    ewi_latest, ewi_model = get_latest_ewi()
    strat_latest = get_latest_strategy()

    messages = []
    
    if ewi_latest and ewi_latest.get("alert_prec25"):
        prob = ewi_latest.get("ewi", 0) * 100
        msg = (
            f"🚨 **UWAGA: EWI ALERT (Zmienność)** 🚨\n"
            f"> System wykrył **{prob:.1f}%** prawdopodobieństwa na drastyczny wzrost zmienności.\n"
            f"> Data wyzwolenia: `{ewi_latest.get('time')}`\n"
            f"> Zmienność RV(30d): `{ewi_latest.get('rv_30d'):.2f}`\n"
            f"> Model OOS: `{ewi_model}`"
        )
        messages.append(msg)
        
    if strat_latest:
        position = "Long" if strat_latest.get("position") == 1 else "Flat"
        if position == "Long":
            msg = (
                f"📈 **Sygnał Strategii On-Chain (MVRV): LONG** 📈\n"
                f"> System znajduje się w strefie opłacalności zakupu.\n"
                f"> Wskaźnik MVRV: `{strat_latest.get('MVRV_Z', 0):.2f}`\n"
                f"> Obecny kapitał (Equity): `{strat_latest.get('strategy_value', 1):.2f}x`\n"
                f"> Ostatnia aktualizacja: `{strat_latest.get('time')}`"
            )
            messages.append(msg)
            
    if not messages:
        print("Brak aktywnych alertów (lub strategia Flat / EWI bezpieczne) do wysłania na webhooka.")
        return
        
    final_message = "\n\n---\n\n".join(messages)
    
    payload = {
        "content": final_message,
        "username": "BTC On-Chain Bot"
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code in [200, 204]:
            print("Webhook wysłany pomyślnie.")
        else:
            print(f"Błąd wysyłania webhooka: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Wyjątek podczas wysyłania webhooka: {e}")

if __name__ == "__main__":
    check_and_send_alerts()
