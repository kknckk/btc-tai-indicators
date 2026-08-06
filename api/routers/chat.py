import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["chat"],
)

class ChatRequest(BaseModel):
    message: str

def get_market_context():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "analytical_pipeline", "results"))
    
    ewi_path = os.path.join(base_dir, "paper10_ewi_v2.json")
    strat_path = os.path.join(base_dir, "strategy_results_v2.json")
    
    context = []
    
    # EWI
    if os.path.exists(ewi_path):
        with open(ewi_path, "r") as f:
            data = json.load(f).get("paper10_ewi_v2", {})
            if "time_series" in data and len(data["time_series"]) > 0:
                latest = data["time_series"][-1]
                prob = latest.get("ewi", 0) * 100
                context.append(f"Early Warning Indicator (EWI) na dzień {latest.get('time')}: prawdopodobieństwo podwyższonej zmienności wynosi {prob:.1f}%. (RV 30d: {latest.get('rv_30d', 0):.2f})")
                
    # Strategy
    if os.path.exists(strat_path):
        with open(strat_path, "r") as f:
            data = json.load(f).get("paper4_v2", {}).get("fixed_literature", {})
            if "equity_curve" in data and len(data["equity_curve"]) > 0:
                latest = data["equity_curve"][-1]
                pos = "LONG" if latest.get("position") == 1 else "FLAT (Cash)"
                mvrv = latest.get("MVRV_Z", 0)
                context.append(f"Strategia MVRV na dzień {latest.get('time')}: Pozycja = {pos}. Aktualny MVRV Z-Score = {mvrv:.2f}. Kapitał (Equity) = {latest.get('strategy_value', 1):.2f}x.")

    return " ".join(context) if context else "Brak zaktualizowanych danych w systemie lokalnym."

@router.post("/")
def chat_with_data(req: ChatRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"response": "⚠️ Klucz `GEMINI_API_KEY` nie został skonfigurowany w środowisku. \n\n*Przykładowa odpowiedź Mock:*\nZ danych systemowych wynika, że rynek znajduje się obecnie w fazie umiarkowanego ryzyka. Wskaźnik EWI jest niski, a wycena MVRV pozostaje na bezpiecznych poziomach."}
        
    try:
        client = genai.Client(api_key=api_key)
        
        system_instruction = (
            "Jesteś zaawansowanym analitykiem ilościowym rynku kryptowalut (Quant Analyst AI). "
            "Twoim celem jest pomoc w interpretacji skomplikowanych danych pochodzących z modeli maszynowego uczenia "
            "i metryk on-chain, takich jak MVRV Z-Score, NUPL, "
            "oraz systemu ostrzegania przed zmiennością (EWI). "
            "Odpowiadaj po polsku, profesjonalnie, zwięźle i na temat. Używaj formatowania Markdown (pogrubienia, listy). "
            f"\n\nBieżący kontekst rynkowy z systemów produkcyjnych (MUSISZ go wziąć pod uwagę, jeśli użytkownik pyta o obecną sytuację): {get_market_context()}"
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=req.message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            )
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd modelu LLM: {str(e)}")
