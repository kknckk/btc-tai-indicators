import os
import json
from fastapi import APIRouter, HTTPException

router = APIRouter()

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "analytical_pipeline", "results"))

@router.get("/api/v1/research/strategy/status")
def get_strategy_status():
    file_path = os.path.join(RESULTS_DIR, "strategy_results_v2.json")
    if not os.path.exists(file_path):
        return {"status": "unavailable", "message": "Plik z wynikami nie istnieje."}
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Extract fixed_literature strategy
        strat_data = data.get("paper4_v2", {}).get("fixed_literature", {})
        equity_curve = strat_data.get("equity_curve", [])
        
        if not equity_curve:
            return {"status": "unavailable", "message": "Brak krzywej kapitału."}
            
        last_day = equity_curve[-1]
        
        return {
            "status": "ok",
            "time": last_day.get("time"),
            "position": "Long" if last_day.get("position") == 1 else "Flat",
            "equity_value": last_day.get("strategy_value"),
            "benchmark_value": last_day.get("bh_value")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/v1/research/ewi/current")
def get_ewi_current():
    file_path = os.path.join(RESULTS_DIR, "paper10_ewi_v2.json")
    if not os.path.exists(file_path):
        return {"status": "unavailable", "message": "Plik z wynikami EWI nie istnieje."}
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        ewi_data = data.get("paper10_ewi_v2", {})
        time_series = ewi_data.get("time_series", [])
        
        if not time_series:
            return {"status": "unavailable", "message": "Brak szeregu czasowego EWI."}
            
        last_day = time_series[-1]
        
        return {
            "status": "ok",
            "time": last_day.get("time"),
            "ewi_probability": last_day.get("ewi"),
            "alert_triggered": bool(last_day.get("alert_prec25") == 1),
            "quality_score": ewi_data.get("alert_quality_score")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
