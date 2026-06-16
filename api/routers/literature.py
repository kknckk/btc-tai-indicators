import os
import json
from fastapi import APIRouter, HTTPException

router = APIRouter(
    prefix="/api/v1/literature",
    tags=["literature"],
)

def load_json_results(filename: str):
    # Domyślny folder to analytical_pipeline/results
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filepath = os.path.join(base_dir, "analytical_pipeline", "results", filename)
    
    if not os.path.exists(filepath):
        return None
        
    with open(filepath, "r") as f:
        return json.load(f)

@router.get("/econometrics")
def get_econometrics_results():
    results = load_json_results("econometrics_results.json")
    if not results:
        raise HTTPException(status_code=404, detail="Wyniki ekonometryczne nie są jeszcze dostępne.")
    return results

@router.get("/strategies")
def get_strategy_results():
    results = load_json_results("strategy_results.json")
    if not results:
        raise HTTPException(status_code=404, detail="Wyniki strategii nie są jeszcze dostępne.")
    return results

@router.get("/ml")
def get_ml_results():
    results = load_json_results("ml_results.json")
    if not results:
        raise HTTPException(status_code=404, detail="Wyniki modeli ML nie są jeszcze dostępne.")
    return results
