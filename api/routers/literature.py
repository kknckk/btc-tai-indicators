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

@router.get("/paper/{paper_id}")
def get_paper_results(paper_id: str):
    # Mapping of paper_id to result file and key
    # Paper 1, 2, 3, 8, 9, 10 are in econometrics (10 is partly PCA but saved there)
    # Paper 4 is in strategies
    # Paper 5, 6, 7 are in ML
    
    mapping = {
        "paper1": ("econometrics_results.json", "paper1"),
        "paper2": ("econometrics_results.json", "paper2"),
        "paper3": ("econometrics_results.json", "paper3"),
        "paper4": ("strategy_results.json", "paper4"),
        "paper5": ("ml_results.json", "paper5"),
        "paper6": ("ml_results.json", "paper6"),
        "paper7": ("ml_results.json", "paper7"),
        "paper8": ("econometrics_results.json", "paper8"),
        "paper9": ("econometrics_results.json", "paper9"),
        "paper10": ("econometrics_results.json", "paper10"),
    }
    
    if paper_id not in mapping:
        raise HTTPException(status_code=404, detail=f"Nieznane ID publikacji: {paper_id}")
        
    filename, key = mapping[paper_id]
    results = load_json_results(filename)
    
    if not results or key not in results:
        raise HTTPException(status_code=404, detail=f"Brak wyników dla {paper_id}.")
        
    return results[key]
