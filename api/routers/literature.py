import os
import json
from fastapi import APIRouter, HTTPException

router = APIRouter(
    prefix="/api/v1/literature",
    tags=["literature"],
)

def load_json_results(filename: str):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filepath = os.path.join(base_dir, "analytical_pipeline", "results", filename)
    
    if not os.path.exists(filepath):
        return None
        
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

@router.get("/paper/{paper_id}")
def get_paper_results(paper_id: str):
    # Mapping of paper_id to result file and key for V2 updates
    mapping = {
        "paper1": ("paper1_garch_v2.json", "paper1_garch_v2"),
        "paper2": ("paper2_netflows_v2.json", "paper2_netflows_v2"),
        "paper3": ("paper3_ardl_v2.json", "paper3_ardl_v2"),
        "paper4": ("strategy_results_v2.json", "paper4_v2"),
        "paper5": ("ml_results.json", "paper5"), # fallback for un-updated papers
        "paper6": ("paper6_dl_v2.json", "paper6_dl_v2"),
        "paper7": ("paper7_classification_v2.json", "paper7_classification_v2"),
        "paper8": ("econometrics_results.json", "paper8"),
        "paper9": ("econometrics_results.json", "paper9"),
        "paper10": ("paper10_ewi_v2.json", "paper10_ewi_v2"),
    }
    
    if paper_id not in mapping:
        raise HTTPException(status_code=404, detail=f"Nieznane ID publikacji: {paper_id}")
        
    filename, key = mapping[paper_id]
    results = load_json_results(filename)
    
    if not results or key not in results:
        # Fallback to older files if v2 is missing
        if paper_id == "paper10":
            old_res = load_json_results("econometrics_results.json")
            if old_res and "paper10" in old_res:
                return old_res["paper10"]
        elif paper_id == "paper4":
            old_res = load_json_results("strategy_results.json")
            if old_res and "paper4" in old_res:
                return old_res["paper4"]
        elif paper_id == "paper3":
            old_res = load_json_results("econometrics_results.json")
            if old_res and "paper3" in old_res:
                return old_res["paper3"]
                
        raise HTTPException(status_code=404, detail=f"Brak wyników dla {paper_id}.")
        
    return results[key]
