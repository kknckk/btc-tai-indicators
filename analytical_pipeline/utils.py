import os
import pandas as pd

def load_merged_data() -> pd.DataFrame:
    """
    Loads all CSVs from data_ingestion/csv, merges them on 'time',
    and sorts by time. This mirrors the logic in upload_csv_to_bq.py.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_dir = os.path.join(base_dir, "data_ingestion", "csv")
    
    if not os.path.exists(csv_dir):
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
        
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if not csv_files:
        raise ValueError("No CSV files found in directory.")

    first_file = csv_files[0]
    merged_df = pd.read_csv(os.path.join(csv_dir, first_file))
    merged_df['time'] = pd.to_datetime(merged_df['time'])

    for filename in csv_files[1:]:
        df = pd.read_csv(os.path.join(csv_dir, filename))
        df['time'] = pd.to_datetime(df['time'])
        merged_df = pd.merge(merged_df, df, on="time", how="outer")

    merged_df = merged_df.sort_values("time").reset_index(drop=True)
    return merged_df

def save_results(df: pd.DataFrame, filename: str):
    """
    Saves a dataframe to the analytical_pipeline/results directory.
    """
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, filename), index=False)
