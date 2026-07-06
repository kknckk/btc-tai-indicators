import pytest
import pandas as pd
import os
import sys

# Dodaj główny katalog do PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analytical_pipeline')))

from utils import load_merged_data

def test_load_merged_data():
    """Prosty test sprawdzający czy utils.load_merged_data zwraca obiekt DataFrame."""
    try:
        df = load_merged_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'time' in df.columns
    except FileNotFoundError:
        pytest.skip("Brak plików danych - test pominięto")
