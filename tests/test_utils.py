import unittest
import pandas as pd
import os
import sys

# Dodaj główny katalog oraz analytical_pipeline do PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analytical_pipeline')))

from utils import load_merged_data


class TestUtils(unittest.TestCase):
    def test_load_merged_data(self):
        """Test sprawdzający czy utils.load_merged_data zwraca obiekt DataFrame ze standardowymi kolumnami."""
        try:
            df = load_merged_data()
            self.assertIsInstance(df, pd.DataFrame, "Zwrócony obiekt musi być instancją pandas.DataFrame")
            self.assertGreater(len(df), 0, "DataFrame nie może być pusty")
            self.assertIn('time', df.columns, "Kolumna 'time' musi znajdować się w DataFrame")
            self.assertIn('PriceUSD', df.columns, "Kolumna 'PriceUSD' musi znajdować się w DataFrame")
        except FileNotFoundError:
            self.skipTest("Brak plików danych w lokalizacji - test pominięto")


if __name__ == '__main__':
    unittest.main()
