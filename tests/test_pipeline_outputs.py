import unittest
import json
import os


class TestPipelineOutputs(unittest.TestCase):
    def setUp(self):
        self.results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analytical_pipeline', 'results'))
        self.econ_file = os.path.join(self.results_dir, 'econometrics_results.json')
        self.strat_file = os.path.join(self.results_dir, 'strategy_results.json')
        self.ml_file = os.path.join(self.results_dir, 'ml_results.json')

    def test_results_files_exist(self):
        """Test sprawdzający, czy wszystkie 3 pliki z wynikami potoku analitycznego istnieją na dysku."""
        self.assertTrue(os.path.exists(self.econ_file), f"Brak pliku {self.econ_file}")
        self.assertTrue(os.path.exists(self.strat_file), f"Brak pliku {self.strat_file}")
        self.assertTrue(os.path.exists(self.ml_file), f"Brak pliku {self.ml_file}")

    def test_econometrics_results_structure(self):
        """Test sprawdzający zawartość i kompletność kluczy ekonometrycznych: paper1, paper2, paper3, paper8, paper9, paper10."""
        with open(self.econ_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertIsInstance(data, dict, "econometrics_results.json musi być słownikiem")
        
        expected_keys = ['paper1', 'paper2', 'paper3', 'paper8', 'paper9', 'paper10']
        for key in expected_keys:
            self.assertIn(key, data, f"Brak klucza '{key}' w econometrics_results.json")
            self.assertIsInstance(data[key], dict, f"Klucz '{key}' w econometrics_results.json nie jest słownikiem")
            self.assertGreater(len(data[key]), 0, f"Słownik '{key}' w econometrics_results.json jest pusty")

        # Sprawdzenie szczegółowe kluczy dla Paper 2 (NetFlows)
        self.assertIn('returns_model', data['paper2'], "Brak sekcji 'returns_model' w paper2")
        self.assertIn('volatility_model', data['paper2'], "Brak sekcji 'volatility_model' w paper2")

        # Sprawdzenie szczegółowe kluczy dla Paper 10 (PCA & EWI)
        self.assertIn('classification_metrics', data['paper10'], "Brak classification_metrics w paper10")
        self.assertIn('accuracy', data['paper10']['classification_metrics'], "Brak accuracy w paper10 classification_metrics")

    def test_strategy_results_structure(self):
        """Test sprawdzający zawartość i kompletność wyników strategii: paper4."""
        with open(self.strat_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertIsInstance(data, dict, "strategy_results.json musi być słownikiem")
        self.assertIn('paper4', data, "Brak klucza 'paper4' w strategy_results.json")
        
        p4 = data['paper4']
        self.assertIn('metrics', p4, "Brak metrics w paper4")
        self.assertIn('strategy', p4['metrics'], "Brak wyliczeń strategii w paper4 metrics")
        self.assertIn('sharpe', p4['metrics']['strategy'], "Brak Sharpe ratio w paper4 strategy")

    def test_ml_results_structure(self):
        """Test sprawdzający zawartość i kompletność modeli ML/DL: paper5, paper6, paper7."""
        with open(self.ml_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertIsInstance(data, dict, "ml_results.json musi być słownikiem")
        
        expected_keys = ['paper5', 'paper6', 'paper7']
        for key in expected_keys:
            self.assertIn(key, data, f"Brak klucza '{key}' w ml_results.json")
            self.assertIsInstance(data[key], dict, f"Klucz '{key}' w ml_results.json nie jest słownikiem")
            self.assertGreater(len(data[key]), 0, f"Słownik '{key}' w ml_results.json jest pusty")

        # Weryfikacja Paper 5 (Prophet)
        self.assertIn('metrics', data['paper5'], "Brak metrics w paper5 (Prophet)")
        self.assertIn('predictions', data['paper5'], "Brak predictions w paper5 (Prophet)")
        self.assertGreater(len(data['paper5']['predictions']), 0, "Lista predykcji Prophet jest pusta")

        # Weryfikacja Paper 7 (Random Forest vs SVM)
        self.assertIn('random_forest', data['paper7'], "Brak random_forest w paper7")
        self.assertIn('feature_importance', data['paper7'], "Brak feature_importance w paper7")


if __name__ == '__main__':
    unittest.main()
