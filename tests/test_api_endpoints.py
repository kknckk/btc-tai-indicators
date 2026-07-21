import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.routers.literature import get_paper_results, get_econometrics_results, get_strategy_results, get_ml_results
from fastapi import HTTPException


class TestApiEndpoints(unittest.TestCase):
    def test_all_10_paper_endpoints_return_data(self):
        """Test sprawdzający, czy wywołanie get_paper_results dla paper1..paper10 zwraca poprawne dane bez rzucania wyjątków."""
        for i in range(1, 11):
            paper_id = f"paper{i}"
            with self.subTest(paper_id=paper_id):
                res = get_paper_results(paper_id)
                self.assertIsInstance(res, dict, f"{paper_id} powinien zwrócić słownik")
                self.assertGreater(len(res), 0, f"{paper_id} nie powinien być pustym słownikiem")

    def test_invalid_paper_id_raises_404(self):
        """Test sprawdzający, czy wywołanie nieznanego ID (np. paper99) zwraca błąd HTTP 404."""
        with self.assertRaises(HTTPException) as ctx:
            get_paper_results("paper99")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("Nieznane ID publikacji", ctx.exception.detail)

    def test_general_endpoints(self):
        """Test sprawdzający ogólne endpointy grupowe /econometrics, /strategies oraz /ml."""
        econ = get_econometrics_results()
        self.assertIsInstance(econ, dict)
        self.assertIn("paper1", econ)

        strat = get_strategy_results()
        self.assertIsInstance(strat, dict)
        self.assertIn("paper4", strat)

        ml = get_ml_results()
        self.assertIsInstance(ml, dict)
        self.assertIn("paper5", ml)


if __name__ == '__main__':
    unittest.main()
