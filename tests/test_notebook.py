import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_data_loading.ipynb"


class TestNotebookValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]

    def test_every_code_cell_is_valid_python(self):
        for index, source in enumerate(self.code_cells):
            with self.subTest(cell=index):
                compile(source, f"notebook-cell-{index}", "exec")

    def test_vectorizer_is_fitted_after_train_test_split(self):
        training_cell = next(
            source for source in self.code_cells
            if "vectorizer.fit_transform(X_train_text)" in source
        )
        self.assertLess(
            training_cell.index("train_test_split("),
            training_cell.index("vectorizer.fit_transform(X_train_text)"),
        )
        self.assertIn("vectorizer.transform(X_test_text)", training_cell)
        self.assertNotIn('fit_transform(df["clean_text"])', "\n".join(self.code_cells))

    def test_prediction_description_does_not_claim_fact_verification(self):
        source = "\n".join(self.code_cells)
        self.assertIn("def describe_model_output", source)
        self.assertIn("does not verify claims", source)
        self.assertNotIn("likely FAKE because it contains emotionally", source)
        self.assertNotIn("likely REAL because it uses factual language", source)

    def test_committed_notebook_has_no_stale_execution_outputs(self):
        for cell in self.notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])


if __name__ == "__main__":
    unittest.main()
