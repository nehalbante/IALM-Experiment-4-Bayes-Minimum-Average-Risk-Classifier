"""Smoke test: run app.py through Streamlit's AppTest and exercise every page.

This is slower than the pure-math tests; it verifies that each navigation
section renders without raising an exception.
"""
from __future__ import annotations

import os
import unittest

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

NAV = [
    "Experiment Overview",
    "Part 1 – Part 10",
    "Dataset Explorer",
    "Classifier Playground",
    "Model Evaluation",
    "Decision Boundary Visualizer",
    "Perceptron Laboratory",
    "Extension Experiments",
    "Mathematical Reference",
]


class TestAppSmoke(unittest.TestCase):
    def test_all_pages_render(self):
        at = AppTest.from_file(APP_PATH, default_timeout=120)
        at.run()
        self.assertEqual(len(at.exception), 0, f"Initial render failed: {at.exception}")
        for page in NAV:
            at.sidebar.radio[0].set_value(page).run()
            self.assertEqual(len(at.exception), 0, f"Page '{page}' failed: {at.exception}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
