"""Unit tests for the core mathematical functions (unittest / pytest compatible).

Run from the ``bayes_risk_analyzer`` folder with either:

    python -m unittest discover tests
    python -m pytest tests
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import bayes, config, data_loader, extensions, lda, metrics  # noqa: E402


class TestThresholdAndBias(unittest.TestCase):
    def test_original_threshold(self):
        xi = bayes.compute_threshold(0.2, 0.8, 8, 3, 40, 1)
        self.assertAlmostEqual(xi, 0.25, places=9)

    def test_original_weights_and_bias(self):
        mu1 = np.array([0.6, 0.7, 0.6])
        mu2 = np.array([0.4, 0.3, 0.3])
        Cinv = 25.0 * np.eye(3)
        w = bayes.compute_weights(mu1, mu2, Cinv)
        np.testing.assert_allclose(w, [5.0, 10.0, 7.5], atol=1e-9)
        b_geom = bayes.compute_b_geom(mu1, mu2, Cinv)
        self.assertAlmostEqual(b_geom, -10.875, places=9)
        b = bayes.compute_bias(b_geom, 0.25)
        self.assertAlmostEqual(b, -9.488706, places=5)

    def test_cheaper_misses(self):
        xi = bayes.compute_threshold(0.2, 0.8, 8, 3, 11, 1)
        self.assertAlmostEqual(xi, 2.66666667, places=6)
        b = bayes.compute_bias(-10.875, xi)
        self.assertAlmostEqual(b, -11.855829, places=5)

    def test_screening_prior(self):
        xi = bayes.compute_threshold(0.05, 0.95, 8, 3, 40, 1)
        self.assertAlmostEqual(xi, 1.1875, places=9)
        b = bayes.compute_bias(-10.875, xi)
        self.assertAlmostEqual(b, -11.04685, places=5)

    def test_haykin_reduces_to_geometric(self):
        xi = bayes.compute_threshold(0.5, 0.5, 0, 1, 1, 0)
        self.assertAlmostEqual(xi, 1.0, places=9)
        b = bayes.compute_bias(-10.875, xi)
        self.assertAlmostEqual(b, -10.875, places=9)


class TestValidation(unittest.TestCase):
    def test_bad_priors(self):
        with self.assertRaises(ValueError):
            bayes.compute_threshold(0.3, 0.8, 8, 3, 40, 1)

    def test_zero_prior(self):
        with self.assertRaises(ValueError):
            bayes.compute_threshold(0.0, 1.0, 8, 3, 40, 1)

    def test_nonpositive_regret(self):
        with self.assertRaises(ValueError):
            bayes.compute_threshold(0.2, 0.8, 40, 3, 40, 1)  # c21 - c11 = 0


class TestFittedClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = data_loader.load_train()
        cls.test = data_loader.load_test()
        cls.X, cls.y = data_loader.split_features_labels(cls.train)
        cls.Xt, cls.yt = data_loader.split_features_labels(cls.test)
        cls.clf = bayes.fit_default_classifier(cls.X, cls.y)

    def test_learned_parameters(self):
        self.assertAlmostEqual(self.clf.p1_, 0.2, places=6)
        self.assertAlmostEqual(self.clf.p2_, 0.8, places=6)
        np.testing.assert_allclose(self.clf.mu1_, [0.6, 0.7, 0.6], atol=1e-4)
        np.testing.assert_allclose(self.clf.mu2_, [0.4, 0.3, 0.3], atol=1e-4)
        np.testing.assert_allclose(self.clf.w_, [5.0, 10.0, 7.5], atol=1e-2)
        self.assertAlmostEqual(self.clf.b_geom_, -10.875, places=3)
        self.assertAlmostEqual(self.clf.xi_, 0.25, places=6)
        self.assertAlmostEqual(self.clf.b_, -9.4887, places=3)

    def test_reference_results(self):
        ev = self.clf.evaluate(self.Xt, self.yt)
        self.assertEqual(ev["caught"], 1947)
        self.assertEqual(ev["missed"], 53)
        self.assertEqual(ev["false_alarms"], 1618)
        self.assertEqual(ev["cleared"], 6382)
        self.assertAlmostEqual(ev["miss_rate"], 0.0265, places=4)
        self.assertAlmostEqual(ev["false_alarm_rate"], 0.2023, places=3)
        self.assertAlmostEqual(ev["error_rate"], 0.1671, places=4)
        self.assertAlmostEqual(ev["avg_cost"], 2.8932, places=4)

    def test_theoretical_results(self):
        th = self.clf.theoretical()
        self.assertAlmostEqual(th["d2"], 7.25, places=6)
        self.assertAlmostEqual(th["miss_rate"], 0.0314, places=3)
        self.assertAlmostEqual(th["false_alarm_rate"], 0.2029, places=3)
        self.assertAlmostEqual(th["avg_cost"], 2.9253, places=3)

    def test_sign_convention(self):
        g = self.clf.decision_function(np.array([[0.75, 0.9, 0.8]]))
        self.assertGreater(g[0], 0)
        self.assertEqual(self.clf.predict(np.array([[0.75, 0.9, 0.8]]))[0], 1)


class TestLdaComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = data_loader.load_train()
        cls.X, cls.y = data_loader.split_features_labels(cls.train)
        cls.clf = bayes.fit_default_classifier(cls.X, cls.y)

    def test_lda_relationship(self):
        model = lda.fit_lda(self.X, self.y)
        params = lda.lda_params(model)
        np.testing.assert_allclose(params["w"], [5.0, 10.0, 7.5], atol=1e-1)
        self.assertAlmostEqual(params["b"], -12.2613, places=3)
        self.assertAlmostEqual(params["b"] + np.log(16), self.clf.b_, places=3)


class TestExtensions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = data_loader.load_train()
        cls.test = data_loader.load_test()
        cls.X, cls.y = data_loader.split_features_labels(cls.train)
        cls.Xt, cls.yt = data_loader.split_features_labels(cls.test)
        cls.clf = bayes.fit_default_classifier(cls.X, cls.y)

    def test_extension_cheaper_misses(self):
        r = extensions.extension_cheaper_misses(self.clf, self.Xt, self.yt)
        self.assertAlmostEqual(r["xi"], 2.666667, places=4)
        self.assertAlmostEqual(r["b"], -11.8558, places=3)

    def test_extension_screening(self):
        r = extensions.extension_screening(self.clf, self.Xt, self.yt)
        self.assertAlmostEqual(r["xi"], 1.1875, places=4)
        self.assertAlmostEqual(r["b"], -11.0469, places=3)

    def test_extension_haykin(self):
        r = extensions.extension_haykin(self.clf, self.Xt, self.yt)
        self.assertAlmostEqual(r["xi"], 1.0, places=9)
        self.assertTrue(r["reduces_to_geometric"])

    def test_extension_feature_importance(self):
        r = extensions.extension_feature_importance(self.clf)
        np.testing.assert_allclose(r["w_new"], [10.0, 10.0, 7.5], atol=1e-2)

    def test_extension_correlated(self):
        r = extensions.extension_correlated(self.clf)
        expected_cinv = np.array([[33.333, 0, -16.667], [0, 25, 0], [-16.667, 0, 33.333]])
        np.testing.assert_allclose(r["Cinv_modified"], expected_cinv, atol=1e-2)
        np.testing.assert_allclose(r["w_new"], [1.667, 10.0, 6.667], atol=1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

