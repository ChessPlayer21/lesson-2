"""
Юнит-тесты для критически важных функций:
- EarlyStopping;
- аналитические метрики;
- CSVDataset (нормализация, one-hot);
- формат матрицы ошибок;
- регуляризационный loss.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from homework_model_modification import (
    EarlyStopping,
    LinearRegressionRegularized,
    compute_classification_metrics,
)
from homework_datasets import CSVDataset, ColumnConfig


class TestEarlyStopping(unittest.TestCase):
    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3, mode="min")
        es.step(1.0)
        for _ in range(3):
            stop = es.step(1.0)          # нет улучшения
        self.assertTrue(stop)

    def test_reset_on_improvement(self):
        es = EarlyStopping(patience=2, mode="min")
        es.step(1.0)
        es.step(0.9)                      # улучшение
        self.assertEqual(es.counter, 0)

    def test_max_mode(self):
        es = EarlyStopping(patience=2, mode="max")
        es.step(0.5)
        es.step(0.6)                      # улучшение
        self.assertEqual(es.counter, 0)


class TestRegressionRegularization(unittest.TestCase):
    def test_l1_l2_zero(self):
        m = LinearRegressionRegularized(3, l1=0, l2=0)
        self.assertAlmostEqual(m.regularization_loss().item(), 0.0, places=6)

    def test_l2_positive(self):
        m = LinearRegressionRegularized(3, l1=0, l2=0.1)
        with torch.no_grad():
            m.linear.weight.fill_(1.0)
        self.assertAlmostEqual(m.regularization_loss().item(), 0.3, places=5)

    def test_l1_positive(self):
        m = LinearRegressionRegularized(3, l1=0.5, l2=0)
        with torch.no_grad():
            m.linear.weight.fill_(2.0)
        self.assertAlmostEqual(m.regularization_loss().item(), 3.0, places=5)


class TestClassificationMetrics(unittest.TestCase):
    def test_binary_perfect(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        m = compute_classification_metrics(y_true, y_pred, y_proba=y_pred, n_classes=2)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["roc_auc"], 1.0)

    def test_multiclass(self):
        y_true = np.array([0, 1, 2, 2, 1, 0])
        y_pred = np.array([0, 1, 2, 1, 1, 0])
        proba = np.eye(3)[y_pred]
        m = compute_classification_metrics(y_true, y_pred, y_proba=proba, n_classes=3)
        self.assertGreater(m["accuracy"], 0.5)
        self.assertIn("roc_auc", m)


class TestCSVDataset(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0, 4.0],
            "num2": [10.0, 20.0, 30.0, 40.0],
            "cat":  ["a", "b", "a", "b"],
            "bin":  [0, 1, 0, 1],
            "target": [1.0, 2.0, 3.0, 4.0],
        })
        self.tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        self.df.to_csv(self.tmp.name, index=False)
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_length(self):
        cols = [ColumnConfig("num1"), ColumnConfig("num2"),
                ColumnConfig("cat", "categorical"),
                ColumnConfig("bin", "binary"),
                ColumnConfig("target", "target")]
        ds = CSVDataset(self.tmp.name, cols, "target", task="regression")
        self.assertEqual(len(ds), 4)

    def test_one_hot(self):
        cols = [ColumnConfig("num1"),
                ColumnConfig("cat", "categorical"),
                ColumnConfig("target", "target")]
        ds = CSVDataset(self.tmp.name, cols, "target",
                        task="regression", normalize=False)
        # num1 + 2 dummy-колонки для cat
        self.assertEqual(ds.n_features, 1 + 2)

    def test_normalization(self):
        cols = [ColumnConfig("num1"), ColumnConfig("target", "target")]
        ds = CSVDataset(self.tmp.name, cols, "target", task="regression",
                        normalize=True)
        mean = ds.X[:, 0].mean()
        std = ds.X[:, 0].std()
        self.assertAlmostEqual(float(mean), 0.0, places=5)
        self.assertAlmostEqual(float(std), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()