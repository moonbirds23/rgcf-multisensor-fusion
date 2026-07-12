import math
import unittest

import numpy as np
import torch

from training.uncertainty_metrics import diagonal_gaussian_nll, diagonal_nees, diagonal_uncertainty_metrics


class DiagonalUncertaintyMetricsTests(unittest.TestCase):
    def test_matches_manual_two_dimensional_calculation(self):
        prediction = np.array([[1.0, 2.0]])
        target = np.zeros_like(prediction)
        covariance = np.array([[1.0, 4.0]])

        self.assertTrue(np.allclose(diagonal_nees(prediction, target, covariance), [2.0]))
        expected_nll = 0.5 * (2.0 * math.log(2.0 * math.pi) + math.log(4.0) + 2.0)
        self.assertTrue(np.allclose(diagonal_gaussian_nll(prediction, target, covariance), [expected_nll]))
        summary = diagonal_uncertainty_metrics(prediction, target, covariance)
        self.assertEqual(summary["anees"], 2.0)
        self.assertEqual(summary["coverage"], 1.0)
        self.assertTrue(np.isclose(summary["nll"], expected_nll))

    def test_mask_and_invalid_rows_and_tensor_inputs(self):
        prediction = torch.tensor([[1.0, 0.0], [float("nan"), 1.0], [3.0, 0.0]])
        target = torch.zeros_like(prediction)
        covariance = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]])

        summary = diagonal_uncertainty_metrics(prediction, target, covariance, valid_mask=torch.tensor([True, True, True]))

        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["valid_count"], 1)
        self.assertEqual(summary["invalid_count"], 2)
        self.assertEqual(summary["anees"], 1.0)
        self.assertEqual(summary["coverage"], 1.0)

    def test_empty_selection_returns_nan(self):
        summary = diagonal_uncertainty_metrics(
            np.array([[1.0, 0.0]]),
            np.zeros((1, 2)),
            np.ones((1, 2)),
            valid_mask=np.array([False]),
        )

        self.assertEqual(summary["valid_count"], 0)
        self.assertTrue(math.isnan(summary["anees"]))
        self.assertTrue(math.isnan(summary["coverage"]))
        self.assertTrue(math.isnan(summary["nll"]))
