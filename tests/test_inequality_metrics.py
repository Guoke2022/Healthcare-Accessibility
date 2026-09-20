from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from utils.inequality_metrics import weighted_gini, theil_index, atkinson_05
from utils.multiscale import theil_city_decomposition


class FormalInequalityMetricsTest(unittest.TestCase):
    def check_three(self, x, expected_gini, expected_theil, expected_atkinson, w=None):
        if w is None:
            w = np.ones(len(x), dtype=float)
        self.assertAlmostEqual(weighted_gini(x, w), expected_gini, places=12)
        self.assertAlmostEqual(theil_index(x, w), expected_theil, places=12)
        self.assertAlmostEqual(atkinson_05(x, w), expected_atkinson, places=12)

    def test_equal_positive(self):
        self.check_three([1, 1], 0.0, 0.0, 0.0)

    def test_one_zero(self):
        self.check_three([0, 1], 0.5, math.log(2.0), 0.5)

    def test_two_zeros(self):
        self.check_three([0, 0, 1], 2.0 / 3.0, math.log(3.0), 2.0 / 3.0)

    def test_all_zero(self):
        self.check_three([0, 0, 0], 0.0, 0.0, 0.0)

    def test_unequal_population_weights(self):
        # 75% of the population has zero accessibility and 25% has accessibility=1.
        self.check_three([0, 1], 0.75, math.log(4.0), 0.75, w=[3.0, 1.0])

    def test_scale_invariance(self):
        a = np.array([0.0, 1.0, 3.0, 10.0])
        b = a * 100.0
        w = np.array([5.0, 2.0, 7.0, 1.0])
        for f in (weighted_gini, theil_index, atkinson_05):
            self.assertAlmostEqual(f(a, w), f(b, w), places=12)

    def test_theil_city_decomposition_with_all_zero_city(self):
        df = pd.DataFrame({
            "acc": [0.0, 0.0, 1.0, 3.0],
            "pop": [5.0, 4.0, 2.0, 1.0],
            "city_name_norm": ["ZeroCity", "ZeroCity", "PositiveCity", "PositiveCity"],
        })
        dec = theil_city_decomposition(df)
        total = theil_index(df["acc"].to_numpy(), df["pop"].to_numpy())
        self.assertTrue(np.isfinite(total))
        self.assertAlmostEqual(
            total,
            dec["pop_theil_within_city"] + dec["pop_theil_between_city"],
            places=12,
        )
        self.assertAlmostEqual(dec["pop_theil_decomp_residual"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
