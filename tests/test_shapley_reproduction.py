from __future__ import annotations

import unittest

import pandas as pd

from recompute_shapley_from_scenarios import decompose_scenario_stats, exact_shapley


class PublicShapleyReproductionTest(unittest.TestCase):
    def test_additive_synthetic_example(self):
        values = {}
        for r in (0, 1):
            for p in (0, 1):
                for h in (0, 1):
                    values[f"A{r}{p}{h}"] = 10.0 + 2.0 * r + 3.0 * p + 5.0 * h
        phi = exact_shapley(values)
        self.assertAlmostEqual(phi["road"], 2.0, places=12)
        self.assertAlmostEqual(phi["population"], 3.0, places=12)
        self.assertAlmostEqual(phi["hospital"], 5.0, places=12)

    def test_interaction_efficiency(self):
        rows = []
        for r in (0, 1):
            for p in (0, 1):
                for h in (0, 1):
                    code = f"A{r}{p}{h}"
                    base = 1.0 + 2.0 * r + 3.0 * p + 5.0 * h + 7.0 * r * p + 11.0 * r * h + 13.0 * p * h + 17.0 * r * p * h
                    rows.append(
                        {
                            "scenario": code,
                            "road_year": 2014 if r == 0 else 2024,
                            "population_year": 2014 if p == 0 else 2024,
                            "hospital_year": 2014 if h == 0 else 2024,
                            "profile": "test",
                            "service_scope": "test",
                            "pop_median": base,
                            "pop_gini": 100.0 - base,
                            "pop_theil": 200.0 - 2.0 * base,
                            "pop_atkinson_05": 300.0 - 3.0 * base,
                        }
                    )
        out = decompose_scenario_stats(pd.DataFrame(rows))
        for _, sub in out.groupby("outcome"):
            total = float(sub["total_change"].iloc[0])
            self.assertAlmostEqual(float(sub["contribution_abs"].sum()), total, places=12)
            self.assertAlmostEqual(float(sub["share_pct"].sum()), 100.0, places=10)


if __name__ == "__main__":
    unittest.main()
