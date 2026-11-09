"""Unit tests for the statistics toolkit."""

import unittest
from pathlib import Path

from stats_toolkit import (
    mean,
    median,
    mode,
    variance,
    std_dev,
    quartiles,
    iqr_outliers,
    load_column
)


class TestCentralTendency(unittest.TestCase):

    def test_mean_of_known_values(self):
        self.assertEqual(mean([10, 20, 30]), 20)

    def test_median_even_and_odd_lengths(self):
        self.assertEqual(median([10, 20, 30]), 20)
        self.assertEqual(median([10, 20, 30, 40]), 25)

    def test_mode_with_ties(self):
        self.assertEqual(mode([2, 2, 3, 3, 4]), [2, 3])

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            mean([])

        with self.assertRaises(ValueError):
            median([])

        with self.assertRaises(ValueError):
            mode([])


class TestSpread(unittest.TestCase):

    def test_sample_vs_population_variance(self):
        values = [2, 4, 6, 8]

        self.assertAlmostEqual(
            variance(values, sample=True),
            6.6666666667
        )

        self.assertAlmostEqual(
            variance(values, sample=False),
            5.0
        )

    def test_std_dev_single_value(self):
        with self.assertRaises(ValueError):
            std_dev([5], sample=True)

        self.assertEqual(
            std_dev([5], sample=False),
            0.0
        )


class TestOutliers(unittest.TestCase):

    def test_no_outliers_in_uniform_data(self):
        result = iqr_outliers([10, 20, 30, 40, 50])

        self.assertEqual(result["outliers"], [])

    def test_detects_obvious_outlier(self):
        result = iqr_outliers(
            [10, 20, 30, 40, 50, 60, 70, 80, 500]
        )

        self.assertEqual(result["outliers"], [500])
        self.assertEqual(result["iqr"], 50)


class TestLoading(unittest.TestCase):

    def test_invalid_and_missing_values_are_rejected(self):
        path = Path("data/agent_deposits.csv")

        values, rejected = load_column(
            path,
            "daily_deposits_kes"
        )

        self.assertEqual(len(values), 35)
        self.assertEqual(len(rejected), 5)

        self.assertIn((10, "-"), rejected)
        self.assertIn((14, "n/a"), rejected)
        self.assertIn((25, "12,40O"), rejected)


if __name__ == "__main__":
    unittest.main()