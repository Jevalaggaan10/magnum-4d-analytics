import unittest
from unittest.mock import patch

import pandas as pd

import backtest
from predictor_suite import MethodResult


class BacktestRegressionTests(unittest.TestCase):
    def test_baseline_uses_actual_distinct_pick_count(self):
        df = pd.DataFrame({'DrawDate': pd.to_datetime(['2024-01-01', '2025-01-01']),
                           '1stPrizeNo': ['0001', '0001'], '2ndPrizeNo': ['0002', '0002'],
                           '3rdPrizeNo': ['0003', '0003']})
        results = {'short': MethodResult('short', [('0001', 1), ('0001', 0)], '')}
        with patch.object(backtest.ps, 'run_all', return_value=results):
            rows = backtest.run_year(df, 2025, top_n=50)
        short = next(r for r in rows if r.method == 'short')
        self.assertAlmostEqual(short.random_baseline, 1 - (1 - 1 / 10000) ** 3)

    def test_invalid_pick_counts_fail_explicitly(self):
        with self.assertRaises(ValueError):
            backtest._random_baseline_rate(10001, 1)


if __name__ == '__main__':
    unittest.main()
