import sys
import unittest
import subprocess
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from consensus import build_consensus
from predictor_suite import MethodResult, _all_numbers_iter, method_arima_digit_sum


class PredictorRegressionTests(unittest.TestCase):
    def test_invalid_and_missing_numbers_never_become_candidates(self):
        df = pd.DataFrame({'1stPrizeNo': [None, float('nan'), '12', '1234.0', 'bad', '0000']})
        self.assertEqual(list(_all_numbers_iter(df)), ['0012', '1234', '0000'])

    def test_missing_arima_column_returns_well_formed_result(self):
        result = method_arima_digit_sum(pd.DataFrame({'DrawDate': []}))
        self.assertEqual(result.top, [])
        self.assertIsInstance(result.explanation, str)

    def test_one_method_cannot_vote_twice_for_same_number(self):
        results = {'frequency': MethodResult('frequency', [('1234', 2), ('1234', 1), ('bad', 3)], '')}
        result = build_consensus(results)
        self.assertEqual(result['number'].tolist(), ['1234'])
        self.assertEqual(result.iloc[0]['agreement_count'], 1)

    def test_consensus_ties_are_reproducible(self):
        a = MethodResult('a', [('5678', 1)], '')
        b = MethodResult('b', [('1234', 1)], '')
        self.assertEqual(build_consensus({'a': a, 'b': b})['number'].tolist(),
                         build_consensus({'b': b, 'a': a})['number'].tolist())

    def test_statistical_import_does_not_load_heavy_ml_runtimes(self):
        result = subprocess.run([sys.executable, '-c',
                                 "import train_model,sys; assert 'tensorflow' not in sys.modules; assert 'xgboost' not in sys.modules"],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_lstm_preparation_includes_the_final_observed_target(self):
        from train_model import LSTMModel
        model = LSTMModel(sequence_length=4)
        df = pd.DataFrame({'DrawDate': pd.to_datetime(['2026-01-01', '2026-01-02']),
                           '1stPrizeNo': ['1234', '5678']})
        features, targets = model._prepare_sequences(df)
        self.assertEqual(features.shape, (4, 4, 10))
        self.assertEqual(targets.tolist(), [5, 6, 7, 8])


if __name__ == '__main__':
    unittest.main()
