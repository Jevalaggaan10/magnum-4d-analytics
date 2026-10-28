import ast
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data_loader import (REPO_ROOT, SLOT_COLS, extract_top3_prizes,
                         load_combined_data, load_data_cache, normalize_prize_number)
from data_sources.base import NormalizedDraw
from data_sources import manager
from feature_engineer import build_last_seen_index, calculate_number_lag_indexed
from user_guess_analyzer import analyze
import randomness_tests


def draw_row(date='20260101', number='0012'):
    return {'DrawDate': date, **{col: number for col in SLOT_COLS}}


class DataAndFeatureRegressionTests(unittest.TestCase):
    def test_prize_normalization_preserves_zero_and_rejects_missing(self):
        for value in [None, float('nan'), pd.NA, '', 'nan', '-1', '12345', '1.5', True]:
            self.assertIsNone(normalize_prize_number(value), repr(value))
        for value, expected in [('12', '0012'), (12.0, '0012'), ('0000', '0000'), ('1234.0', '1234')]:
            self.assertEqual(normalize_prize_number(value), expected)

    def test_all_23_cache_slots_keep_leading_zeros_and_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            row = draw_row()
            row['SpecialNo1'] = ''
            pd.DataFrame([row]).to_csv(Path(directory) / 'damacai.csv', index=False)
            loaded = load_data_cache('damacai', directory)
        self.assertIsNone(loaded.iloc[0]['SpecialNo1'])
        self.assertEqual(loaded.iloc[0]['ConsolationNo10'], '0012')
        self.assertEqual(loaded.iloc[0]['DrawDate'], pd.Timestamp('2026-01-01'))

    def test_one_operator_is_deduplicated_without_pooling_other_series(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data_cache').mkdir()
            pd.DataFrame([draw_row()]).to_csv(root / '4D.txt', index=False)
            pd.DataFrame([draw_row(number='1234')]).to_csv(root / 'data_cache' / 'magnum.csv', index=False)
            pd.DataFrame([draw_row(number='5678')]).to_csv(root / 'data_cache' / 'damacai.csv', index=False)
            (root / 'MDTdec28.xlsx').touch()
            with patch('data_loader.load_mdt_excel') as excel:
                loaded = load_combined_data(root)
            excel.assert_not_called()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.iloc[0]['1stPrizeNo'], '1234')
        self.assertEqual(loaded['Operator'].unique().tolist(), ['magnum'])

    def test_invalid_top_three_rows_are_excluded(self):
        rows = [draw_row(), draw_row('20260102')]
        rows[1]['2ndPrizeNo'] = None
        self.assertEqual(len(extract_top3_prizes(pd.DataFrame(rows))), 1)

    def test_last_seen_is_latest_across_all_prize_columns(self):
        df = pd.DataFrame({'DrawDate': pd.to_datetime(['2026-01-01', '2026-01-10']),
                           '1stPrizeNo': ['1111', '1234'], '2ndPrizeNo': ['2222', '3333'],
                           '3rdPrizeNo': ['1234', '4444']})
        index = build_last_seen_index(df)
        self.assertEqual(calculate_number_lag_indexed(index, '1234', pd.Timestamp('2026-01-11')), 1)

    def test_missing_required_prize_cannot_be_fabricated_as_zero(self):
        with self.assertRaises(ValueError):
            NormalizedDraw('20260101', '', '0001', '0002').to_row()

    def test_sync_counts_malformed_draw_as_failure(self):
        class InvalidSource:
            def fetch_dates(self):
                return ['20260101']
            def fetch_draw(self, date):
                return NormalizedDraw(date, '', '0001', '0002')
        with patch.dict(manager.SOURCES, {'test': InvalidSource}), \
             patch.object(manager, '_load_cache', return_value=pd.DataFrame()), \
             patch.object(manager, '_update_last_sync'):
            report = manager.sync('test', recent=1, verbose=False)
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.fetched, 0)

    def test_legacy_ensemble_handles_an_unavailable_model(self):
        from predict import ensemble_predictions
        self.assertEqual(ensemble_predictions({'missing': [], 'working': ['1234']}), [('1234', 1.0)])

    def test_legacy_sampler_rejects_impossible_unique_sample(self):
        from predict_final import get_digit_probability_numbers
        with self.assertRaises(ValueError):
            get_digit_probability_numbers(pd.DataFrame([draw_row(number='1111')]), n=2)

    def test_guess_baseline_matches_columns_actually_analyzed(self):
        df = pd.DataFrame([draw_row(), draw_row('20260102')])
        df['DrawDate'] = pd.to_datetime(df['DrawDate'], format='%Y%m%d')
        report = analyze('0012', df)
        self.assertAlmostEqual(report.random_baseline, 1 - (1 - 1 / 10000) ** 3)

    def test_short_randomness_battery_does_not_crash(self):
        df = pd.DataFrame([draw_row(), draw_row('20260102')])
        df['DrawDate'] = pd.to_datetime(df['DrawDate'], format='%Y%m%d')
        results = randomness_tests.run_all(df)
        self.assertIn('autocorrelation_ljung_box', results)

    def test_sparse_full_number_test_does_not_report_unreliable_p_value(self):
        result = randomness_tests.chi_square_full_number(pd.DataFrame([draw_row()]))
        self.assertTrue(math.isnan(result['p_value']))


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Streamlit is optional for core regression checks')
class StreamlitCacheRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import streamlit as st
        source = ast.parse((REPO_ROOT / 'app.py').read_text(encoding='utf-8-sig'))
        names = {'run_predictors', 'build_consensus_table'}
        selected = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
        cls.namespace = {'st': st, 'pd': pd}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(REPO_ROOT / 'app.py'), 'exec'), cls.namespace)

    def test_consensus_cache_changes_when_method_results_change(self):
        from predictor_suite import MethodResult
        fn = self.namespace['build_consensus_table']
        fn.clear()
        first = fn({'frequency': MethodResult('frequency', [('1234', 1)], '')}, 50)
        second = fn({'frequency': MethodResult('frequency', [('5678', 1)], '')}, 50)
        self.assertEqual(first['number'].tolist(), ['1234'])
        self.assertEqual(second['number'].tolist(), ['5678'])

    def test_prediction_cache_sees_corrections_with_unchanged_row_count(self):
        fn = self.namespace['run_predictors']
        fn.clear()
        with patch('predictor_suite.run_all', return_value={}) as run:
            fn(pd.DataFrame({'1stPrizeNo': ['1234']}), False, 50)
            fn(pd.DataFrame({'1stPrizeNo': ['5678']}), False, 50)
            self.assertEqual(run.call_count, 2)


if __name__ == '__main__':
    unittest.main()
