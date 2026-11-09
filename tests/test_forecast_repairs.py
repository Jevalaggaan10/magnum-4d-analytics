"""Regression scenarios from the November 2026 reliability review."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecast_evaluation import promotion_decision, summarize
from probability_tracker import advance_forecasts
from scripts import evaluate_forecasts as runner
from scripts import track_predictions as tracker

ROOT = Path(__file__).resolve().parents[1]


class RepairRegressionTests(unittest.TestCase):
    def test_incompatible_series_cannot_pool_into_successful_promotion(self):
        rows = []
        for i, date in enumerate(pd.date_range('2020-01-01', periods=1000)):
            rows.append({'for_date': date.strftime('%Y%m%d'), 'status': 'verified',
                         'coverage': 1, 'confirmation_version': 'a' if i < 500 else 'b',
                         'settings_sha256': 'a' if i < 500 else 'b', 'metrics': {
                             'champion': {'log_loss': 9.21, 'hit_draw': 0, 'conditional_control': .02},
                             'shadow': {'log_loss': 9., 'hit_draw': 1},
                             'simplest_nonuniform': {'log_loss': 9.1, 'hit_draw': 0}}})
        result = promotion_decision(rows, development_qualified=True,
                                    correctness_verified=True, replicates=20)
        self.assertFalse(result['probability_promotion'])
        self.assertFalse(result['exact_hit_advantage'])
        self.assertNotEqual(result['status'], 'evaluated')

    def test_invalid_frozen_source_blocks_generation_without_writing_artifacts(self):
        state = json.loads((ROOT / 'predictions.json').read_text())
        settings = json.loads((ROOT / 'forecast_config.json').read_text())
        settings['source_hashes']['forecasting.py'] = '0' * 64
        with tempfile.TemporaryDirectory() as directory:
            advance_forecasts(state, tracker.load_corpus(state),
                             now=datetime.fromisoformat('2026-11-12T12:00:00+08:00'),
                             artifact_dir=Path(directory), settings=settings)
            self.assertFalse('20261114' in state['probability_forecasts'])
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(state['forecast_integrity']['status'], 'blocked')

    def test_cli_recovers_published_target_outside_recent_five_results(self):
        state = json.loads((ROOT / 'predictions.json').read_text())
        record = copy.deepcopy(state['probability_forecasts']['20261107'])
        expected_actual = record['actual']
        record['status'] = 'published'
        for key in ['actual', 'metrics', 'checked_at']:
            record.pop(key, None)
        state['probability_forecasts'] = {'20261107': record}
        state['observed_draws'].pop('20261107', None)
        original_forecasts = copy.deepcopy(record['forecasts'])
        recent = {date: expected_actual for date in
                  ['20261114', '20261115', '20261118', '20261121', '20261122']}
        now = datetime.fromisoformat('2026-11-25T22:00:00+08:00')
        with patch.object(tracker, 'datetime', wraps=datetime) as clock, \
                patch.object(tracker, 'load_state', return_value=state), \
                patch.object(tracker, 'fetch_recent_draws', return_value=recent), \
                patch.object(tracker, 'fetch_single_draw', return_value=expected_actual), \
                contextlib.redirect_stdout(io.StringIO()):
            clock.now.return_value = now
            tracker.main(['--dry-run'])
        recovered = state['probability_forecasts']['20261107']
        self.assertEqual(recovered['status'], 'verified')
        self.assertEqual(recovered['actual'], expected_actual)
        self.assertEqual(recovered['forecasts'], original_forecasts)
        self.assertIn('20261107', state['observation_provenance'])

    def test_noisy_development_gain_keeps_uniform_champion(self):
        base = float(np.log(10000))
        uniform = [{'for_date': d.strftime('%Y%m%d'), 'log_loss': base,
                    'brier': .9999, 'coverage': 1, 'health': {'status': 'ok'}}
                   for d in pd.date_range('2018-01-01', periods=200)]
        noisy = [dict(row, log_loss=base + (.1 if i < 100 else -.104))
                 for i, row in enumerate(uniform)]

        def completed_attempt(history, config, output, label, **kwargs):
            if config['model'] == 'uniform':
                rows = uniform
            elif config == {'model': 'full_code', 'prior_mass': 10000}:
                rows = noisy
            else:
                rows = [dict(row, log_loss=base + .2) for row in uniform]
            return {'label': label, 'config': config, 'status': 'complete',
                    'summary': summarize(rows), 'seconds': 0}, rows

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, 'attempt', side_effect=completed_attempt):
            selected, _ = runner.run_search(pd.DataFrame(), Path(directory),
                                           {'compute': {'round_seconds': [600]}}, max_rounds=1)
        self.assertEqual(selected['selected'], {'model': 'uniform'})
