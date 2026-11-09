import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from data_sources.damacai import Damacai
from scripts import track_predictions as tracker
from test_tracker import slots


class ResultRecoveryTests(unittest.TestCase):
    def test_adapter_rejects_result_for_another_draw_date(self):
        source = Damacai(polite_sleep_seconds=0)
        response = {'drawDate': '08/11/2026', 'p1': '1234', 'p2': '5678', 'p3': '9012'}
        with patch.object(source, '_signed_link', return_value='https://example.invalid'), \
                patch.object(source.session, 'get') as fetch:
            fetch.return_value.json.return_value = response
            self.assertIsNone(source.fetch_draw('20261107'))

    def test_historical_recovery_requires_date_evidence(self):
        source = Damacai(polite_sleep_seconds=0)
        with patch.object(source, '_signed_link', return_value='https://example.invalid'), \
                patch.object(source.session, 'get') as fetch:
            fetch.return_value.json.return_value = {'p1': '1234', 'p2': '5678', 'p3': '9012'}
            self.assertIsNone(source.fetch_draw('20261107', require_date_evidence=True))
            fetch.return_value.json.return_value['drawDate'] = '07/11/2026'
            self.assertEqual(source.fetch_draw('20261107', require_date_evidence=True).draw_date, '20261107')

    def test_bounded_retries_do_not_starve_other_dates(self):
        dates = ['20261107', '20261108', '20261111']
        state = {'probability_forecasts': {
            date: {'status': 'published', 'target_time': f'{date[:4]}-{date[4:6]}-{date[6:]}T19:00:00+08:00'}
            for date in dates}}
        corpus = pd.DataFrame(columns=['DrawDate', *tracker.SLOT_COLS])
        now = datetime.fromisoformat('2026-11-12T12:00:00+08:00')
        with patch.object(tracker, 'fetch_single_draw', side_effect=TimeoutError('endpoint unavailable')):
            tracker.recover_probability_results(state, corpus, now=now, budget=1)
            tracker.recover_probability_results(state, corpus, now=now, budget=1)
            tracker.recover_probability_results(state, corpus, now=now, budget=1)
        self.assertEqual(set(state['result_recovery']), set(dates))
        self.assertTrue(all(report['attempts'] == 1 for report in state['result_recovery'].values()))

    def test_partial_failure_keeps_success_and_provenance(self):
        state = {'probability_forecasts': {
            date: {'status': 'published', 'target_time': stamp}
            for date, stamp in [('20261107', '2026-11-07T19:00:00+08:00'),
                                ('20261108', '2026-11-08T19:00:00+08:00')]}}
        corpus = pd.DataFrame(columns=['DrawDate', *tracker.SLOT_COLS])
        with patch.object(tracker, 'fetch_single_draw', side_effect=[TimeoutError('offline'), slots()]):
            results = tracker.recover_probability_results(
                state, corpus, now='2026-11-09T12:00:00+08:00')
        self.assertEqual(set(results), {'20261108'})
        self.assertIn('content_sha256', state['observation_provenance']['20261108'])
        self.assertEqual(state['result_recovery']['20261107']['status'], 'failed')

    def test_future_complete_and_verified_targets_are_not_fetched(self):
        state = {'probability_forecasts': {
            '20261107': {'status': 'verified', 'target_time': '2026-11-07T19:00:00+08:00'},
            '20261108': {'status': 'published', 'target_time': '2026-11-08T19:00:00+08:00'},
            '20261111': {'status': 'published', 'target_time': '2026-11-11T19:00:00+08:00'}}}
        corpus = pd.DataFrame([dict(slots(), DrawDate=pd.Timestamp('2026-11-08'))])
        with patch.object(tracker, 'fetch_single_draw', side_effect=AssertionError('unexpected fetch')):
            results = tracker.recover_probability_results(state, corpus, now='2026-11-11T18:59:59+08:00')
        self.assertEqual(results, {})
        self.assertNotIn('result_recovery', state)

    def test_incomplete_and_conflicting_results_stay_unverified(self):
        state = {'probability_forecasts': {
            '20261107': {'status': 'published', 'target_time': '2026-11-07T19:00:00+08:00'},
            '20261108': {'status': 'published', 'target_time': '2026-11-08T19:00:00+08:00'}}}
        partial = dict(slots(), SpecialNo1='')
        corpus = pd.DataFrame([dict(partial, DrawDate=pd.Timestamp('2026-11-08'))])
        with patch.object(tracker, 'fetch_single_draw', side_effect=[partial, slots()]):
            results = tracker.recover_probability_results(state, corpus, now='2026-11-09T12:00:00+08:00')
        self.assertEqual(results, {})
        self.assertEqual(state['result_recovery']['20261107']['status'], 'incomplete')
        self.assertEqual(state['result_recovery']['20261108']['status'], 'quarantined')
        self.assertEqual(len(state['quarantined_observations']), 1)
