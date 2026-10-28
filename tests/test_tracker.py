import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import track_predictions as tracker


def slots():
    return {col: f'{i:04d}' for i, col in enumerate(tracker.SLOT_COLS)}


class TrackerRegressionTests(unittest.TestCase):
    def test_corpus_hash_has_platform_independent_dates_and_newlines(self):
        train = pd.DataFrame({'DrawDate': pd.to_datetime(['2026-01-01']),
                              '1stPrizeNo': ['0001'], '2ndPrizeNo': ['0002'], '3rdPrizeNo': ['0003']})
        canonical = b'DrawDate,1stPrizeNo,2ndPrizeNo,3rdPrizeNo\n2026-01-01,0001,0002,0003\n'
        self.assertEqual(tracker.training_metadata(train)['corpus_sha256'], hashlib.sha256(canonical).hexdigest())

    def test_invalid_draw_slots_do_not_turn_into_zero(self):
        actual = tracker._draw_to_slots({'1stPrizeNo': None, '2ndPrizeNo': '', '3rdPrizeNo': '0'})
        self.assertEqual(actual['1stPrizeNo'], '')
        self.assertEqual(actual['2ndPrizeNo'], '')
        self.assertEqual(actual['3rdPrizeNo'], '0000')

    def test_cache_lookup_accepts_parsed_timestamps(self):
        row = dict(slots(), DrawDate=pd.Timestamp('2026-10-28'))
        actual = tracker.draw_from_local_cache(pd.DataFrame([row]), '20261028')
        self.assertIsNotNone(actual)
        self.assertEqual(actual['1stPrizeNo'], '0000')

    def test_numeric_zero_from_live_adapter_is_a_valid_prize(self):
        from data_sources.damacai import Damacai
        source = Damacai(polite_sleep_seconds=0)
        with patch.object(source, '_signed_link', return_value='https://example.invalid'), \
             patch.object(source.session, 'get') as fetch:
            fetch.return_value.json.return_value = {'p1': 0, 'p2': 1234, 'p3': 5678}
            row = source.fetch_draw('20261028').to_row()
        self.assertEqual(row['1stPrizeNo'], '0000')

    def test_corrupt_ledger_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / 'state.json'
            state_path.write_text('{broken', encoding='utf-8')
            with patch.object(tracker, 'STATE_FILE', state_path):
                with self.assertRaises(ValueError):
                    tracker.load_state()

    def test_offline_mode_does_not_fetch_a_pending_result(self):
        state = {'pending': {'for_date': '20991028', 'picks': ['1234'], 'random_control': ['5678']}, 'history': []}
        corpus = pd.DataFrame([dict(slots(), DrawDate=pd.Timestamp('2026-10-28'), Source='damacai')])
        with patch.object(tracker, 'load_state', return_value=state), \
             patch.object(tracker, 'load_corpus', return_value=corpus), \
             patch.object(tracker, 'load_data_cache', return_value=None), \
             patch.object(tracker, 'fetch_single_draw') as fetch, \
             patch.object(tracker, 'render_log', return_value='dry-run'):
            self.assertEqual(tracker.main(['--no-sync', '--dry-run']), 0)
            fetch.assert_not_called()

    def test_online_mode_never_fetches_a_future_pending_result(self):
        pending = {'for_date': '20991028', 'picks': ['1234'], 'random_control': ['5678']}
        state = {'pending': pending.copy(), 'history': []}
        corpus = pd.DataFrame([dict(slots(), DrawDate=pd.Timestamp('2026-10-28'), Source='damacai')])
        with patch.object(tracker, 'load_state', return_value=state), \
             patch.object(tracker, 'load_corpus', return_value=corpus), \
             patch.object(tracker, 'fetch_recent_draws', return_value={}), \
             patch.object(tracker, 'fetch_single_draw') as fetch, \
             patch.object(tracker, 'render_log', return_value='dry-run'):
            tracker.main(['--dry-run'])
        fetch.assert_not_called()
        self.assertEqual(state['pending'], pending)

    def test_corpus_uses_same_operator_and_newly_observed_draws(self):
        state = {'history': [], 'observed_draws': {'20261028': slots()}}
        with tempfile.TemporaryDirectory() as directory, patch.object(tracker, 'REPO_ROOT', Path(directory)):
            df = tracker.load_corpus(state)
        self.assertEqual(df['Source'].unique().tolist(), ['damacai'])
        self.assertEqual(df['DrawDate'].max(), pd.Timestamp('2026-10-28'))

    def test_replay_incorporates_only_earlier_observations(self):
        recorded = []
        draws = [{'for_date': d, 'slots': slots()} for d in ['20261001', '20261002', '20261003']]
        seed = pd.DataFrame([dict(slots(), DrawDate=pd.Timestamp('2026-09-30'), Source='damacai')])
        def fake_picks(train):
            recorded.append(train['DrawDate'].tolist())
            return ['1234'] * 10, ['frequency']
        with patch.object(tracker, 'load_corpus', return_value=seed), \
             patch.object(tracker, 'generate_picks', side_effect=fake_picks):
            rows, skipped = tracker.walk_forward_replay(draws, state={})
        self.assertFalse(skipped)
        self.assertEqual([len(x) for x in recorded], [1, 2, 3])
        for dates, row in zip(recorded, rows):
            self.assertTrue(all(d < pd.Timestamp(row['for_date']) for d in dates))

    def test_partial_result_does_not_resolve_the_published_prediction(self):
        pending = {'for_date': '20261028', 'picks': ['1234'], 'random_control': ['5678']}
        state = {'pending': pending.copy(), 'history': []}
        corpus = pd.DataFrame([dict(slots(), DrawDate=pd.Timestamp('2026-10-27'), Source='damacai')])
        with patch.object(tracker, 'load_state', return_value=state), \
             patch.object(tracker, 'load_corpus', return_value=corpus), \
             patch.object(tracker, 'fetch_recent_draws', return_value={'20261028': {'1stPrizeNo': '1234'}}), \
             patch.object(tracker, 'render_log', return_value='dry-run'):
            tracker.main(['--dry-run'])
        self.assertEqual(state['pending'], pending)
        self.assertEqual(state['history'], [])

    def test_state_metadata_is_saved_when_markdown_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / 'state.json'
            log_path = Path(directory) / 'log.md'
            log_path.write_text('same markdown', encoding='utf-8')
            state_path.write_text(json.dumps({'history': [], 'metadata': 1}), encoding='utf-8')
            with patch.object(tracker, 'STATE_FILE', state_path), patch.object(tracker, 'LOG_FILE', log_path):
                tracker.save_state_and_log({'history': [], 'metadata': 2}, 'same markdown')
            self.assertEqual(json.loads(state_path.read_text())['metadata'], 2)


if __name__ == '__main__':
    unittest.main()
