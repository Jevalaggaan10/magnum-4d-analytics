import contextlib
import copy
import io
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data_loader import SLOT_COLS
from forecast_identity import PROTOCOL_PATH, REQUIRED_SOURCES, validate_execution
from probability_tracker import advance_forecasts, confirm_publication
from scripts import track_predictions as tracker
from forecast_fixtures import ROOT, settings
from test_forecasting import history


class SourceIntegrityTests(unittest.TestCase):
    def test_corrupt_missing_and_outside_files_never_approve_execution(self):
        config = settings()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (*REQUIRED_SOURCES, PROTOCOL_PATH):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / relative).read_bytes())
            validate_execution(config, root)
            (root / PROTOCOL_PATH).write_text('changed protocol')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                validate_execution(config, root)
            (root / PROTOCOL_PATH).write_bytes((ROOT / PROTOCOL_PATH).read_bytes())
            (root / 'forecasting.py').unlink()
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                validate_execution(config, root)
        for invalid in [dict(config, source_hashes={}),
                        dict(config, source_hashes={**config['source_hashes'], '../outside': '0' * 64})]:
            with self.assertRaises(ValueError):
                validate_execution(invalid)

    def test_source_mismatch_preserves_artifact_scoring_but_blocks_new_publication(self):
        config = settings()
        state = {'pending': None, 'history': []}
        with tempfile.TemporaryDirectory() as directory:
            advance_forecasts(state, history(), now=datetime.fromisoformat('2026-11-03T12:00:00+08:00'),
                             artifact_dir=Path(directory), settings=config)
            self.assertTrue(confirm_publication(state, '2026-11-03T12:01:00+08:00',
                                                settings=config, evidence='fixture push'))
            original = copy.deepcopy(state['probability_forecasts']['20261107']['forecasts'])
            actual = history(1)
            actual.loc[0, 'DrawDate'] = pd.Timestamp('2026-11-07')
            actual.loc[0, SLOT_COLS] = '0000'
            config['source_hashes']['forecasting.py'] = '0' * 64
            approved = advance_forecasts(state, actual, now=datetime.fromisoformat('2026-11-07T21:00:00+08:00'),
                                         artifact_dir=Path(directory), settings=config)
        self.assertFalse(approved)
        self.assertEqual(state['probability_forecasts']['20261107']['status'], 'verified')
        self.assertEqual(state['probability_forecasts']['20261107']['forecasts'], original)
        self.assertNotIn('20261108', state['probability_forecasts'])

    def test_cli_persists_block_reason_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = settings(source_hashes={})
            (root / 'settings.json').write_text(json.dumps(config))
            (root / 'state.json').write_text(json.dumps({'pending': None, 'history': []}))
            with patch.object(tracker, 'SETTINGS_FILE', root / 'settings.json'), \
                    patch.object(tracker, 'STATE_FILE', root / 'state.json'), \
                    patch.object(tracker, 'LOG_FILE', root / 'log.md'), \
                    patch.object(tracker, 'ARTIFACT_DIR', root / 'artifacts'), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = tracker.main(['--no-sync'])
            saved = json.loads((root / 'state.json').read_text())
            self.assertEqual(code, 2)
            self.assertEqual(saved['forecast_integrity']['status'], 'blocked')
            self.assertIn('generation blocked', (root / 'log.md').read_text())
            self.assertFalse((root / 'artifacts').exists())

    def test_confirmation_does_not_publish_changed_procedure_or_missing_evidence(self):
        config = settings()
        state = {'pending': None, 'history': []}
        with tempfile.TemporaryDirectory() as directory:
            advance_forecasts(state, history(), now=datetime.fromisoformat('2026-11-03T12:00:00+08:00'),
                             artifact_dir=Path(directory), settings=config)
        original = copy.deepcopy(state)
        confirm_publication(state, '2026-11-03T12:01:00+08:00', settings=config)
        self.assertEqual(state['probability_forecasts']['20261107']['status'], 'awaiting-publication')
        changed = dict(config, shadow={'model': 'digit'})
        confirm_publication(original, '2026-11-03T12:01:00+08:00', settings=changed, evidence='push')
        self.assertEqual(original['probability_forecasts']['20261107']['coverage'], 0)
