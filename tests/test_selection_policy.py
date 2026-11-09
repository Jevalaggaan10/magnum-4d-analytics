import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from forecast_evaluation import development_decision
from scripts.evaluate_forecasts import SELECTION_POLICY, run_search


class SelectionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads(SELECTION_POLICY.read_text())
        self.policy['replicates'] = 1000
        self.baseline = [{'for_date': d.strftime('%Y%m%d'), 'log_loss': 9.21,
                          'coverage': 1, 'health': {'status': 'ok'}}
                         for d in pd.date_range('2018-01-01', periods=200)]

    def test_clear_paired_gain_can_advance(self):
        rows = [dict(row, log_loss=9.19) for row in self.baseline]
        result = development_decision(rows, self.baseline, self.policy)
        self.assertTrue(result['accepted'])
        self.assertEqual(set(result['intervals']), {'13', '26'})

    def test_both_dependence_sensitivities_must_support_gain(self):
        rows = [dict(row, log_loss=9.21 + (.03 if i < 100 else -.174))
                for i, row in enumerate(self.baseline)]
        result = development_decision(rows, self.baseline, self.policy)
        self.assertLess(result['intervals']['13']['upper'], 0)
        self.assertGreater(result['intervals']['26']['upper'], 0)
        self.assertFalse(result['accepted'])

    def test_unhealthy_unaligned_duplicate_and_subthreshold_candidates_do_not_advance(self):
        good = [dict(row, log_loss=9.19) for row in self.baseline]
        cases = [good[:-1], good + [good[0]],
                 [dict(row, health={'status': 'fallback'}) for row in good],
                 [dict(row, coverage=0) for row in good],
                 [dict(row, log_loss=9.2095) for row in good],
                 [dict(row, log_loss=float('nan')) for row in good], self.baseline]
        for rows in cases:
            with self.subTest(rows=len(rows), first=rows[0]):
                self.assertFalse(development_decision(rows, self.baseline, self.policy)['accepted'])

    def test_expired_selection_budget_does_not_emit_partial_decision(self):
        rows = [dict(row, log_loss=9.19) for row in self.baseline]
        with self.assertRaises(TimeoutError):
            development_decision(rows, self.baseline, self.policy, deadline=0)

    def test_resume_cannot_reuse_another_data_or_fold_identity(self):
        protocol = {'compute': {'round_seconds': [1]}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            output.mkdir()
            (output / 'search-identity.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                run_search(pd.DataFrame(), output, protocol, max_rounds=1)
            legacy = Path(directory) / 'legacy'
            (legacy / 'development').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                run_search(pd.DataFrame(), Path(directory) / 'new', protocol,
                           max_rounds=1, resume_from=legacy)
