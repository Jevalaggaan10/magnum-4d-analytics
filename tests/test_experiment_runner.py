import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from test_forecasting import history


class RunnerTests(unittest.TestCase):
    def test_budget_failure_is_recorded_without_deleting_partial_evidence(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.evaluate_forecasts'),'bounded runner missing')
        from scripts.evaluate_forecasts import attempt
        with tempfile.TemporaryDirectory() as d:
            record,rows=attempt(history(4),{'model':'uniform'},Path(d),'tiny',start='2025-01-01',end='2025-02-01',seconds=0)
            self.assertEqual(record['status'],'timeout')
            self.assertEqual(rows,[])
            self.assertTrue((Path(d)/'tiny'/'attempt.json').exists())

    def test_frozen_configuration_count_and_unique_labels(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.evaluate_forecasts'),'bounded runner missing')
        from scripts.evaluate_forecasts import round_configs
        families={'baseline':{'model':'digit','alpha':100},'logistic':{'model':'logistic'},'xgboost':{'model':'xgboost'}}
        self.assertEqual([len(round_configs(r,families)) for r in range(1,6)],[5,7,6,6,6])

    def test_round_review_serializes_numpy_decisions(self):
        import numpy as np
        from scripts.evaluate_forecasts import write_json
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'review.json'
            write_json(p,{'qualifying_gain':np.float64(.0)>=.001,'seconds':np.float64(1)})
            self.assertEqual(json.loads(p.read_text())['qualifying_gain'],False)
