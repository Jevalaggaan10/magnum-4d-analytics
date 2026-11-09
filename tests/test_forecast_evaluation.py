import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from test_forecasting import history
from test_confirmation_identity import record


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('forecast_evaluation'),'draw evaluator missing')
        import forecast_evaluation
        self.e=forecast_evaluation

    def test_evaluator_saves_exact_forecasts_and_keeps_draw_units(self):
        with tempfile.TemporaryDirectory() as d:
            rows=self.e.evaluate(history(6),{'model':'uniform'},start='2025-01-04',end='2025-01-16',output=Path(d))
            self.assertEqual(len(rows),5)
            self.assertEqual([r['training_rows'] for r in rows],[1,2,3,4,5])
            self.assertTrue(all(r['slots']==23 for r in rows))
            self.assertTrue(all(r['coverage']==1 for r in rows))
            self.assertTrue((Path(d)/'predictions.jsonl').exists())
            from forecasting import load_forecast,score_forecast
            result=load_forecast(rows[0]['forecast'],Path(d)/'probabilities')
            self.assertAlmostEqual(score_forecast(result,history(6).iloc[1].to_dict())['log_loss'],rows[0]['log_loss'])

    def test_failed_forecast_stays_in_denominator(self):
        rows=self.e.evaluate(history(3),{'model':'bad'},start='2025-01-04',end='2025-01-07')
        self.assertEqual(len(rows),2)
        self.assertTrue(all(r['health']['status']=='fallback' for r in rows))
        self.assertEqual(self.e.summarize(rows)['fallback_draws'],2)

    def test_paired_loss_interval_uses_draw_difference(self):
        rows=[{'for_date':f'{i:04d}','log_loss':9.2} for i in range(52)]
        base=[dict(r,log_loss=9.3) for r in rows]
        interval=self.e.paired_interval(rows,base,block_length=13,replicates=1000)
        self.assertAlmostEqual(interval['mean'],-.1)
        self.assertAlmostEqual(interval['upper'],-.1)

    def test_ensemble_weights_reject_future_calibration(self):
        rows=[[{'for_date':'20250104','observed_slot_probabilities':[p]*23}] for p in [.0001,.0002]]
        with self.assertRaises(ValueError):self.e.fit_mixture(rows,cutoff='2025-01-01')
        weights=self.e.fit_mixture(rows,cutoff='2025-01-05',minimum_draws=1)
        self.assertGreater(weights[1],weights[0])
        self.assertAlmostEqual(sum(weights),1)

    def test_synthetic_uniform_and_injected_signal_sanity(self):
        rng=np.random.default_rng(1907)
        h=history(180)
        from data_loader import SLOT_COLS
        for col in SLOT_COLS:h[col]=[f'{i:04d}' for i in rng.integers(0,10000,len(h))]
        rows=self.e.evaluate(h,{'model':'digit','alpha':100},start='2025-12-01',end='2026-12-31')
        self.assertGreater(self.e.summarize(rows)['mean_log_loss'],9.0)
        self.assertLess(self.e.summarize(rows)['mean_log_loss'],9.5)
        for col in SLOT_COLS:h[col]='0000'
        signal=self.e.evaluate(h,{'model':'full_code','prior_mass':10000},start='2025-12-01',end='2026-12-31')
        self.assertLess(self.e.summarize(signal)['mean_log_loss'],5)

    def test_publication_clock_and_coverage_checkpoint(self):
        status=self.e.confirmation_status([record(str(i)) for i in range(999)])
        self.assertEqual(status['status'],'pending')
        rows=[record(str(i)) for i in range(1000)]
        rows[10]['coverage']=0
        self.assertEqual(self.e.confirmation_status(rows)['status'],'incomplete-coverage')

    def test_confirmation_gate_cannot_promote_on_loss_alone(self):
        from forecast_evaluation import promotion_decision
        rows=[{**record(str(i)),'metrics':{
            'champion':{'log_loss':9.21,'hit_draw':0},'shadow':{'log_loss':9.,'hit_draw':0},
            'simplest_nonuniform':{'log_loss':9.1}},'conditional_control':.02} for i in range(1000)]
        result=promotion_decision(rows,development_qualified=True,correctness_verified=True,replicates=100)
        self.assertTrue(result['probability_promotion'])
        self.assertFalse(result['exact_hit_advantage'])

    def test_confirmation_uses_control_from_saved_slot_metrics(self):
        from forecast_evaluation import promotion_decision
        rows=[{**record(str(i)),'metrics':{
            'champion':{'log_loss':9.21,'hit_draw':0,'conditional_control':.02},
            'shadow':{'log_loss':9.,'hit_draw':0,'conditional_control':.02},
            'simplest_nonuniform':{'log_loss':9.1,'conditional_control':.02}}} for i in range(1000)]
        result=promotion_decision(rows,development_qualified=True,correctness_verified=True,replicates=100)
        self.assertTrue(result['probability_promotion'])
        self.assertFalse(result['exact_hit_advantage'])
