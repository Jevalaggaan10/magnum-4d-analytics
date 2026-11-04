"""Behavior checks for the probability boundary, chronology and artifacts."""
import hashlib
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import SLOT_COLS


def history(n=4):
    return pd.DataFrame([dict(DrawDate=pd.Timestamp('2025-01-01') + pd.Timedelta(days=i*3),
                              Operator='damacai', **{c: f'{(i+j)%10:04d}' for j,c in enumerate(SLOT_COLS)})
                         for i in range(n)])


class ProbabilityBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('forecasting'), 'shared probability interface missing')
        import forecasting
        self.f = forecasting

    def test_uniform_scores_and_ten_distinct_codes(self):
        result = self.f.forecast(history(), operator='damacai', target_time='2025-02-01', config={'model':'uniform'})
        scores = self.f.score_forecast(result, {c:'0000' for c in SLOT_COLS})
        self.assertAlmostEqual(scores['log_loss'], math.log(10000), places=11)
        self.assertAlmostEqual(scores['brier'], .9999, places=11)
        self.assertEqual(result.picks, [f'{i:04d}' for i in range(10)])
        self.assertEqual(scores['slot_hits'], 23)
        self.assertEqual(scores['hit_draw'], 1)

    def test_shrinkage_support_missing_zero_and_future_invariance(self):
        h = history(); h.loc[0,'SpecialNo1'] = None
        config = {'model':'full_code','prior_mass':10000.0}
        a = self.f.forecast(h, operator='damacai', target_time='2025-01-08', config=config)
        h.loc[3, SLOT_COLS] = '9999'
        b = self.f.forecast(h, operator='damacai', target_time='2025-01-08', config=config)
        np.testing.assert_array_equal(a.probabilities, b.probabilities)
        self.assertTrue(np.all(a.probabilities > 0))
        np.testing.assert_allclose(a.probabilities.sum(axis=1), 1, atol=1e-10, rtol=0)
        self.assertEqual(a.metadata['training_rows'], 2)  # partial first draw excluded; no fabricated zero
        self.assertEqual(a.metadata['corpus_cutoff'], '2025-01-07')
        self.assertEqual(a.metadata['corpus_sha256'], b.metadata['corpus_sha256'])

    def test_any_slot_changes_training_hash(self):
        h=history(); a=self.f.forecast(h,operator='damacai',target_time='2025-02-01',config={'model':'uniform'})
        h.loc[0,'ConsolationNo10']='9876'
        b=self.f.forecast(h,operator='damacai',target_time='2025-02-01',config={'model':'uniform'})
        self.assertNotEqual(a.metadata['corpus_sha256'],b.metadata['corpus_sha256'])

    def test_sparse_empty_tier_and_unsupported_model_health(self):
        for config in [{'model':'uniform'},{'model':'digit','alpha':100,'pooling':'tier'}, {'model':'bad'}]:
            r=self.f.forecast(history(0),operator='damacai',target_time='2025-02-01',config=config)
            self.assertTrue(np.all(r.probabilities>0))
            self.assertEqual(len(set(r.picks)),10)
        self.assertEqual(r.metadata['health']['status'],'fallback')
        self.assertIn('bad',r.metadata['health']['error'])

    def test_partial_actual_cannot_be_scored(self):
        r=self.f.forecast(history(),operator='damacai',target_time='2025-02-01',config={'model':'uniform'})
        with self.assertRaises(ValueError): self.f.score_forecast(r,{'1stPrizeNo':'0000'})

    def test_artifact_roundtrip_detects_corruption(self):
        r=self.f.forecast(history(),operator='damacai',target_time='2025-02-01',config={'model':'digit'})
        with tempfile.TemporaryDirectory() as d:
            ref=self.f.save_forecast(r,Path(d))
            loaded=self.f.load_forecast(ref,Path(d))
            np.testing.assert_array_equal(r.probabilities,loaded.probabilities)
            p=Path(d)/ref['artifact']; p.write_bytes(p.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):self.f.load_forecast(ref,Path(d))

    def test_exact_conditional_control_and_independence_labels(self):
        self.assertEqual(self.f.conditional_control_probability(0),0)
        self.assertAlmostEqual(self.f.conditional_control_probability(1),.001,places=12)
        r=self.f.forecast(history(),operator='damacai',target_time='2025-02-01',config={'model':'uniform'})
        self.assertIsNone(r.any_hit_probability())
        self.assertAlmostEqual(r.any_hit_probability(assume_slot_independence=True),1-.999**23)

    def test_invalid_distributions_and_mixed_operators_rejected(self):
        for value in [np.full((1,10000),.1),np.full((1,10000),np.nan),np.zeros((1,10000))]:
            with self.assertRaises(ValueError):self.f.validate_probabilities(value)
        h=history();h.loc[0,'Operator']='magnum'
        with self.assertRaises(ValueError):self.f.forecast(h,operator='damacai',target_time='2025-02-01',config={'model':'uniform'})

    def test_next_draw_features_exclude_current_label_preserve_context(self):
        x,y,dates=self.f.supervised_examples(history(5),calendar=False)
        altered=history(5);altered.loc[3,SLOT_COLS]='9999'
        x2,y2,dates2=self.f.supervised_examples(altered,calendar=False)
        np.testing.assert_array_equal(x[dates==pd.Timestamp('2025-01-10')],x2[dates2==pd.Timestamp('2025-01-10')])
        self.assertFalse(np.array_equal(y[dates==pd.Timestamp('2025-01-10')],y2[dates2==pd.Timestamp('2025-01-10')]))
        self.assertEqual(len(set(dates)),4)

    def test_incremental_statistics_equal_cold_batch(self):
        h=history(10); engine=self.f.ForecastEngine(h)
        for n in [2,5,10]:
            t=h.iloc[n-1].DrawDate+pd.Timedelta(days=1)
            for config in [{'model':'full_code'},{'model':'digit','half_life_days':365,'pooling':'tier'}]:
                cached=engine.forecast(operator='damacai',target_time=t,config=config)
                cold=self.f.forecast(h.iloc[:n],operator='damacai',target_time=t,config=config)
                np.testing.assert_allclose(cached.probabilities,cold.probabilities,atol=1e-14)


class AlignmentAndCliTests(unittest.TestCase):
    def test_rolling_features_do_not_include_own_outcome(self):
        from feature_engineer import create_rolling_features
        h=history(); a=create_rolling_features(h)
        h.loc[2,SLOT_COLS]='9999';b=create_rolling_features(h)
        self.assertEqual(a.loc[2,'digit_9_roll_7'],b.loc[2,'digit_9_roll_7'])

    def test_documented_predict_cli_runs_without_optional_training(self):
        import subprocess,sys
        code="""import runpy,sys,pandas as pd
from unittest.mock import patch
frame=pd.DataFrame({'DrawDate':pd.to_datetime(['2025-01-01']),'1stPrizeNo':['0000'],'2ndPrizeNo':['0001'],'3rdPrizeNo':['0002']})
sys.argv=['predict.py']
with patch('train_model.train_all_models',return_value={}),patch('data_loader.load_combined_data',return_value=frame):
 runpy.run_path('predict.py',run_name='__main__')
"""
        p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

class TrainedCategoricalTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'optional ML environment')
    def test_trained_digit_heads_score_complete_future_codes_and_cache_parity(self):
        import forecasting as f
        rng=np.random.default_rng(792)
        h=history(65)
        for col in SLOT_COLS:h[col]=[f'{i:04d}' for i in rng.integers(0,10000,len(h))]
        engine=f.ForecastEngine(h)
        config={'model':'logistic','C':.01}
        a=engine.forecast(operator='damacai',target_time='2025-07-10',config=config)
        self.assertEqual(a.metadata['health']['status'],'ok',a.metadata['health'])
        self.assertEqual(a.metadata['estimator_fit_rows'],64)
        b=f.forecast(h,operator='damacai',target_time='2025-07-10',config=config)
        np.testing.assert_allclose(a.probabilities,b.probabilities,atol=1e-12)
        self.assertEqual(len(a.picks),10)

    def test_future_calibration_fails_visibly(self):
        import forecasting as f
        r=f.forecast(history(),operator='damacai',target_time='2025-02-01',config={
            'model':'ensemble','components':[{'model':'uniform'}],'weights':[1.],
            'calibration_cutoff':'2025-02-01'})
        self.assertEqual(r.metadata['health']['status'],'fallback')
        self.assertIn('future',r.metadata['health']['error'])

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'optional ML environment')
    def test_cached_estimator_retains_fit_warning_health(self):
        import warnings
        from unittest.mock import patch
        from sklearn.linear_model import LogisticRegression
        import forecasting as f
        h=history(70);rng=np.random.default_rng(7)
        for c in SLOT_COLS:h[c]=[f'{i:04d}' for i in rng.integers(0,10000,len(h))]
        engine=f.ForecastEngine(h);original=LogisticRegression.fit
        def noisy(model,*args,**kwargs):
            warnings.warn('fit health test warning')
            return original(model,*args,**kwargs)
        with patch.object(LogisticRegression,'fit',noisy):
            a=engine.forecast(operator='damacai',target_time='2026-01-01',config={'model':'logistic','C':.01})
        b=engine.forecast(operator='damacai',target_time='2026-01-02',config={'model':'logistic','C':.01})
        self.assertEqual(a.metadata['health']['status'],'degraded')
        self.assertEqual(b.metadata['health']['status'],'degraded')
