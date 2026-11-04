import copy
import importlib.util
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from test_forecasting import history


class ProbabilityTrackerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('probability_tracker'),'versioned probability tracker missing')
        import probability_tracker
        self.t=probability_tracker

    def test_midnight_same_day_deadline_and_delayed_saturday(self):
        for stamp,want in [('2026-11-01T00:00:00+08:00','20261101'),
                           ('2026-11-01T01:43:03+08:00','20261101'),
                           ('2026-11-01T17:59:59+08:00','20261101'),
                           ('2026-11-01T18:00:00+08:00','20261104')]:
            self.assertEqual(self.t.choose_target(datetime.fromisoformat(stamp))['for_date'],want)

    def test_special_requires_prior_verified_announcement(self):
        now=datetime.fromisoformat('2026-11-02T12:00:00+08:00')
        special={'20261103':{'url':'https://www.damacai.com.my/news/announcement','announced_at':'2026-11-01T12:00:00+08:00'}}
        self.assertEqual(self.t.choose_target(now,specials=special)['for_date'],'20261103')
        with self.assertRaises(ValueError):self.t.choose_target(now,specials={'20261103':{'url':''}})

    def test_legacy_pending_is_protected_and_dry_run_writes_nothing(self):
        state={'pending':{'for_date':'20261104','picks':['1234']},'history':[]}
        original=copy.deepcopy(state)
        with tempfile.TemporaryDirectory() as d:
            self.t.advance_forecasts(state,history(4),now=datetime.fromisoformat('2026-11-03T12:00:00+08:00'),
                artifact_dir=Path(d),settings={'champion':{'model':'uniform'},'start_date':'20261107'},dry_run=True)
            self.assertEqual(list(Path(d).iterdir()),[])
        self.assertEqual(state['pending'],original['pending'])
        self.assertEqual(state['history'],[])
        self.assertEqual(state['probability_forecasts']['20261107']['picks'],[f'{i:04d}' for i in range(10)])

    def test_published_forecast_not_retrained_when_more_history_arrives(self):
        state={'pending':None,'history':[]}
        with tempfile.TemporaryDirectory() as d:
            settings={'champion':{'model':'uniform'},'start_date':'20261107'}
            self.t.advance_forecasts(state,history(4),now=datetime.fromisoformat('2026-11-03T12:00:00+08:00'),artifact_dir=Path(d),settings=settings)
            original=copy.deepcopy(state['probability_forecasts']['20261107'])
            self.t.advance_forecasts(state,history(8),now=datetime.fromisoformat('2026-11-04T12:00:00+08:00'),artifact_dir=Path(d),settings=settings)
            self.assertEqual(state['probability_forecasts']['20261107'],original)

    def test_late_publication_is_coverage_failure(self):
        state={'probability_forecasts':{'20261101':{'for_date':'20261101','publication_deadline':'2026-11-01T18:00:00+08:00','status':'awaiting-publication'}}}
        self.t.confirm_publication(state,datetime.fromisoformat('2026-11-01T18:01:00+08:00'))
        self.assertEqual(state['probability_forecasts']['20261101']['coverage'],0)
        self.assertEqual(state['probability_forecasts']['20261101']['status'],'missed-publication')

    def test_conflicting_result_quarantined_not_replaced(self):
        from data_loader import SLOT_COLS
        h=history(1);old={c:h.iloc[0][c] for c in SLOT_COLS};new=dict(old,SpecialNo1='9999')
        state={'observed_draws':{'20250101':old}}
        accepted=self.t.accept_observations(state,{'20250101':new},h,now=datetime.fromisoformat('2025-01-02T12:00:00+08:00'))
        self.assertEqual(accepted,{})
        self.assertEqual(state['observed_draws']['20250101'],old)
        self.assertEqual(len(state['quarantined_observations']),1)

    def test_complete_published_forecast_scores_original_artifact(self):
        import pandas as pd
        from data_loader import SLOT_COLS
        state={'pending':None,'history':[]}
        with tempfile.TemporaryDirectory() as d:
            settings={'champion':{'model':'uniform'},'start_date':'20261107'}
            self.t.advance_forecasts(state,history(4),now=datetime.fromisoformat('2026-11-03T12:00:00+08:00'),artifact_dir=Path(d),settings=settings)
            self.t.confirm_publication(state,datetime.fromisoformat('2026-11-03T12:01:00+08:00'),evidence='CI-test')
            actual=history(1);actual.loc[0,'DrawDate']=pd.Timestamp('2026-11-07');actual.loc[0,SLOT_COLS]='0000'
            self.t.advance_forecasts(state,actual,now=datetime.fromisoformat('2026-11-07T21:00:00+08:00'),artifact_dir=Path(d),settings=settings)
            record=state['probability_forecasts']['20261107']
            self.assertEqual(record['status'],'verified')
            self.assertEqual(record['metrics']['champion']['slot_hits'],23)
            self.assertEqual(record['forecasts']['champion']['metadata']['training_rows'],4)

    def test_explicit_special_accepts_same_day_prior_announcement(self):
        state={'pending':None,'history':[]}
        with tempfile.TemporaryDirectory() as d:
            self.t.advance_forecasts(state,history(),now=datetime.fromisoformat('2026-11-03T16:00:00+08:00'),
                artifact_dir=Path(d),dry_run=True,settings={'champion':{'model':'uniform'},'requested_date':'20261103',
                'specials':{'20261103':{'url':'https://www.damacai.com.my/news/a','announced_at':'2026-11-03T14:00:00+08:00'}}})
        self.assertIn('20261103',state['probability_forecasts'])
