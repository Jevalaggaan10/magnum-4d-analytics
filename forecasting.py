"""Categorical next-draw forecasts. No probability is inferred from legacy ranks.

Four-digit codes are categorical. Digit heads impose within-code factorization;
slot pooling does not assert conditional independence between prize slots.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data_loader import SLOT_COLS, normalize_prize_number

VERSION = 'damacai-v3'
FEATURE_VERSION = 'lagged-digits-v1'
MYT = ZoneInfo('Asia/Kuala_Lumpur')
CODES = np.array([f'{i:04d}' for i in range(10000)])
DIGITS = np.array([[int(d) for d in n] for n in CODES], dtype=np.int8)
TIERS = np.array([0, 1, 2] + [3]*10 + [4]*10)
UNIFORM = np.full((1, 10000), .0001, dtype=np.float64)
EPSILON = 1e-6


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def target_timestamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError('invalid target time')
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(MYT)
    return stamp.tz_convert(MYT)


def validate_probabilities(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 10000 or p.shape[0] not in (1, 5, 23):
        raise ValueError('expected 1, 5 or 23 complete 10000-code distributions')
    if not np.isfinite(p).all() or not (p > 0).all():
        raise ValueError('categorical probabilities must be finite with positive support')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError('categorical distributions must sum to one')
    return p


@dataclass
class ForecastResult:
    probabilities: np.ndarray
    slot_distribution: list[int]
    picks: list[str]
    metadata: dict

    def __post_init__(self):
        self.probabilities = validate_probabilities(self.probabilities)
        if len(self.slot_distribution) != 23 or any(i < 0 or i >= len(self.probabilities) for i in self.slot_distribution):
            raise ValueError('invalid slot distribution mapping')
        if len(self.picks) != 10 or len(set(self.picks)) != 10 or any(normalize_prize_number(n) != n for n in self.picks):
            raise ValueError('forecast requires ten distinct four-digit codes')

    def any_hit_probability(self, *, assume_slot_independence=False):
        if not assume_slot_independence:
            return None
        indexes = [int(n) for n in self.picks]
        mass = self.probabilities[:, indexes].sum(axis=1)[self.slot_distribution]
        return float(-np.expm1(np.log1p(-np.minimum(mass, 1)).sum()))

    def single_slot_probability(self, code, slot='1stPrizeNo'):
        if normalize_prize_number(code) != code:
            raise ValueError('code must contain four ASCII digits')
        return float(self.probabilities[self.slot_distribution[SLOT_COLS.index(slot)], int(code)])


def _result(p, metadata):
    p = validate_probabilities(p)
    mapping = [0]*23 if len(p) == 1 else TIERS.tolist() if len(p) == 5 else list(range(23))
    expected = p[mapping].sum(axis=0)
    # Code order is lexical and np.lexsort resolves score ties deterministically.
    indexes = np.lexsort((np.arange(10000), -expected))[:10]
    return ForecastResult(p, mapping, CODES[indexes].tolist(), metadata)


def _prepare(history):
    if history.empty:
        return pd.DataFrame(columns=['DrawDate', *SLOT_COLS]), 0
    if 'DrawDate' not in history:
        raise ValueError('history lacks DrawDate')
    h = history.copy()
    h['DrawDate'] = pd.to_datetime(h['DrawDate'], errors='raise').dt.tz_localize(None).dt.normalize()
    if h['DrawDate'].isna().any() or h['DrawDate'].duplicated().any():
        raise ValueError('history requires unique valid draw dates; resolve source conflicts explicitly')
    for col in SLOT_COLS:
        h[col] = h[col].map(normalize_prize_number) if col in h else None
    valid = h[SLOT_COLS].notna().all(axis=1)
    return h.loc[valid].sort_values('DrawDate').reset_index(drop=True), int((~valid).sum())


class ForecastEngine:
    """Reusable sufficient statistics; every cache is local to one frozen corpus.

    Future rows may be present for replay, but each target sees a strict prefix.
    Cache keys include exact earlier bytes, configuration and deterministic refit
    anchor. No process-global fitted estimator or calendar-clock dependence.
    """
    def __init__(self, history):
        self.history, self.excluded_partial_rows = _prepare(history)
        self.raw = (self.history[SLOT_COLS].to_numpy().astype('U4').astype(np.int32)
                    if len(self.history) else np.empty((0,23), dtype=np.int32))
        self.dates = self.history.DrawDate.to_numpy(dtype='datetime64[ns]')
        self._n = 0
        self._counts = np.zeros((5,10000), dtype=np.float64)
        header = ('DrawDate,' + ','.join(SLOT_COLS) + '\n').encode()
        running = hashlib.sha256(header)
        self.hashes = [running.hexdigest()]
        for date, codes in zip(self.history.DrawDate, self.raw):
            running.update((date.strftime('%Y-%m-%d') + ',' + ','.join(CODES[codes]) + '\n').encode())
            self.hashes.append(running.hexdigest())
        self._fits = {}
        self._fit_health = {}
        self._examples = {}

    def _counts_at(self, n, target, half_life=None):
        if half_life:
            if half_life <= 0:
                raise ValueError('half life must be positive')
            days = (np.datetime64(target.tz_localize(None).normalize()) - self.dates[:n]) / np.timedelta64(1,'D')
            weights = np.exp2(-days/half_life)
            return np.stack([np.bincount(self.raw[:n, TIERS == tier].ravel(),
                                        weights=np.repeat(weights, (TIERS == tier).sum()), minlength=10000)
                             for tier in range(5)])
        if n < self._n:
            self._counts.fill(0)
            self._n = 0
        for tier in range(5):
            self._counts[tier] += np.bincount(self.raw[self._n:n, TIERS == tier].ravel(), minlength=10000)
        self._n = n
        return self._counts.copy()

    def forecast(self, *, operator, target_time, config=None):
        config = dict(config or {'model':'uniform'})
        target = target_timestamp(target_time)
        n = int(np.searchsorted(self.dates, np.datetime64(target.tz_localize(None).normalize())))
        prefix = self.history.iloc[:n]
        if 'Operator' in prefix and not prefix.Operator.eq(operator).all():
            raise ValueError('mixed/wrong operator history')
        if operator not in ('damacai','magnum','unverified_excel','fouryes3'):
            raise ValueError('unknown operator')
        metadata = {'pipeline_version':VERSION, 'model_version':'categorical-v1',
                    'training_operator':operator, 'target_time':target.isoformat(),
                    'generated_at':datetime.now(timezone.utc).isoformat(),
                    'training_rows':n, 'corpus_cutoff':prefix.DrawDate.max().strftime('%Y-%m-%d') if n else None,
                    'corpus_sha256':self.hashes[n], 'feature_version':FEATURE_VERSION,
                    'feature_sha256':digest((FEATURE_VERSION + self.hashes[n] + target.isoformat()).encode()),
                    'configuration':config, 'config_sha256':digest(canonical_json(config).encode()),
                    'seeds':config.get('seeds',[17,29,43]), 'calibration_version':config.get('calibration_version','identity-v1'),
                    'selection_objective':'expected_matched_slots', 'joint_model':None,
                    'excluded_partial_history_rows':self.excluded_partial_rows,
                    'health':{'status':'ok','warnings':[]}}
        model = config.get('model','uniform')
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                if model == 'uniform':
                    p = UNIFORM.copy()
                elif model in ('full_code','digit'):
                    counts = self._counts_at(n,target,config.get('half_life_days'))
                    global_counts = counts.sum(axis=0,keepdims=True)
                    pooling = config.get('pooling','shared')
                    if pooling not in ('shared','tier'):
                        raise ValueError('pooling must be shared or tier')
                    active = global_counts if pooling == 'shared' else counts
                    if model == 'full_code':
                        prior = float(config.get('prior_mass',10000))
                        if prior <= 0: raise ValueError('prior mass must be positive')
                        global_p = (global_counts + prior/10000)/(global_counts.sum()+prior)
                        tau = float(config.get('tier_strength',prior))
                        if tau <= 0: raise ValueError('tier strength must be positive')
                        p = ((active + tau*global_p)/(active.sum(axis=1,keepdims=True)+tau)
                             if pooling == 'tier' else global_p)
                    else:
                        alpha = float(config.get('alpha',100))
                        if alpha <= 0: raise ValueError('digit pseudocount must be positive')
                        digits = np.stack([active[:,DIGITS[:,pos] == d].sum(axis=1)
                                           for pos in range(4) for d in range(10)],axis=1).reshape(len(active),4,10)
                        if pooling == 'tier' and 'tier_strength' in config:
                            shared = digits.sum(axis=0,keepdims=True)+alpha
                            shared /= shared.sum(axis=2,keepdims=True)
                            tau = float(config['tier_strength'])
                            digits = digits + tau*shared
                        else:
                            digits = digits + alpha
                        digits /= digits.sum(axis=2,keepdims=True)
                        p = np.prod(np.stack([digits[:,i,DIGITS[:,i]] for i in range(4)]),axis=0)
                    weight = float(config.get('uniform_weight',0))
                    if not 0 <= weight <= 1: raise ValueError('invalid uniform weight')
                    p = (1-weight)*p + weight*UNIFORM
                elif model in ('logistic','xgboost'):
                    p, fit_info = self._learner(n,target,config)
                    metadata.update(fit_info)
                elif model == 'ensemble':
                    components = config['components']; weights = np.array(config['weights'],dtype=float)
                    if len(weights)!=len(components) or (weights<0).any() or not np.isclose(weights.sum(),1):
                        raise ValueError('invalid ensemble weights')
                    fitted_through = config.get('calibration_cutoff')
                    if fitted_through and pd.Timestamp(fitted_through).date() >= target.date():
                        raise ValueError('calibration contains current/future targets')
                    p = np.zeros((23,10000))
                    health = []
                    for component, weight in zip(components,weights):
                        result = self.forecast(operator=operator,target_time=target,config=component)
                        health.append(result.metadata['health'])
                        if weight and result.metadata['health']['status'] != 'ok':
                            raise ValueError('ensemble component unavailable; entire procedure falls back to uniform')
                        p += weight*result.probabilities[result.slot_distribution]
                    metadata['component_health'] = health
                else:
                    raise ValueError(f'unknown model: {model}')
                validate_probabilities(p)
                p = (1-EPSILON)*p + EPSILON*UNIFORM
                metadata['health']['warnings'] = [str(w.message) for w in caught]
                if caught:
                    metadata['health']['status'] = 'degraded'
        except (ValueError,ImportError,RuntimeError,ArithmeticError) as error:
            p = UNIFORM.copy()
            metadata['health'] = {'status':'fallback','error':f'{type(error).__name__}: {error}','warnings':[]}
        if model in ('logistic','xgboost') and 'estimator_fit_cutoff' in metadata:
            fit_key = (metadata['estimator_fit_cutoff'], metadata['config_sha256'])
            metadata['health'] = self._fit_health.setdefault(fit_key, metadata['health'])
        metadata.setdefault('estimator_fit_cutoff',metadata['corpus_cutoff'])
        metadata.setdefault('estimator_fit_rows',n)
        return _result(p,metadata)

    def _learner(self,n,target,config):
        anchor = n//32*32
        if anchor < 32:
            raise ValueError('fewer than 32 complete draws for categorical learner')
        calendar = bool(config.get('calendar',False))
        key = (self.hashes[anchor], canonical_json(config))
        if key not in self._fits:
            from threadpoolctl import threadpool_limits
            if calendar not in self._examples:
                self._examples[calendar] = supervised_examples(self.history,calendar=calendar)
            X,y,dates = self._examples[calendar]
            end = self.history.iloc[anchor-1].DrawDate
            start = self.history.iloc[max(1,anchor-1024)].DrawDate
            mask = (dates>=start)&(dates<=end)
            X,y = X[mask],y[mask]
            models=[]
            with threadpool_limits(limits=2):
                if config['model']=='logistic':
                    from sklearn.pipeline import make_pipeline
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.linear_model import LogisticRegression
                    for pos in range(4):
                        m=make_pipeline(StandardScaler(),LogisticRegression(C=config.get('C',.1),solver='lbfgs',max_iter=150,tol=1e-5))
                        m.fit(X,y[:,pos]); models.append(m)
                else:
                    import xgboost as xgb
                    for seed in config.get('seeds',[17,29,43]):
                        heads=[]
                        for pos in range(4):
                            labels=np.unique(y[:,pos])
                            if len(labels)!=10: raise ValueError('boosting training lacks digit classes')
                            m=xgb.XGBClassifier(objective='multi:softprob',num_class=10,
                                max_depth=config.get('depth',2),n_estimators=config.get('rounds',25),
                                learning_rate=.05,reg_lambda=10,reg_alpha=1,subsample=.8,colsample_bytree=.8,
                                random_state=seed,n_jobs=2,tree_method='hist',verbosity=0)
                            m.fit(X,y[:,pos]);heads.append(m)
                        models.append(heads)
            self._fits = {key: models}  # bounded fitted memory, no unbounded fold cache
        models=self._fits[key]
        features = _target_features(self.raw[:n],self.dates[:n],target,calendar)
        if config['model']=='logistic':
            probs=[]
            for pos,m in enumerate(models):
                p=np.full((5,10),1e-12);p[:,m.classes_.astype(int)]=m.predict_proba(features)
                p/=p.sum(axis=1,keepdims=True);probs.append(p)
            digit=np.stack(probs,axis=1)
            p=np.prod(np.stack([digit[:,pos,DIGITS[:,pos]] for pos in range(4)]),axis=0)
        else:
            seed_p=[]
            for heads in models:
                digit=np.stack([m.predict_proba(features) for m in heads],axis=1)
                seed_p.append(np.prod(np.stack([digit[:,pos,DIGITS[:,pos]] for pos in range(4)]),axis=0))
            p=np.mean(seed_p,axis=0)
        # Floats from optional learners may not sum to exactly one in float64.
        p=np.asarray(p,dtype=np.float64)+1e-15;p/=p.sum(axis=1,keepdims=True)
        return p, {'estimator_fit_cutoff':self.history.iloc[anchor-1].DrawDate.strftime('%Y-%m-%d'),
                   'estimator_fit_rows':anchor,'fit_training_target_rows':min(anchor-1,1024)}


def _target_features(raw,dates,target,calendar):
    features=[]
    for window in [len(raw),30,90]:
        codes=raw[-window:] if window else raw
        digits=DIGITS[codes.ravel()]
        counts=np.stack([np.bincount(digits[:,pos],minlength=10) for pos in range(4)])+1
        features.extend((counts/counts.sum(axis=1,keepdims=True)).ravel())
    gap=float((np.datetime64(target.tz_localize(None).normalize())-dates[-1])/np.timedelta64(1,'D')) if len(dates) else 0
    features.append(gap)
    if calendar:
        features.extend([target.weekday(),np.sin(2*np.pi*target.dayofyear/365.25),np.cos(2*np.pi*target.dayofyear/365.25)])
    return np.concatenate([np.tile(features,(5,1)),np.eye(5)],axis=1).astype(np.float64)


def supervised_examples(history, *, calendar=False):
    h,_=_prepare(history)
    raw=h[SLOT_COLS].to_numpy().astype('U4').astype(np.int32)
    dates=h.DrawDate.to_numpy(dtype='datetime64[ns]')
    X=[];y=[];target_dates=[]
    for i in range(1,len(h)):
        features=_target_features(raw[:i],dates[:i],target_timestamp(h.iloc[i].DrawDate),calendar)
        X.append(features[TIERS]); y.append(DIGITS[raw[i]])
        target_dates.extend([h.iloc[i].DrawDate]*23)
    if not X: return np.empty((0,129 if calendar else 126)),np.empty((0,4),dtype=int),np.array([],dtype='datetime64[ns]')
    return np.concatenate(X),np.concatenate(y),np.array(target_dates,dtype='datetime64[ns]')


def forecast(history, *, operator, target_time, config=None):
    return ForecastEngine(history).forecast(operator=operator,target_time=target_time,config=config)


def conditional_control_probability(unique_suffixes, picks=10):
    if not 0<=unique_suffixes<=10000 or not 0<=picks<=10000:
        raise ValueError('invalid control size')
    if unique_suffixes>10000-picks:return 1.0
    return float(-math.expm1(sum(math.log1p(-unique_suffixes/(10000-i)) for i in range(picks))))


def score_forecast(result, actual):
    codes=[actual.get(col) for col in SLOT_COLS]
    if any(not isinstance(n,str) or len(n)!=4 or not n.isascii() or not n.isdigit() for n in codes):
        raise ValueError('scoring requires complete actual 23-slot draw')
    p=result.probabilities; mapping=np.array(result.slot_distribution); idx=np.array([int(n) for n in codes])
    observed=p[mapping,idx]
    hits=np.isin(codes,result.picks)
    return {'log_loss':float(-np.log(observed).mean()),
            'brier':float((np.square(p).sum(axis=1)[mapping]-2*observed+1).mean()),
            'hit_draw':int(hits.any()),'slot_hits':int(hits.sum()),
            'top3_hit':int(hits[:3].any()),'first_hit':int(hits[0]),
            'unique_picks':len(set(result.picks)), 'slots':23,
            'conditional_control':conditional_control_probability(len(set(codes))),
            'observed_slot_probabilities':observed.tolist()}


def save_forecast(result, directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    content=digest(result.probabilities.astype('<f8').tobytes()+bytes(result.slot_distribution))
    path=directory/(content+'.npz')
    if not path.exists():
        import io
        data=io.BytesIO()
        np.savez_compressed(data,probabilities=result.probabilities,slot_distribution=np.array(result.slot_distribution,dtype=np.int8))
        temporary=path.with_suffix('.tmp');temporary.write_bytes(data.getvalue());temporary.replace(path)
    return {'artifact':path.name,'artifact_sha256':digest(path.read_bytes()),'probability_sha256':content,
            'picks':result.picks,'metadata':result.metadata}


def load_forecast(reference,directory):
    base=Path(directory).resolve();path=(base/reference['artifact']).resolve()
    if not path.is_relative_to(base):raise ValueError('artifact outside store')
    if digest(path.read_bytes())!=reference['artifact_sha256']:raise ValueError('probability artifact hash mismatch')
    with np.load(path,allow_pickle=False) as data:
        result=ForecastResult(data['probabilities'],data['slot_distribution'].tolist(),reference['picks'],reference['metadata'])
    content=digest(result.probabilities.astype('<f8').tobytes()+bytes(result.slot_distribution))
    if content!=reference['probability_sha256']:raise ValueError('probability contents mismatch')
    return result
