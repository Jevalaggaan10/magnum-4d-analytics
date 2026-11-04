"""Draw-level proper scoring, controls, chronological calibration and inference."""
from __future__ import annotations

import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd

from data_loader import SLOT_COLS
from forecasting import (ForecastEngine, DIGITS, canonical_json, save_forecast,
                         score_forecast, target_timestamp)

BINS = np.array([0,.00005,.0001,.0002,.0005,.001,.01,1.00000001])


def calibration_diagnostics(result, actual):
    """Aggregate code events; uncertainty must resample entire draw records."""
    p=result.probabilities
    multiplicity=np.bincount(result.slot_distribution,minlength=len(p))
    expected=np.zeros(7);observed=np.zeros(7);events=np.zeros(7)
    for i,distribution in enumerate(p):
        bins=np.clip(np.searchsorted(BINS,distribution,side='right')-1,0,6)
        expected+=multiplicity[i]*np.bincount(bins,weights=distribution,minlength=7)
        events+=multiplicity[i]*np.bincount(bins,minlength=7)
    digit_expected=np.zeros((4,10));digit_observed=np.zeros((4,10))
    for slot,col in enumerate(SLOT_COLS):
        distribution=p[result.slot_distribution[slot]];code=int(actual[col])
        which=int(np.clip(np.searchsorted(BINS,distribution[code],side='right')-1,0,6))
        observed[which]+=1
        for pos in range(4):
            digit_expected[pos]+=np.bincount(DIGITS[:,pos],weights=distribution,minlength=10)
            digit_observed[pos,DIGITS[code,pos]]+=1
    chosen=[int(n) for n in result.picks]
    selected_mass=float(sum(p[i,chosen].sum() for i in result.slot_distribution))
    return {'bin_edges':BINS.tolist(),'expected_events':expected.tolist(),
            'observed_events':observed.tolist(),'code_events':events.tolist(),
            'digit_expected':digit_expected.tolist(),'digit_observed':digit_observed.tolist(),
            'expected_selected_slot_hits':selected_mass}


def controls(for_date,actual,count=1000):
    actual_codes=[actual[col] for col in SLOT_COLS]
    unique=set(actual_codes)
    date_seed=int(str(for_date).replace('-',''))
    date_picks={f'{i:04d}' for i in random.Random(date_seed).sample(range(10000),10)}
    # Independent target/replicate streams, never module-global RNG or best seed.
    rng=np.random.default_rng(np.random.SeedSequence([17029,date_seed]))
    hit_draws=slot_hits=0
    for _ in range(count):
        picks={f'{i:04d}' for i in rng.choice(10000,10,replace=False)}
        hit_draws+=bool(picks & unique)
        slot_hits+=sum(n in picks for n in actual_codes)
    return {'date_control_hit':int(bool(date_picks & unique)),
            'date_control_slot_hits':sum(n in date_picks for n in actual_codes),
            'random_controls':count,'random_control_hit_mean':hit_draws/count if count else None,
            'random_control_slot_hits_mean':slot_hits/count if count else None,
            'uniform_pick_hit':int(bool({f'{i:04d}' for i in range(10)} & unique))}


def evaluate(history,config,*,start,end,output=None,operator='damacai',control_count=1000,deadline=None):
    engine=ForecastEngine(history)
    targets=engine.history.loc[engine.history.DrawDate.between(pd.Timestamp(start),pd.Timestamp(end))]
    rows=[]
    stream=None
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        stream=(output/'predictions.jsonl').open('w',encoding='utf-8')
    try:
        for _,actual in targets.iterrows():
            if deadline is not None and time.monotonic()>=deadline:
                raise TimeoutError('candidate evaluation time budget exhausted; partial fold set ineligible')
            target=actual.DrawDate.strftime('%Y%m%d');stamp=time.perf_counter()
            result=engine.forecast(operator=operator,target_time=actual.DrawDate+pd.Timedelta(hours=19),config=config)
            scores=score_forecast(result,actual.to_dict())
            row={'for_date':target,**scores,'training_rows':result.metadata['training_rows'],
                 'training_cutoff':result.metadata['corpus_cutoff'],'corpus_sha256':result.metadata['corpus_sha256'],
                 'health':result.metadata['health'],'coverage':1,'seconds':time.perf_counter()-stamp,
                 'calibration':calibration_diagnostics(result,actual),**controls(target,actual,control_count)}
            if output is not None:
                row['forecast']=save_forecast(result,output/'probabilities')
                stream.write(canonical_json(row)+'\n');stream.flush()
            rows.append(row)
    finally:
        if stream:stream.close()
    return rows


def summarize(rows):
    valid=[r for r in rows if r.get('coverage') and 'log_loss' in r]
    return {'scheduled_or_eligible_draws':len(rows),'scored_draws':len(valid),
            'coverage':len(valid)/len(rows) if rows else 0,
            'fallback_draws':sum(r.get('health',{}).get('status')=='fallback' for r in valid),
            'degraded_draws':sum(r.get('health',{}).get('status')=='degraded' for r in valid),
            'mean_log_loss':float(np.mean([r['log_loss'] for r in valid])) if valid else None,
            'mean_brier':float(np.mean([r['brier'] for r in valid])) if valid else None,
            'hit_draws':sum(r.get('hit_draw',0) for r in valid),
            'slot_hits':sum(r.get('slot_hits',0) for r in valid),
            'top3_hit_draws':sum(r.get('top3_hit',0) for r in valid),
            'first_hit_draws':sum(r.get('first_hit',0) for r in valid),
            'forecast_seconds':sum(r.get('seconds',0) for r in valid)}


def block_interval(values,*,block_length=13,replicates=10000,alpha=.0125,seed=17029):
    values=np.asarray(values,dtype=float)
    if not len(values) or not np.isfinite(values).all():raise ValueError('finite paired draw values required')
    if block_length<1 or replicates<1:raise ValueError('positive blocks/replicates required')
    n=len(values);length=min(block_length,n);blocks=int(np.ceil(n/length))
    rng=np.random.default_rng(seed);means=[]
    for begin in range(0,replicates,256):
        starts=rng.integers(0,n-length+1,size=(min(256,replicates-begin),blocks))
        indices=(starts[:,:,None]+np.arange(length)).reshape(len(starts),-1)[:,:n]
        means.extend(values[indices].mean(axis=1))
    return {'mean':float(values.mean()),'lower':float(np.quantile(means,alpha/2)),
            'upper':float(np.quantile(means,1-alpha/2)),'draws':n,'block_length':block_length,
            'replicates':replicates,'confidence':1-alpha,
            'assumptions':'short-range dependence and locally stationary paired draw differences; exploratory historical evidence'}


def paired_interval(rows,baseline,*,metric='log_loss',block_length=13,replicates=10000):
    a={r['for_date']:r for r in rows};b={r['for_date']:r for r in baseline}
    if a.keys()!=b.keys():raise ValueError('paired comparisons require identical eligible draw dates')
    return block_interval([a[d][metric]-b[d][metric] for d in sorted(a)],block_length=block_length,replicates=replicates)


def fit_mixture(component_rows,*,cutoff,minimum_draws=100):
    """Learn simplex weights from strictly earlier OOF actual-slot probabilities."""
    from scipy.optimize import minimize
    if not component_rows:raise ValueError('no components')
    keys=[r['for_date'] for r in component_rows[0]]
    if any([r['for_date'] for r in rows]!=keys for rows in component_rows):raise ValueError('OOF targets must align')
    if any(pd.Timestamp(d).date()>=pd.Timestamp(cutoff).date() for d in keys):raise ValueError('current/future calibration targets')
    uniform=np.zeros(len(component_rows));uniform[0]=1
    if len(keys)<minimum_draws:return uniform.tolist()
    actual=np.array([[r['observed_slot_probabilities'] for r in rows] for rows in component_rows])
    if not np.isfinite(actual).all() or not (actual>0).all():raise ValueError('invalid OOF probabilities')
    def objective(w):return float(-np.log(np.tensordot(w,actual,axes=1)).mean()+.001*np.square(w-uniform).sum())
    fit=minimize(objective,uniform,method='SLSQP',bounds=[(0,1)]*len(uniform),
                 constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'ftol':1e-10,'maxiter':200})
    if not fit.success:raise RuntimeError('mixture optimizer failed: '+fit.message)
    weights=np.maximum(fit.x,0);return (weights/weights.sum()).tolist()


def confirmation_status(records,checkpoint=1000):
    rows=sorted(records,key=lambda r:r['for_date'])[:checkpoint]
    if len(rows)<checkpoint:return {'status':'pending','scheduled_targets':len(rows),'remaining':checkpoint-len(rows)}
    if any(not r.get('coverage') for r in rows):return {'status':'incomplete-coverage','scheduled_targets':checkpoint}
    return {'status':'ready','scheduled_targets':checkpoint}


def calibration_summary(rows,replicates=10000):
    if not rows:return {}
    diagnostics=[r['calibration'] for r in rows]
    expected=np.array([r['expected_events'] for r in diagnostics]);observed=np.array([r['observed_events'] for r in diagnostics])
    total=np.array([r['code_events'] for r in diagnostics]).sum(axis=0)
    intervals=[]
    for i in range(7):
        intervals.append(block_interval(observed[:,i]-expected[:,i],replicates=replicates))
    return {'bin_edges':BINS.tolist(),'expected_events':expected.sum(axis=0).tolist(),
            'observed_events':observed.sum(axis=0).tolist(),'code_events':total.tolist(),
            'draw_level_observed_minus_expected_intervals':intervals,
            'digit_expected':np.sum([r['digit_expected'] for r in diagnostics],axis=0).tolist(),
            'digit_observed':np.sum([r['digit_observed'] for r in diagnostics],axis=0).tolist(),
            'limitation':'pooled calibration does not validate individual rare-code probabilities or predictive information'}


def promotion_decision(records,*,development_qualified=False,correctness_verified=False,replicates=10000):
    """Fixed checkpoint only. Never expose interim comparative confirmation scores."""
    status=confirmation_status(records)
    result={**status,'probability_promotion':False,'exact_hit_advantage':False}
    if status['status']!='ready':return result
    rows=sorted(records,key=lambda r:r['for_date'])[:1000]
    if any(r.get('status')!='verified' or not all(k in r.get('metrics',{}) for k in ['champion','shadow','simplest_nonuniform']) for r in rows):
        return {**result,'status':'incomplete-scoring'}
    contrasts={
        'loss_uniform':[r['metrics']['shadow']['log_loss']-np.log(10000) for r in rows],
        'loss_simplest':[r['metrics']['shadow']['log_loss']-r['metrics']['simplest_nonuniform']['log_loss'] for r in rows],
        'hit_uniform':[r['metrics']['shadow']['hit_draw']-r['metrics']['champion']['hit_draw'] for r in rows],
        'hit_control':[r['metrics']['shadow']['hit_draw']-(r['conditional_control'] if 'conditional_control' in r
                         else r['metrics']['champion']['conditional_control']) for r in rows]}
    intervals={str(length):{name:block_interval(values,block_length=length,replicates=replicates) for name,values in contrasts.items()} for length in [13,26]}
    probability=development_qualified and correctness_verified and all(
        v[name]['mean']<=-.001 and v[name]['upper']<0 for v in intervals.values() for name in ['loss_uniform','loss_simplest'])
    hits=probability and all(v[name]['lower']>0 for v in intervals.values() for name in ['hit_uniform','hit_control'])
    return {**result,'status':'evaluated','probability_promotion':bool(probability),'exact_hit_advantage':bool(hits),'intervals':intervals}
