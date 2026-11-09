#!/usr/bin/env python3
"""Offline, frozen-protocol experiments. Never writes the prospective ledger."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import yaml
from data_loader import load_data_cache, SLOT_COLS
from forecasting import canonical_json, digest, forecast
from forecast_evaluation import (evaluate,summarize,paired_interval,block_interval,
                                 fit_mixture,calibration_summary,development_decision)

SELECTION_POLICY=ROOT/'docs/FORECAST_SELECTION_POLICY.json'


def complexity(config,policy):
    family=policy['family_complexity_order'].index(config['model'])
    # Stronger regularization, fewer trees/components and shared pooling come first.
    detail=(int(config.get('pooling')=='tier'),int(bool(config.get('half_life_days'))),
            int(bool(config.get('calendar'))),config.get('C',0),config.get('depth',0),
            config.get('rounds',0),len(config.get('components',[])),
            -config.get('alpha',0),-config.get('prior_mass',0),-config.get('uniform_weight',0))
    return family,detail,canonical_json(config)


def search_identity(history,protocol,years):
    packages={}
    for name in ['numpy','pandas','scipy','scikit-learn','xgboost','threadpoolctl']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    return {'data_sha256':digest(history.to_csv(index=False,date_format='%Y-%m-%d',lineterminator='\n').encode()),
            'protocol_sha256':digest(json.dumps(protocol,sort_keys=True,default=str).encode()),
            'selection_policy_sha256':digest(SELECTION_POLICY.read_bytes()),'years':list(years),
            'source_hashes':{p:digest((ROOT/p).read_bytes()) for p in
                             ['forecasting.py','forecast_evaluation.py','forecast_identity.py',
                              'data_loader.py','scripts/evaluate_forecasts.py']},
            'python':platform.python_version(),'packages':packages}


@contextmanager
def time_limit(seconds):
    if seconds<=0:raise TimeoutError('no compute allocation remains')
    def expired(signum,frame):raise TimeoutError('candidate compute allocation exhausted')
    previous=signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,
        default=lambda v: v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf-8')


def round_configs(round_number,families):
    if round_number==1:
        return [{'model':'uniform'},*[{'model':'full_code','prior_mass':p} for p in [10000,100000]],
                *[{'model':'digit','alpha':p} for p in [10,100]]]
    if round_number==2:
        baseline=families['baseline']
        return [*[dict(baseline,uniform_weight=w) for w in [.5,.9,.99]],
                *[dict(baseline,uniform_weight=.9,half_life_days=h) for h in [365,1095]],
                *[dict(baseline,uniform_weight=.9,pooling='tier',tier_strength=t) for t in [1000,10000]]]
    if round_number==3:
        return [{'model':'logistic','C':c,'calendar':calendar} for c in [.01,.1,1.] for calendar in [False,True]]
    if round_number==4:
        return [{'model':'xgboost','depth':d,'rounds':n,'seeds':[17,29,43]} for d in [1,2,3] for n in [25,75]]
    subsets=[['baseline'],['logistic'],['xgboost'],['baseline','logistic'],['baseline','xgboost'],['baseline','logistic','xgboost']]
    return [{'model':'ensemble','components':[{'model':'uniform'},*[families[name] for name in subset]],
             'component_names':['uniform',*subset]} for subset in subsets if all(name in families for name in subset)]


HYPOTHESES={
 1:('Smoothed historical marginals contain stable information beyond uniform.','Reject useful gain if forward log loss fails to improve by .001 nats.'),
 2:('Stronger shrinkage, limited calendar decay or tier sharing improves sparse estimates.','Reject added assumption without .001 paired development gain.'),
 3:('Lagged digit frequencies predict the next complete draw beyond marginal counts.','Reject temporal signal if properly aligned digit heads do not improve loss.'),
 4:('Shallow nonlinear interactions improve aligned categorical digit forecasts.','Reject complexity without practical gain and acceptable runtime.'),
 5:('Distinct earlier OOF models improve proper loss through a regularized mixture.','Reject component if ablation is within uncertainty or weights shrink to uniform.')}


def attempt(history,config,output,label,*,start,end,seconds,control_count=1000):
    directory=Path(output)/label;directory.mkdir(parents=True,exist_ok=False)
    record={'label':label,'config':config,'config_sha256':digest(canonical_json(config).encode()),
            'start':str(start),'end':str(end),'started_at':datetime.now(timezone.utc).isoformat(),
            'allocated_seconds':seconds,'status':'running'}
    write_json(directory/'attempt.json',record);stamp=time.monotonic();rows=[]
    try:
        with time_limit(seconds):
            rows=evaluate(history,config,start=start,end=end,output=directory,control_count=control_count,
                          deadline=stamp+seconds)
        record['summary']=summarize(rows)
        record['status']='complete' if rows and not record['summary']['fallback_draws'] and not record['summary']['degraded_draws'] else 'unhealthy'
    except TimeoutError as error:
        record['status']='timeout';record['error']=str(error)
    except Exception as error:
        record['status']='failed';record['error']=f'{type(error).__name__}: {error}'
        (directory/'traceback.txt').write_text(traceback.format_exc())
    record['seconds']=time.monotonic()-stamp
    record['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if sys.platform!='darwin' else 1)
    write_json(directory/'attempt.json',record)
    print(f"{label}: {record['status']}, {record['seconds']:.2f}s",flush=True)
    return record,rows if record['status']=='complete' else []


def calibrated_config(history,config,*,year,output,seconds):
    if config['model']!='ensemble':return config
    stamp=time.monotonic();component_rows=[]
    for i,component in enumerate(config['components']):
        record,rows=attempt(history,component,output,f'calibration-{i}',start=f'{year-2}-01-01',end=f'{year-1}-12-31',
                            seconds=max(0,seconds-(time.monotonic()-stamp)),control_count=0)
        if record['status']!='complete':raise RuntimeError('earlier OOF component failed')
        component_rows.append(rows)
    weights=fit_mixture(component_rows,cutoff=f'{year}-01-01')
    return {**config,'weights':weights,'calibration_cutoff':f'{year-1}-12-31','calibration_version':'earlier-oof-simplex-v1'}


def run_search(history,output,protocol,*,years=(2018,2019),max_rounds=5,seconds=5100,resume_from=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    stamp=time.monotonic();records=[];complete={};families={};no_gain=0
    champion={'model':'uniform'};champion_loss=np.log(10000);best_nonuniform=None;best_nonuniform_loss=float('inf')
    champion_rows=None;policy=json.loads(SELECTION_POLICY.read_text())
    identity=search_identity(history,protocol,years)
    identity_path=output/'search-identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('search identity changed; use a new output directory')
    if not identity_path.exists() and (output/'ledger.json').exists():
        raise ValueError('legacy search has no compatible identity evidence')
    if resume_from:
        previous=Path(resume_from)/'development/search-identity.json'
        if not previous.exists() or json.loads(previous.read_text())!=identity:
            raise ValueError('resume data, code, protocol, policy or fold identity mismatch')
    write_json(identity_path,identity)
    prior = {}
    if (output/'ledger.json').exists():
        for record in json.loads((output/'ledger.json').read_text()):
            prior[canonical_json(record['config'])]=record
    for path in output.glob('r*-*/attempt.json'):
        record=json.loads(path.read_text())
        if record['status']=='running':
            record['status']='interrupted'
            record['error']='execution interrupted; partial target set ineligible'
            record['seconds']=0
            write_json(path,record)
            prior[canonical_json(record['config'])]=record
    if resume_from:
        source=Path(resume_from)
        for record in json.loads((source/'development/ledger.json').read_text()):
            prior[canonical_json(record['config'])]=record
    seconds-=sum(r['seconds'] for r in prior.values())
    for round_number in range(1,max_rounds+1):
        round_stamp=time.monotonic();round_seconds=protocol['compute']['round_seconds'][round_number-1]
        configs=sorted(round_configs(round_number,families),key=lambda config:complexity(config,policy))
        hypothesis,falsify=HYPOTHESES[round_number]
        before=champion_loss;round_completed=True;family_best={}
        write_json(output/f'round-{round_number}.json',{'hypothesis':hypothesis,'falsification':falsify,
                   'configs':configs,'allocated_seconds':min(round_seconds,seconds-(time.monotonic()-stamp))})
        for index,config in enumerate(configs):
            remaining=min(round_seconds-(time.monotonic()-round_stamp),seconds-(time.monotonic()-stamp))
            if remaining<=0 or len(records)>=30:
                round_completed=False;break
            label=f'r{round_number}-{index:02d}'
            prepared=config
            if config['model']=='ensemble':
                # Evaluate each inner year using only its earlier OOF calibration.
                all_rows=[];subrecords=[];failed=False
                for year in years:
                    fold=output/(label+f'-{year}');fold.mkdir(exist_ok=True)
                    try:
                        prepared=calibrated_config(history,config,year=year,output=fold,
                             seconds=max(0,remaining-(time.monotonic()-round_stamp)))
                        rec,rows=attempt(history,prepared,fold,'validation',start=f'{year}-01-01',end=f'{year}-12-31',
                             seconds=max(0,min(round_seconds-(time.monotonic()-round_stamp),seconds-(time.monotonic()-stamp))))
                        subrecords.append(rec);all_rows+=rows
                        if rec['status']!='complete':failed=True;break
                    except Exception as error:
                        failed=True;subrecords.append({'status':'failed','error':str(error)});break
                rec={'label':label,'config':config,'status':'failed' if failed else 'complete','folds':subrecords,
                     'summary':summarize(all_rows),'seconds':sum(r.get('seconds',0) for r in subrecords)}
                rows=[] if failed else all_rows
                write_json(output/(label+'-attempt.json'),rec)
            elif canonical_json(config) in prior:
                rec=prior[canonical_json(config)].copy()
                source=Path(rec.get('reused_from', str((Path(resume_from)/'development' if resume_from else output)/rec['label'])))
                rows=[json.loads(line) for line in (source/'predictions.jsonl').read_text().splitlines()] if rec['status']=='complete' else []
                rec['reused_from']=str(source)
            else:
                rec,rows=attempt(history,config,output,label,start=f'{min(years)}-01-01',end=f'{max(years)}-12-31',seconds=remaining)
            records.append(rec)
            write_json(output/'ledger.json',records)
            if not rows:continue
            complete[label]={'config':config,'rows':rows}
            loss=rec['summary']['mean_log_loss']
            if config['model']!='uniform' and loss<best_nonuniform_loss:
                best_nonuniform=config;best_nonuniform_loss=loss
            if config['model']!='uniform' and (not family_best or loss<family_best['loss']):
                family_best={'loss':loss,'config':config}
            if config['model']=='uniform':
                champion_rows=rows
                decision={'accepted':False,'reason':'initial-uniform-reference'}
            else:
                remaining=min(round_seconds-(time.monotonic()-round_stamp),seconds-(time.monotonic()-stamp))
                decision_start=time.monotonic()
                try:
                    with time_limit(remaining):
                        decision=development_decision(rows,champion_rows,policy,deadline=time.monotonic()+remaining)
                except TimeoutError:
                    decision={'accepted':False,'reason':'selection-timeout'}
                    round_completed=False
                decision.update(candidate=config,incumbent=champion)
                if decision['accepted']:
                    champion=config;champion_loss=loss;champion_rows=rows
                rec['seconds']+=time.monotonic()-decision_start
            rec['selection_decision']=decision
            write_json(output/(label+'-decision.json'),decision)
            write_json(output/'ledger.json',records)
            if not round_completed:break
        if round_number==1:
            baselines=[v for v in complete.values() if v['config']['model'] in ('digit','full_code')]
            if not baselines:break
            families['baseline']=min(baselines,key=lambda v:summarize(v['rows'])['mean_log_loss'])['config']
        if round_number in (3,4) and family_best:
            families['logistic' if round_number==3 else 'xgboost']=family_best['config']
        gain=before-champion_loss
        if round_completed:no_gain=no_gain+1 if gain<.001 else 0
        write_json(output/f'round-{round_number}-review.json',{'completed':round_completed,'qualifying_gain':gain>=.001,
                   'gain_nats':gain,'consecutive_no_gain':no_gain,'champion':champion,
                   'critique':'Historical development selection is exploratory; fit warnings/failures disqualify candidates. No edge inferred from ranks.'})
        if not round_completed or no_gain>=3 or time.monotonic()-stamp>=seconds:break
    selection={'selected':champion,'development_loss':float(champion_loss),'simplest_nonuniform':families.get('baseline'),
               'selection_policy_sha256':digest(SELECTION_POLICY.read_bytes()),
               'best_observed_nonuniform':best_nonuniform,'best_observed_nonuniform_loss':best_nonuniform_loss if np.isfinite(best_nonuniform_loss) else None,
               'attempted':len(records),'seconds':time.monotonic()-stamp,'rounds_completed':round_number,
               'stop_reason':'three_no_gain_rounds' if no_gain>=3 else 'budget_or_round_limit',
               'historical_evidence':'exploratory; no prospective confirmation'}
    write_json(output/'selection.json',selection)
    return selection,complete


def frozen_evaluation(history,selection,output,protocol,*,seconds):
    output=Path(output);output.mkdir(exist_ok=True);stamp=time.monotonic();outer=[]
    freeze={'selection':selection,'frozen_at':datetime.now(timezone.utc).isoformat(),
            'selection_sha256':digest(canonical_json(selection).encode()),'no_outer_feedback':True,
            'search_identity':search_identity(history,protocol,(2020,2023))}
    if (output/'freeze.json').exists():
        previous=json.loads((output/'freeze.json').read_text())
        if any(previous.get(key)!=freeze[key] for key in ['selection_sha256','search_identity']):
            raise ValueError('frozen evaluation identity changed')
    if not (output/'freeze.json').exists():
        write_json(output/'freeze.json',freeze)
    for year in range(2020,2024):
        remaining=seconds-(time.monotonic()-stamp)
        if remaining<=0:break
        fold=output/str(year);fold.mkdir(exist_ok=True)
        if (fold/'outer/attempt.json').exists():
            outer.append({'year':year,'evaluation':json.loads((fold/'outer/attempt.json').read_text())});continue
        # Replay the frozen selection procedure, limited to rounds reached in development.
        annual,unused=run_search(history,fold/'inner',protocol,years=(year-2,year-1),
                                max_rounds=selection['rounds_completed'],seconds=remaining*.75/(2024-year))
        config=annual['selected']
        if config['model']=='ensemble':config=calibrated_config(history,config,year=year,output=fold/'calibration',seconds=seconds-(time.monotonic()-stamp))
        rec,rows=attempt(history,config,fold,'outer',start=f'{year}-01-01',end=f'{year}-12-31',seconds=seconds-(time.monotonic()-stamp))
        outer.append({'year':year,'selection':annual,'evaluation':rec})
        write_json(output/'outer.json',outer)
    remaining=seconds-(time.monotonic()-stamp)
    if remaining>0:
        rec,rows=attempt(history,selection['selected'],output,'inspected-retrospective',start='2024-01-01',end='2026-11-01',seconds=remaining)
    return outer


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',default='docs/FORECAST_PROTOCOL.yaml')
    parser.add_argument('--phase',choices=['development','full','smoke'],default='development')
    parser.add_argument('--offline',action='store_true',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--data',default='experiments/data-v3/damacai-through-20261101.csv')
    parser.add_argument('--resume-from',help='reuse unchanged attempts after an infrastructure failure; retains original evidence')
    args=parser.parse_args(argv)
    output=Path(args.output).resolve()
    if output==ROOT or ROOT not in output.parents:parser.error('output must be a separate directory inside repository')
    output.mkdir(parents=True,exist_ok=False)
    protocol_bytes=Path(args.protocol).read_bytes();protocol=yaml.safe_load(protocol_bytes)
    expected=json.loads((ROOT/'experiments/data-v3/manifest.json').read_text())
    if digest(protocol_bytes)!=expected['protocol_sha256']:raise ValueError('protocol changed after data manifest freeze')
    if digest(Path(args.data).read_bytes())!=expected['frozen_csv_sha256']:raise ValueError('frozen data hash mismatch')
    history=pd.read_csv(args.data,dtype={c:str for c in SLOT_COLS});history.DrawDate=pd.to_datetime(history.DrawDate);history['Operator']='damacai'
    started=time.monotonic()
    manifest={'protocol_sha256':digest(protocol_bytes),'data_sha256':digest(Path(args.data).read_bytes()),
              'selection_policy_sha256':digest(SELECTION_POLICY.read_bytes()),
              'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'working_source_hashes':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in [ROOT/'forecasting.py',ROOT/'forecast_evaluation.py',Path(__file__).resolve()]},
              'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
              'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
              'started_at':datetime.now(timezone.utc).isoformat(),'args':vars(args),'model_compute_seconds':0}
    write_json(output/'manifest.json',manifest)
    if args.phase=='smoke':
        rec,rows=attempt(history,{'model':'uniform'},output,'uniform',start='2026-10-28',end='2026-11-01',seconds=60)
        selection={'selected':{'model':'uniform'},'smoke_only':True}
        if rec['status']!='complete':
            manifest['smoke_failure']=rec['status']
            write_json(output/'manifest.json',manifest)
            return 1
    else:
        selection,complete=run_search(history,output/'development',protocol,seconds=5100,resume_from=args.resume_from)
        write_json(output/'frozen-selection.json',selection)
        if complete:
            uniform=next(v['rows'] for v in complete.values() if v['config']['model']=='uniform')
            comparisons={}
            for label,value in complete.items():
                comparisons[label]={str(length):paired_interval(value['rows'],uniform,block_length=length) for length in [13,26]}
            write_json(output/'development-comparisons.json',comparisons)
            best=min(complete.values(),key=lambda v:summarize(v['rows'])['mean_log_loss'])
            write_json(output/'calibration.json',calibration_summary(best['rows']))
            differences=np.array([r['log_loss']-np.log(10000) for r in best['rows']])
            uncertainty=block_interval(differences,alpha=.05)
            write_json(output/'power.json',{'future_targets':1000,'development_draws':len(differences),
                'loss_mde_approx_80pct_power':float((1.96+.84)*np.std(differences,ddof=1)/np.sqrt(1000)),
                'limitation':'iid variance approximation; block intervals separately quantify dependence; small hit-rate lifts remain underpowered',
                'expected_uniform_hit_draws_1000':float(1000*(1-.999**23)),'development_interval':uncertainty})
        if args.phase=='full':frozen_evaluation(history,selection,output/'frozen-evaluation',protocol,seconds=min(2100,7200-(time.monotonic()-started)))
    manifest['model_compute_seconds']=time.monotonic()-started
    manifest['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if sys.platform!='darwin' else 1)
    manifest['completed_at']=datetime.now(timezone.utc).isoformat()
    write_json(output/'manifest.json',manifest)
    print(canonical_json(selection))
    return 0


if __name__=='__main__':raise SystemExit(main())
