"""Publication-aware v3 records beside, never replacing, legacy predictions."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pandas as pd

from data_loader import SLOT_COLS, normalize_prize_number
from forecasting import (forecast,save_forecast,load_forecast,score_forecast,
                         target_timestamp,digest,canonical_json)
from forecast_evaluation import controls, confirmation_status, promotion_decision

REGULAR_WEEKDAYS={2,5,6}


def choose_target(now,*,blocked_dates=(),specials=None,start_date=None):
    now=target_timestamp(now)
    specials=specials or {}
    verified={}
    for date,announcement in specials.items():
        if not announcement.get('url','').startswith('https://www.damacai.com.my/') or not announcement.get('announced_at'):
            raise ValueError('special draw requires official URL and prior announcement timestamp')
        announced=target_timestamp(announcement['announced_at'])
        draw=pd.Timestamp(datetime.strptime(date,'%Y%m%d')).tz_localize(now.tzinfo)+pd.Timedelta(hours=19)
        if announced>=draw-pd.Timedelta(hours=1) or announced>now:raise ValueError('special announcement is late or in future')
        verified[date]=announcement
    for days in range(370):
        day=now.normalize()+pd.Timedelta(days=days)
        date=day.strftime('%Y%m%d')
        target=day+pd.Timedelta(hours=19);deadline=day+pd.Timedelta(hours=18)
        if (day.weekday() in REGULAR_WEEKDAYS or date in verified) and now<deadline and date not in blocked_dates and (not start_date or date>=start_date):
            return {'for_date':date,'target_time':target.isoformat(),'publication_deadline':deadline.isoformat(),
                    'schedule_source':'https://www.damacai.com.my/general-information/',
                    'special_announcement':verified.get(date)}
    raise ValueError('no eligible scheduled target within one year')


def accept_observations(state,recent,corpus,*,now):
    accepted={}
    existing={r.DrawDate.strftime('%Y%m%d'):{c:normalize_prize_number(r.get(c)) for c in SLOT_COLS}
              for _,r in corpus.iterrows()}
    for date,actual in recent.items():
        if not all(normalize_prize_number(actual.get(c)) is not None for c in SLOT_COLS):continue
        normalized={c:normalize_prize_number(actual[c]) for c in SLOT_COLS}
        if date in existing and existing[date]!=normalized:
            record={'for_date':date,'existing':existing[date],'new':normalized,
                    'retrieved_at':target_timestamp(now).isoformat(),'source':'damacai',
                    'reason':'conflicting complete result; frozen history retained'}
            queue=state.setdefault('quarantined_observations',[])
            if not any(q['for_date']==date and q['new']==normalized for q in queue):queue.append(record)
            continue
        # No result from a draw whose scheduled start is still in the future.
        draw=target_timestamp(pd.Timestamp(datetime.strptime(date,'%Y%m%d'))+pd.Timedelta(hours=19))
        if draw>target_timestamp(now):continue
        accepted[date]=normalized
        state.setdefault('observation_provenance',{}).setdefault(date,{
            'source':'damacai','retrieved_at':target_timestamp(now).isoformat(),
            'content_sha256':digest(canonical_json(normalized).encode())})
    return accepted


def confirm_publication(state,now,*,evidence=None):
    """Called only AFTER first successful push; CI URL is retained as evidence."""
    now=target_timestamp(now)
    for record in state.get('probability_forecasts',{}).values():
        if record.get('status')!='awaiting-publication':continue
        if now<target_timestamp(record['publication_deadline']):
            record.update(status='published',published_at=now.isoformat(),publication_evidence=evidence,
                          coverage=1)
        else:
            record.update(status='missed-publication',coverage=0,publication_checked_at=now.isoformat())


def _actual_by_date(corpus):
    return {r.DrawDate.strftime('%Y%m%d'):r.to_dict() for _,r in corpus.iterrows()}


def advance_forecasts(state,corpus,*,now,artifact_dir,settings,dry_run=False,network_ok=True):
    now=target_timestamp(now);records=state.setdefault('probability_forecasts',{})
    actuals=_actual_by_date(corpus)
    # Resolve saved artifacts, never refit a published prediction to score it.
    for date,record in records.items():
        if record.get('status')=='awaiting-publication' and now>=target_timestamp(record['publication_deadline']):
            record.update(status='missed-publication',coverage=0)
        if record.get('status')=='published' and date in actuals and now>=target_timestamp(record['target_time']):
            actual=actuals[date]
            if not all(normalize_prize_number(actual.get(c)) is not None for c in SLOT_COLS):continue
            metrics={}
            for name,reference in record['forecasts'].items():
                result=load_forecast(reference,artifact_dir)
                metrics[name]=score_forecast(result,actual)
            record.update(status='verified',actual={c:actual[c] for c in SLOT_COLS},metrics=metrics,
                          checked_at=now.isoformat(),**controls(date,actual))
    state['probability_confirmation'] = promotion_decision(list(records.values()),
        development_qualified=settings.get('development_qualified', False),
        correctness_verified=settings.get('correctness_verified', False))
    start=settings.get('start_date')
    if start:
        day=target_timestamp(pd.Timestamp(datetime.strptime(start,'%Y%m%d')))
        while day.normalize()<=now.normalize():
            date=day.strftime('%Y%m%d')
            if (day.weekday() in REGULAR_WEEKDAYS or date in settings.get('specials',{})) and now>=day+pd.Timedelta(hours=18) and date not in records:
                records[date]={'for_date':date,'status':'missing-forecast','coverage':0,
                               'reason':'no completed publication before deadline','pipeline_version':'damacai-v3'}
            day+=pd.Timedelta(days=1)
    blocked=set()
    if state.get('pending'):blocked.add(state['pending']['for_date'])
    target=choose_target(now,blocked_dates=blocked,specials=settings.get('specials'),start_date=start)
    if settings.get('requested_date'):
        requested=settings['requested_date']
        requested_day=target_timestamp(pd.Timestamp(datetime.strptime(requested,'%Y%m%d')))
        if requested in blocked or now>=requested_day+pd.Timedelta(hours=18):
            raise ValueError('target already published or publication deadline passed')
        if requested_day.weekday() not in REGULAR_WEEKDAYS and requested not in settings.get('specials',{}):
            raise ValueError('special target lacks verified announcement')
        target=choose_target(now,blocked_dates=blocked,specials=settings.get('specials'),start_date=requested)
        if target['for_date']!=requested:
            raise ValueError('explicit date is not an eligible target')
    date=target['for_date']
    if date in records or not network_ok:return
    models={'champion':settings.get('champion',{'model':'uniform'})}
    if settings.get('shadow'):models['shadow']=settings['shadow']
    if settings.get('simplest_nonuniform'):models['simplest_nonuniform']=settings['simplest_nonuniform']
    references={}
    for name,config in models.items():
        result=forecast(corpus,operator='damacai',target_time=target['target_time'],config=config)
        reference=({'picks':result.picks,'metadata':result.metadata,'artifact':None,'dry_run':True}
                   if dry_run else save_forecast(result,artifact_dir))
        references[name]=reference
    records[date]={**target,'forecast_id':'damacai-v3:'+date,
                   'pipeline_version':'damacai-v3','generated_at':now.isoformat(),
                   'status':'awaiting-publication','coverage':0,'source':'damacai',
                   'picks':references['champion']['picks'],'forecasts':references,
                   'confirmation_version':settings.get('confirmation_version'),
                   'settings_sha256':digest(canonical_json(settings).encode())}


def render_probability_log(state):
    records=state.get('probability_forecasts',{})
    if not records:return ''
    lines=['','## Probability forecasts — damacai-v3','',
           'Ten picks maximize expected matched slots. Single-slot probabilities are categorical; any-hit independence approximation is not a verified joint model.',
           '', '| Target | Publication/result status | Picks | Training cutoff |',
           '|---|---|---|---|']
    for date,record in sorted(records.items()):
        meta=record.get('forecasts',{}).get('champion',{}).get('metadata',{})
        lines.append(f"| {date} | {record['status']} | {', '.join(record.get('picks',[])) or '—'} | {meta.get('corpus_cutoff','—')} |")
    status=confirmation_status(list(records.values()))
    lines+=['',f"Confirmation: **{status['status']}**, {status['scheduled_targets']}/1000 scheduled targets. Aggregate comparisons withheld before fixed checkpoint.",
            'Published legacy records above retain their original experiment identities.']
    return '\n'.join(lines)+'\n'
