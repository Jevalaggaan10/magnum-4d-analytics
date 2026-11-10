"""Isolated exploratory Magnum audit. Never writes the live ledger or model settings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import requests

from data_loader import SLOT_COLS, load_4d_txt
from forecasting import ForecastEngine, canonical_json, conditional_control_probability, score_forecast
from forecast_evaluation import block_interval
from scripts.track_predictions import generate_picks

OUT = Path(__file__).resolve().parent


class ArchiveTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            self.cell = []

    def handle_data(self, text):
        if self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(' '.join(' '.join(self.cell).split()))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def archive():
    source = OUT / 'lottochaos.html'
    if not source.exists():
        response = requests.get('https://lottochaos.com/en/magnum-4d/archive/', timeout=30)
        response.raise_for_status()
        source.write_text(response.content.decode('utf-8'))
    parser = ArchiveTable()
    parser.feed(source.read_text())
    rows = []
    for cells in parser.rows:
        if not cells or not re.fullmatch(r'\d{2}\.\d{2}\.\d{4}', cells[0]):
            continue
        numbers = cells[1].split() + cells[2].split() + cells[3].split()
        assert len(numbers) == 23 and all(re.fullmatch('[0-9]{4}', number) for number in numbers)
        date = datetime.strptime(cells[0], '%d.%m.%Y')
        assert date.date() <= datetime.now(timezone.utc).date()
        rows.append(dict(zip(SLOT_COLS, numbers), DrawDate=date, Operator='magnum'))
    frame = pd.DataFrame(rows).sort_values('DrawDate').reset_index(drop=True)
    assert frame.DrawDate.is_unique and len(frame) >= 40
    frame.to_csv(OUT / 'magnum-2026-secondary-archive.csv', index=False, date_format='%Y-%m-%d')
    return frame


def corroborate(targets):
    directory = OUT / 'secondary-pages'
    directory.mkdir(exist_ok=True)
    reports = []
    session = requests.Session()
    for _, actual in targets.sort_values('DrawDate', ascending=False).iterrows():
        date = actual.DrawDate.strftime('%Y-%m-%d')
        path = directory / (date + '.html')
        url = f'https://starbucks4d.com/4d-result/magnum/{date}/'
        try:
            if not path.exists():
                response = session.get(url, timeout=30)
                response.raise_for_status()
                path.write_text(response.content.decode('utf-8'))
                time.sleep(1)
            raw = path.read_text()
            codes = re.findall(r'data-n="([0-9]{4})"', raw)
            assert len(codes) == 23, f'{date}: found {len(codes)} result numbers'
            expected = [actual[col] for col in SLOT_COLS]
            # Special/consolation publication order can vary across archives.
            same = codes[:3] == expected[:3] and sorted(codes[3:13]) == sorted(expected[3:13]) and sorted(codes[13:]) == sorted(expected[13:])
            reports.append({'date': date, 'url': url, 'all_23_agree': same,
                            'response_sha256': hashlib.sha256(raw.encode()).hexdigest()})
        except Exception as error:
            reports.append({'date': date, 'url': url, 'all_23_agree': False,
                            'error': f'{type(error).__name__}: {error}'})
        write_json('source-crosschecks.json', reports)
        print('crosscheck', date, reports[-1]['all_23_agree'], flush=True)
    return reports


def evaluate(frame):
    config = json.loads((ROOT / 'forecast_config.json').read_text())
    spec = {'as_of': datetime.now(timezone.utc).isoformat(), 'draws': 40, 'slots': 23,
            'kind': 'retrospective exploratory transfer benchmark; not prospective Magnum forecasts',
            'models': {name: config[name] for name in ['champion', 'shadow', 'simplest_nonuniform']},
            'legacy': 'unchanged generate_picks; descriptive consensus, no probability metrics',
            'model_selection': 'none; settings copied before evaluating target outcomes',
            'sources': ['https://lottochaos.com/en/magnum-4d/archive/',
                        'https://starbucks4d.com/4d-result/magnum/'],
            'source_limit': 'two secondary archives; official result retrieval unavailable; shared upstream possible',
            'current_repository': 'cbf189b', 'random_control_replicates': 1000, 'control_seed': 17029}
    write_json('analysis-spec.json', spec)
    bundled = load_4d_txt(ROOT / '4D.txt')
    bundled['Operator'] = 'magnum'
    all_history = pd.concat([bundled, frame], ignore_index=True).sort_values('DrawDate')
    assert not all_history.DrawDate.duplicated().any()
    engine = ForecastEngine(all_history)
    targets = frame.tail(40)
    targets.to_csv(OUT / 'latest-40-magnum-draws.csv', index=False, date_format='%Y-%m-%d')
    rng = np.random.default_rng(17029)
    control_hits = np.zeros(1000, dtype=int)
    comparisons = []
    for _, actual in targets.iterrows():
        date = actual.DrawDate.strftime('%Y%m%d')
        codes = [actual[col] for col in SLOT_COLS]
        row = {'for_date': date, 'actual': {col: actual[col] for col in SLOT_COLS}, 'models': {}}
        for name, model in spec['models'].items():
            result = engine.forecast(operator='magnum', target_time=actual.DrawDate + pd.Timedelta(hours=19), config=model)
            assert result.metadata['corpus_cutoff'] < actual.DrawDate.strftime('%Y-%m-%d')
            metrics = score_forecast(result, actual)
            row['models'][name] = {**metrics, 'picks': result.picks, 'health': result.metadata['health'],
                                   'training_rows': result.metadata['training_rows'],
                                   'training_cutoff': result.metadata['corpus_cutoff'],
                                   'corpus_sha256': result.metadata['corpus_sha256']}
        prefix = all_history.loc[all_history.DrawDate < actual.DrawDate]
        picks, methods = generate_picks(prefix)
        assert len(picks) == len(set(picks)) == 10
        hit_codes = sorted(set(picks) & set(codes))
        row['models']['legacy_consensus'] = {'picks': picks, 'hit_draw': int(bool(hit_codes)),
            'slot_hits': sum(code in picks for code in codes), 'top3_hit': int(bool(set(picks) & set(codes[:3]))),
            'first_hit': int(codes[0] in picks), 'hits': hit_codes, 'methods': methods,
            'training_rows': len(prefix), 'training_cutoff': prefix.DrawDate.max().strftime('%Y-%m-%d')}
        probability = conditional_control_probability(len(set(codes)))
        for i in range(1000):
            ticket = {f'{number:04d}' for number in rng.choice(10000, 10, replace=False)}
            control_hits[i] += bool(ticket.intersection(codes))
        row['random_ticket_hit_probability'] = probability
        comparisons.append(row)
        (OUT / 'comparisons.jsonl').write_text(''.join(canonical_json(r) + '\n' for r in comparisons))
        print('evaluated', date, {name: r['slot_hits'] for name, r in row['models'].items()}, flush=True)
    summary = {'window': [targets.DrawDate.min().strftime('%Y-%m-%d'), targets.DrawDate.max().strftime('%Y-%m-%d')],
               'draws': 40, 'prize_slots': 920, 'published_magnum_forecasts': 0, 'models': {},
               'random_ticket_expected_hit_draws': sum(r['random_ticket_hit_probability'] for r in comparisons),
               'random_ticket_probability_of_no_hits': float(np.prod([1-r['random_ticket_hit_probability'] for r in comparisons])),
               'random_control_replicates': 1000, 'random_control_mean_hit_draws': float(control_hits.mean()),
               'random_control_95pct_range': np.quantile(control_hits, [.025, .975]).tolist(),
               'evidence_status': 'exploratory only; not prospective and not operator-certified'}
    for name in comparisons[0]['models']:
        rows = [r['models'][name] for r in comparisons]
        result = {metric: sum(row[metric] for row in rows) for metric in ['hit_draw', 'slot_hits', 'top3_hit', 'first_hit']}
        if name != 'legacy_consensus':
            result.update(mean_log_loss=float(np.mean([row['log_loss'] for row in rows])),
                          mean_brier=float(np.mean([row['brier'] for row in rows])))
            difference = [row['log_loss'] - np.log(10000) for row in rows]
            result['loss_difference_intervals'] = {str(length): block_interval(difference, block_length=length)
                                                   for length in [13, 26]}
        summary['models'][name] = result
    write_json('summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crosscheck', action='store_true')
    args = parser.parse_args()
    data = archive()
    if args.crosscheck:
        checks = corroborate(data.tail(40))
        assert len(checks) == 40 and all(row['all_23_agree'] for row in checks), 'archive disagreement: inspect source-crosschecks.json'
    else:
        evaluate(data)
