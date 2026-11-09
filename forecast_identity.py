"""Scientific series identity and explicit compatibility for published evidence."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from forecasting import FEATURE_VERSION, canonical_json, digest, target_timestamp

ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = 'docs/FORECAST_PROTOCOL.yaml'
REQUIRED_SOURCES = (
    'forecasting.py', 'forecast_evaluation.py', 'probability_tracker.py',
    'forecast_identity.py', 'data_loader.py', 'data_sources/base.py',
    'data_sources/damacai.py', 'scripts/track_predictions.py',
    'requirements-tracker.txt', 'constraints-py312.txt', 'data/damacai.csv',
    'data/provenance.json',
)


def models(settings):
    return {name: settings[name] for name in ('champion', 'shadow', 'simplest_nonuniform')
            if settings.get(name)}


def procedure(settings):
    """Runtime flags and individual growing corpus hashes are provenance, not identity."""
    return {
        'confirmation_version': settings.get('confirmation_version'),
        'start_date': settings.get('start_date'),
        'checkpoint': settings.get('checkpoint_scheduled_targets', 1000),
        'operator': 'damacai', 'slots': 'all23', 'picks': 10,
        'models': models(settings), 'feature_version': FEATURE_VERSION,
        'calibration': 'declared-in-model-configuration',
        'lineage': 'tracked-damacai-snapshot-plus-accepted-observations-v1',
        'selection_objective': 'expected_matched_slots',
        'protocol_sha256': settings.get('protocol_sha256'),
        'selection_evidence_sha256': settings.get('selection_evidence_sha256'),
    }


def procedure_hash(settings):
    return digest(canonical_json(procedure(settings)).encode())


def execution_hash(settings):
    return digest(canonical_json({'source_hashes': settings.get('source_hashes'),
                                  'protocol_sha256': settings.get('protocol_sha256')}).encode())


def record_identity(settings):
    return {'confirmation_version': settings['confirmation_version'],
            'procedure_sha256': procedure_hash(settings),
            'execution_revision_sha256': execution_hash(settings)}


def published_fingerprint(record):
    fields = ('for_date', 'forecast_id', 'target_time', 'publication_deadline',
              'special_announcement', 'generated_at', 'picks', 'forecasts',
              'confirmation_version', 'settings_sha256', 'published_at', 'publication_evidence')
    return digest(canonical_json({key: record.get(key) for key in fields}).encode())


def compatibility(settings, root=ROOT):
    """Legacy mappings require an exact pinned, append-only evidence document."""
    relative = settings.get('compatibility_manifest')
    if not relative:
        return {}
    base = Path(root).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError('compatibility manifest outside repository')
    raw = path.read_bytes()
    if digest(raw) != settings.get('compatibility_sha256'):
        raise ValueError('compatibility manifest hash mismatch')
    manifest = json.loads(raw)
    if (manifest.get('confirmation_version') != settings.get('confirmation_version') or
            manifest.get('procedure_sha256') != procedure_hash(settings)):
        raise ValueError('compatibility manifest belongs to a different procedure')
    return manifest


def validate_execution(settings, root=ROOT):
    """Check approved bytes; never refresh frozen hashes from the working tree."""
    base = Path(root).resolve()
    if not settings.get('confirmation_version') or not settings.get('start_date'):
        raise ValueError('declared confirmation series and start date required')
    if settings.get('checkpoint_scheduled_targets') != 1000:
        raise ValueError('confirmation checkpoint must contain 1000 scheduled targets')
    if 'champion' not in models(settings):
        raise ValueError('frozen champion configuration required')
    hashes = settings.get('source_hashes', {})
    if not isinstance(hashes, dict) or not set(REQUIRED_SOURCES).issubset(hashes):
        raise ValueError('frozen source manifest is incomplete')
    expected = {**hashes, PROTOCOL_PATH: settings.get('protocol_sha256')}
    if settings.get('selection_evidence'):
        expected[settings['selection_evidence']] = settings.get('selection_evidence_sha256')
    for relative, checksum in expected.items():
        path = (base / relative).resolve()
        if not path.is_relative_to(base) or not re.fullmatch('[0-9a-f]{64}', checksum or ''):
            raise ValueError(f'invalid frozen file evidence: {relative}')
        if not path.is_file() or digest(path.read_bytes()) != checksum:
            raise ValueError(f'frozen file mismatch: {relative}')
    manifest = compatibility(settings, root)
    if manifest and execution_hash(settings) not in manifest.get('compatible_revisions', []):
        raise ValueError('execution revision has no compatibility attestation')
    return execution_hash(settings)


def scheduled_dates(settings, end):
    start = settings.get('start_date')
    if not start or not end:
        return []
    specials = settings.get('specials', {})
    for date, evidence in specials.items():
        deadline=target_timestamp(pd.Timestamp(date))+pd.Timedelta(hours=18)
        if (not evidence.get('url','').startswith('https://www.damacai.com.my/') or
                not evidence.get('announced_at') or
                target_timestamp(evidence['announced_at'])>=deadline):
            raise ValueError('special target lacks prior official announcement evidence')
    return [day.strftime('%Y%m%d') for day in pd.date_range(pd.Timestamp(start), pd.Timestamp(end))
            if day.weekday() in (2, 5, 6) or day.strftime('%Y%m%d') in specials]


def confirmation_rows(records, settings=None):
    """Validate before filtering a checkpoint; never replace invalid rows with successes."""
    rows = list(records)
    selected = settings.get('confirmation_version') if settings else None
    versions = {row.get('confirmation_version') for row in rows}
    if selected:
        # Unlabelled rows cannot safely be classified as an archived series.
        if None in versions:
            raise ValueError('record has no confirmation series')
        rows = [row for row in rows if row.get('confirmation_version') == selected]
    elif rows and (len(versions) != 1 or not all(versions)):
        raise ValueError('one explicitly identified confirmation series is required')
    dates = [row['for_date'] for row in rows]
    if len(set(dates)) != len(dates):
        raise ValueError('duplicate confirmation target')
    expected = procedure_hash(settings) if settings else None
    manifest = compatibility(settings) if settings else {}
    identities = set()
    revisions = set()
    for row in rows:
        if row.get('status')=='missing-forecast' and not row.get('coverage') and not row.get('forecasts'):
            # A missed scheduled target has no executed forecast to attest.
            continue
        identity = row.get('procedure_sha256')
        revision = row.get('execution_revision_sha256')
        if not identity:
            evidence = manifest.get('legacy_records', {}).get(row['for_date'])
            if evidence != published_fingerprint(row):
                raise ValueError('missing or unsupported legacy identity')
            identity = manifest.get('procedure_sha256')
            revision = 'legacy-reviewed'
        if (not re.fullmatch('[0-9a-f]{64}', identity or '') or
                (revision!='legacy-reviewed' and not re.fullmatch('[0-9a-f]{64}',revision or ''))):
            raise ValueError('invalid procedure or execution identity')
        if expected and identity != expected:
            raise ValueError('incompatible procedure within confirmation series')
        if settings and revision != 'legacy-reviewed':
            approved = {execution_hash(settings), *manifest.get('compatible_revisions', [])}
            if revision not in approved:
                raise ValueError('unapproved execution revision')
        identities.add(identity)
        revisions.add(revision)
        if settings and row.get('coverage') and not row.get('forecasts'):
            raise ValueError('covered forecast lacks model evidence')
        if settings and row.get('forecasts'):
            if set(row['forecasts']) != set(models(settings)):
                raise ValueError('forecast model set differs from frozen procedure')
            for name, config in models(settings).items():
                meta = row['forecasts'][name]['metadata']
                if (meta.get('configuration') != config or
                        meta.get('training_operator') != 'damacai' or
                        meta.get('feature_version') != FEATURE_VERSION):
                    raise ValueError('forecast configuration or lineage mismatch')
    if len(identities) > 1:
        raise ValueError('incompatible procedures within confirmation series')
    if not settings and len(revisions)>1:
        raise ValueError('multiple execution revisions require explicit compatibility evidence')
    rows.sort(key=lambda row: row['for_date'])
    if settings and rows:
        # Missing regular dates stay in the denominator even if absent from a ledger.
        by_date = {row['for_date']: row for row in rows}
        specials=dict(settings.get('specials',{}))
        specials.update({row['for_date']:row['special_announcement'] for row in rows
                         if row.get('special_announcement')})
        expected_dates = scheduled_dates({**settings,'specials':specials}, rows[-1]['for_date'])
        if any(date not in expected_dates for date in by_date):
            raise ValueError('target outside declared confirmation schedule')
        rows = [by_date.get(date, {'for_date': date, 'coverage': 0,
                                 'status': 'missing-forecast'}) for date in expected_dates]
    return rows
