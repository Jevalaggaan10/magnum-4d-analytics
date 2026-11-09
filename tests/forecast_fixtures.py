"""Isolated approved settings for tracker behavior tests."""
from pathlib import Path

from forecasting import digest
from forecast_identity import PROTOCOL_PATH, REQUIRED_SOURCES

ROOT = Path(__file__).resolve().parents[1]


def settings(**overrides):
    result = {'champion': {'model': 'uniform'}, 'confirmation_version': 'test-series',
              'start_date': '20261107', 'checkpoint_scheduled_targets': 1000,
              'protocol_sha256': digest((ROOT / PROTOCOL_PATH).read_bytes()),
              'source_hashes': {p: digest((ROOT / p).read_bytes()) for p in REQUIRED_SOURCES}}
    result.update(overrides)
    return result
