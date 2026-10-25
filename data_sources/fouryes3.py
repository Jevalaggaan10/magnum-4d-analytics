"""4dyes3.com adapter — config-driven because the real spec is not public.

The base widget URL (https://4dyes3.com/en/api-widget) returns HTTP 403 to
unauthenticated fetches, so this adapter reads endpoint paths and field
mappings from `fouryes3_config.yaml` and tries them politely. If requests
fail it marks the source as unavailable instead of crashing the app.

To enable: paste a working curl into `fouryes3_config.yaml`, optionally
set `FOURYES3_API_KEY` env var, then call:
    python -m data_sources.manager sync fouryes3 --recent 30
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests
import yaml

from .base import LotterySource, NormalizedDraw


CONFIG_PATH = Path(__file__).with_name("fouryes3_config.yaml")


def _load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class FourYes3(LotterySource):
    name = "fouryes3"

    def __init__(self, config_path: Path = CONFIG_PATH, polite_sleep_seconds: float = 1.0, timeout: float = 15.0):
        self.config = _load_config(config_path)
        self.timeout = timeout
        self.polite_sleep = polite_sleep_seconds

        headers = dict(self.config.get("headers") or {})
        env_key = os.environ.get("FOURYES3_API_KEY")
        if env_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {env_key}"

        self.session = requests.Session()
        self.session.headers.update(headers)
        self.base_url = (self.config.get("base_url") or "").rstrip("/")
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _get_json(self, url: str, params: dict | None = None) -> dict | list | None:
        try:
            r = self.session.get(url, params=params, timeout=self.timeout)
            if r.status_code >= 400:
                self._last_error = f"HTTP {r.status_code} on {url}"
                return None
            return r.json()
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            return None

    def fetch_dates(self) -> list[str]:
        cfg = self.config.get("list_endpoint") or {}
        path = cfg.get("path")
        if not (self.base_url and path):
            self._last_error = "missing base_url/list_endpoint.path in fouryes3_config.yaml"
            return []

        data = self._get_json(self.base_url + path)
        if data is None:
            return []

        dates_field = cfg.get("dates_field") or "data"
        date_key = cfg.get("date_key") or "date"

        items = data.get(dates_field) if isinstance(data, dict) else data
        if not isinstance(items, list):
            self._last_error = f"expected list at '{dates_field}', got {type(items).__name__}"
            return []

        out: list[str] = []
        for item in items:
            if isinstance(item, str):
                out.append(item.replace("-", "").strip())
            elif isinstance(item, dict):
                v = item.get(date_key)
                if isinstance(v, str):
                    out.append(v.replace("-", "").strip())
        return sorted({d for d in out if d}, reverse=True)

    def fetch_draw(self, draw_date: str) -> NormalizedDraw | None:
        cfg = self.config.get("draw_endpoint") or {}
        path_template = cfg.get("path")
        if not (self.base_url and path_template):
            self._last_error = "missing base_url/draw_endpoint.path in fouryes3_config.yaml"
            return None

        time.sleep(self.polite_sleep)
        url = self.base_url + path_template.format(date=draw_date)
        data = self._get_json(url)
        if data is None:
            return None

        fields = cfg.get("fields") or {}
        node = data
        if isinstance(data, dict) and (op_field := cfg.get("operator_field")):
            preferred = cfg.get("preferred_operator")
            if preferred and isinstance(data.get(op_field), list):
                match = [d for d in data[op_field] if str(d.get("name", "")).lower() == preferred.lower()]
                if match:
                    node = match[0]

        def pick(key: str):
            v = node.get(fields.get(key, key)) if isinstance(node, dict) else None
            return v

        first = str(pick("first") or "")
        second = str(pick("second") or "")
        third = str(pick("third") or "")
        special = pick("special") or []
        consolation = pick("consolation") or []
        if isinstance(special, dict):
            special = list(special.values())
        if isinstance(consolation, dict):
            consolation = list(consolation.values())

        if not (first or second or third):
            return None
        return NormalizedDraw(
            draw_date=draw_date,
            first=first,
            second=second,
            third=third,
            special=[str(x) for x in (special or [])][:10],
            consolation=[str(x) for x in (consolation or [])][:10],
            source=self.name,
        )

    def available(self) -> bool:
        return bool(self.fetch_dates())


if __name__ == "__main__":
    src = FourYes3()
    dates = src.fetch_dates()
    if not dates:
        print(f"4dyes3 unavailable: {src.last_error}")
    else:
        print(f"4dyes3 OK, {len(dates)} dates, first 5: {dates[:5]}")
