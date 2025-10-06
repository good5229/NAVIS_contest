#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic KOSIS indicator collector (regional by year) using Param API.

Config: data/kosis/config.json
Output: data/kosis/{name}.csv with columns: region, year, value

Notes:
- Uses KOSIS OpenAPI Param/statisticsParameterData.do similar to existing fiscal collector
- Expects per-indicator: orgId, tblId, itmId(s), objL1 region codes (space-separated), period range
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import requests
import pandas as pd
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'kosis'
CONFIG_PATH = DATA_DIR / 'config.json'
BASE_URL = 'https://kosis.kr/openapi/Param/statisticsParameterData.do'


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def fetch_year(ind: Dict[str, Any], api_key: str, year: int) -> List[Dict[str, Any]]:
    params = {
        'apiKey': api_key,
        'method': 'getList',
        'format': 'json',
        'jsonVD': 'Y',
        'prdSe': 'Y',
        'startPrdDe': str(year),
        'endPrdDe': str(year),
        'orgId': ind['orgId'],
        'tblId': ind['tblId'],
        'itmId': ind['itmId'],
        'objL1': ind['objL1'],
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    if r.status_code != 200:
        return []
    try:
        data = r.json()
        if isinstance(data, list):
            rows = []
            for item in data:
                region_name = item.get('C1_NM') or item.get('C1')
                if not region_name or region_name == '전국':
                    continue
                val = item.get('DT')
                try:
                    value = float(val)
                except Exception:
                    continue
                rows.append({'region': region_name, 'year': year, 'value': value})
            return rows
        return []
    except Exception:
        return []


def save_csv(name: str, rows: List[Dict[str, Any]]) -> Path:
    out = DATA_DIR / f'{name}.csv'
    df = pd.DataFrame(rows)
    if df.empty:
        return out
    df = df.groupby(['region', 'year'], as_index=False)['value'].mean()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def main() -> None:
    # Load .env if present (gitignored)
    dotenv_path = PROJECT_ROOT / '.env'
    if dotenv_path.exists():
        try:
            for line in dotenv_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip(); v = v.strip()
                if k and v and k not in os.environ:
                    os.environ[k] = v
        except Exception:
            pass
    cfg = load_config(CONFIG_PATH)
    api_key = cfg.get('apiKey', '').strip() or os.environ.get('KOSIS_API_KEY', '').strip()
    if not api_key:
        print('[kosis_collector] Missing apiKey in config.json')
        return
    indicators = cfg.get('indicators', [])
    if not indicators:
        print('[kosis_collector] No indicators configured')
        return
    for ind in indicators:
        name = ind['name']
        start = int(ind.get('startYear', 2019))
        end = int(ind.get('endYear', start))
        all_rows: List[Dict[str, Any]] = []
        for y in range(start, end + 1):
            rows = fetch_year(ind, api_key, y)
            all_rows.extend(rows)
        out = save_csv(name, all_rows)
        print(f'[kosis_collector] Saved {name} -> {out} ({len(all_rows)} rows)')


if __name__ == '__main__':
    main()


