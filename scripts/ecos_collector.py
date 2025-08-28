#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECOS indicator collector (configurable).

Purpose:
- Fetch ECOS indicators by region and year according to a JSON config
- Save each indicator as CSV in data/ecos/{indicator_name}.csv with columns: region, year, value

Notes:
- Requires ECOS API key in environment variable ECOS_API_KEY
- Config path (default): data/ecos/config.json (use provided config.sample.json as a template)
- The collector is conservative: if API key or config is missing, it exits without error
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'ecos'
DEFAULT_CONFIG = DATA_DIR / 'config.json'


def load_config(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def collect_indicator(indicator: Dict[str, Any], api_key: str) -> Optional[pd.DataFrame]:
    base_url: str = indicator.get('base_url', 'https://ecos.bok.or.kr/api/StatisticSearch/json')
    name: str = indicator['name']
    common_params: Dict[str, Any] = indicator.get('common_params', {})
    regions: List[Dict[str, Any]] = indicator.get('regions', [])
    year_range: Dict[str, int] = indicator.get('year_range', {})
    start_year = int(year_range.get('start', 2015))
    end_year = int(year_range.get('end', start_year))
    response_mapping: Dict[str, str] = indicator.get('response_mapping', {
        'items_path': 'StatisticSearch.row',
        'year_field': 'TIME',
        'value_field': 'DATA_VALUE'
    })

    # Build outputs
    all_rows: List[Dict[str, Any]] = []

    # We assume ECOS requires API key path style: /apiKey/format/lang/...
    # But here we support query params style for simplicity; many ECOS proxies accept this
    for reg in regions:
        region_name = reg['region']
        region_params = reg.get('params', {})
        for year in range(start_year, end_year + 1):
            params = {
                **common_params,
                **region_params,
                'apiKey': api_key,
                'p_year': str(year),
            }
            try:
                r = requests.get(base_url, params=params, timeout=30)
                if r.status_code != 200:
                    continue
                data = r.json()
                # Navigate to items
                items_path = response_mapping.get('items_path', 'StatisticSearch.row')
                # support dotted path
                cur: Any = data
                for part in items_path.split('.'):
                    if isinstance(cur, dict):
                        cur = cur.get(part, None)
                    else:
                        cur = None
                    if cur is None:
                        break
                if not isinstance(cur, list):
                    continue
                year_field = response_mapping.get('year_field', 'TIME')
                value_field = response_mapping.get('value_field', 'DATA_VALUE')
                # pick rows matching the exact year (string compare tolerant)
                for item in cur:
                    yval = str(item.get(year_field, '')).strip()
                    if yval != str(year):
                        continue
                    val_str = str(item.get(value_field, '')).strip()
                    try:
                        val = float(val_str)
                    except Exception:
                        continue
                    all_rows.append({'region': region_name, 'year': year, 'value': val})
            except Exception:
                continue

    if not all_rows:
        return None
    df = pd.DataFrame(all_rows)
    # Aggregate duplicates if any (mean)
    df = df.groupby(['region', 'year'], as_index=False)['value'].mean()
    return df


def save_indicator_csv(df: pd.DataFrame, name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f'{name}.csv'
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_config(config_path)
    api_key = os.environ.get('ECOS_API_KEY', '').strip()

    if cfg is None:
        print(f'[ecos_collector] No config found at {config_path}, skipping.')
        return
    if not api_key:
        print('[ecos_collector] ECOS_API_KEY not set, skipping API collection.')
        return

    indicators: List[Dict[str, Any]] = cfg.get('indicators', [])
    if not indicators:
        print('[ecos_collector] No indicators in config, nothing to do.')
        return

    saved = 0
    for ind in indicators:
        name = ind.get('name')
        if not name:
            continue
        df = collect_indicator(ind, api_key)
        if df is None or df.empty:
            print(f'[ecos_collector] No data for {name}.')
            continue
        out = save_indicator_csv(df, name)
        print(f'[ecos_collector] Saved {name} -> {out} ({len(df)} rows)')
        saved += 1

    if saved == 0:
        print('[ecos_collector] No files saved.')


if __name__ == '__main__':
    main()


