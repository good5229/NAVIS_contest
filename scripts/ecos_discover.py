#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECOS discovery tool: find STAT_CODE and item codes by keyword.

Usage:
  ECOS_API_KEY=... python3 scripts/ecos_discover.py

It will print candidate tables and item dimensions for the following keywords:
  - 실업률 (unemployment)
  - 취업자 (employment level)
  - 소비자물가 (CPI)
  - 제조업생산지수 (industrial production index)

Endpoints used:
  - StatisticTableList
  - StatisticItemList/{STAT_CODE}
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import requests


BASE = "https://ecos.bok.or.kr/api"
API_KEY = os.environ.get("ECOS_API_KEY", "").strip()

KEYWORDS = [
    "실업률",
    "취업자",
    "소비자물가",
    "제조업생산지수",
]


def get(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def list_tables() -> List[Dict[str, Any]]:
    url = f"{BASE}/StatisticTableList/{API_KEY}/json/kr/1/10000/"
    data = get(url)
    # data['StatisticTableList']['row']
    tbl = data.get('StatisticTableList', {})
    rows = tbl.get('row', []) if isinstance(tbl, dict) else []
    return rows


def list_items(stat_code: str) -> List[Dict[str, Any]]:
    url = f"{BASE}/StatisticItemList/{API_KEY}/json/kr/1/10000/{stat_code}/"
    data = get(url)
    itm = data.get('StatisticItemList', {})
    rows = itm.get('row', []) if isinstance(itm, dict) else []
    return rows


def main() -> None:
    if not API_KEY:
        print("[discover] ECOS_API_KEY not set.")
        sys.exit(1)
    print("[discover] Listing tables...")
    tables = list_tables()
    print(f"[discover] {len(tables)} tables fetched")
    for kw in KEYWORDS:
        print(f"\n=== Keyword: {kw} ===")
        candidates = [t for t in tables if kw in str(t.get('STAT_NAME', ''))]
        if not candidates:
            print("No tables found")
            continue
        for t in candidates[:10]:
            code = t.get('STAT_CODE')
            name = t.get('STAT_NAME')
            cycle = t.get('CYCLE')
            print(f"- {code} | {cycle} | {name}")
            try:
                items = list_items(code)
                # print dimension sample
                dims = set()
                for it in items[:20]:
                    for k in it.keys():
                        if k.startswith('ITEM_') or k.startswith('C'):
                            dims.add(k)
                print(f"  items: {len(items)} rows, sample dims: {sorted(dims)}")
            except Exception as e:
                print(f"  item list error: {e}")


if __name__ == '__main__':
    main()


