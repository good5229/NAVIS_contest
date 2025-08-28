#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute BDS-like baseline from official datasets only (KOSIS/ECOS).

Currently uses KOSIS datasets available in the repo:
- data/kosis/kosis_gdp_data_2023.csv (regional GDP proxy)
- data/fiscal_autonomy/kosis_fiscal_autonomy_data.csv (fiscal ratios)

Method (deterministic, transparent):
- Choose latest common year across indicators (prefers 2023 if available)
- Normalize each indicator across regions (z-score -> min-max 0..1)
- Weighted sum to 0..10 scale for BDS baseline

Weights (can be tuned / documented):
- GDP proxy: 0.6
- Fiscal autonomy ratio: 0.4

Output:
- data/bds/bds_baseline.json
  { "latestYear": YYYY, "baselines": { region: number (0..10) } }

Note:
- If ECOS datasets are added later, extend INDICATORS with ECOS sources and recompute.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


DATA_DIR = Path('data')
KOSIS_GDP_CSV = DATA_DIR / 'kosis' / 'kosis_gdp_data_2023.csv'
FISCAL_CSV = DATA_DIR / 'fiscal_autonomy' / 'kosis_fiscal_autonomy_data.csv'
ECOS_DIR = DATA_DIR / 'ecos'
KOSIS_INDICATORS_DIR = DATA_DIR / 'kosis'
OUT_JSON = DATA_DIR / 'bds' / 'bds_baseline.json'

# Weights configuration (literature-inspired defaults)
# If ECOS indicators exist and are listed in ECOS_INDICATOR_CONFIG, we use those weights.
# Otherwise, any extra ECOS indicators will share the remaining ECOS weight equally.
GDP_WEIGHT = 0.30
FISCAL_WEIGHT = 0.20

# Per-indicator ECOS config: key is expected CSV stem (filename without .csv)
# higher_is_better=False means we invert after normalization (1 - norm)
ECOS_INDICATOR_CONFIG = {
    # Labor market
    'ecos_unemployment_rate': {
        'weight': 0.05,
        'higher_is_better': False
    },
    'ecos_employment_level': {
        'weight': 0.20,
        'higher_is_better': True
    },
    # Prices
    'ecos_cpi': {
        'weight': 0.05,
        'higher_is_better': False  # 낮을수록 안정적이라고 가정
    },
    # Production / activity
    'ecos_industrial_production_index': {
        'weight': 0.20,
        'higher_is_better': True
    },
    # KOSIS regional counterparts accepted with same semantics
    'kosis_unemployment_rate': {
        'weight': 0.05,
        'higher_is_better': False
    },
    'kosis_employment_level': {
        'weight': 0.20,
        'higher_is_better': True
    },
    'kosis_cpi': {
        'weight': 0.05,
        'higher_is_better': False
    },
    'kosis_industrial_production_index': {
        'weight': 0.20,
        'higher_is_better': True
    },
}

# If the sum of configured ECOS weights is less than this cap, the remainder can be assigned to any extra ECOS
# indicators found (equally). If more than cap, weights will be renormalized.
ECOS_WEIGHT_CAP = 0.50


def load_kosis_gdp() -> Optional[pd.DataFrame]:
    if not KOSIS_GDP_CSV.exists():
        return None
    df = pd.read_csv(KOSIS_GDP_CSV)
    # Try to detect expected columns
    # Heuristics: columns containing 'region' and numeric value (GDP)
    cols = [c for c in df.columns]
    region_col = None
    value_col = None
    year_col = None
    # simple guesses
    for c in cols:
        low = c.lower()
        if region_col is None and ('region' in low or '지역' in low or '행정구역' in low or '시도' in low):
            region_col = c
        if year_col is None and ('year' in low or '연도' in low):
            year_col = c
    # value col: last numeric column or the first column with many numbers
    numeric_candidates = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_candidates:
        value_col = numeric_candidates[-1]
    else:
        # try to coerce a likely value column
        for c in cols:
            try:
                pd.to_numeric(df[c])
                value_col = c
                break
            except Exception:
                continue
    if region_col is None:
        # fallback known column names
        for c in cols:
            if c.strip() in ['지역', '시도', '행정구역', 'Region', '지역명']:
                region_col = c
                break
    # If year column missing, assume single-year dataset; inject 2023
    if year_col is None:
        df['year'] = 2023
        year_col = 'year'
    # Rename
    df = df.rename(columns={region_col: 'region', year_col: 'year', value_col: 'gdp_value'})
    # keep only needed columns
    keep = ['region', 'year', 'gdp_value']
    df = df[keep].copy()
    # filter out aggregates like 전국 if included
    df = df[df['region'].notna()]
    return df


def load_fiscal_autonomy() -> pd.DataFrame:
    df = pd.read_csv(FISCAL_CSV)
    # expected columns: year, region, fiscal_autonomy_ratio
    needed = ['year', 'region', 'fiscal_autonomy_ratio']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in fiscal CSV: {missing}")
    df = df[needed].copy()
    return df


def min_max_scale(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    min_v = float(s.min())
    max_v = float(s.max())
    if max_v == min_v:
        return pd.Series([0.5] * len(s), index=s.index)  # all equal
    return (s - min_v) / (max_v - min_v)


def detect_region_year_value_cols(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)
    region_col = None
    year_col = None
    value_col = None
    for c in cols:
        low = str(c).lower()
        if region_col is None and ('region' in low or '지역' in low or '행정구역' in low or '시도' in low or '지역명' in low):
            region_col = c
        if year_col is None and ('year' in low or '연도' in low or '기간' in low):
            year_col = c
    # numeric candidate for value
    numeric_candidates = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_candidates:
        value_col = numeric_candidates[-1]
    else:
        for c in cols:
            try:
                pd.to_numeric(df[c])
                value_col = c
                break
            except Exception:
                continue
    if year_col is None:
        # try to coerce a year column from typical names
        for c in cols:
            if str(c).strip() in ['year', '연도']:
                year_col = c
                break
    if region_col is None:
        raise ValueError('Region column not found')
    if year_col is None:
        raise ValueError('Year column not found')
    if value_col is None:
        raise ValueError('Value column not found')
    return {'region': region_col, 'year': year_col, 'value': value_col}


def load_ecos_indicators() -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    if not ECOS_DIR.exists():
        return dfs
    for csv in ECOS_DIR.glob('*.csv'):
        try:
            df_raw = pd.read_csv(csv)
            cols = detect_region_year_value_cols(df_raw)
            df = df_raw.rename(columns={cols['region']: 'region', cols['year']: 'year', cols['value']: 'value'})
            # ensure types
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
            df = df[df['year'].notna()].copy()
            df['year'] = df['year'].astype(int)
            df = df[['region', 'year', 'value']]
            df['__indicator_name'] = csv.stem
            dfs.append(df)
        except Exception:
            continue
    return dfs


def load_kosis_indicators() -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    if not KOSIS_INDICATORS_DIR.exists():
        return dfs
    for csv in KOSIS_INDICATORS_DIR.glob('*.csv'):
        try:
            df_raw = pd.read_csv(csv)
            cols = detect_region_year_value_cols(df_raw)
            df = df_raw.rename(columns={cols['region']: 'region', cols['year']: 'year', cols['value']: 'value'})
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
            df = df[df['year'].notna()].copy()
            df['year'] = df['year'].astype(int)
            df = df[['region', 'year', 'value']]
            df['__indicator_name'] = csv.stem  # expect names like kosis_unemployment_rate
            dfs.append(df)
        except Exception:
            continue
    return dfs


def compute_bds_baseline() -> Dict:
    gdp_df = load_kosis_gdp()
    fiscal_df = load_fiscal_autonomy()
    ecos_dfs = load_ecos_indicators()
    kosis_dfs = load_kosis_indicators()

    # Determine target year: intersect years
    fiscal_years = set(int(y) for y in fiscal_df['year'].unique())
    candidate_years = fiscal_years.copy()
    use_gdp = False
    if gdp_df is not None:
        gdp_years = set(int(y) for y in gdp_df['year'].unique())
        candidate_years = candidate_years & gdp_years
        use_gdp = True
    use_ecos = False
    for edf in ecos_dfs:
        edf_years = set(int(y) for y in edf['year'].unique())
        candidate_years = candidate_years & edf_years
        use_ecos = True
    for kdf in kosis_dfs:
        kyears = set(int(y) for y in kdf['year'].unique())
        candidate_years = candidate_years & kyears
        use_ecos = True
    if candidate_years:
        target_year = sorted(candidate_years)[-1]
    else:
        # If no full intersection, fall back to fiscal (and GDP if available) only
        if gdp_df is not None:
            common = sorted(fiscal_years & set(int(y) for y in gdp_df['year'].unique()))
            if common:
                target_year = common[-1]
                use_ecos = False
                ecos_dfs = []
                use_gdp = True
            else:
                target_year = max(fiscal_years)
                use_gdp = False
                use_ecos = False
                ecos_dfs = []
        else:
            target_year = max(fiscal_years)
            use_gdp = False
            use_ecos = False
            ecos_dfs = []

    fiscal_y = fiscal_df[fiscal_df['year'] == target_year].copy()
    fiscal_y = fiscal_y[fiscal_y['region'] != '전국']

    # Merge indicators
    merged = fiscal_y.rename(columns={'fiscal_autonomy_ratio': 'fiscal_autonomy'})[['region', 'fiscal_autonomy']]
    if use_gdp:
        gdp_y = gdp_df[gdp_df['year'] == target_year].copy()
        gdp_y = gdp_y[gdp_y['region'] != '전국']
        merged = merged.merge(gdp_y[['region', 'gdp_value']], on='region', how='inner')
    else:
        merged['gdp_value'] = None

    # Attach ECOS indicators
    ecos_names: List[str] = []
    if use_ecos:
        # ECOS files
        for edf in ecos_dfs:
            name = edf['__indicator_name'].iloc[0]
            ecos_names.append(name)
            edf_y = edf[edf['year'] == target_year].copy()
            edf_y = edf_y[edf_y['region'].notna()]
            col_name = f"{name}"
            merged = merged.merge(edf_y[['region', 'value']].rename(columns={'value': col_name}), on='region', how='inner')
        # KOSIS regional counterparts
        for kdf in kosis_dfs:
            name = kdf['__indicator_name'].iloc[0]
            ecos_names.append(name)
            kdf_y = kdf[kdf['year'] == target_year].copy()
            kdf_y = kdf_y[kdf_y['region'].notna()]
            col_name = f"{name}"
            merged = merged.merge(kdf_y[['region', 'value']].rename(columns={'value': col_name}), on='region', how='inner')

    # Normalize to 0..1
    merged['fiscal_norm'] = min_max_scale(merged['fiscal_autonomy'])
    if use_gdp:
        merged['gdp_norm'] = min_max_scale(merged['gdp_value'])
    else:
        merged['gdp_norm'] = 0.0

    ecos_norm_cols: List[str] = []
    if use_ecos and ecos_names:
        for name in ecos_names:
            col = f"{name}"
            ncol = f"{name}__norm"
            merged[ncol] = min_max_scale(merged[col])
            # 방향성 반영: 낮을수록 좋은 지표는 역변환
            cfg = ECOS_INDICATOR_CONFIG.get(name, None)
            if cfg and (cfg.get('higher_is_better') is False):
                merged[ncol] = 1.0 - merged[ncol]
            ecos_norm_cols.append(ncol)
        # average ECOS normalized values
        if ecos_norm_cols:
            # 가중 평균: 지표별 가중치 적용
            weights: List[float] = []
            extra_names: List[str] = []
            configured_sum = 0.0
            for name in ecos_names:
                cfg = ECOS_INDICATOR_CONFIG.get(name)
                if cfg and 'weight' in cfg:
                    w = float(cfg['weight'])
                    configured_sum += w
                else:
                    extra_names.append(name)
            # 남는 몫을 extra에 균등 분배
            remaining = max(ECOS_WEIGHT_CAP - configured_sum, 0.0)
            extra_w = (remaining / len(extra_names)) if extra_names else 0.0
            # 열 순서에 맞춰 가중치 리스트 구성
            for name in ecos_names:
                cfg = ECOS_INDICATOR_CONFIG.get(name)
                if cfg and 'weight' in cfg:
                    weights.append(float(cfg['weight']))
                else:
                    weights.append(extra_w)
            # 정규화(합=1) 후 가중 평균
            total = sum(weights) if sum(weights) > 0 else 1.0
            weights = [w / total for w in weights]
            ecos_matrix = merged[[f"{n}__norm" for n in ecos_names]].values
            merged['ecos_norm_mean'] = (ecos_matrix * pd.Series(weights)).sum(axis=1)
        else:
            merged['ecos_norm_mean'] = 0.0
    else:
        merged['ecos_norm_mean'] = 0.0

    # Weighted sum -> 0..10
    w_gdp = GDP_WEIGHT
    w_fiscal = FISCAL_WEIGHT
    # ECOS 가중치는 위에서 지표내 가중평균으로 이미 반영되었으므로 생태계 블록 전체에 대한 상위 가중치를 적용
    # 블록 가중치는 ECOS_WEIGHT_CAP을 상한으로 사용
    w_ecos = ECOS_WEIGHT_CAP if (use_ecos and len(ecos_norm_cols) > 0) else 0.0
    # Normalize weights to sum to 1.0
    total_w = w_gdp + w_fiscal + w_ecos
    if total_w == 0:
        # fallback equal weights among available components
        components = 0
        if use_gdp:
            components += 1
        components += 1  # fiscal
        if use_ecos and len(ecos_norm_cols) > 0:
            components += 1
        w_gdp = (1.0 / components) if use_gdp else 0.0
        w_fiscal = 1.0 / components
        w_ecos = (1.0 / components) if (use_ecos and len(ecos_norm_cols) > 0) else 0.0
        total_w = w_gdp + w_fiscal + w_ecos
    w_gdp /= total_w
    w_fiscal /= total_w
    w_ecos /= total_w
    merged['bds_score'] = (merged['gdp_norm'] * w_gdp + merged['fiscal_norm'] * w_fiscal + merged['ecos_norm_mean'] * w_ecos) * 10.0

    baselines = {row['region']: float(row['bds_score']) for _, row in merged.iterrows()}
    return {"latestYear": int(target_year), "baselines": baselines}


def main() -> None:
    result = compute_bds_baseline()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {OUT_JSON} with {len(result['baselines'])} regions, year {result['latestYear']}")


if __name__ == '__main__':
    main()


