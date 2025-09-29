#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECOS 데이터를 포함하여 BDS 재계산
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

def update_bds_with_ecos():
    """ECOS 데이터를 포함하여 BDS 지수 재계산"""
    
    # 기존 데이터 로드
    print("기존 데이터 로드 중...")
    
    # 1. KOSIS 재정자립도 데이터
    fiscal_file = Path('data/fiscal_autonomy/kosis_fiscal_autonomy_data.csv')
    fiscal_df = pd.read_csv(fiscal_file)
    fiscal_2023 = fiscal_df[fiscal_df['year'] == 2023].copy()
    fiscal_2023 = fiscal_2023[fiscal_2023['region'] != '전국']
    
    # 2. KOSIS GDP 데이터
    gdp_file = Path('data/kosis/kosis_gdp_data_2023.csv')
    gdp_df = pd.read_csv(gdp_file)
    
    # 3. ECOS 데이터
    ecos_cpi_file = Path('data/ecos/ecos_cpi.csv')
    ecos_mfg_file = Path('data/ecos/ecos_industrial_production_index.csv')
    
    if not ecos_cpi_file.exists() or not ecos_mfg_file.exists():
        print("ECOS 데이터가 없습니다. 기존 BDS를 유지합니다.")
        return
    
    ecos_cpi_df = pd.read_csv(ecos_cpi_file)
    ecos_mfg_df = pd.read_csv(ecos_mfg_file)
    
    # 2023년 데이터만 사용
    ecos_cpi_2023 = ecos_cpi_df[ecos_cpi_df['year'] == 2023]
    ecos_mfg_2023 = ecos_mfg_df[ecos_mfg_df['year'] == 2023]
    
    print(f"재정자립도 데이터: {len(fiscal_2023)} 건")
    print(f"GDP 데이터: {len(gdp_df)} 건")
    print(f"ECOS CPI 데이터: {len(ecos_cpi_2023)} 건")
    print(f"ECOS 제조업 데이터: {len(ecos_mfg_2023)} 건")
    
    # 데이터 병합
    print("데이터 병합 중...")
    
    # 재정자립도 + GDP
    merged = fiscal_2023.merge(gdp_df, left_on='region', right_on='region', how='inner')
    
    # ECOS 데이터 병합
    merged = merged.merge(ecos_cpi_2023[['region', 'value']].rename(columns={'value': 'cpi'}), 
                         on='region', how='left')
    merged = merged.merge(ecos_mfg_2023[['region', 'value']].rename(columns={'value': 'manufacturing_index'}), 
                         on='region', how='left')
    
    print(f"병합된 데이터: {len(merged)} 건")
    
    # 정규화 (0-1 스케일)
    print("데이터 정규화 중...")
    
    def normalize(series):
        """Min-Max 정규화"""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    # 각 지표 정규화
    merged['fiscal_norm'] = normalize(merged['fiscal_autonomy_ratio'])
    merged['gdp_norm'] = normalize(merged['gdp_2023'])
    
    # ECOS 지표 정규화 (CPI는 낮을수록 좋으므로 역정규화)
    merged['cpi_norm'] = 1 - normalize(merged['cpi'])  # 역정규화
    merged['manufacturing_norm'] = normalize(merged['manufacturing_index'])
    
    # BDS 계산 (가중 평균)
    print("BDS 지수 계산 중...")
    
    # 가중치 설정
    weights = {
        'gdp': 0.30,
        'fiscal': 0.25,
        'cpi': 0.20,
        'manufacturing': 0.25
    }
    
    # BDS 점수 계산 (0-10 스케일)
    merged['bds_score'] = (
        merged['gdp_norm'] * weights['gdp'] +
        merged['fiscal_norm'] * weights['fiscal'] +
        merged['cpi_norm'] * weights['cpi'] +
        merged['manufacturing_norm'] * weights['manufacturing']
    ) * 10.0
    
    # 결과 저장
    print("결과 저장 중...")
    
    # BDS 베이스라인 업데이트
    bds_result = {
        "latestYear": 2023,
        "baselines": {row['region']: float(row['bds_score']) for _, row in merged.iterrows()},
        "data_sources": {
            "kosis_fiscal": "재정자립도 (25%)",
            "kosis_gdp": "지역내총생산 (30%)",
            "ecos_cpi": "소비자물가지수 (20%)",
            "ecos_manufacturing": "제조업생산지수 (25%)"
        },
        "updated": "2025-09-29"
    }
    
    # JSON 파일 저장
    bds_file = Path('data/bds/bds_baseline.json')
    with open(bds_file, 'w', encoding='utf-8') as f:
        json.dump(bds_result, f, ensure_ascii=False, indent=2)
    
    # CSV 파일로도 저장
    bds_csv = merged[['region', 'bds_score', 'fiscal_autonomy_ratio', 'gdp_2023', 'cpi', 'manufacturing_index']].copy()
    bds_csv.to_csv(Path('data/bds/bds_detailed_2023.csv'), index=False)
    
    print(f"BDS 업데이트 완료!")
    print(f"상위 5개 지역:")
    top_regions = merged.nlargest(5, 'bds_score')[['region', 'bds_score']]
    for _, row in top_regions.iterrows():
        print(f"  {row['region']}: {row['bds_score']:.2f}")
    
    print(f"\n하위 5개 지역:")
    bottom_regions = merged.nsmallest(5, 'bds_score')[['region', 'bds_score']]
    for _, row in bottom_regions.iterrows():
        print(f"  {row['region']}: {row['bds_score']:.2f}")

if __name__ == "__main__":
    update_bds_with_ecos()
